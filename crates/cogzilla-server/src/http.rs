use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_kernel::{
    ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram, OutputFormat,
    ResampleMode, SpectralExpr, TerrainEncoding,
};
use cogzilla_proj::mercator::lonlat_deg_to_mercator;
use cogzilla_proj::{TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::{BodyExt, Full};
use hyper::header::{
    ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderName, HeaderValue,
    IF_NONE_MATCH,
};
use hyper::{Method, Request, Response, StatusCode};

mod algorithm_math;
mod algorithm_params;
mod pgstac;
mod array_type;
#[cfg(test)]
mod asset_query_pins;
mod metrics;
mod paint;
mod html;
pub(crate) mod pixel_selection;
mod cql2_text;
mod geo;
mod point_expr;
mod query;
mod server;
mod wms;
mod zarr;
mod zonal;

use query::{ParsedQuery, parse_bool_encoded as parse_bool_param, parse_query, parse_resample_mode};

pub use self::server::{serve, serve_listener, serve_listener_with_shutdown};
pub(crate) use self::geo::{parse_coord_crs_with_epsg, tile_box_intersects_cog};
use self::geo::{
    bbox_to_wgs84, cog_native_epsg_and_bounds, coord_to_lonlat, dataset_native_crs_and_bounds,
    normalize_render_bbox, query_mosaicjson_wgs84_bbox, reproject_wgs84_bounds,
    tile_intersects_cog, wgs84_bboxes_intersect,
};
#[cfg(test)]
use self::geo::cog_standard_epsg;
use self::wms::handle_cog_wms;
#[cfg(test)]
use self::wms::{build_wms_capabilities, parse_wms_format};

use crate::catalog::{MosaicJsonCatalog, SceneCatalog, load_catalog_json_str};
use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};
use self::paint::Paint;
use self::pixel_selection::{Fold, Selected, Selection};
use crate::stac::{
    StacItem, StacSource, get_colormap_image, get_colormap_json, get_tms_json,
    list_colormaps_json, list_tms_json, resolve_stac_assets,
};
use crate::titiler::{
    PolygonMask, StatsOptions, build_cog_info, build_cog_info_geojson, build_cog_validate,
    build_map_html, build_tilejson, build_wmts_capabilities, build_wmts_capabilities_multi,
    compute_f32_band_stats, compute_f32_band_stats_with_engine,
    compute_statistics_with_options,
    encode_auto_image, encode_geotiff_f32, encode_geotiff_rgba, encode_rgba_output,
    get_algorithm_json,
    list_algorithms_json, native_bbox_dims_in_crs, query_point,
    render_crs_bbox, render_mercator_bbox, render_multi_asset_bbox, render_native_preview,
    render_native_preview_stats, render_wgs84_bbox, render_wgs84_bbox_stats,
};

fn percent_decode(input: &str) -> String {
    percent_decode_opts(input, true)
}

/// A query key as Starlette reads it: percent-decoded (`dst%5Fcrs` is `dst_crs`), borrowed
/// when there is nothing to decode.
pub(crate) fn query_key(k: &str) -> std::borrow::Cow<'_, str> {
    if k.contains('%') {
        std::borrow::Cow::Owned(percent_decode(k))
    } else {
        std::borrow::Cow::Borrowed(k)
    }
}

fn percent_decode_opts(input: &str, plus_as_space: bool) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = (bytes[i + 1] as char).to_digit(16);
            let h2 = (bytes[i + 2] as char).to_digit(16);
            if let (Some(a), Some(b)) = (h1, h2) {
                out.push(((a << 4) | b) as u8);
                i += 3;
                continue;
            }
        } else if plus_as_space && bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse an unsigned integer string, also accepting integer-valued decimal strings such as
/// `"256.0"` or `"12.0"` (matching TiTiler / Pydantic coercion).
fn parse_u32_like(s: &str) -> Option<u32> {
    let t = s.trim();
    if let Ok(v) = t.parse::<u32>() {
        return Some(v);
    }
    if t.contains('.')
        && let Ok(f) = t.parse::<f64>()
        && f.is_finite()
        && f >= 0.0
        && f <= u32::MAX as f64
        && f.fract() == 0.0
    {
        return Some(f as u32);
    }
    None
}

fn parse_u16_like(s: &str) -> Option<u16> {
    parse_u32_like(s).and_then(|v| u16::try_from(v).ok())
}

fn parse_u8_like(s: &str) -> Option<u8> {
    parse_u32_like(s).and_then(|v| u8::try_from(v).ok())
}

/// Parse TiTiler's `{y}[@{scale}x][.{ext}]` segment into `(y, scale_opt, ext_opt)`.
fn parse_y_scale_ext(seg: &str) -> Option<(u32, Option<u32>, Option<&str>)> {
    // Handle `1583.0.png` or `1583.0` as well as `1583.png` / `1583@2x.png`.
    let (left, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if !after.is_empty() && after.bytes().all(|b| b.is_ascii_alphabetic() || b == b'3' || b == b'2') && !after.bytes().all(|b| b.is_ascii_digit()) {
            (&seg[..dot], Some(after))
        } else {
            (seg, None)
        }
    } else {
        (seg, None)
    };
    let (y_str, scale) = match left.split_once('@') {
        Some((ys, sc)) => {
            let s_num = parse_u32_like(sc.trim_end_matches('x'))?;
            (ys, Some(s_num))
        }
        None => (left, None),
    };
    let y = parse_u32_like(y_str)?;
    Some((y, scale, ext))
}

fn parse_wxh_ext(seg: &str) -> Result<(u32, u32, Option<&str>), String> {
    let (dims, ext) = match seg.split_once('.') {
        Some((d, e)) => (d, Some(e)),
        None => (seg, None),
    };
    let Some((ws, hs)) = dims.split_once('x') else {
        return Err(format!("invalid dimensions '{seg}', expected {{w}}x{{h}}[.{{ext}}]"));
    };
    let (Some(w), Some(h)) = (parse_u32_like(ws), parse_u32_like(hs)) else {
        return Err(format!("invalid dimensions '{seg}', expected {{w}}x{{h}}[.{{ext}}]"));
    };
    if w == 0 || h == 0 {
        return Err(format!("requested dimensions {w}x{h} must be >= 1x1"));
    }
    if w > 4096 || h > 4096 {
        return Err(format!(
            "requested dimensions {w}x{h} exceed maximum allowed size 4096x4096"
        ));
    }
    Ok((w, h, ext))
}

fn parse_point_coord(coord: &str) -> Option<(f64, f64)> {
    let dec = percent_decode(coord);
    let (lons, lats) = dec.split_once(',')?;
    let x = lons.trim().parse::<f64>().ok()?;
    let y = lats.trim().parse::<f64>().ok()?;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn parse_bbox4(seg: &str) -> Option<([f64; 4], Option<&str>)> {
    // Handle `minx,miny,maxx,maxy.png` or `minx,miny,maxx,maxy` (allowing %20 spaces around commas)
    let (coords_str, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if matches!(
            after,
            "png" | "pngraw" | "webp" | "f32" | "bin" | "raw" | "npy" | "jpg" | "jpeg" | "tif" | "tiff"
        ) {
            (&seg[..dot], Some(after))
        } else {
            (seg, None)
        }
    } else {
        (seg, None)
    };
    let dec = percent_decode(coords_str);
    let parts: Vec<&str> = dec.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let b = [
        parts[0].trim().parse::<f64>().ok()?,
        parts[1].trim().parse::<f64>().ok()?,
        parts[2].trim().parse::<f64>().ok()?,
        parts[3].trim().parse::<f64>().ok()?,
    ];
    if !b.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((b, ext))
}

/// Validate and normalize a parsed bbox `[minx, miny, maxx, maxy]` against its coordinate CRS.
/// - For geographic `coord_crs` (`None` or `CrsModel::Geographic`), `minx > maxx` with longitudes
///   in `[-180, 180]` is unwrapped across the antimeridian per RFC 7946 §5.2 (`maxx += 360.0`),
///   producing the exact same bbox as the `[minx, miny, maxx + 360, maxy]` form.
/// - For projected `coord_crs`, `minx >= maxx` is rejected with `400 Bad Request` stating that
///   `minx` must be less than `maxx`.
fn resolve_input_bbox(
    mut bbox: [f64; 4],
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<[f64; 4], EngineError> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return Err(EngineError::BadRequest(
            "invalid bbox: coordinates must be finite".into(),
        ));
    }
    if bbox[1] >= bbox[3] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: miny ({}) must be less than maxy ({})",
            bbox[1], bbox[3]
        )));
    }
    let is_geo = matches!(coord_crs, None | Some(cogzilla_proj::CrsModel::Geographic));
    if is_geo {
        // Symmetric ±2-turn longitude normalisation (matching point routes) before RFC 7946 §5.2
        // antimeridian unwrap: shift both bounds by the same whole multiple of 360° if needed.
        if (bbox[0] < -180.0 || bbox[0] > 180.0) && (bbox[2] < -180.0 || bbox[2] > 180.0)
            && let (Some(n0), Some(n2)) = (
                cogzilla_proj::normalize_geographic_lon(bbox[0]),
                cogzilla_proj::normalize_geographic_lon(bbox[2]),
            ) {
                let shift = n0 - bbox[0];
                if (bbox[2] + shift - n2).abs() < 1e-9 || (bbox[2] - bbox[0]) <= 360.0 {
                    bbox[0] += shift;
                    bbox[2] += shift;
                }
            }
    }
    if bbox[0] > bbox[2] {
        if is_geo && bbox[0] >= -180.0 && bbox[0] <= 180.0 && bbox[2] >= -180.0 && bbox[2] <= 180.0
        {
            bbox[2] += 360.0;
        } else {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: minx ({}) must be less than maxx ({})",
                bbox[0], bbox[2]
            )));
        }
    } else if bbox[0] == bbox[2] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: minx ({}) must be less than maxx ({})",
            bbox[0], bbox[2]
        )));
    }
    Ok(bbox)
}

/// A `pixel_selection` by name: TiTiler's names in any case, and `std` for `stdev`.
fn parse_pixel_selection(name: &str) -> Option<Selection> {
    let name = name.to_ascii_lowercase();
    Selection::parse(if name == "std" { "stdev" } else { &name })
}

/// Whether a request's values are the raster's own whole numbers, which `mean` and `median`
/// are cast back to; only then does the cast lose anything.
fn reads_whole_numbers(unscale: bool, bands: &BandRequest, sample_type: cogzilla_tiff::SampleType) -> bool {
    !unscale
        && matches!(
            bands,
            BandRequest::Single { .. } | BandRequest::Rgb { .. } | BandRequest::MultiBand { .. }
        )
        && !matches!(
            sample_type,
            cogzilla_tiff::SampleType::Float32 | cogzilla_tiff::SampleType::Float64
        )
}

/// Reject requests that ask for something this server would otherwise ignore.
///
/// TiTiler answers these with a 422. Returning a 200 with the option silently
/// dropped gives a client the wrong picture and no way to notice.
/// Routes that honour `pixel_selection` (either by compositing scenes through `pixel_selection::Fold`
/// or by forwarding the query parameter into generated mosaic tile URL templates). Every other route
/// refuses `?pixel_selection=` with `400 Bad Request` instead of silently ignoring it.
fn route_supports_pixel_selection(segs: &[&str]) -> bool {
    match segs {
        ["mosaic", "tiles", ..] => true,
        ["mosaic", _, "tiles", ..] => true,
        ["mosaicjson", "tiles", .., "assets"] => false,
        ["mosaicjson", "tiles", ..] => true,
        ["mosaicjson", "bbox", _] => true,
        ["mosaicjson", "bbox", _, seg] if *seg != "assets" => true,
        ["mosaicjson", s] if *s == "feature" || s.starts_with("feature.") => true,
        ["mosaicjson", "feature", ..] => true,
        ["mosaicjson", "statistics"] => true,
        ["mosaicjson", "viewer"] => true,
        ["mosaicjson", _, "tilejson.json" | "map.html" | "WMTSCapabilities.xml"]
        | ["mosaicjson", "tilejson.json" | "map.html" | "WMTSCapabilities.xml"] => true,
        _ => false,
    }
}

/// Whether a route applies `algorithm`, as TiTiler's do that take its post-processing:
/// tiles and what points at them, pictures of a box, a feature or the whole, statistics.
fn route_applies_algorithm(segs: &[&str]) -> bool {
    if !matches!(segs.first(), Some(&("cog" | "stac" | "mosaicjson"))) {
        return false;
    }
    let named = |route: &str| segs.get(1).is_some_and(|s| *s == route || s.strip_prefix(route).is_some_and(|ext| ext.starts_with('.')));
    matches!(segs.last(), Some(&("tilejson.json" | "map.html" | "WMTSCapabilities.xml")))
        || (segs.get(1) == Some(&"tiles") && segs.len() > 3)
        || ["preview", "bbox", "feature", "statistics"].iter().any(|r| named(r))
}

fn validate_query(segs: &[&str], pq: &ParsedQuery) -> Result<(), String> {
    let family = segs.first().copied().unwrap_or("");
    if !matches!(family, "cog" | "stac" | "mosaicjson" | "mosaic" | "tiles" | "colorMaps") {
        return Ok(());
    }
    if let Some(e) = pq.errors.first() {
        return Err(e.clone());
    }
    if let Some(alg) = pq.algorithm.as_deref()
        && !crate::algorithms::is_supported_algorithm(alg)
    {
        return Err(format!(
            "algorithm '{alg}' is not supported (supported: {})",
            crate::algorithms::supported_algorithms_display()
        ));
    }

    // {family}/tiles/{tms}[/{z}/{x}/{y}], {family}/{tms}/tilejson.json, {family}/{tms}/map.html,
    // {family}/{tms}/WMTSCapabilities.xml
    if family != "tiles" && family != "colorMaps" {
        let last = segs.last().copied().unwrap_or("");
        let tms = if matches!(last, "tilejson.json" | "map.html" | "WMTSCapabilities.xml")
            && segs.len() >= 3
        {
            Some(segs[segs.len() - 2])
        } else {
            match segs.iter().position(|s| *s == "tiles") {
                Some(i) if last == "assets" => {
                    // `/mosaicjson/tiles/{tms}/{z}/{x}/{y}/assets` has len - i == 6;
                    // `/mosaicjson/tiles/{z}/{x}/{y}/assets` (len - i == 5) has no TMS segment.
                    if segs.len() - i == 6 {
                        Some(segs[i + 1])
                    } else {
                        None
                    }
                }
                Some(i) if matches!(segs.len() - i, 2 | 5) => Some(segs[i + 1]),
                _ => None,
            }
        };
        if let Some(tms) = tms
            && crate::tms::lookup_tms(tms).is_none() {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported (see /tileMatrixSets)"
                ));
            }
    }

    let has = |names: &[&str]| pq.keys.iter().find(|k| names.contains(&k.as_str())).cloned();
    // Only match route sub-endpoints on `/cog/...`, `/stac/...`, and `/mosaicjson/...` so
    // dataset/catalog names like `/tiles/statistics/0/0/0.png` are never misclassified.
    let route = |name: &str| {
        matches!(segs[0], "cog" | "stac" | "mosaicjson")
            && segs
                .get(1)
                .is_some_and(|s| *s == name || s.starts_with(&format!("{name}.")))
    };
    let is_ogc_map =
        matches!(segs[0], "cog" | "stac" | "mosaicjson") && segs.get(1) == Some(&"map");
    let is_tileset_meta = matches!(segs[0], "cog" | "stac" | "mosaicjson")
        && segs.get(1) == Some(&"tiles")
        && segs.len() <= 3;
    if family != "colorMaps"
        && let Some(k) = has(&["orientation"]) {
            return Err(format!("parameter '{k}' is only supported on /colorMaps routes"));
        }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("wms") || route("statistics") || family == "colorMaps")
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, wms, statistics, and colorMaps routes"
            ));
        }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("statistics"))
        && let Some(k) = has(&["dst_crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, and statistics routes"
            ));
        }
    if !(route("bounds") || route("info") || is_ogc_map || route("wms") || is_tileset_meta)
        && let Some(k) = has(&["crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on bounds, info, tiles, map, and wms routes"
            ));
        }
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["algorithm"])
            && !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics"))) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
        if !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics")))
            && let Some(k) = has(&["expression", "expr", "index"]) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
    } else if let Some(k) = has(&["p", "histogram_bins", "histogram_range", "categorical", "c"]) {
        return Err(format!("parameter '{k}' is only supported on statistics routes"));
    }
    if route("point") {
        if let Some(k) = has(&["algorithm"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if !(route("bbox") || route("feature") || route("statistics") || (segs[0] == "mosaicjson" && route("assets")))
        && let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    if route("bbox") && segs.len() >= 4
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    if (route("preview") || route("feature")) && segs.len() >= 3
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    let last_seg = segs.last().copied().unwrap_or("");
    if !matches!(last_seg, "tilejson.json" | "map.html" | "WMTSCapabilities.xml") && !is_tileset_meta {
        if let Some(k) = has(&["minzoom", "maxzoom", "tile_format", "tile_scale"]) {
            return Err(format!(
                "parameter '{k}' is only supported on tilejson.json, map.html, tiles, and WMTSCapabilities.xml routes"
            ));
        }
    } else if let (Some(minz), Some(maxz)) = (pq.minzoom, pq.maxzoom)
        && minz > maxz
    {
        return Err(format!("minzoom ({minz}) must be <= maxzoom ({maxz})"));
    }
    Ok(())
}

/// Apply optional `?width=` / `?height=` overrides onto aspect-preserving default dimensions.
fn resolve_query_dims(default_w: u32, default_h: u32, q_w: Option<u32>, q_h: Option<u32>) -> (u32, u32) {
    match (q_w, q_h) {
        (Some(w), Some(h)) => (w.clamp(1, 4096), h.clamp(1, 4096)),
        (Some(w), None) => {
            let w = w.clamp(1, 4096);
            let h = ((w as f64 * (default_h as f64 / default_w.max(1) as f64)).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, Some(h)) => {
            let h = h.clamp(1, 4096);
            let w = ((h as f64 * (default_w as f64 / default_h.max(1) as f64)).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, None) => (default_w, default_h),
    }
}

/// Strip TileJSON/WMTS-only query parameters so they are not forwarded into tile URL templates.
fn filter_tile_template_query(raw_query: &str) -> String {
    raw_query
        .split('&')
        .filter(|pair| {
            let k = query_key(pair.split('=').next().unwrap_or(""));
            !pair.is_empty()
                && !matches!(
                    k.as_ref(),
                    "minzoom" | "maxzoom" | "tile_format" | "tile_scale" | "use_epsg" | "zooms"
                )
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Apply TiTiler `minzoom`, `maxzoom`, `tile_format`, and `tile_scale` query overrides to a
/// TileJSON document, stripping those TileJSON-only params from the forwarded tile URL template.
fn apply_tilejson_overrides(
    tj: &mut serde_json::Value,
    pq: &ParsedQuery,
    host: &str,
    route_prefix: &str,
    tms: &str,
    raw_query: &str,
) {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let filtered_query: String = raw_query
        .split('&')
        .filter(|pair| {
            let k = query_key(pair.split('=').next().unwrap_or(""));
            !pair.is_empty() && !matches!(k.as_ref(), "minzoom" | "maxzoom" | "tile_format" | "tile_scale")
        })
        .collect::<Vec<_>>()
        .join("&");
    let scale_suffix = match pq.tile_scale {
        Some(sc) if sc > 1 => format!("@{sc}x"),
        _ => String::new(),
    };
    let ext_suffix = pq
        .tile_format
        .as_deref()
        .map(|f| format!(".{f}"))
        .unwrap_or_default();
    let tile_url = if filtered_query.is_empty() {
        format!("{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}")
    } else {
        format!(
            "{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}?{filtered_query}"
        )
    };
    tj["tiles"] = serde_json::json!([tile_url]);
    let cur_min = tj.get("minzoom").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let cur_max = tj.get("maxzoom").and_then(|v| v.as_u64()).unwrap_or(22) as u8;
    let (new_min, new_max) = resolve_zoom_range(cur_min, cur_max, pq.minzoom, pq.maxzoom);
    tj["minzoom"] = serde_json::json!(new_min);
    tj["maxzoom"] = serde_json::json!(new_max);
    if let Some(center) = tj.get_mut("center").and_then(|v| v.as_array_mut())
        && center.len() >= 3
    {
        center[2] = serde_json::json!(new_min);
    }
}

fn resolve_zoom_range(
    def_min: u8,
    def_max: u8,
    req_min: Option<u8>,
    req_max: Option<u8>,
) -> (u8, u8) {
    match (req_min, req_max) {
        (Some(mn), Some(mx)) => (mn, mx.max(mn)),
        (Some(mn), None) => (mn, def_max.max(mn)),
        (None, Some(mx)) => (def_min.min(mx), mx),
        (None, None) => (def_min, def_max),
    }
}

fn apply_info_bounds_crs(
    info: &mut serde_json::Value,
    wgs84_bounds: [f64; 4],
    bounds_crs: Option<&(cogzilla_proj::CrsModel, u16)>,
) -> Result<(), EngineError> {
    if let Some((dst_crs, dst_epsg)) = bounds_crs {
        let proj_bounds = reproject_wgs84_bounds(wgs84_bounds, dst_crs)?;
        info["bounds"] = serde_json::json!(proj_bounds);
        info["crs"] = serde_json::json!(format!(
            "http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"
        ));
    }
    Ok(())
}

const GEOTIFF_CONTENT_TYPE: &str = "image/tiff; application=geotiff";

/// Return the widest `SampleType` across one or more stacked COGs (matching numpy band stacking).
fn widest_sample_type(cogs: &[Arc<cogzilla_io::OpenedCog>]) -> cogzilla_tiff::SampleType {
    cogs.iter()
        .map(|c| c.index.full_res().sample_type)
        .reduce(array_type::wider_type)
        .unwrap_or(cogzilla_tiff::SampleType::Uint8)
}

/// Return true when a `.tif`/`.tiff` request should emit a native IEEE Float32 GeoTIFF
/// (`SampleFormat=3`, `GDAL_NODATA="nan"`) instead of an 8-bit visual RGB/RGBA GeoTIFF.
fn wants_f32_geotiff(
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
) -> bool {
    if ct != GEOTIFF_CONTENT_TYPE {
        return false;
    }
    if !pq.rescale.is_empty()
        || pq.colormap_explicit
        || pq.custom_colormap_json.is_some()
        || pq.color_formula.is_some()
    {
        return false;
    }
    match bands {
        BandRequest::Terrain { .. } => false,
        BandRequest::Dem { algorithm, .. } => !matches!(
            algorithm,
            cogzilla_kernel::DemAlgorithm::Hillshade { .. }
                | cogzilla_kernel::DemAlgorithm::Contours { .. }
        ),
        BandRequest::Single { .. } => {
            pq.unscale
                || (source_cog.is_none_or(|c| c.index.color_map.is_none())
                    && !matches!(sample_type, cogzilla_tiff::SampleType::Uint8))
        }
        BandRequest::Rgb { .. } => {
            pq.unscale || !matches!(sample_type, cogzilla_tiff::SampleType::Uint8)
        }
        BandRequest::CustomExpr { .. }
        | BandRequest::CustomExprRgb { .. }
        | BandRequest::CustomExprMulti { .. }
        | BandRequest::MultiBand { .. }
        | BandRequest::Spectral1 { .. }
        | BandRequest::Spectral2 { .. }
        | BandRequest::Spectral3 { .. }
        | BandRequest::SarRgb { .. } => true,
    }
}

fn geotiff_render_format(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
    allow_f64: bool,
) -> OutputFormat {
    let can_exact_f64 = allow_f64
        && sample_type.needs_f64()
        && !pq.unscale
        && !array_type::shown(pq)
        && pq.algorithm.is_none()
        && pq.expr_raw.is_none()
        && pq.buffer == 0.0
        && matches!(pq.resample, ResampleMode::Nearest | ResampleMode::Bilinear)
        && matches!(pq.reproject, ResampleMode::Nearest | ResampleMode::Bilinear)
        && matches!(
            bands,
            BandRequest::Single { .. } | BandRequest::Rgb { .. } | BandRequest::MultiBand { .. }
        );
    if wants_f32_geotiff(ct, pq, source_cog, bands, sample_type) {
        if can_exact_f64 {
            OutputFormat::RawF64
        } else {
            OutputFormat::RawF32
        }
    } else if fmt == OutputFormat::Npy && can_exact_f64 {
        OutputFormat::NpyF64
    } else {
        fmt
    }
}

/// Return true when an unrescaled `Uint16` `.png` request should emit a 16-bit PNG
/// matching TiTiler / GDAL's PNG driver instead of downscaling to 8-bit.
fn wants_u16_png(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
) -> bool {
    fmt == OutputFormat::Png
        && ct == "image/png"
        && sample_type == cogzilla_tiff::SampleType::Uint16
        && !pq.unscale
        && pq.rescale.is_empty()
        && !pq.colormap_explicit
        && pq.custom_colormap_json.is_none()
        && pq.color_formula.is_none()
        && pq.algorithm.is_none()
        && pq.expr_raw.is_none()
        && !source_cog.is_some_and(|c| has_embedded_palette(pq, c, bands))
        && matches!(
            bands,
            BandRequest::Single { .. } | BandRequest::Rgb { .. } | BandRequest::MultiBand { .. }
        )
        && (TileEngine::value_bands(bands) + usize::from(pq.return_mask)) <= 4
}

fn render_output_format(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_tiff::SampleType,
    allow_f64: bool,
) -> OutputFormat {
    if wants_u16_png(fmt, ct, pq, source_cog, bands, sample_type) {
        OutputFormat::RawF32
    } else {
        geotiff_render_format(fmt, ct, pq, source_cog, bands, sample_type, allow_f64)
    }
}

fn wrap_geotiff_fmt(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    w: u32,
    h: u32,
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    if ct == GEOTIFF_CONTENT_TYPE {
        if matches!(fmt, OutputFormat::RawF64) {
            let n_px = (w as usize).saturating_mul(h as usize).max(1);
            let n_bands = (data.len() / (n_px * 8)).max(1);
            let vals: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            crate::geotiff::write_geotiff_typed_f64(
                &vals,
                n_bands,
                w,
                h,
                cogzilla_tiff::SampleType::Float64,
                bbox,
                epsg,
                return_mask,
            )
        } else if matches!(fmt, OutputFormat::RawF32) {
            encode_geotiff_f32(w, h, &data, bbox, epsg, return_mask)
        } else {
            encode_geotiff_rgba(w, h, &data, bbox, epsg, return_mask)
        }
    } else if ct == "image/png" && matches!(fmt, OutputFormat::RawF32) {
        if let Some(n_px) = (w as usize).checked_mul(h as usize).filter(|&n| n > 0)
            && let Some(row_bytes) = n_px.checked_mul(4)
        {
            let n_bands = data.len() / row_bytes;
            if n_bands >= 1
                && data.len() == n_bands * row_bytes
                && (n_bands + usize::from(return_mask)) <= 4
            {
                let vals: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                return pgstac::png_of_16_bits(&vals, n_bands, w as usize, h as usize, return_mask);
            }
        }
        debug_assert!(
            false,
            "wrap_geotiff_fmt: unexpected RawF32 buffer length {} for {}x{} 16-bit PNG",
            data.len(),
            w,
            h
        );
        pgstac::png_of_16_bits(&[], 1, w.max(1) as usize, h.max(1) as usize, return_mask)
    } else {
        data
    }
}

/// Whether `data` may be an array the renderers wrote in float32: a float32 `.npy`, or a
/// GeoTIFF (whose samples [`array_in_its_type`] reads before it changes anything).
fn may_be_f32_array(data: &[u8]) -> bool {
    let npy = data.starts_with(b"\x93NUMPY") && data.get(10..80).is_some_and(|dict| dict.windows(4).any(|w| w == b"<f4'"));
    npy || data.starts_with(b"II*\0")
}

/// An array the renderers wrote in float32, as `.npy` or as a GeoTIFF, rewritten in `kind`,
/// the type TiTiler writes it in ([`array_type`]); with `mask`, its last band is the mask.
/// Left as it is without a type, or when it is not a float32 array of the bands it says.
fn array_in_its_type(data: Bytes, kind: Option<cogzilla_tiff::SampleType>, mask: bool) -> Bytes {
    array_in_its_type_rounded(data, kind, mask, None)
}

/// [`array_in_its_type`], each value first rounded by `rounding` and put in `kind` as
/// numpy's `astype` puts it ([`array_type::astype`]): what `cast`, `ceil` and `floor` do.
fn array_in_its_type_rounded(
    data: Bytes,
    kind: Option<cogzilla_tiff::SampleType>,
    mask: bool,
    rounding: Option<fn(f64) -> f64>,
) -> Bytes {
    let Some(kind) = kind else { return data };
    let tif = data.starts_with(b"II*\0");
    let typed = match rounding {
        None if tif => crate::geotiff::retype_f32_geotiff(&data, kind, mask),
        None => cogzilla_kernel::retype_f32_npy(&data, kind, mask),
        Some(round) => {
            let prepare = |v: f64| array_type::astype(kind, round(v));
            if tif {
                crate::geotiff::retype_f32_geotiff_with(&data, kind, mask, prepare)
            } else {
                cogzilla_kernel::retype_f32_npy_with(&data, kind, mask, prepare)
            }
        }
    };
    typed.unwrap_or(data)
}

/// What a `/cog` or `/stac` array is decided by.
#[derive(Clone)]
enum ArrayAsked {
    /// The values in their type: the rule, the assets asked for (none on `/cog`), the band
    /// math, whether a mask was asked for, and the assets' own band math and the request
    /// expression over it, typed one after the other (`ParsedQuery::asset_math_stages`).
    Typed(array_type::ArrayRule, Vec<String>, Option<String>, bool, Option<(Vec<Option<String>>, String)>),
    /// The picture's bytes, for a `.npy` asked with a rescale, a colour map or a colour
    /// formula: the request, which says how they are painted.
    Shown(Box<ParsedQuery>),
}

/// `pq` with its assets' own band math folded into the request expression as
/// [`expand_stac_asset_bidx`] folds it, but with each asset's bands counted from one: `None`
/// when no asset has band math, or the request expression is not over its results.
fn fold_asset_math_for_typing(pq: &ParsedQuery) -> Option<ParsedQuery> {
    if pq.asset_exprs.is_empty() {
        return None;
    }
    let names: Vec<String> = pq.assets.iter().map(|a| a.split('|').next().unwrap_or(a).to_string()).collect();
    let own: Vec<Option<String>> = stac_entry_options(pq, &names).into_iter().map(|(own, _)| own).collect();
    let mut folded = pq.clone();
    // The array is typed as numpy types it: each asset's band math over its own raster, the
    // results stacked, then the request expression, if any, over them.
    folded.expr_raw = Some(pq.expr_raw.clone().unwrap_or_else(|| {
        own.iter().map(|e| e.as_deref().unwrap_or("b1")).collect::<Vec<_>>().join(";")
    }));
    folded.asset_math_stages = Some((own, pq.expr_raw.clone().unwrap_or_default()));
    folded.asset_exprs.clear();
    Some(folded)
}

/// What decides the array of a request whose path ends in `last`.
fn array_asked(pq: &ParsedQuery, last: Option<&&str>) -> Option<ArrayAsked> {
    // The assets' own band math is typed as the one expression it becomes over the stack
    // (`expand_stac_asset_bidx`): which band each part reads does not change its type.
    let folded;
    let pq = match fold_asset_math_for_typing(pq) {
        Some(f) => {
            folded = f;
            &folded
        }
        None => pq,
    };
    if let Some(rule) = array_type::ArrayRule::of(pq) {
        return Some(ArrayAsked::Typed(rule, pq.assets.clone(), pq.expr_raw.clone(), pq.return_mask, pq.asset_math_stages.clone()));
    }
    let npy = last.is_some_and(|last| last.ends_with(".npy")) || pq.format_param.as_deref() == Some("npy");
    let shown = npy && array_type::shown(pq);
    shown.then(|| ArrayAsked::Shown(Box::new(pq.clone())))
}

/// A float32 `.npy` of raw values painted as the request paints them ([`paint_selected`]),
/// as TiTiler writes an array asked with a rescale, a colour map or a colour formula: in
/// bytes, preserving all bands for multi-band rescaled arrays (or 1 grey / 3 RGB bands when
/// painted through a colour map or colour formula), then the mask (`255` where a pixel is
/// there) when it was asked for. `first` is the raster read first, for a colour map of its
/// own. `None` when `data` is not such an array.
fn shown_array(pq: &ParsedQuery, first: Option<&cogzilla_io::OpenedCog>, data: &[u8]) -> Option<Bytes> {
    let (shape, start) = cogzilla_kernel::f32_npy_shape(data)?;
    let [planes, height, width] = shape[..] else { return None };
    let pixels = height.checked_mul(width)?;
    let mask = pq.return_mask;
    let bands = planes.checked_sub(usize::from(mask)).filter(|b| *b > 0)?;
    let byte_len = planes.checked_mul(pixels)?.checked_mul(4)?;
    let end = start.checked_add(byte_len)?;
    let body = data.get(start..end)?;
    let floats: Vec<f32> = body.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
    let (values, there) = floats.split_at(bands * pixels);
    // A pixel that is not there is no value, whatever the renderer filled it with.
    let values: Vec<f32> = if mask {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let m = there[i % pixels];
                if !m.is_finite() || m == 0.0 { f32::NAN } else { *v }
            })
            .collect()
    } else {
        values.to_vec()
    };
    let request = if bands == 1 && pq.expr_raw.is_none() && pq.asset_exprs.is_empty() {
        let rescale = pq.rescale.first().copied().unwrap_or((0.0, 255.0));
        BandRequest::Single { band_1based: 1, rescale, colormap: pq.colormap }
    } else {
        BandRequest::MultiBand { bands_1based: (1..=bands as u16).collect() }
    };
    let custom = match first {
        Some(first) => resolve_custom_lut(pq, first, &request).ok()?,
        None => pq
            .custom_colormap_json
            .as_deref()
            .map(|json| CustomColormap::from_json_str_with_type(json, !pq.rescale.is_empty(), pq.colormap_type_linear))
            .transpose()
            .ok()?,
    };
    // Preserve all N bands for 2-band and 4+-band rescaled arrays when no colormap/formula is active.
    if bands != 1
        && bands != 3
        && pq.colormap == ColormapName::Grayscale
        && custom.is_none()
        && pq.color_formula.is_none()
    {
        let out_planes = bands + usize::from(mask);
        let hdr = cogzilla_kernel::encode_npy_header("|u1", &[out_planes, height, width]);
        let mut out = Vec::with_capacity(hdr.len() + out_planes * pixels);
        out.extend_from_slice(&hdr);
        let mut valid = if mask { vec![0u8; pixels] } else { Vec::new() };
        for b in 0..bands {
            let (min, max) = pq.rescale.get(b).or(pq.rescale.first()).copied().unwrap_or((0.0, 255.0));
            let plane = &values[b * pixels..(b + 1) * pixels];
            for (p, &v) in plane.iter().enumerate() {
                if v.is_finite() {
                    out.push(paint::rescale_byte(v, min, max));
                    if mask {
                        valid[p] = 255;
                    }
                } else {
                    out.push(0);
                }
            }
        }
        if mask {
            out.extend_from_slice(&valid);
        }
        return Some(Bytes::from(out));
    }
    let selected = pixel_selection::Selected { bands, values };
    let grey = bands < 3 && pq.colormap == ColormapName::Grayscale && custom.is_none();
    let rgba = paint_selected(selected, pixels, &request, pq, custom.as_ref()).ok()?;
    let shown = if grey { 1 } else { 3 };
    let planes = shown + usize::from(mask);
    let hdr = cogzilla_kernel::encode_npy_header("|u1", &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels);
    out.extend_from_slice(&hdr);
    for band in 0..shown {
        out.extend(rgba.chunks_exact(4).map(|px| px[band]));
    }
    if mask {
        out.extend(rgba.chunks_exact(4).map(|px| if px[3] == 0 { 0 } else { 255 }));
    }
    Some(Bytes::from(out))
}

/// Apply `ArrayAsked` (typed `.npy`/`.tif` or painted `|u1` `.npy`) over already-opened `cogs`.
/// Single shared entry point used by `/cog`, `/stac`, and `/mosaicjson` so all three route
/// families produce identical dtypes, mask sentinels, and GeoTIFF nodata metadata.
fn apply_array_asked(
    asked: Option<ArrayAsked>,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    expression_override: Option<&str>,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    if cogs.is_empty() {
        return data;
    }
    match asked {
        ArrayAsked::Typed(rule, _, expression, mask, stages) => {
            let reads: Vec<_> = cogs.iter().map(|cog| cog.index.full_res().sample_type).collect();
            let kind = match &stages {
                Some((own, request)) if own.len() == reads.len() => {
                    rule.staged_kind(&reads, own, request)
                }
                _ => rule.kind(&reads, expression_override.or(expression.as_deref())),
            };
            let rounding = kind.and_then(|kind| rule.rounding(kind));
            array_in_its_type_rounded(data, kind, mask, rounding)
        }
        ArrayAsked::Shown(pq) => {
            shown_array(&pq, cogs.first().map(|cog| &**cog), &data).unwrap_or(data)
        }
    }
}

/// A `/cog` answer in the type TiTiler writes its array in: the rasters at `urls` are
/// opened again (they are cached) for their types. Anything but a float32 array, and an
/// array whose rasters cannot be opened, is left as it is.
async fn cog_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    let Ok(cogs) = open_query_urls(engine, urls, None).await else {
        return data;
    };
    apply_array_asked(Some(asked), &cogs, None, data)
}

/// A `/stac` answer in the type TiTiler writes its array in, from the types of the item's
/// assets that were read and the band math as it was rewritten over them. As
/// [`cog_array_in_its_type`].
async fn stac_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    item: &crate::stac::StacItem,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    let (assets, expression) = match &asked {
        ArrayAsked::Typed(_, assets, expression, _, _) => (assets.clone(), expression.clone()),
        ArrayAsked::Shown(pq) => (pq.assets.clone(), pq.expr_raw.clone()),
    };
    let Ok((_, cogs, expression)) =
        resolve_stac_assets(engine, item, &assets, expression.as_deref(), None).await
    else {
        return data;
    };
    apply_array_asked(Some(asked), &cogs, expression.as_deref(), data)
}

/// A `/mosaicjson` (or `/mosaic`) answer in the type TiTiler writes its array in, using the
/// exact same [`apply_array_asked`] writer as `/cog` and `/stac`. Zero overhead for non-array
/// image formats (short-circuits immediately on `!may_be_f32_array(&data)`).
async fn mosaic_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    selection: pixel_selection::Selection,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    if let ArrayAsked::Typed(_, _, _, mask, _) = &asked {
        match selection {
            pixel_selection::Selection::Stdev => {
                return array_in_its_type(data, Some(cogzilla_tiff::SampleType::Float32), *mask);
            }
            pixel_selection::Selection::Count => {
                let kind = if urls.len().min(64) <= 255 {
                    cogzilla_tiff::SampleType::Uint8
                } else {
                    cogzilla_tiff::SampleType::Uint16
                };
                return array_in_its_type(data, Some(kind), *mask);
            }
            _ => {}
        }
    }
    let cap = urls.len().min(64);
    let mut cogs = Vec::with_capacity(cap);
    for u in urls.iter().take(cap) {
        if let Ok(cog) = engine.store.open_uri(u).await {
            cogs.push(cog);
        }
    }
    apply_array_asked(Some(asked), &cogs, None, data)
}

/// A rendered `/cog` tile, its array in its type: what the tile cache keeps.
async fn cog_tile_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    rendered: impl std::future::Future<Output = Result<(Bytes, &'static str), EngineError>>,
) -> Result<(Bytes, &'static str), EngineError> {
    let (data, ct) = rendered.await?;
    Ok((cog_array_in_its_type(engine, asked, urls, data).await, ct))
}

/// A rendered `/stac` tile, its array in its type: what the tile cache keeps. The item is
/// the one the tile was drawn from, named by `url` or read from pgSTAC.
async fn stac_tile_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    source: StacSource,
    rendered: impl std::future::Future<Output = Result<(Bytes, &'static str), EngineError>>,
) -> Result<(Bytes, &'static str), EngineError> {
    let (data, ct) = rendered.await?;
    if asked.is_none() || !may_be_f32_array(&data) {
        return Ok((data, ct));
    }
    match source.load(engine).await {
        Ok(item) => Ok((stac_array_in_its_type(engine, asked, &item, data).await, ct)),
        Err(_) => Ok((data, ct)),
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_image_output(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    auto_fmt: bool,
    out_w: u32,
    out_h: u32,
    bbox: [f64; 4],
    epsg: u16,
    jpeg_quality: u8,
    return_mask: bool,
) -> (Bytes, &'static str) {
    if auto_fmt {
        encode_auto_image(out_w as usize, out_h as usize, &data, jpeg_quality, return_mask)
    } else {
        (wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, bbox, epsg, return_mask), ct)
    }
}

fn attach_x_assets(resp: &mut Response<Full<Bytes>>, assets: &[String]) {
    if assets.is_empty() || !resp.status().is_success() {
        return;
    }
    let joined = assets
        .iter()
        .map(|u| cogzilla_io::redact_http_url(u))
        .collect::<Vec<_>>()
        .join(",");
    if let Ok(hv) = hyper::header::HeaderValue::from_str(&joined) {
        resp.headers_mut().insert("x-assets", hv);
    }
}

fn attach_x_failed_assets(resp: &mut Response<Full<Bytes>>, failed: &[String]) {
    if failed.is_empty() || !resp.status().is_success() {
        return;
    }
    let joined = failed
        .iter()
        .map(|u| cogzilla_io::redact_http_url(u))
        .collect::<Vec<_>>()
        .join(",");
    if let Ok(hv) = hyper::header::HeaderValue::from_str(&joined) {
        resp.headers_mut().insert("x-failed-assets", hv);
    }
}

fn parse_ext_format(
    ext: &str,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    match ext {
        "png" | "pngraw" | "" => Ok((OutputFormat::Png, "image/png")),
        "tif" | "tiff" => Ok((OutputFormat::RawRgba, GEOTIFF_CONTENT_TYPE)),
        "webp" => Ok((
            OutputFormat::Webp {
                quality: webp_quality,
            },
            "image/webp",
        )),
        "jpg" | "jpeg" => Ok((OutputFormat::Jpeg { quality: jpeg_quality }, "image/jpeg")),
        "f32" | "bin" | "raw" => Ok((OutputFormat::RawF32, "application/octet-stream")),
        "npy" => Ok((OutputFormat::Npy, "application/x-binary")),
        other => Err(EngineError::BadRequest(format!(
            "unsupported format '.{other}' (use .png, .webp, .jpg, or .f32)"
        ))),
    }
}

/// Fold GDAL `scale`/`offset` into rescale bounds when `?unscale=true` (zero per-pixel cost!).
/// Supports both single multi-band COGs and multi-asset STAC/multi-COG composites.
/// `slot_bands`: the band each of several `cogs` is read for, when not its first
/// ([`expand_stac_asset_bidx`]), whose scale and offset are then that band's.
fn apply_unscale_cogs(bands: &mut BandRequest, cogs: &[Arc<cogzilla_io::OpenedCog>], slot_bands: Option<&[u16]>) {
    if cogs.is_empty() {
        return;
    }
    let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
        let i = b1.saturating_sub(1) as usize;
        let (sc, off) = if cogs.len() > 1 {
            let slot = i.min(cogs.len() - 1);
            let c = &cogs[slot];
            let band = slot_bands.and_then(|s| s.get(slot)).map_or(0, |b| usize::from(b.saturating_sub(1)));
            (
                c.index.scales.get(band).copied().unwrap_or(1.0),
                c.index.offsets.get(band).copied().unwrap_or(0.0),
            )
        } else {
            (
                cogs[0].index.scales.get(i).copied().unwrap_or(1.0),
                cogs[0].index.offsets.get(i).copied().unwrap_or(0.0),
            )
        };
        if (sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9 || sc.abs() < 1e-12 {
            (mn, mx)
        } else {
            ((mn - off) / sc, (mx - off) / sc)
        }
    };
    match bands {
        BandRequest::Single {
            band_1based,
            rescale,
            ..
        } => {
            *rescale = adj(*band_1based, rescale.0, rescale.1);
        }
        BandRequest::Rgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            for k in 0..3 {
                let (mn, mx) = adj(bands_1based[k], rescale_min[k], rescale_max[k]);
                rescale_min[k] = mn;
                rescale_max[k] = mx;
            }
        }
        _ => {}
    }
}

/// Compute aspect-preserving `(width, height)` inside `max_dim` for format-only `/bbox` and
/// `POST /feature` routes (matching TiTiler instead of stretching into a fixed square).
fn aspect_preserving_dims(bbox: [f64; 4], max_dim: u32) -> (u32, u32) {
    let max_d = max_dim.clamp(1, 4096) as f64;
    let dx = (bbox[2] - bbox[0]).abs();
    let dy = (bbox[3] - bbox[1]).abs();
    if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
        let d = max_dim.clamp(1, 4096);
        return (d, d);
    }
    if dx >= dy {
        let w = max_d.round() as u32;
        let h = ((max_d * (dy / dx)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w.max(1), h)
    } else {
        let h = max_d.round() as u32;
        let w = ((max_d * (dx / dy)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w, h.max(1))
    }
}

/// Compute native-resolution `(width, height)` for `dst_bbox` capped at `max_dim`,
/// matching TiTiler / GDAL `WarpedVRT` without upsampling past native pixel resolution.
fn capped_native_bbox_dims(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    dst_crs: &cogzilla_proj::CrsModel,
    wgs_bbox: [f64; 4],
    dst_bbox: [f64; 4],
    max_dim: u32,
) -> (u32, u32) {
    let cap = max_dim.clamp(1, 4096);
    let Ok(crs) = engine.get_or_build_crs(cog) else {
        return aspect_preserving_dims(dst_bbox, cap);
    };
    let (nat_w, nat_h) = native_bbox_dims_in_crs(cog.index.full_res(), &crs, dst_crs, wgs_bbox, dst_bbox);
    if nat_w.max(nat_h) <= cap {
        (nat_w.max(1), nat_h.max(1))
    } else if nat_w >= nat_h {
        let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (cap, h)
    } else {
        let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (w, cap)
    }
}

/// Default rescale range matching TiTiler's `dtype_ranges` when `?rescale=` is omitted.
fn default_rescale_for_dtype(st: cogzilla_tiff::SampleType, single: bool) -> (f32, f32) {
    use cogzilla_tiff::SampleType;
    match st {
        SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint8 => (0.0, 255.0),
        SampleType::Int8 => (-128.0, 127.0),
        SampleType::Uint16 => (0.0, 65535.0),
        SampleType::Int16 => (-32768.0, 32767.0),
        SampleType::Uint32 => (0.0, 4_294_967_295.0),
        SampleType::Int32 => (-2_147_483_648.0, 2_147_483_647.0),
        SampleType::Uint64 => (0.0, u64::MAX as f32),
        SampleType::Int64 => (i64::MIN as f32, i64::MAX as f32),
        SampleType::Float32 | SampleType::Float64 => {
            if single {
                (0.0, 1000.0)
            } else {
                (0.0, 1.0)
            }
        }
    }
}

fn build_band_request(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
) -> Result<BandRequest, EngineError> {
    if let Some(def) = pq.algorithm.as_deref().and_then(crate::algorithms::lookup_algorithm) {
        use crate::algorithms::AlgorithmId;
        let band_1based = pq.bidx.first().copied().unwrap_or(1);
        match def.id {
            AlgorithmId::Terrarium => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::Terrarium,
                });
            }
            AlgorithmId::TerrainRgb => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::TerrainRgb,
                });
            }
            AlgorithmId::Hillshade => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 255.0));
                let colormap = if pq.colormap_explicit {
                    pq.colormap
                } else {
                    ColormapName::from_str_opt("gray").unwrap_or(ColormapName::Viridis)
                };
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Hillshade {
                        azimuth_deg: pq.azimuth,
                        altitude_deg: pq.altitude,
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap,
                });
            }
            AlgorithmId::Slope => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 90.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Slope {
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap: pq.colormap,
                });
            }
            AlgorithmId::Aspect => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 360.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Aspect,
                    rescale,
                    colormap: pq.colormap,
                });
            }
            AlgorithmId::Contours => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    encoding: TerrainEncoding::Contours {
                        increment: (pq.contour_increment.round() as i32).max(1),
                        thickness: pq.contour_thickness.min(255) as u8,
                        minz: pq.contour_minz.round() as i32,
                        maxz: pq.contour_maxz.round() as i32,
                    },
                });
            }
            AlgorithmId::NormalizedIndex
            | AlgorithmId::Cast
            | AlgorithmId::Ceil
            | AlgorithmId::Floor
            | AlgorithmId::Min
            | AlgorithmId::Max
            | AlgorithmId::Median
            | AlgorithmId::Mean
            | AlgorithmId::Std
            | AlgorithmId::Var
            | AlgorithmId::Sum
            | AlgorithmId::Grayscale
            | AlgorithmId::Bitonal => {
                let alg_lower = def.name.to_ascii_lowercase();
                let alg = alg_lower.as_str();
                let bands: Vec<u16> = if !pq.bidx.is_empty() {
                    pq.bidx.clone()
                } else {
                    (1..=default_bands.clamp(1, 16)).collect()
                };
                let whole = algorithm_math::whole_numbers(sample_type, pq.unscale);
                let source_range = default_rescale_for_dtype(sample_type, true);
                if matches!(def.id, AlgorithmId::Cast | AlgorithmId::Ceil | AlgorithmId::Floor) && bands.len() >= 3 {
                    // Each of the first three bands, as an RGB picture.
                    let one = |b: u16| -> Result<(ExprProgram, (f32, f32)), EngineError> {
                        let (expr, range) = algorithm_math::integer_band(alg, b, whole, pq.cast_dtype.as_deref());
                        Ok((ExprProgram::compile(&expr).map_err(EngineError::BadRequest)?, range.unwrap_or(source_range)))
                    };
                    let ((p0, dr), (p1, _), (p2, _)) = (one(bands[0])?, one(bands[1])?, one(bands[2])?);
                    let r0 = pq.rescale.first().copied().unwrap_or(dr);
                    let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
                    let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
                    return Ok(BandRequest::CustomExprRgb {
                        progs: [p0, p1, p2],
                        rescale_min: [r0.0, r1.0, r2.0],
                        rescale_max: [r0.1, r1.1, r2.1],
                    });
                }
                let expr_str = algorithm_math::band_math(alg, &bands, whole, pq.cast_dtype.as_deref(), pq.threshold)
                    .map_err(EngineError::BadRequest)?;
                let default_rescale = match def.id {
                    AlgorithmId::NormalizedIndex => (-1.0, 1.0),
                    AlgorithmId::Cast | AlgorithmId::Ceil | AlgorithmId::Floor => {
                        algorithm_math::integer_band(alg, bands[0], whole, pq.cast_dtype.as_deref()).1.unwrap_or(source_range)
                    }
                    AlgorithmId::Bitonal => (0.0, 255.0),
                    AlgorithmId::Sum => (source_range.0, source_range.1 * bands.len() as f32),
                    _ => source_range,
                };
                let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
                let rescale = pq.rescale.first().copied().unwrap_or(default_rescale);
                return Ok(BandRequest::CustomExpr {
                    prog,
                    rescale,
                    colormap: pq.colormap,
                });
            }
        }
    }
    if let Some(raw_expr) = &pq.expr_raw {
        if let Some(preset) = SpectralExpr::from_str_opt(raw_expr) {
            return Ok(match preset {
                SpectralExpr::Db => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-25.0, 0.0));
                    let b = pq.bidx.first().copied().unwrap_or(1);
                    BandRequest::Spectral1 {
                        expr: preset,
                        band_1based: b,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                SpectralExpr::SarRgb => {
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    let (rmin, rmax) = match pq.rescale.len() {
                        0 => ([-20.0, -25.0, 0.0], [0.0, -5.0, 15.0]),
                        1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                        3 => (
                            [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                            [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                        ),
                        n => {
                            return Err(EngineError::BadRequest(format!(
                                "expected 1 or 3 rescale pairs for 3-band SAR RGB output, got {n}"
                            )));
                        }
                    };
                    BandRequest::SarRgb {
                        bands_1based: bs,
                        rescale_min: rmin,
                        rescale_max: rmax,
                    }
                }
                SpectralExpr::Evi => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-0.2, 0.8));
                    let bs = if pq.bidx.len() >= 3 {
                        [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
                    } else {
                        [1, 2, 3]
                    };
                    BandRequest::Spectral3 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                _ => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    BandRequest::Spectral2 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
            });
        }
        // Arbitrary TiTiler expression -> compile to bytecode VM!
        // Supports single expressions (`(b1-b2)/(b1+b2)`) and `;`-separated 3-band RGB
        // expressions (`b3*2;b2*2;b1*2`).
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(EngineError::BadRequest("empty expression".into()));
        }
        if parts.len() == 3 {
            if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
                return Err(EngineError::BadRequest(
                    "colormap cannot be applied to 3-band RGB output".into(),
                ));
            }
            let p0 = ExprProgram::compile(parts[0]).map_err(EngineError::BadRequest)?;
            let p1 = ExprProgram::compile(parts[1]).map_err(EngineError::BadRequest)?;
            let p2 = ExprProgram::compile(parts[2]).map_err(EngineError::BadRequest)?;
            let (rmin, rmax) = match pq.rescale.len() {
                0 => {
                    let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                    ([lo; 3], [hi; 3])
                }
                1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                3 => (
                    [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                    [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                ),
                n => {
                    return Err(EngineError::BadRequest(format!(
                        "rescale count ({n}) does not match band count (3); expected 1 or 3 rescale ranges"
                    )));
                }
            };
            return Ok(BandRequest::CustomExprRgb {
                progs: [p0, p1, p2],
                rescale_min: rmin,
                rescale_max: rmax,
            });
        } else if parts.len() != 1 {
            if parts.len() > 15 {
                return Err(EngineError::BadRequest(format!(
                    "multi-band tile expression count ({}) exceeds maximum of 15",
                    parts.len()
                )));
            }
            if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
                return Err(EngineError::BadRequest(
                    "colormap cannot be applied to multi-band expression output".into(),
                ));
            }
            let mut progs = Vec::with_capacity(parts.len());
            for p in &parts {
                progs.push(ExprProgram::compile(p).map_err(EngineError::BadRequest)?);
            }
            return Ok(BandRequest::CustomExprMulti { progs });
        }
        if pq.rescale.len() > 1 {
            return Err(EngineError::BadRequest(format!(
                "rescale count ({}) does not match band count (1); expected 1 rescale range",
                pq.rescale.len()
            )));
        }
        let prog = ExprProgram::compile(parts[0]).map_err(EngineError::BadRequest)?;
        let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
        return Ok(BandRequest::CustomExpr {
            prog,
            rescale,
            colormap: pq.colormap,
        });
    }

    if !pq.bidx.is_empty() && pq.bidx.len() != 1 && pq.bidx.len() != 3 {
        if pq.bidx.len() > 15 {
            return Err(EngineError::BadRequest(format!(
                "bidx count {} exceeds maximum of 15 bands",
                pq.bidx.len()
            )));
        }
        return Ok(BandRequest::MultiBand {
            bands_1based: pq.bidx.clone(),
        });
    }

    if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
        if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
            return Err(EngineError::BadRequest(
                "colormap cannot be applied to 3-band RGB output".into(),
            ));
        }
        let bs = if pq.bidx.len() == 3 {
            [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
        } else {
            [1, 2, 3]
        };
        let (rmin, rmax) = match pq.rescale.len() {
            0 => {
                let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                ([lo; 3], [hi; 3])
            }
            1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
            3 => (
                [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
            ),
            n => {
                return Err(EngineError::BadRequest(format!(
                    "rescale count ({n}) does not match band count (3); expected 1 or 3 rescale ranges"
                )));
            }
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        if pq.rescale.len() > 1 {
            return Err(EngineError::BadRequest(format!(
                "expected 1 rescale pair for single-band output, got {}",
                pq.rescale.len()
            )));
        }
        let b = pq.bidx.first().copied().unwrap_or(1);
        let default_range = default_rescale_for_dtype(sample_type, true);
        let rescale = pq.rescale.first().copied().unwrap_or(default_range);
        Ok(BandRequest::Single {
            band_1based: b,
            rescale,
            colormap: pq.colormap,
        })
    }
}

/// Match TiTiler / rio-tiler resampling semantics: when the source CRS matches the target CRS,
/// `resampling` applies; when reprojecting across CRSes, `reproject` applies (both default to
/// `Nearest`). Uses `engine.get_or_build_crs` (cached) to avoid rebuilding `CrsModel` per tile.
fn resolve_resample(
    engine: Option<&TileEngine>,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    target_epsg: u16,
) -> ResampleMode {
    let Some(cog) = source_cog else {
        return pq.resample;
    };
    let gk = &cog.index.geo_keys;
    let is_projected = gk.model_type == Some(1)
        || gk.projected_cs_type.is_some()
        || gk.proj_coord_trans.is_some();
    if (target_epsg != 32767 && gk.projected_cs_type == Some(target_epsg))
        || (!is_projected && gk.geographic_type == Some(target_epsg))
    {
        return pq.resample;
    }
    let crs_res = if let Some(eng) = engine {
        eng.get_or_build_crs(cog).ok()
    } else {
        cogzilla_proj::CrsModel::from_geo_keys(gk).ok()
    };
    let same_crs = match crs_res {
        Some(cogzilla_proj::CrsModel::Epsg3857) => target_epsg == 3857,
        Some(cogzilla_proj::CrsModel::Geographic) => matches!(
            cogzilla_proj::CrsModel::from_epsg(target_epsg),
            Ok(cogzilla_proj::CrsModel::Geographic)
        ),
        _ => false,
    };
    if same_crs { pq.resample } else { pq.reproject }
}

fn build_ogc_tileset_json(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    tms_def: &crate::tms::TmsDef,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
    minzoom: Option<u8>,
    maxzoom: Option<u8>,
) -> serde_json::Value {
    let suffix = if !forwarded_prefix.is_empty() && !host.ends_with(forwarded_prefix) {
        forwarded_prefix
    } else {
        ""
    };
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{suffix}")
    } else {
        format!("http://{host}{suffix}")
    };
    let q_suffix = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tile_q: String = filter_tile_template_query(raw_query)
        .split('&')
        .filter(|pair| !pair.is_empty() && query_key(pair.split('=').next().unwrap_or("")) != "crs")
        .collect::<Vec<_>>()
        .join("&");
    let tile_q_suffix = if tile_q.is_empty() {
        String::new()
    } else {
        format!("?{tile_q}")
    };
    let [w, s, e, n] = bounds.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    let limits: Vec<serde_json::Value> = tms_def
        .matrices
        .iter()
        .filter(|m| {
            let Ok(z) = m.id.parse::<u8>() else {
                return false;
            };
            minzoom.is_none_or(|mn| z >= mn) && maxzoom.is_none_or(|mx| z <= mx)
        })
        .map(|m| {
            let (min_col, max_col, min_row, max_row) =
                tms_def.tile_limits_for_wgs84_bounds(m, [w, s, e, n]);
            serde_json::json!({
                "tileMatrix": m.id,
                "minTileRow": min_row,
                "maxTileRow": max_row,
                "minTileCol": min_col,
                "maxTileCol": max_col,
            })
        })
        .collect();
    serde_json::json!({
        "title": tms_def.id,
        "id": tms_def.id,
        "tileMatrixSetId": tms_def.id,
        "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms_def.id),
        "dataType": "map",
        "crs": tms_def.crs_uri,
        "accessConstraints": "unclassified",
        "boundingBox": {
            "lowerLeft": [w, s],
            "upperRight": [e, n],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
        "tileMatrixSetLimits": limits,
        "links": [
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}{q_suffix}", tms_def.id),
                "rel": "self",
                "type": "application/json",
                "title": tms_def.id,
            },
            {
                "href": format!("{origin}/tileMatrixSets/{}", tms_def.id),
                "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                "type": "application/json",
                "title": format!("Definition of '{}' tile matrix set", tms_def.id),
            },
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}/{{tileMatrix}}/{{tileCol}}/{{tileRow}}{tile_q_suffix}", tms_def.id),
                "rel": "item",
                "type": "image/png",
                "title": "Tiles",
                "templated": true,
            },
        ],
    })
}

fn build_ogc_tilesets_list(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
) -> serde_json::Value {
    let tilesets: Vec<serde_json::Value> = crate::tms::ALL_TMS
        .iter()
        .map(|t| {
            build_ogc_tileset_json(
                host,
                forwarded_prefix,
                route_prefix,
                t,
                raw_query,
                bounds,
                None,
                None,
            )
        })
        .collect();
    serde_json::json!({ "tilesets": tilesets })
}

/// A 405 response with an optional `Allow` header, shared across all route families.
fn method_not_allowed_opt(allow: Option<&str>) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": "Method Not Allowed" }).to_string();
    let mut resp = plain_resp(
        StatusCode::METHOD_NOT_ALLOWED,
        Bytes::from(body),
        "application/json",
    );
    if let Some(Ok(hv)) = allow.map(HeaderValue::from_str) {
        resp.headers_mut().insert(hyper::header::ALLOW, hv);
    }
    resp
}

fn method_not_allowed_resp(allow: &'static str) -> Response<Full<Bytes>> {
    method_not_allowed_opt(Some(allow))
}

fn strip_null_object_fields(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.retain(|_, val| !val.is_null());
    }
    v
}

fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Howard Hinnant's civil-from-days algorithm
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn percent_encode_component(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0F) as usize] as char);
        }
    }
    out
}

fn build_stac_render_entry(
    host: &str,
    forwarded_prefix: &str,
    stac_url: &str,
    render_id: &str,
    render: &serde_json::Value,
) -> serde_json::Value {
    let suffix = if !forwarded_prefix.is_empty() && !host.ends_with(forwarded_prefix) {
        forwarded_prefix
    } else {
        ""
    };
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{suffix}")
    } else {
        format!("http://{host}{suffix}")
    };
    let enc_url = percent_encode_component(stac_url);
    let enc_rid = percent_encode_component(render_id);
    let mut qs_parts = vec![format!("url={enc_url}")];
    if let Some(obj) = render.as_object() {
        for (k, v) in obj {
            let ek = percent_encode_component(k);
            match k.as_str() {
                "title" | "minmax_zoom" | "tilematrixsets" => continue,
                "assets" => {
                    if let Some(arr) = v.as_array() {
                        for a in arr {
                            if let Some(s) = a.as_str() {
                                qs_parts.push(format!("assets={}", percent_encode_component(s)));
                            }
                        }
                    }
                }
                "rescale" => {
                    if let Some(arr) = v.as_array() {
                        for pair in arr {
                            if let Some(p) = pair.as_array()
                                && p.len() == 2
                            {
                                qs_parts.push(format!("rescale={},{}", p[0], p[1]));
                            }
                        }
                    }
                }
                "bidx" => {
                    if let Some(arr) = v.as_array() {
                        for b in arr {
                            qs_parts.push(format!("bidx={b}"));
                        }
                    }
                }
                "colormap" => match v {
                    serde_json::Value::String(s) => {
                        qs_parts.push(format!("colormap={}", percent_encode_component(s)));
                    }
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        qs_parts.push(format!(
                            "colormap={}",
                            percent_encode_component(&v.to_string())
                        ));
                    }
                    _ => {}
                },
                _ => match v {
                    serde_json::Value::String(s) => {
                        qs_parts.push(format!("{ek}={}", percent_encode_component(s)));
                    }
                    serde_json::Value::Number(n) => qs_parts.push(format!("{ek}={n}")),
                    serde_json::Value::Bool(b) => qs_parts.push(format!("{ek}={b}")),
                    _ => {}
                },
            }
        }
    }
    let render_qs = qs_parts.join("&");
    serde_json::json!({
        "valid": true,
        "params": render,
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": format!("{origin}/stac/renders/{enc_rid}?url={enc_url}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/json",
                "title": "TileJSON",
                "href": format!("{origin}/stac/WebMercatorQuad/tilejson.json?{render_qs}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/xml",
                "title": "WMTS",
                "href": format!("{origin}/stac/WebMercatorQuad/WMTSCapabilities.xml?{render_qs}"),
            },
        ],
    })
}

fn data_band_count(ifd: &cogzilla_tiff::IfdMeta) -> u16 {
    if cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
        .is_some()
    {
        ifd.samples_per_pixel.saturating_sub(1).max(1)
    } else {
        ifd.samples_per_pixel
    }
}

fn effective_bbox_max_size(pq: &ParsedQuery) -> u32 {
    if pq.keys.iter().any(|k| k == "max_size") {
        pq.max_size
    } else {
        // TiTiler defaults max_size=None on /bbox and /feature (full native resolution, capped at 4096)
        4096
    }
}

fn maybe_upgrade_multiband_default(
    bands: BandRequest,
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> BandRequest {
    if pq.bidx.is_empty()
        && pq.expr_raw.is_none()
        && pq.algorithm.is_none()
        && (default_bands == 2 || (4..=15).contains(&default_bands))
        && (matches!(fmt, OutputFormat::Npy | OutputFormat::RawF32)
            || (ct == GEOTIFF_CONTENT_TYPE
                && pq.rescale.is_empty()
                && !pq.colormap_explicit
                && pq.custom_colormap_json.is_none()
                && pq.color_formula.is_none()))
    {
        return BandRequest::MultiBand {
            bands_1based: (1..=default_bands).collect(),
        };
    }
    bands
}

/// TiTiler's `cast`, `ceil` and `floor` keep every band. An array of other than one or
/// three bands (which are band math), on a route that writes arrays in their type
/// (`apply_array_asked`), reads them as they are, and each value is rounded and wrapped as
/// it is typed (`ArrayRule::rounding`).
fn keep_algorithm_bands(
    bands: BandRequest,
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> BandRequest {
    let array = fmt == OutputFormat::Npy || ct == GEOTIFF_CONTENT_TYPE;
    let asked = if pq.bidx.is_empty() { usize::from(default_bands) } else { pq.bidx.len() };
    if matches!(pq.algorithm.as_deref(), Some("cast" | "ceil" | "floor"))
        && array
        && !array_type::shown(pq)
        && pq.asset_exprs.is_empty()
        && (asked == 2 || (4..=15).contains(&asked))
    {
        let bands_1based = if pq.bidx.is_empty() { (1..=default_bands).collect() } else { pq.bidx.clone() };
        return BandRequest::MultiBand { bands_1based };
    }
    bands
}

/// The bands of a request on a route that writes arrays in their type (`/cog`, `/stac`,
/// `/mosaicjson`).
fn build_band_request_for_output(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    fmt: OutputFormat,
    ct: &str,
) -> Result<BandRequest, EngineError> {
    let bands = build_band_request(pq, default_bands, sample_type)?;
    let bands = maybe_upgrade_multiband_default(bands, fmt, ct, pq, default_bands);
    Ok(keep_algorithm_bands(bands, fmt, ct, pq, default_bands))
}

fn build_tile_request(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    is_crs84: bool,
    source_cog: Option<&cogzilla_io::OpenedCog>,
) -> Result<(TileRequest, &'static str), EngineError> {
    let tms_id = if is_crs84 {
        "WorldCRS84Quad"
    } else {
        "WebMercatorQuad"
    };
    build_tile_request_for_tms(
        engine,
        tile,
        scale_opt,
        ext_opt,
        pq,
        default_bands,
        sample_type,
        tms_id,
        source_cog,
        false,
    )
}

/// A tile request; `typed_arrays` when the route writes its arrays in their type
/// (`apply_array_asked`), which some algorithms' bands rely on ([`keep_algorithm_bands`]).
#[allow(clippy::too_many_arguments)]
fn build_tile_request_for_tms(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_tiff::SampleType,
    tms_id: &str,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    typed_arrays: bool,
) -> Result<(TileRequest, &'static str), EngineError> {
    let (cols, rows, target_epsg, base_tile_size) =
        if let Some(tms_def) = crate::tms::lookup_tms(tms_id) {
            let Some(m) = tms_def.matrix(tile.z) else {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} is not defined for tile matrix set '{}'",
                    tile.z, tms_id
                )));
            };
            (m.matrix_width, m.matrix_height, tms_def.epsg, m.tile_width)
        } else {
            if tile.z > 30 {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} exceeds maximum supported zoom (30)",
                    tile.z
                )));
            }
            let rows = 1u64 << (tile.z as u32);
            (rows, rows, 3857u16, 256u32)
        };
    if (tile.x as u64) >= cols || (tile.y as u64) >= rows {
        return Err(EngineError::BadRequest(format!(
            "tile coordinate ({}, {}) out of range for {tms_id} z={} (expected x < {cols}, y < {rows})",
            tile.x, tile.y, tile.z
        )));
    }
    let eff_ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
    let (format, content_type) = parse_ext_format(eff_ext, pq.webp_quality, pq.jpeg_quality)?;
    let eff_default_bands = match source_cog {
        Some(c) if default_bands == c.index.full_res().samples_per_pixel => {
            data_band_count(c.index.full_res())
        }
        _ => default_bands,
    };
    let bands = if typed_arrays {
        build_band_request_for_output(pq, eff_default_bands, sample_type, format, content_type)?
    } else {
        let bands = build_band_request(pq, eff_default_bands, sample_type)?;
        maybe_upgrade_multiband_default(bands, format, content_type, pq, eff_default_bands)
    };
    let format = geotiff_render_format(format, content_type, pq, source_cog, &bands, sample_type, false);
    let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
    let u16_png = !auto_fmt && wants_u16_png(format, content_type, pq, source_cog, &bands, sample_type);
    if !matches!(format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64) {
        match bands {
            BandRequest::MultiBand { ref bands_1based }
                if !(u16_png
                    || pq.pixel_selection.by_last_band()
                        && matches!(bands_1based.len(), 2 | 4)) =>
            {
                return Err(EngineError::BadRequest(
                    "multi-band bidx (2 or 4..=15 bands) is only supported for .tif, .npy, or .f32 output"
                        .into(),
                ));
            }
            BandRequest::CustomExprMulti { ref progs }
                if !(pq.pixel_selection.by_last_band() && matches!(progs.len(), 2 | 4)) =>
            {
                return Err(EngineError::BadRequest(format!(
                    "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                    progs.len()
                )));
            }
            _ => {}
        }
    }
    let has_explicit_tile_size = pq
        .keys
        .iter()
        .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
    let tile_size = match scale_opt {
        Some(sc) if (1..=4).contains(&sc) => {
            let sz = sc.saturating_mul(base_tile_size);
            if sz > 4096 {
                return Err(EngineError::BadRequest(format!(
                    "scaled tile size {sz} exceeds maximum (4096) for {tms_id}"
                )));
            }
            sz
        }
        Some(other) => {
            return Err(EngineError::BadRequest(format!(
                "invalid tile scale @{other}x (expected 1..=4)"
            )));
        }
        None => {
            if has_explicit_tile_size {
                pq.tile_size
            } else {
                base_tile_size
            }
        }
    };
    let buffered_dim = (tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    if buffered_dim > 4096 {
        return Err(EngineError::BadRequest(format!(
            "buffered tile size {buffered_dim} (tile_size={tile_size}, buffer={}) exceeds maximum (4096) for {tms_id}",
            pq.buffer
        )));
    }
    Ok((
        TileRequest {
            tile,
            tile_size,
            bands,
            resample: resolve_resample(engine, pq, source_cog, target_epsg),
            format,
            unscale: pq.unscale,
        },
        content_type,
    ))
}

/// The range of an integer sample type (`None` for floats), for the statistics read.
pub(crate) fn array_type_range(kind: cogzilla_tiff::SampleType) -> Option<(f64, f64)> {
    array_type::integer_range(kind)
}

fn build_stats_options(pq: &ParsedQuery, allow_expr: bool) -> Result<StatsOptions, EngineError> {
    build_stats_options_with_bands(pq, allow_expr, 0)
}

fn build_stats_options_with_bands(
    pq: &ParsedQuery,
    allow_expr: bool,
    default_bands: u16,
) -> Result<StatsOptions, EngineError> {
    let mut opts = StatsOptions { resample: pq.resample, ..StatsOptions::default() };
    if !pq.percentiles.is_empty() {
        opts.percentiles = pq.percentiles.clone();
    }
    if let Some(bins) = pq.histogram_bins {
        opts.histogram_bins = bins;
    }
    opts.histogram_edges = pq.histogram_edges.clone();
    opts.histogram_range = pq.histogram_range;
    opts.categorical = pq.categorical;
    if !pq.categories.is_empty() {
        opts.categories = Some(pq.categories.clone());
    }
    if !pq.bidx.is_empty() {
        opts.bidx = Some(pq.bidx.clone());
    }
    opts.width = pq.width;
    opts.height = pq.height;
    opts.max_size_explicit = !pq.max_size_zero && pq.keys.iter().any(|k| k == "max_size");
    opts.area_weighted = pq.area_weighted;
    if allow_expr
        && let Some(raw) = pq.expr_raw.as_deref()
    {
        let mut compiled = Vec::new();
        for part in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let prog = ExprProgram::compile(part).map_err(EngineError::BadRequest)?;
            compiled.push((part.to_string(), prog));
            if compiled.len() > 16 {
                return Err(EngineError::BadRequest(
                    "expression exceeds max 16 semicolon-separated outputs".into(),
                ));
            }
        }
        if !compiled.is_empty() {
            opts.expressions = Some(compiled);
        }
    } else if allow_expr
        && let Some(alg_raw) = pq.algorithm.as_deref()
    {
        let bands: Vec<u16> = if !pq.bidx.is_empty() {
            pq.bidx.clone()
        } else {
            (1..=default_bands.clamp(1, 16)).collect()
        };
        let alg = alg_raw.to_ascii_lowercase();
        // The bands' type is not known here: they are taken as possibly fractional.
        let expr_str = algorithm_math::band_math(&alg, &bands, None, pq.cast_dtype.as_deref(), pq.threshold)
            .map_err(|e| match alg.as_str() {
                "normalizedindex" => EngineError::BadRequest(e),
                _ => EngineError::BadRequest(format!("algorithm '{alg_raw}' is not supported on statistics routes")),
            })?;
        let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
        opts.expressions = Some(vec![(alg_raw.to_string(), prog)]);
    }
    Ok(opts)
}

/// Apply optional `?expression=` (supporting `;`-separated multi-band outputs) or `?bidx=`
/// filtering to a TiTiler `/point` JSON payload.
fn apply_point_filters(
    pt: &mut serde_json::Value,
    expr_raw: Option<&str>,
    bidx: &[u16],
) -> Result<(), EngineError> {
    let Some(vals) = pt.get("values").and_then(|v| v.as_array()).cloned() else {
        return Ok(());
    };
    let names: Vec<serde_json::Value> = pt
        .get("band_names")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let descs: Vec<serde_json::Value> = pt
        .get("band_descriptions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| names.clone());

    if let Some(raw_expr) = expr_raw {
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(EngineError::BadRequest("empty expression".into()));
        }
        if parts.len() > 16 {
            return Err(EngineError::BadRequest(
                "expression exceeds max 16 semicolon-separated outputs".into(),
            ));
        }
        let n_bands = vals.len();
        let mut out_vals = Vec::with_capacity(parts.len());
        let mut out_names = Vec::with_capacity(parts.len());
        for expr in parts {
            let prog = ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
            let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
            for &b in &prog.used_bands {
                if b == 0 || (b as usize) > n_bands {
                    return Err(EngineError::BadRequest(format!(
                        "expression references b{b} but point query only has {n_bands} bands"
                    )));
                }
                let f = vals[(b - 1) as usize]
                    .as_f64()
                    .map(|x| x as f32)
                    .unwrap_or(f32::NAN);
                slot_vals.push(f);
            }
            let res = prog.eval(&slot_vals);
            out_vals.push(if res.is_finite() {
                serde_json::json!(res)
            } else {
                serde_json::Value::Null
            });
            out_names.push(serde_json::json!(expr));
        }
        let band_names: Vec<serde_json::Value> = (1..=out_vals.len())
            .map(|i| serde_json::json!(format!("b{i}")))
            .collect();
        pt["values"] = serde_json::Value::Array(out_vals);
        pt["band_names"] = serde_json::Value::Array(band_names);
        pt["band_descriptions"] = serde_json::Value::Array(out_names);
        return Ok(());
    }

    if !bidx.is_empty() {
        let n_bands = vals.len();
        let mut f_vals = Vec::with_capacity(bidx.len());
        let mut f_names = Vec::with_capacity(bidx.len());
        let mut f_descs = Vec::with_capacity(bidx.len());
        for &b in bidx {
            if b == 0 || (b as usize) > n_bands {
                return Err(EngineError::BadRequest(format!(
                    "band {b} out of range (dataset has {n_bands} bands)"
                )));
            }
            let idx = (b - 1) as usize;
            f_vals.push(vals[idx].clone());
            f_names.push(names.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
            f_descs.push(descs.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
        }
        pt["values"] = serde_json::Value::Array(f_vals);
        pt["band_names"] = serde_json::Value::Array(f_names);
        pt["band_descriptions"] = serde_json::Value::Array(f_descs);
    }
    Ok(())
}

fn validate_mosaic_tile_params(_pq: &ParsedQuery, tile_size: u32) -> Result<(), EngineError> {
    if !(64..=2048).contains(&tile_size) {
        return Err(EngineError::BadRequest(format!(
            "mosaic tile_size must be in 64..=2048 (got {tile_size})"
        )));
    }
    Ok(())
}

fn default_raw_band_count(bands: &BandRequest, pixel_selection: Selection) -> usize {
    if pixel_selection == Selection::Count {
        return 1;
    }
    let raw_nb = match bands {
        BandRequest::Rgb { .. }
        | BandRequest::SarRgb { .. }
        | BandRequest::Terrain { .. }
        | BandRequest::CustomExprRgb { .. } => 3,
        BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
        BandRequest::CustomExprMulti { progs } => progs.len().max(1),
        _ => 1,
    };
    if pixel_selection.by_last_band() {
        raw_nb.saturating_sub(1).max(1)
    } else {
        raw_nb
    }
}

/// Hard memory budget (64 MiB) for `pixel_selection=median`, which must buffer
/// per-scene planes in order to compute exact per-pixel medians.
const MAX_MEDIAN_BUFFER_BYTES: usize = 64 * 1024 * 1024;

async fn render_mosaic_tile_pixel_selection(
    engine: &TileEngine,
    urls: &[String],
    treq: TileRequest,
    pq: &ParsedQuery,
    failed_out: &mut Vec<String>,
) -> Result<Bytes, EngineError> {
    let urls = &urls[..urls.len().min(64)];
    let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    let bbox_3857 = web_mercator_buffered_3857_bbox(
        treq.tile.z,
        treq.tile.x,
        treq.tile.y,
        pq.buffer,
        treq.tile_size,
    );
    let n_px = (out_dim as usize) * (out_dim as usize);
    let mut first_lut: Option<cogzilla_kernel::CustomColormap> = None;

    // Only `median` holds its scenes, within MAX_MEDIAN_BUFFER_BYTES; the other methods
    // keep a running figure for each value.
    let method = pq.pixel_selection;
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;

    for (scene_i, url) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                failed_out.push(url.clone());
                last_err = Some(crate::engine::asset_fetch_error(url, e));
                continue;
            }
        };
        let cogs = vec![cog];
        let mut scene_bands = treq.bands.clone();
        let data_bands = data_band_count(cogs[0].index.full_res());
        if method.by_last_band()
            && pq.bidx.is_empty()
            && pq.expr_raw.is_none()
            && pq.algorithm.is_none()
            && data_bands >= 2
            && matches!(scene_bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        {
            scene_bands = BandRequest::MultiBand {
                bands_1based: (1..=data_bands).collect(),
            };
        }
        if pq.unscale {
            apply_unscale_cogs(&mut scene_bands, &cogs, None);
        }
        if first_lut.is_none()
            && !(matches!(method, Selection::Count | Selection::Stdev)
                && pq.custom_colormap_json.is_none())
        {
            first_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
        }
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
        let raw = match render_mercator_bbox(
            engine,
            &cogs,
            bbox_3857,
            out_dim,
            out_dim,
            &scene_bands,
            resample,
            OutputFormat::RawF32,
            pq.nodata,
            None,
            None,
            pq.return_mask,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => {
                any_ok = true;
                continue;
            }
            Err(EngineError::Io(e))
                if e.is_fatal() =>
            {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                failed_out.push(url.clone());
                last_err = Some(e);
                continue;
            }
        };
        any_ok = true;
        if n_px == 0 || raw.len() % (n_px * 4) != 0 {
            continue;
        }
        let nb = raw.len() / (n_px * 4);
        if nb == 0 {
            continue;
        }
        let fold = match &mut fold {
            Some(fold) => fold,
            None => {
                if method.by_last_band() && nb < 2 {
                    return Err(EngineError::BadRequest(
                        "pixel_selection by the last band needs at least two bands".into(),
                    ));
                }
                if Fold::<f32>::working_bytes(method, nb, n_px) > MAX_MEDIAN_BUFFER_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce tile_size, buffer, or band count",
                        MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                    )));
                }
                let integer =
                    reads_whole_numbers(pq.unscale, &scene_bands, cogs[0].index.full_res().sample_type);
                let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                fold.insert(Fold::new(method, nb, n_px, integer).absent_as(absent))
            }
        };
        if fold.bands() != nb {
            continue;
        }
        taken += 1;
        if Fold::<f32>::held_bytes(method, taken, nb, n_px) > MAX_MEDIAN_BUFFER_BYTES {
            return Err(EngineError::BadRequest(format!(
                "pixel_selection=median exceeds memory budget ({} MiB limit); reduce tile_size, buffer, or band count",
                MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
            )));
        }
        fold.add_le_bytes(&raw);
    }

    if !any_ok
        && let Some(err) = last_err
    {
        return Err(err);
    }
    #[cfg(feature = "gpu")]
    let gpu_selected: Option<Selected<f32>> = if let Some(ref gpu) = engine.gpu
        && let Some(ref f) = fold
        && let Some((nb, scenes)) = f.median_scenes()
        && scenes.len() <= 32
    {
        let band_planes: Vec<Vec<&[f32]>> = (0..nb)
            .map(|b| {
                let b_off = b * n_px;
                scenes.iter().map(|s| &s[b_off..b_off + n_px]).collect()
            })
            .collect();
        gpu.try_composite_mosaic(&crate::gpu_backend::MosaicCompositeRequest {
            width: out_dim,
            height: out_dim,
            input: crate::gpu_backend::MosaicCompositeInput::Bands(&band_planes),
            method: cogzilla_gpu::GpuPixelSelection::Median,
            output: crate::gpu_backend::MosaicCompositeOutput::F32,
        })
        .and_then(crate::gpu_backend::MosaicCompositeResult::into_f32_bands)
        .map(|gpu_bands| {
            engine
                .stats
                .gpu_dispatches
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut values = Vec::with_capacity(nb * n_px);
            for band_out in gpu_bands {
                values.extend_from_slice(&band_out);
            }
            Selected { bands: nb, values }
        })
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_selected: Option<Selected<f32>> = None;

    let offload_cpu = gpu_selected.is_none()
        && fold.is_some()
        && n_px >= 256 * 256
        && (matches!(method, Selection::Median) || n_px >= 512 * 512);
    crate::engine::block_in_place_if_multithread(offload_cpu, || {
        let selected = gpu_selected
            .or_else(|| fold.and_then(Fold::finish))
            .unwrap_or_else(|| {
                let bands = default_raw_band_count(&treq.bands, method);
                Selected { bands, values: vec![f32::NAN; bands * n_px] }
            });

        if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut out = bytes::BytesMut::with_capacity(selected.values.len() * 4);
            for v in &selected.values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if matches!(treq.format, OutputFormat::Npy) {
                return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    selected.bands,
                    out_dim,
                    out_dim,
                    pq.return_mask,
                ));
            }
            return Ok(raw);
        }

        let rgba = paint_selected(selected, n_px, &treq.bands, pq, first_lut.as_ref())?;
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = engine.gpu {
            return Ok(gpu.encode_rgba_output(
                out_dim as usize,
                out_dim as usize,
                &rgba,
                treq.format,
                pq.return_mask,
            ));
        }
        Ok(encode_rgba_output(
            out_dim as usize,
            out_dim as usize,
            &rgba,
            treq.format,
            pq.return_mask,
        ))
    })
}

/// The picture of what a pixel selection left: `pixels` RGBA pixels, by the rules in
/// [`paint`]. Three bands or more are red, green and blue (the first three); fewer are the
/// first band under the request's colour map. Kept from the single-scene renderer, where
/// rio-tiler has no equivalent or does otherwise: a raster's own palette and a colour map
/// given as JSON (`custom`, which the kernel looks up from the value). A range given
/// upside down is the ramp inverted, which is `rescale_byte`'s doing.
fn paint_selected(
    mut selected: Selected,
    pixels: usize,
    bands: &BandRequest,
    pq: &ParsedQuery,
    custom: Option<&CustomColormap>,
) -> Result<Vec<u8>, EngineError> {
    let formula = pq.parse_color_formula()?;
    let three = selected.bands >= 3;
    // The request's own ranges and colour map where it has them for this many bands;
    // otherwise bytes as they are.
    let paint = match (Paint::of(bands), bands) {
        (Some(paint @ Paint::Rgb { .. }), _) if three => paint,
        (Some(paint @ Paint::Gray { .. }), _) if !three => paint,
        (_, BandRequest::CustomExprRgb { rescale_min, rescale_max, .. }) if three => {
            Paint::Rgb { min: *rescale_min, max: *rescale_max }
        }
        (_, BandRequest::MultiBand { .. }) if three && !pq.rescale.is_empty() => {
            let r0 = pq.rescale[0];
            let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
            let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
            Paint::Rgb { min: [r0.0, r1.0, r2.0], max: [r0.1, r1.1, r2.1] }
        }
        (_, BandRequest::MultiBand { .. }) if !three && !pq.rescale.is_empty() => {
            let (min, max) = pq.rescale[0];
            Paint::Gray { min, max, colormap: pq.colormap }
        }
        _ if three => Paint::Rgb { min: [0.0; 3], max: [255.0; 3] },
        _ => Paint::Gray { min: 0.0, max: 255.0, colormap: pq.colormap },
    };
    if three {
        selected.bands = 3;
        let mut rgba = paint.rgba(&selected, pixels);
        if let Some(formula) = &formula {
            formula.apply_buffer_in_place(&mut rgba, 4);
        }
        return Ok(rgba);
    }
    if let (Some(custom), Paint::Gray { min, max, .. }) = (custom, paint) {
        let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(min, max);
        let mut rgba = vec![0u8; pixels * 4];
        for (px, v) in rgba.chunks_exact_mut(4).zip(&selected.values) {
            if v.is_finite() {
                px.copy_from_slice(&custom.lookup(*v, min, inv));
            }
        }
        // Under a colour map of the file's or the request's own, the formula is for the
        // finished picture.
        if let Some(formula) = &formula {
            formula.apply_buffer_in_place(&mut rgba, 4);
        }
        return Ok(rgba);
    }
    Ok(match &formula {
        Some(formula) => paint.rgba_through(&selected, pixels, &paint::formula_levels(formula)),
        None => paint.rgba(&selected, pixels),
    })
}

async fn render_mosaic_tile_styled(
    engine: &TileEngine,
    urls: &[String],
    mut treq: TileRequest,
    pq: &ParsedQuery,
    first_has_colormap: bool,
    failed_out: &mut Vec<String>,
) -> Result<Bytes, EngineError> {
    if pq.pixel_selection != Selection::First {
        return render_mosaic_tile_pixel_selection(engine, urls, treq, pq, failed_out).await;
    }
    let mut needs_bbox_mosaic = pq.nodata.is_some()
        || pq.custom_colormap_json.is_some()
        || pq.buffer > 0.0
        || matches!(treq.format, OutputFormat::Npy)
        || (pq.color_formula.is_some()
            && !matches!(
                treq.bands,
                BandRequest::Rgb { .. }
                    | BandRequest::SarRgb { .. }
                    | BandRequest::CustomExprRgb { .. }
                    | BandRequest::Terrain { .. }
            ))
        || matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        )
        || (treq.tile_size != 256 && treq.tile_size != 512);
    if !needs_bbox_mosaic
        && !pq.colormap_explicit
        && pq.custom_colormap_json.is_none()
        && first_has_colormap
    {
        needs_bbox_mosaic = true;
    }

    if needs_bbox_mosaic {
        let urls = &urls[..urls.len().min(64)];
        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
        let bbox_3857 = web_mercator_buffered_3857_bbox(
            treq.tile.z,
            treq.tile.x,
            treq.tile.y,
            pq.buffer,
            treq.tile_size,
        );
        let n_px = (out_dim as usize) * (out_dim as usize);
        if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut acc_f32: Option<(usize, Vec<f32>)> = None;
            let mut filled = 0usize;
            let mut any_ok = false;
            let mut last_err: Option<EngineError> = None;
            for (scene_i, url) in urls.iter().enumerate() {
                if scene_i > 0 {
                    tokio::task::consume_budget().await;
                }
                let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                    Ok(c) => c,
                    Err(e) if e.is_fatal() => {
                        return Err(EngineError::Io(e));
                    }
                    Err(e) => {
                        failed_out.push(url.clone());
                        last_err = Some(crate::engine::asset_fetch_error(url, e));
                        continue;
                    }
                };
                let cogs = vec![cog];
                let mut scene_bands = treq.bands.clone();
                if pq.unscale {
                    apply_unscale_cogs(&mut scene_bands, &cogs, None);
                }
                let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
                let raw = match render_mercator_bbox(
                    engine,
                    &cogs,
                    bbox_3857,
                    out_dim,
                    out_dim,
                    &scene_bands,
                    resample,
                    OutputFormat::RawF32,
                    pq.nodata,
                    None,
                    None,
                    pq.return_mask,
                    pq.unscale,
                )
                .await
                {
                    Ok(b) => b,
                    Err(EngineError::NotFound(_)) => {
                        any_ok = true;
                        continue;
                    }
                    Err(EngineError::Io(e))
                        if e.is_fatal() =>
                    {
                        return Err(EngineError::Io(e));
                    }
                    Err(e) => {
                        failed_out.push(url.clone());
                        last_err = Some(e);
                        continue;
                    }
                };
                any_ok = true;
                if n_px == 0 || raw.len() % (n_px * 4) != 0 {
                    continue;
                }
                let nb = raw.len() / (n_px * 4);
                if nb == 0 {
                    continue;
                }
                let (acc_nb, acc) = acc_f32.get_or_insert_with(|| (nb, vec![f32::NAN; nb * n_px]));
                if *acc_nb != nb {
                    continue;
                }
                for i in 0..n_px {
                    if acc[i].is_nan() {
                        let mut all_valid = true;
                        for b in 0..nb {
                            let boff = (b * n_px + i) * 4;
                            let v = f32::from_le_bytes([
                                raw[boff],
                                raw[boff + 1],
                                raw[boff + 2],
                                raw[boff + 3],
                            ]);
                            if v.is_nan() {
                                all_valid = false;
                                break;
                            }
                        }
                        if all_valid {
                            for b in 0..nb {
                                let boff = (b * n_px + i) * 4;
                                acc[b * n_px + i] = f32::from_le_bytes([
                                    raw[boff],
                                    raw[boff + 1],
                                    raw[boff + 2],
                                    raw[boff + 3],
                                ]);
                            }
                            filled += 1;
                        }
                    }
                }
                if filled >= n_px {
                    break;
                }
            }
            if !any_ok
                && let Some(err) = last_err
            {
                return Err(err);
            }
            let default_nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
            let (nb, acc) = acc_f32.unwrap_or_else(|| (default_nb, vec![f32::NAN; default_nb * n_px]));
            let mut out = bytes::BytesMut::with_capacity(acc.len() * 4);
            for v in acc {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if matches!(treq.format, OutputFormat::Npy) {
                return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    nb,
                    out_dim,
                    out_dim,
                    pq.return_mask,
                ));
            }
            return Ok(raw);
        }
        let cf = pq.parse_color_formula()?;
        let mut comp = vec![0u8; n_px * 4];
        let mut filled = 0usize;
        let mut any_ok = false;
        let mut last_err: Option<EngineError> = None;

        for (scene_i, url) in urls.iter().enumerate() {
            if scene_i > 0 {
                tokio::task::consume_budget().await;
            }
            let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                Ok(c) => c,
                Err(e) if e.is_fatal() => {
                    return Err(EngineError::Io(e));
                }
                Err(e) => {
                    failed_out.push(url.clone());
                    last_err = Some(crate::engine::asset_fetch_error(url, e));
                    continue;
                }
            };
            let cogs = vec![cog];
            let mut scene_bands = treq.bands.clone();
            if pq.unscale {
                apply_unscale_cogs(&mut scene_bands, &cogs, None);
            }
            let scene_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
            let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
            let scene_rgba = match render_mercator_bbox(
                engine,
                &cogs,
                bbox_3857,
                out_dim,
                out_dim,
                &scene_bands,
                resample,
                OutputFormat::RawRgba,
                pq.nodata,
                scene_lut.as_ref(),
                cf.as_ref(),
                true,
                pq.unscale,
            )
            .await
            {
                Ok(b) => b,
                Err(EngineError::NotFound(_)) => {
                    any_ok = true;
                    continue;
                }
                Err(EngineError::Io(e))
                    if e.is_fatal() =>
                {
                    return Err(EngineError::Io(e));
                }
                Err(e) => {
                    failed_out.push(url.clone());
                    last_err = Some(e);
                    continue;
                }
            };
            any_ok = true;
            if scene_rgba.len() != n_px * 4 {
                continue;
            }
            for i in 0..n_px {
                let o = i * 4;
                if comp[o + 3] == 0 && scene_rgba[o + 3] > 0 {
                    comp[o..o + 4].copy_from_slice(&scene_rgba[o..o + 4]);
                    filled += 1;
                }
            }
            if filled >= n_px {
                break;
            }
        }
        if !any_ok
            && let Some(err) = last_err
        {
            return Err(err);
        }
        return Ok(encode_rgba_output(
            out_dim as usize,
            out_dim as usize,
            &comp,
            treq.format,
            pq.return_mask,
        ));
    }

    if (pq.color_formula.is_some() || !pq.return_mask)
        && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
    {
        let cf = pq.parse_color_formula()?;
        let orig_format = treq.format;
        treq.format = OutputFormat::RawRgba;
        let (raw, failed) = engine.render_mosaic_tile_with_report(urls, &treq).await?;
        failed_out.extend(failed);
        let dim = treq.tile_size as usize;
        let data = if let Some(cf) = cf {
            let mut buf = raw.to_vec();
            cf.apply_buffer_in_place(&mut buf, 4);
            encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
        } else {
            encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
        };
        Ok(data)
    } else {
        let (raw, failed) = engine.render_mosaic_tile_with_report(urls, &treq).await?;
        failed_out.extend(failed);
        Ok(raw)
    }
}

/// The band math of the `nth` time (from 0) `asset_name` is asked for, or of the last time.
fn stac_asset_expr_nth<'a>(pq: &'a ParsedQuery, asset_name: &str, nth: usize) -> Option<&'a str> {
    let mut of_it = pq.asset_exprs.iter().filter(|(n, _)| n == asset_name).map(|(_, e)| e.as_str());
    let last = of_it.clone().next_back();
    of_it.nth(nth).or(last)
}

/// How many times each of `names` came before it in `names`.
fn occurrences(names: &[String]) -> Vec<usize> {
    names.iter().enumerate().map(|(i, n)| names[..i].iter().filter(|m| *m == n).count()).collect()
}

/// `expr` with each band token (`b3`, or `B3` as rio-tiler takes it) replaced by what `to`
/// makes of its number, counted from one; a token `to` has nothing for is left as it is.
fn substitute_band_tokens(expr: &str, mut to: impl FnMut(usize) -> Option<String>) -> String {
    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &expr[start..i];
            let replaced = ident.strip_prefix(['b', 'B']).filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).and_then(|n| n.parse().ok()).and_then(&mut to);
            out.push_str(replaced.as_deref().unwrap_or(ident));
        } else {
            let ch = expr[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// The parts of an expression, `;` apart.
fn expression_parts(expr: &str) -> impl Iterator<Item = &str> {
    expr.split(';').map(str::trim).filter(|p| !p.is_empty())
}

/// A request expression over the results of the assets' own band math: each `bK` of it is
/// the Kth result, worked out in brackets.
fn compose_over_results(request: &str, results: &[String]) -> Result<String, EngineError> {
    let mut missing = None;
    let composed = substitute_band_tokens(request, |k| match results.get(k.wrapping_sub(1)) {
        Some(part) => Some(format!("({part})")),
        None => {
            missing.get_or_insert(k);
            None
        }
    });
    match missing {
        Some(k) => Err(EngineError::BadRequest(format!("band {k} out of range ({} bands)", results.len()))),
        None => Ok(composed),
    }
}

/// Whether `expr` names any of `names` (as `{asset}` or `{asset}_b{k}`), even where the name
/// is spelled as its band would be (an asset named `b2`).
fn names_in_expression(expr: &str, names: &[String]) -> bool {
    let found = std::cell::Cell::new(false);
    let _ = crate::stac::rewrite_stac_expr_to(expr, names, |asset, _, _| {
        found.set(true);
        Ok(asset)
    });
    found.get()
}

/// Each of `names`' own band math and own bands: by entry when the query's entries are
/// `names` (an asset asked for twice keeps what each time asked), else by name. An entry's
/// own `|bidx=` wins over what `asset_bidx` gives its asset.
fn stac_entry_options(pq: &ParsedQuery, names: &[String]) -> Vec<(Option<String>, Option<Vec<u16>>)> {
    let by_entry = pq.asset_entries.len() == names.len() && pq.asset_entries.iter().zip(names).all(|(e, n)| &e.name == n);
    if by_entry {
        return pq
            .asset_entries
            .iter()
            .map(|e| {
                let named = || pq.asset_bidx_by_name.iter().find(|(n, _)| n == &e.name).map(|(_, b)| b.clone());
                (e.expression.clone(), e.bands.clone().or_else(named))
            })
            .collect();
    }
    let nth = occurrences(names);
    names
        .iter()
        .zip(nth)
        .map(|(name, nth)| {
            (stac_asset_expr_nth(pq, name, nth).map(str::to_string), stac_asset_selected_bands(pq, name).map(<[u16]>::to_vec))
        })
        .collect()
}

/// The bands of a `/stac` request's assets as the renderer reads them, and its expression over
/// them, as rio-tiler stacks a STAC item: of several assets, every band of each (or those its
/// `|bidx=` picks), one after another, so that `bN` is the Nth band of the stack; an asset's
/// own band math (`|expression=`) is over all of its own bands, at its place in the stack, and
/// a request `expression` is then over the results. Asset names in the request expression
/// stand for the asset's first band in the stack (`{asset}_b{k}` for its band `k`), and `bN`
/// for the Nth band of the stack. Under `asset_as_band` each asset must give one band.
///
/// Returns the rasters to read, one per band read when that is not one band per asset,
/// with the band each is read for.
fn expand_stac_asset_bidx(
    pq: &mut ParsedQuery,
    names: &[String],
    cogs: Vec<Arc<cogzilla_io::OpenedCog>>,
    rewritten: Option<String>,
    read_names: bool,
) -> Result<(Vec<Arc<cogzilla_io::OpenedCog>>, Option<Vec<u16>>), EngineError> {
    let original = std::mem::replace(&mut pq.expr_raw, rewritten);
    let named = read_names && original.as_deref().is_some_and(|expr| names_in_expression(expr, names));
    let options = stac_entry_options(pq, names);
    let mut picked: Vec<Vec<u16>> = Vec::with_capacity(names.len());
    for ((name, cog), (own, sel)) in names.iter().zip(&cogs).zip(&options) {
        let nb = data_band_count(cog.index.full_res());
        // An asset's band math is over all of its bands, and wins over its `bidx`.
        let bands: Vec<u16> = match (own, sel) {
            (Some(_), _) | (None, None) => (1..=nb).collect(),
            (None, Some(sel)) => {
                if let Some(&b) = sel.iter().find(|&&b| b == 0 || b > nb) {
                    return Err(EngineError::BadRequest(format!(
                        "band {b} out of range for asset '{name}' (dataset has {nb} bands)"
                    )));
                }
                sel.clone()
            }
        };
        let gives = own.as_deref().map_or(bands.len(), |own| expression_parts(own).count());
        if pq.asset_as_band && gives != 1 {
            return Err(EngineError::BadRequest(format!(
                "Can't use `asset_as_band` for multibands asset ('{name}' gives {gives} bands)"
            )));
        }
        picked.push(bands);
    }
    if names.len() == 1 {
        match &options[0] {
            (Some(own), _) => {
                let results: Vec<String> = expression_parts(own).map(str::to_string).collect();
                pq.expr_raw = Some(match pq.expr_raw.take() {
                    None => own.clone(),
                    Some(request) => compose_over_results(&request, &results)?,
                });
                pq.asset_exprs.clear();
                pq.asset_bidx.clear();
                pq.bidx.clear();
            }
            (None, Some(sel)) if pq.bidx.is_empty() => pq.bidx = sel.clone(),
            _ => {}
        }
        return Ok((cogs, None));
    }
    let offsets: Vec<usize> = picked
        .iter()
        .scan(0usize, |at, bands| {
            let first = *at;
            *at += bands.len();
            Some(first)
        })
        .collect();
    if options.iter().any(|(own, _)| own.is_some()) {
        if named {
            return Err(EngineError::BadRequest(
                "an expression over the results of the assets' own band math takes b1..bn, not asset names".into(),
            ));
        }
        let mut results: Vec<String> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let count = picked[i].len();
            match &options[i].0 {
                Some(own) => {
                    for part in expression_parts(own) {
                        let mut beyond = None;
                        let shifted = substitute_band_tokens(part, |j| {
                            if (1..=count).contains(&j) {
                                Some(format!("b{}", offsets[i] + j))
                            } else {
                                beyond.get_or_insert(j);
                                None
                            }
                        });
                        if let Some(j) = beyond {
                            return Err(EngineError::BadRequest(format!(
                                "band {j} out of range for asset '{name}' (dataset has {count} bands)"
                            )));
                        }
                        results.push(shifted);
                    }
                }
                None => results.extend((1..=count).map(|k| format!("b{}", offsets[i] + k))),
            }
        }
        pq.expr_raw = Some(match pq.expr_raw.take() {
            None => results.join(";"),
            Some(request) => compose_over_results(&request, &results)?,
        });
        pq.asset_exprs.clear();
        pq.bidx.clear();
    } else if named && let Some(expr) = original {
        // An asset's name is its first band in the stack, `{asset}_b{k}` its band `k`; `bN`
        // stays the Nth band of the stack.
        pq.expr_raw = Some(crate::stac::rewrite_stac_expr_to(&expr, names, |asset, name, band| {
            let at = asset - 1;
            let place = match band {
                None => Some(0),
                Some(k) => picked[at].iter().position(|&b| usize::from(b) == k),
            };
            place.map(|p| offsets[at] + p + 1).ok_or_else(|| {
                EngineError::BadRequest(format!("band {} of asset '{name}' is not read", band.unwrap_or(0)))
            })
        })?);
    }
    if picked.iter().all(|bands| bands.as_slice() == [1]) {
        return Ok((cogs, None));
    }
    let slots: usize = picked.iter().map(Vec::len).sum();
    if slots > 16 {
        return Err(EngineError::BadRequest(format!(
            "{slots} bands over the assets are more than the 16 read at once; choose bands with `|bidx=`"
        )));
    }
    // Without band math every band read is an output, and an answer holds at most 15.
    if pq.expr_raw.is_none() && pq.bidx.is_empty() && slots > 15 {
        return Err(EngineError::BadRequest(format!(
            "{slots} bands over the assets are more than the 15 an answer holds; choose bands with `|bidx=` or `expression`"
        )));
    }
    let mut expanded_cogs = Vec::with_capacity(slots);
    let mut slot_bands = Vec::with_capacity(slots);
    for (cog, bands) in cogs.into_iter().zip(picked) {
        for b in bands {
            expanded_cogs.push(cog.clone());
            slot_bands.push(b);
        }
    }
    Ok((expanded_cogs, Some(slot_bands)))
}

#[allow(clippy::too_many_arguments)]
async fn render_stac_bbox_opt(
    engine: &TileEngine,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    slot_bands: Option<&[u16]>,
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    fmt: OutputFormat,
    nodata: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    cf: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    if let Some(ab) = slot_bands {
        render_multi_asset_bbox(
            engine,
            cogs,
            ab,
            bbox,
            dst_crs,
            native_preview,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else if native_preview {
        render_native_preview(
            engine,
            cogs,
            bbox,
            out_w,
            out_h,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else {
        render_crs_bbox(
            engine,
            cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    }
}

/// Resolve the effective output `(CrsModel, epsg)` for `/{cog,stac,mosaicjson}/bbox` routes:
/// explicit `?dst_crs=` wins, otherwise default to `?coord_crs=` (matching TiTiler 2.4 / `rio-tiler`),
/// falling back to `EPSG:4326`.
fn resolve_bbox_dst_crs(pq: &ParsedQuery) -> Option<(cogzilla_proj::CrsModel, u16)> {
    pq.dst_crs
        .clone()
        .or_else(|| pq.coord_crs.clone().zip(pq.coord_crs_epsg))
}

fn stac_asset_selected_bands<'a>(pq: &'a ParsedQuery, asset_name: &str) -> Option<&'a [u16]> {
    pq.asset_bidx
        .iter()
        .find(|(n, _)| n == asset_name)
        .map(|(_, bands)| bands.as_slice())
}

/// Composite up to 16 intersecting MosaicJSON scenes for `/mosaicjson/bbox` and `/mosaicjson/feature`
/// using first-non-transparent alpha-over (or first-non-NaN for f32/npy) blending in target CRS.
#[allow(clippy::too_many_arguments)]
async fn render_mosaic_bbox_composite(
    engine: &TileEngine,
    urls: &[String],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    pq: &ParsedQuery,
    fmt: OutputFormat,
    ct: &'static str,
    epsg: u16,
) -> Result<(Bytes, Vec<String>), EngineError> {
    if urls.is_empty() {
        return Err(EngineError::NotFound("no mosaic scenes intersect requested bounds".into()));
    }
    let out_w = out_w.clamp(1, 4096);
    let out_h = out_h.clamp(1, 4096);
    let urls = &urls[..urls.len().min(64)];
    let method = pq.pixel_selection;
    let total_px = (out_w as usize).saturating_mul(out_h as usize);
    let mut eff_fmt: Option<OutputFormat> = None;
    let mut first_bands: Option<BandRequest> = None;
    let mut first_lut: Option<cogzilla_kernel::CustomColormap> = None;
    let mut acc_rgba: Option<Vec<u8>> = None;
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut opaque_count = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;
    let mut used_assets: Vec<String> = Vec::new();

    for (scene_i, u) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                last_err = Some(crate::engine::asset_fetch_error(u, e));
                continue;
            }
        };
        let cogs = [cog];
        let full = cogs[0].index.full_res();
        let data_bands = data_band_count(full);
        let mut bands =
            build_band_request_for_output(pq, data_bands, full.sample_type, fmt, ct)?;
        let scene_fmt = *eff_fmt.get_or_insert_with(|| {
            geotiff_render_format(fmt, ct, pq, Some(&cogs[0]), &bands, full.sample_type, false)
        });
        let is_f32_out = matches!(scene_fmt, OutputFormat::RawF32 | OutputFormat::Npy);
        if !is_f32_out {
            match bands {
                BandRequest::MultiBand { ref bands_1based }
                    if !(method.by_last_band() && matches!(bands_1based.len(), 2 | 4)) =>
                {
                    return Err(EngineError::BadRequest(
                        "multi-band bidx (2 or 4..=15 bands) is only supported for .tif, .npy, or .f32 output"
                            .into(),
                    ));
                }
                BandRequest::CustomExprMulti { ref progs }
                    if !(method.by_last_band() && matches!(progs.len(), 2 | 4)) =>
                {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                        progs.len()
                    )));
                }
                _ => {}
            }
        }
        if first_bands.is_none() {
            first_bands = Some(bands.clone());
        }
        if method.by_last_band()
            && pq.bidx.is_empty()
            && pq.expr_raw.is_none()
            && pq.algorithm.is_none()
            && data_bands >= 2
            && matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        {
            bands = BandRequest::MultiBand {
                bands_1based: (1..=data_bands).collect(),
            };
        }
        if pq.unscale {
            apply_unscale_cogs(&mut bands, &cogs, None);
        }
        let custom_lut = if matches!(method, Selection::Count | Selection::Stdev)
            && pq.custom_colormap_json.is_none()
        {
            None
        } else {
            resolve_custom_lut(pq, &cogs[0], &bands)?
        };
        if first_lut.is_none() && custom_lut.is_some() {
            first_lut = custom_lut.clone();
        }
        let cf = pq.parse_color_formula()?;
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), epsg);
        if urls.len() == 1 && method == Selection::First {
            let d = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                scene_fmt,
                pq.nodata,
                custom_lut.as_ref(),
                cf.as_ref(),
                pq.return_mask,
                pq.unscale,
            )
            .await?;
            return Ok((
                wrap_geotiff_fmt(
                    d,
                    ct,
                    scene_fmt,
                    out_w,
                    out_h,
                    bbox,
                    epsg,
                    pq.return_mask,
                ),
                vec![u.clone()],
            ));
        }
        if is_f32_out || method != Selection::First {
            let raw = match render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                OutputFormat::RawF32,
                pq.nodata,
                None,
                None,
                pq.return_mask,
                pq.unscale,
            )
            .await
            {
                Ok(b) => b,
                Err(EngineError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            any_ok = true;
            if total_px == 0 || raw.len() % (total_px * 4) != 0 {
                continue;
            }
            let nb = raw.len() / (total_px * 4);
            if nb == 0 {
                continue;
            }
            let fold = match &mut fold {
                Some(f) => f,
                None => {
                    if method.by_last_band() && nb < 2 {
                        return Err(EngineError::BadRequest(
                            "pixel_selection by the last band needs at least two bands".into(),
                        ));
                    }
                    if Fold::<f32>::working_bytes(method, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                        return Err(EngineError::BadRequest(format!(
                            "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce width, height, or band count",
                            MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                        )));
                    }
                    let integer = reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                    let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                    fold.insert(Fold::new(method, nb, total_px, integer).absent_as(absent))
                }
            };
            if fold.bands() != nb {
                continue;
            }
            taken += 1;
            if Fold::<f32>::held_bytes(method, taken, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                return Err(EngineError::BadRequest(format!(
                    "pixel_selection=median exceeds memory budget ({} MiB limit); reduce width, height, or band count",
                    MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                )));
            }
            fold.add_le_bytes(&raw);
            used_assets.push(u.clone());
            if fold.is_done() {
                break;
            }
            continue;
        }
        let raw = match render_crs_bbox(
            engine,
            &cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            &bands,
            resample,
            OutputFormat::RawRgba,
            pq.nodata,
            custom_lut.as_ref(),
            cf.as_ref(),
            true,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        };
        any_ok = true;
        used_assets.push(u.clone());
        if raw.len() != total_px * 4 {
            continue;
        }
        let acc = acc_rgba.get_or_insert_with(|| vec![0u8; total_px * 4]);
        for i in 0..total_px {
            let off = i * 4;
            if acc[off + 3] == 0 && raw[off + 3] > 0 {
                acc[off..off + 4].copy_from_slice(&raw[off..off + 4]);
                if raw[off + 3] == 255 {
                    opaque_count += 1;
                }
            }
        }
        if opaque_count >= total_px {
            break;
        }
    }
    if !any_ok {
        return Err(last_err.unwrap_or_else(|| {
            EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
        }));
    }
    let final_fmt = eff_fmt.unwrap_or(fmt);
    if matches!(final_fmt, OutputFormat::RawF32 | OutputFormat::Npy) || method != Selection::First {
        let (out_nb, mut combined): (usize, Vec<f32>) = match fold.and_then(Fold::finish) {
            Some(Selected { bands, values }) => (bands, values),
            None => {
                let nb = first_bands
                    .as_ref()
                    .map(|b| default_raw_band_count(b, method))
                    .unwrap_or(1);
                (nb, vec![f32::NAN; nb * total_px])
            }
        };
        if method == Selection::Count
            && let Some(mask) = poly_mask
        {
            let kept = mask.rasterize_all_touched(out_w as usize, out_h as usize, bbox, dst_crs);
            for (v, &inside) in combined.iter_mut().zip(&kept) {
                if !inside {
                    *v = f32::NAN;
                }
            }
        }
        if matches!(final_fmt, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut raw_buf = bytes::BytesMut::with_capacity(combined.len() * 4);
            for v in combined {
                raw_buf.extend_from_slice(&v.to_le_bytes());
            }
            let raw = raw_buf.freeze();
            if matches!(final_fmt, OutputFormat::Npy) {
                return Ok((
                    cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                        raw,
                        out_nb,
                        out_h,
                        out_w,
                        pq.return_mask,
                    ),
                    used_assets,
                ));
            }
            return Ok((
                wrap_geotiff_fmt(
                    raw,
                    ct,
                    final_fmt,
                    out_w,
                    out_h,
                    bbox,
                    epsg,
                    pq.return_mask,
                ),
                used_assets,
            ));
        }
        let fallback_bands = BandRequest::Single {
            band_1based: 1,
            rescale: (0.0, 255.0),
            colormap: pq.colormap,
        };
        let paint_bands = first_bands.as_ref().unwrap_or(&fallback_bands);
        let rgba = paint_selected(
            Selected {
                bands: out_nb,
                values: combined,
            },
            total_px,
            paint_bands,
            pq,
            first_lut.as_ref(),
        )?;
        let data = encode_rgba_output(out_w as usize, out_h as usize, &rgba, final_fmt, pq.return_mask);
        return Ok((
            wrap_geotiff_fmt(
                data,
                ct,
                final_fmt,
                out_w,
                out_h,
                bbox,
                epsg,
                pq.return_mask,
            ),
            used_assets,
        ));
    }
    let acc = acc_rgba.unwrap_or_else(|| vec![0u8; total_px * 4]);
    #[cfg(feature = "gpu")]
    let data = if let Some(gpu) = engine.gpu.as_ref() {
        gpu.encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask)
    } else {
        encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask)
    };
    #[cfg(not(feature = "gpu"))]
    let data = encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask);
    Ok((
        wrap_geotiff_fmt(
            data,
            ct,
            final_fmt,
            out_w,
            out_h,
            bbox,
            epsg,
            pq.return_mask,
        ),
        used_assets,
    ))
}

/// Compute statistics over a mosaicked set of assets folded through `pixel_selection::Fold`
/// (honouring all 9 `pixel_selection` methods, `bidx`, `unscale`, `nodata`, `width`/`height`,
/// `max_size`, and optional `poly_mask`), returning `(stats_json, used_assets)`.
async fn compute_mosaic_statistics_with_fold(
    engine: &TileEngine,
    urls: &[String],
    wgs_bbox: [f64; 4],
    poly_mask: Option<&PolygonMask>,
    pq: &ParsedQuery,
    stats_opts: &crate::titiler::StatsOptions,
) -> Result<(serde_json::Value, Vec<String>), EngineError> {
    if urls.is_empty() {
        return Err(EngineError::NotFound(
            "no mosaic scenes intersect requested geometry".into(),
        ));
    }
    let urls = &urls[..urls.len().min(64)];
    let method = pq.pixel_selection;
    let dst_crs = cogzilla_proj::CrsModel::Geographic;
    let resample = pq.resample;

    // Exact analytical native-grid path for POST /mosaicjson/statistics when no explicit
    // output width/height resampling was requested and all contributing assets share one grid
    // (always true for a single-asset mosaic, guaranteeing exact parity with /cog/statistics).
    if let Some(mask) = poly_mask
        && pq.width.is_none()
        && pq.height.is_none()
    {
        let mut opened: Vec<(String, Arc<cogzilla_io::OpenedCog>)> =
            Vec::with_capacity(urls.len());
        let mut open_err: Option<EngineError> = None;
        for u in urls {
            match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                Ok(c) => opened.push((u.clone(), c)),
                Err(e) if e.is_fatal() => {
                    return Err(EngineError::Io(e));
                }
                Err(e) => open_err = Some(crate::engine::asset_fetch_error(u, e)),
            }
        }
        if opened.is_empty() {
            return Err(open_err.unwrap_or_else(|| {
                EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
            }));
        }
        let mut exact_opts = stats_opts.clone();
        if exact_opts.hist_dtype.is_none() {
            let band_dtype = crate::histogram::HistDtype::for_sample_type(
                opened[0].1.index.full_res().sample_type,
                pq.unscale,
            );
            exact_opts.hist_dtype = Some(if method == Selection::Count {
                crate::histogram::HistDtype::Float64
            } else if let Some(expr) = pq.expr_raw.as_deref() {
                crate::histogram::HistDtype::for_expression(expr, band_dtype)
            } else {
                band_dtype
            });
        }
        if let Some(exact) = crate::zonal::compute_native_exact_mosaic_fold_stats(
            engine,
            &opened,
            mask,
            pq.max_size,
            pq.unscale,
            pq.nodata,
            method,
            MAX_MEDIAN_BUFFER_BYTES,
            &exact_opts,
        )
        .await?
        {
            return Ok(exact);
        }
    }

    let mut eff_bbox = wgs_bbox;
    let mut out_dims: Option<(u32, u32)> = None;
    let mut first_sample_type: Option<cogzilla_tiff::SampleType> = None;
    let mut selected_bidx: Vec<u16> = Vec::new();
    let mut band_descs: Vec<String> = Vec::new();
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;
    let mut used_assets: Vec<String> = Vec::new();

    for (scene_i, u) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                last_err = Some(crate::engine::asset_fetch_error(u, e));
                continue;
            }
        };
        let cogs = [cog];
        let full = cogs[0].index.full_res();
        first_sample_type.get_or_insert(full.sample_type);
        let (out_w, out_h) = match out_dims {
            Some(d) => d,
            None => {
                // Expand degenerate Point / axis-aligned LineString bboxes by one native
                // pixel along the zero-extent axis so render_crs_bbox can sample them.
                if eff_bbox[2] <= eff_bbox[0] || eff_bbox[3] <= eff_bbox[1] {
                    let (px_dx, px_dy) = if matches!(
                        engine.get_or_build_crs(&cogs[0]),
                        Ok(cogzilla_proj::CrsModel::Geographic)
                    ) && full.pixel_to_crs.a.abs() > 1e-12
                        && full.pixel_to_crs.e.abs() > 1e-12
                    {
                        (full.pixel_to_crs.a.abs(), full.pixel_to_crs.e.abs())
                    } else if let Ok(crs) = engine.get_or_build_crs(&cogs[0]) {
                        let b = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        (
                            ((b[2] - b[0]).abs() / full.width.max(1) as f64).max(1e-6),
                            ((b[3] - b[1]).abs() / full.height.max(1) as f64).max(1e-6),
                        )
                    } else {
                        (1e-5, 1e-5)
                    };
                    if eff_bbox[2] <= eff_bbox[0] {
                        eff_bbox[0] -= 0.5 * px_dx;
                        eff_bbox[2] += 0.5 * px_dx;
                    }
                    if eff_bbox[3] <= eff_bbox[1] {
                        eff_bbox[1] -= 0.5 * px_dy;
                        eff_bbox[3] += 0.5 * px_dy;
                    }
                }
                let cap = if pq.max_size == 0 {
                    4096
                } else {
                    pq.max_size.clamp(1, 4096)
                };
                let (def_w, def_h) = if matches!(
                    engine.get_or_build_crs(&cogs[0]),
                    Ok(cogzilla_proj::CrsModel::Geographic)
                ) && full.pixel_to_crs.b.abs() < 1e-12
                    && full.pixel_to_crs.d.abs() < 1e-12
                    && full.pixel_to_crs.a.abs() > 1e-12
                    && full.pixel_to_crs.e.abs() > 1e-12
                {
                    let nat_w = ((eff_bbox[2] - eff_bbox[0]).abs() / full.pixel_to_crs.a.abs())
                        .round()
                        .max(1.0) as u32;
                    let nat_h = ((eff_bbox[3] - eff_bbox[1]).abs() / full.pixel_to_crs.e.abs())
                        .round()
                        .max(1.0) as u32;
                    if nat_w.max(nat_h) <= cap {
                        (nat_w, nat_h)
                    } else if nat_w >= nat_h {
                        let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
                            .ceil()
                            .clamp(1.0, cap as f64) as u32;
                        (cap, h)
                    } else {
                        let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
                            .ceil()
                            .clamp(1.0, cap as f64) as u32;
                        (w, cap)
                    }
                } else {
                    capped_native_bbox_dims(engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, eff_bbox, eff_bbox, cap)
                };
                let (w, h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let dims = (w.clamp(1, 4096), h.clamp(1, 4096));
                out_dims = Some(dims);
                selected_bidx = if pq.bidx.is_empty() {
                    (1..=data_band_count(full).max(1)).collect()
                } else {
                    for &b in &pq.bidx {
                        if b == 0 || b > full.samples_per_pixel {
                            return Err(EngineError::BadRequest(format!(
                                "band index b{b} out of range (dataset has {} bands)",
                                full.samples_per_pixel
                            )));
                        }
                    }
                    pq.bidx.clone()
                };
                band_descs = selected_bidx
                    .iter()
                    .map(|&b| {
                        cogs[0]
                            .index
                            .band_descriptions
                            .get((b - 1) as usize)
                            .filter(|s| !s.is_empty())
                            .cloned()
                            .unwrap_or_else(|| format!("b{b}"))
                    })
                    .collect();
                dims
            }
        };
        let total_px = (out_w as usize).saturating_mul(out_h as usize);
        let bands = BandRequest::MultiBand {
            bands_1based: selected_bidx.clone(),
        };
        let raw = match render_crs_bbox(
            engine,
            &cogs,
            eff_bbox,
            &dst_crs,
            out_w,
            out_h,
            None,
            &bands,
            resample,
            OutputFormat::RawF32,
            pq.nodata,
            None,
            None,
            true,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        };
        any_ok = true;
        if total_px == 0 || raw.len() % (total_px * 4) != 0 {
            continue;
        }
        let nb = raw.len() / (total_px * 4);
        if nb == 0 {
            continue;
        }
        let fold = match &mut fold {
            Some(f) => f,
            None => {
                if method.by_last_band() && nb < 2 {
                    return Err(EngineError::BadRequest(
                        "pixel_selection by the last band needs at least two bands".into(),
                    ));
                }
                if Fold::<f32>::working_bytes(method, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce max_size, width, height, or band count",
                        MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                    )));
                }
                let integer = reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                fold.insert(Fold::new(method, nb, total_px, integer).absent_as(absent))
            }
        };
        if fold.bands() != nb {
            continue;
        }
        taken += 1;
        if Fold::<f32>::held_bytes(method, taken, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
            return Err(EngineError::BadRequest(format!(
                "pixel_selection=median exceeds memory budget ({} MiB limit); reduce max_size, width, height, or band count",
                MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
            )));
        }
        fold.add_le_bytes(&raw);
        used_assets.push(u.clone());
        if fold.is_done() {
            break;
        }
    }

    if !any_ok {
        return Err(last_err.unwrap_or_else(|| {
            EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
        }));
    }

    let (out_w, out_h) = out_dims.unwrap_or((1, 1));
    let total_px = (out_w as usize).saturating_mul(out_h as usize);
    let (out_nb, combined): (usize, Vec<f32>) = match fold.and_then(Fold::finish) {
        Some(Selected { bands, values }) => (bands, values),
        None => {
            let nb = if method == Selection::Count {
                1
            } else if method.by_last_band() {
                selected_bidx.len().saturating_sub(1).max(1)
            } else {
                selected_bidx.len().max(1)
            };
            (nb, vec![f32::NAN; nb * total_px])
        }
    };
    let mut eff_opts = stats_opts.clone();
    if eff_opts.hist_dtype.is_none()
        && let Some(st) = first_sample_type
    {
        let band_dtype = crate::histogram::HistDtype::for_sample_type(st, pq.unscale);
        eff_opts.hist_dtype = Some(if method == Selection::Count {
            crate::histogram::HistDtype::Float64
        } else if let Some(expr) = pq.expr_raw.as_deref() {
            crate::histogram::HistDtype::for_expression(expr, band_dtype)
        } else {
            band_dtype
        });
    }
    if let Some(mask) = poly_mask {
        let mut keys_descs: Vec<(String, String)> = Vec::with_capacity(out_nb);
        for bi in 0..out_nb {
            let key = if method == Selection::Count {
                "b1".to_string()
            } else {
                let b_num = selected_bidx.get(bi).copied().unwrap_or((bi + 1) as u16);
                format!("b{b_num}")
            };
            let desc = if method == Selection::Count {
                "count".to_string()
            } else {
                band_descs.get(bi).cloned().unwrap_or_else(|| key.clone())
            };
            keys_descs.push((key, desc));
        }
        let plane_refs: Vec<(&str, &str, &[f32])> = keys_descs
            .iter()
            .enumerate()
            .map(|(bi, (k, d))| {
                (
                    k.as_str(),
                    d.as_str(),
                    &combined[bi * total_px..(bi + 1) * total_px],
                )
            })
            .collect();
        let json = crate::zonal::compute_exact_grid_polygon_stats(
            &plane_refs,
            eff_bbox,
            out_w as usize,
            out_h as usize,
            mask,
            &eff_opts,
        );
        return Ok((json, used_assets));
    }

    let mut out = serde_json::Map::new();
    for bi in 0..out_nb {
        let plane = &combined[bi * total_px..(bi + 1) * total_px];
        let key = if method == Selection::Count {
            "b1".to_string()
        } else {
            let b_num = selected_bidx.get(bi).copied().unwrap_or((bi + 1) as u16);
            format!("b{b_num}")
        };
        let desc = if method == Selection::Count {
            "count".to_string()
        } else {
            band_descs.get(bi).cloned().unwrap_or_else(|| key.clone())
        };
        out.insert(key, compute_f32_band_stats(plane, &desc, &eff_opts));
    }
    Ok((serde_json::Value::Object(out), used_assets))
}

fn web_mercator_buffered_3857_bbox(z: u8, x: u32, y: u32, buffer: f32, tile_size: u32) -> [f64; 4] {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
    let pad_x = (max_x - min_x) * (buffer as f64 / tile_size as f64);
    let pad_y = (max_y - min_y) * (buffer as f64 / tile_size as f64);
    [min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y]
}

fn resolve_custom_lut(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> Result<Option<CustomColormap>, EngineError> {
    if let Some(ref j) = pq.custom_colormap_json {
        return CustomColormap::from_json_str_with_type(
            j,
            !pq.rescale.is_empty(),
            pq.colormap_type_linear,
        )
        .map(Some)
        .map_err(EngineError::BadRequest);
    }
    if !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && let Some(ref pal) = cog.index.color_map
    {
        return Ok(Some(CustomColormap::from_rgba_palette(pal)));
    }
    Ok(None)
}

fn has_embedded_palette(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> bool {
    pq.custom_colormap_json.is_none()
        && !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && cog.index.color_map.is_some()
}

fn plain_resp(
    status: StatusCode,
    body: impl Into<Bytes>,
    content_type: &'static str,
) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(body.into()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    resp.headers_mut()
        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

async fn load_mosaicjson(
    engine: &TileEngine,
    uri: &str,
) -> Result<Arc<MosaicJsonCatalog>, EngineError> {
    cogzilla_io::check_uri_allowed(uri)?;
    if let Some(cached) = engine.mosaicjson_cache.get(uri) {
        return Ok(cached.clone());
    }
    let bytes = engine
        .store
        .fetch_uri_bytes(uri, MAX_POST_BODY_BYTES)
        .await?;
    let decoded_buf: Vec<u8>;
    let json_bytes: &[u8] = if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        decoded_buf = cogzilla_codec::decompress_gzip_limited(&bytes, MAX_POST_BODY_BYTES)
            .map_err(|e| EngineError::BadRequest(format!("invalid gzipped MosaicJSON: {e}")))?;
        &decoded_buf
    } else {
        &bytes
    };
    let text = String::from_utf8_lossy(json_bytes);
    let cat = Arc::new(MosaicJsonCatalog::from_json_str(&text).map_err(EngineError::BadRequest)?);
    if engine.mosaicjson_cache.len() >= 256 {
        engine.mosaicjson_cache.clear();
    }
    engine.mosaicjson_cache.insert(uri.to_string(), cat.clone());
    Ok(cat)
}

const MAX_POST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Maximum features accepted in a `POST /*/statistics` `FeatureCollection` (`0` = unlimited,
/// still bounded by `MAX_POST_BODY_BYTES` and per-feature window/block caps).
fn max_statistics_features() -> usize {
    crate::config::Config::global().max_statistics_features
}

/// Timeout in seconds for reading a POST request body (`0` disables the timeout).
fn body_timeout_secs() -> u64 {
    crate::config::Config::global().body_timeout_secs
}

fn check_feature_collection_len(len: usize) -> Result<(), EngineError> {
    let cap = max_statistics_features();
    if cap > 0 && len > cap {
        return Err(EngineError::BadRequest(format!(
            "FeatureCollection exceeds max {cap} features (got {len})"
        )));
    }
    Ok(())
}

fn is_null_geometry_feature(feat: &serde_json::Value) -> bool {
    feat.get("type").and_then(|v| v.as_str()) == Some("Feature")
        && matches!(feat.get("geometry"), Some(serde_json::Value::Null))
}

fn parse_feature_mask_or_null(
    feat: &serde_json::Value,
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<(PolygonMask, [f64; 4]), EngineError> {
    if is_null_geometry_feature(feat) {
        return Ok((PolygonMask::empty(), [0.0; 4]));
    }
    PolygonMask::from_geojson_in_crs(feat, coord_crs)
}

fn wrap_feature_error(idx: usize, feat: &serde_json::Value, err: EngineError) -> EngineError {
    let prefix = match feat.get("id") {
        Some(serde_json::Value::String(s)) => format!("features[{idx}] (id={s:?})"),
        Some(v) if !v.is_null() => format!("features[{idx}] (id={v})"),
        _ => format!("features[{idx}]"),
    };
    match err {
        EngineError::BadRequest(msg) => EngineError::BadRequest(format!("{prefix}: {msg}")),
        other => other,
    }
}

fn insert_feature_stats(feat: &mut serde_json::Value, stats: serde_json::Value) {
    if let Some(props) = feat.get_mut("properties").and_then(|v| v.as_object_mut()) {
        props.insert("statistics".into(), stats);
    } else if let Some(obj) = feat.as_object_mut() {
        obj.insert(
            "properties".into(),
            serde_json::json!({"statistics": stats}),
        );
    }
}

fn fc_worker_count(n: usize) -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(2, 16)
        .min(n)
}

/// Shared worker pool for `POST /*/statistics` on GeoJSON `FeatureCollection` payloads.
/// Parses every feature's polygon mask against `fc_crs`, runs `compute` across bounded
/// tokio workers, inserts each feature's `"statistics"` property, and returns per-feature
/// sidecar outputs `T` (e.g. used asset lists for `X-Assets`).
async fn run_feature_collection_stats<T, F, Fut>(
    geojson: &mut serde_json::Value,
    fc_crs: Option<&cogzilla_proj::CrsModel>,
    compute: F,
) -> Result<Vec<T>, EngineError>
where
    T: Send + 'static,
    F: Fn(Arc<PolygonMask>, [f64; 4]) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<(serde_json::Value, T), EngineError>> + Send + 'static,
{
    let features = geojson
        .get_mut("features")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| EngineError::BadRequest("FeatureCollection missing 'features' array".into()))?;
    check_feature_collection_len(features.len())?;
    let n_feats = features.len();
    let mut parsed = Vec::with_capacity(n_feats);
    for (idx, feat) in features.iter().enumerate() {
        let (mask, bbox) = parse_feature_mask_or_null(feat, fc_crs)
            .map_err(|e| wrap_feature_error(idx, feat, e))?;
        parsed.push((Arc::new(mask), bbox));
    }
    let mut extras = Vec::with_capacity(n_feats);
    if n_feats <= 1 {
        for (idx, (mask, bbox)) in parsed.into_iter().enumerate() {
            let (stats, extra) = compute(mask, bbox)
                .await
                .map_err(|e| wrap_feature_error(idx, &features[idx], e))?;
            insert_feature_stats(&mut features[idx], stats);
            extras.push(extra);
        }
    } else {
        let parsed = Arc::new(parsed);
        let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let workers = fc_worker_count(n_feats);
        let compute = Arc::new(compute);
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..workers {
            let parsed_c = parsed.clone();
            let next_c = next_idx.clone();
            let compute_c = compute.clone();
            set.spawn(async move {
                let mut local = Vec::new();
                loop {
                    let idx = next_c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= parsed_c.len() {
                        break;
                    }
                    let (ref mask, bbox) = parsed_c[idx];
                    let (stats, extra) = compute_c(mask.clone(), bbox)
                        .await
                        .map_err(|e| (idx, e))?;
                    local.push((idx, stats, extra));
                }
                Ok::<_, (usize, EngineError)>(local)
            });
        }
        while let Some(joined) = set.join_next().await {
            let batch = joined
                .map_err(|e| EngineError::BadRequest(format!("statistics worker failed: {e}")))?
                .map_err(|(idx, e)| wrap_feature_error(idx, &features[idx], e))?;
            for (idx, stats, extra) in batch {
                insert_feature_stats(&mut features[idx], stats);
                extras.push(extra);
            }
        }
    }
    Ok(extras)
}

async fn read_limited_body_with_timeout<B>(body: B, timeout_secs: u64) -> Result<Bytes, EngineError>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    use bytes::Buf;
    let read_fut = async move {
        let mut out = bytes::BytesMut::new();
        tokio::pin!(body);
        while let Some(frame_res) = body.frame().await {
            let frame =
                frame_res.map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?;
            if let Ok(mut data) = frame.into_data() {
                if out.len().saturating_add(data.remaining()) > MAX_POST_BODY_BYTES {
                    return Err(EngineError::PayloadTooLarge(format!(
                        "request body exceeds {} MiB limit",
                        MAX_POST_BODY_BYTES / (1024 * 1024)
                    )));
                }
                while data.has_remaining() {
                    let chunk = data.chunk();
                    out.extend_from_slice(chunk);
                    let n = chunk.len();
                    data.advance(n);
                }
            }
        }
        Ok(out.freeze())
    };
    if timeout_secs == 0 {
        read_fut.await
    } else {
        tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), read_fut)
            .await
            .map_err(|_| {
                EngineError::RequestTimeout(format!(
                    "request body read timed out after {timeout_secs}s"
                ))
            })?
    }
}

async fn read_limited_body<B>(body: B) -> Result<Bytes, EngineError>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    read_limited_body_with_timeout(body, body_timeout_secs()).await
}

/// Open the first readable scene in a mosaic candidate list so a single missing/corrupt scene at
/// index 0 does not fail the entire mosaic tile when subsequent scenes are valid.
async fn open_first_mosaic_cog(
    engine: &TileEngine,
    urls: &[String],
) -> Result<Arc<cogzilla_io::OpenedCog>, EngineError> {
    let mut last_err: Option<EngineError> = None;
    for u in urls.iter().take(16) {
        match engine.store.open_uri(u).await {
            Ok(c) => return Ok(c),
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => last_err = Some(crate::engine::asset_fetch_error(u, e)),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        EngineError::BadGateway("mosaic contains no readable scenes".into())
    }))
}

/// Open the `?url=` rasters; `nodata` (the request's `?nodata=`) replaces each file's own
/// nodata, as GDAL's override does (see [`cogzilla_io::CogStore::with_nodata`]).
async fn open_query_urls(
    engine: &TileEngine,
    urls: &[String],
    nodata: Option<f32>,
) -> Result<Vec<Arc<cogzilla_io::OpenedCog>>, EngineError> {
    if urls.is_empty() {
        return Err(EngineError::BadRequest(
            "missing ?url=... query parameter".to_string(),
        ));
    }
    if urls.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "too many ?url= parameters ({}, maximum 16)",
            urls.len()
        )));
    }
    if urls.len() == 1 {
        let cog = engine.store.open_uri(&urls[0]).await?;
        return Ok(vec![engine.store.with_nodata(&cog, nodata)]);
    }
    let mut js = tokio::task::JoinSet::new();
    for (idx, u) in urls.iter().enumerate() {
        let store = engine.store.clone();
        let url = u.clone();
        js.spawn(async move { Ok::<_, cogzilla_io::IoError>((idx, store.open_uri(&url).await?)) });
    }
    let mut slots: Vec<Option<Arc<cogzilla_io::OpenedCog>>> = vec![None; urls.len()];
    while let Some(res) = js.join_next().await {
        let (idx, cog) = res.map_err(|e| {
            EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                "open_uri task failed: {e}"
            ))))
        })??;
        slots[idx] = Some(cog);
    }
    Ok(slots.into_iter().map(|s| engine.store.with_nodata(&s.unwrap(), nodata)).collect())
}

/// Convenience entry point for serverless / embedded runtimes (such as AWS Lambda)
/// that already hold the request body in memory as `Bytes`.
pub async fn handle_bytes_request(
    engine: Arc<TileEngine>,
    req: Request<Bytes>,
) -> Response<Bytes> {
    let (parts, body) = req.into_parts();
    let req_full = Request::from_parts(parts, Full::new(body));
    let Ok(resp) = handle_request(engine, req_full).await;
    let (rparts, rbody) = resp.into_parts();
    let bytes = rbody
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();
    Response::from_parts(rparts, bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreferredEncoding {
    None,
    Gzip,
    Brotli,
}

/// Negotiate `Accept-Encoding` across all header lines per RFC 9110 §5.3 & §12.5.3,
/// supporting `br`, `gzip`/`x-gzip`, `identity`, and `*`. Prefers `br` over `gzip` when both
/// have equal explicit positive q-values, and falls back to `gzip` for bare `*`.
fn negotiate_encoding<'a>(vals: impl IntoIterator<Item = &'a str>) -> PreferredEncoding {
    let mut br_q: Option<f32> = None;
    let mut gzip_q: Option<f32> = None;
    let mut identity_q: Option<f32> = None;
    let mut star_q: Option<f32> = None;
    for val in vals {
        for part in val.split(',') {
            let mut iter = part.split(';');
            let enc = iter.next().unwrap_or("").trim();
            let is_br = enc.eq_ignore_ascii_case("br");
            let is_gzip = enc.eq_ignore_ascii_case("gzip") || enc.eq_ignore_ascii_case("x-gzip");
            let is_identity = enc.eq_ignore_ascii_case("identity");
            let is_star = enc == "*";
            if !is_br && !is_gzip && !is_identity && !is_star {
                continue;
            }
            let mut q_val = 1.0f32;
            let mut valid = true;
            for param in iter {
                let p = param.trim();
                if let Some(qv) = p.strip_prefix("q=").or_else(|| p.strip_prefix("Q=")) {
                    match qv.trim().parse::<f32>() {
                        Ok(q) if q.is_finite() && (0.0..=1.0).contains(&q) => q_val = q,
                        _ => {
                            valid = false;
                            break;
                        }
                    }
                }
            }
            if !valid {
                continue;
            }
            if is_br {
                br_q = Some(br_q.map_or(q_val, |prev| prev.max(q_val)));
            } else if is_gzip {
                gzip_q = Some(gzip_q.map_or(q_val, |prev| prev.max(q_val)));
            } else if is_identity {
                identity_q = Some(identity_q.map_or(q_val, |prev| prev.max(q_val)));
            } else {
                star_q = Some(star_q.map_or(q_val, |prev| prev.max(q_val)));
            }
        }
    }
    let eff_br = br_q.or(star_q).unwrap_or(0.0);
    let eff_gzip = gzip_q.or(star_q).unwrap_or(0.0);
    let best_compressed = eff_br.max(eff_gzip);
    if best_compressed <= 0.0 || identity_q.is_some_and(|iq| iq > best_compressed) {
        return PreferredEncoding::None;
    }
    if eff_br > eff_gzip {
        PreferredEncoding::Brotli
    } else if eff_gzip > eff_br {
        PreferredEncoding::Gzip
    } else if br_q.is_some() {
        PreferredEncoding::Brotli
    } else {
        PreferredEncoding::Gzip
    }
}

fn is_compressible_content_type(ct: &str) -> bool {
    let media = ct.split(';').next().unwrap_or("").trim();
    media.eq_ignore_ascii_case("application/json")
        || media.eq_ignore_ascii_case("application/geo+json")
        || media.eq_ignore_ascii_case("application/xml")
        || media.eq_ignore_ascii_case("application/vnd.oai.openapi+json")
        || media.eq_ignore_ascii_case("application/x-binary")
        || media.eq_ignore_ascii_case("application/octet-stream")
        || media.eq_ignore_ascii_case("image/tiff")
        || media.eq_ignore_ascii_case("image/svg+xml")
        || media.starts_with("text/")
}

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Extract a valid caller-supplied `X-Request-Id` header (1..=128 printable non-whitespace ASCII
/// bytes) or generate a compact lock-free hex ID (`<8-hex-process-seed>-<12-hex-seq>`).
fn resolve_request_id(headers: &hyper::HeaderMap) -> HeaderValue {
    if let Some(hv) = headers.get(&X_REQUEST_ID) {
        let bytes = hv.as_bytes();
        if !bytes.is_empty()
            && bytes.len() <= 128
            && bytes.iter().all(|&b| (0x21..=0x7E).contains(&b))
        {
            return hv.clone();
        }
    }
    static SEED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seed = *SEED.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let pid = std::process::id() as u64;
        (nanos ^ (nanos >> 32) ^ (pid.wrapping_mul(0x9E37_79B9))) as u32
    });
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) & 0x0000_FFFF_FFFF_FFFF;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 21];
    for i in 0..8 {
        buf[7 - i] = HEX[((seed >> (i * 4)) & 0xF) as usize];
    }
    buf[8] = b'-';
    for i in 0..12 {
        buf[20 - i] = HEX[((seq >> (i * 4)) & 0xF) as usize];
    }
    HeaderValue::from_bytes(&buf).expect("ASCII hex request id")
}

pub async fn handle_request<B>(
    engine: Arc<TileEngine>,
    mut req: Request<B>,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let t0 = std::time::Instant::now();
    let req_id = resolve_request_id(req.headers());
    let req_method = req.method().clone();
    // O(1) `Bytes`-backed URI clone (zero heap allocation) for structured access logging.
    let req_uri = req.uri().clone();
    let is_head = req_method == Method::HEAD;
    let is_post = req_method == Method::POST;
    let is_get_or_head = matches!(req_method, Method::GET | Method::HEAD);
    let is_health_or_metrics =
        matches!(req_uri.path(), "/healthz" | "/readyz" | "/metrics" | "/stats");
    let is_control_plane = is_health_or_metrics
        || matches!(
            req_uri.path(),
            "/admin/clear-cache" | "/searches" | "/searches/list" | "/searches/register"
        );
    let bypass_aimd = is_control_plane || req_method == Method::OPTIONS;
    let mut aimd_permit = if bypass_aimd {
        None
    } else {
        match engine.aimd.try_acquire() {
            Some(permit) => Some(permit),
            None => {
                let body = if is_head {
                    Bytes::new()
                } else {
                    Bytes::from_static(br#"{"detail":"server overloaded; retry shortly"}"#)
                };
                let mut resp = plain_resp(
                    StatusCode::TOO_MANY_REQUESTS,
                    body,
                    "application/json",
                );
                resp.headers_mut()
                    .insert(hyper::header::RETRY_AFTER, HeaderValue::from_static("1"));
                if let Ok(hv) = HeaderValue::from_str(&engine.aimd.current_limit().to_string()) {
                    resp.headers_mut()
                        .insert(HeaderName::from_static("x-cogzilla-aimd-limit"), hv);
                }
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    tracing::debug!(
                        request_id = req_id.to_str().unwrap_or(""),
                        method = %req_method,
                        path = %req_uri.path(),
                        status = 429u16,
                        dur_ms = format_args!("{dur_ms:.2}"),
                        "http request"
                    );
                }
                resp.headers_mut().insert(X_REQUEST_ID, req_id);
                resp.headers_mut().insert(
                    hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS,
                    HeaderValue::from_static("X-Request-Id"),
                );
                return Ok(resp);
            }
        }
    };
    let preferred_enc = negotiate_encoding(
        req.headers()
            .get_all(hyper::header::ACCEPT_ENCODING)
            .iter()
            .filter_map(|v| v.to_str().ok()),
    );
    engine.sync_tile_cache_epoch();
    let retry_head = if is_get_or_head {
        Some((req.method().clone(), req.uri().clone(), req.headers().clone()))
    } else {
        None
    };
    // `f=html` or `Accept: text/html` on a route that describes something: its JSON, written out as a page.
    let offers_page = is_get_or_head && html::offers_html(req.uri().path());
    let accept_hdr = req
        .headers()
        .get(hyper::header::ACCEPT)
        .and_then(|v| v.to_str().ok());
    let raw_q = req.uri().query().unwrap_or("");
    let colormap_prefers_image = {
        let mut parts = req.uri().path().trim_matches('/').split('/').filter(|s| !s.is_empty());
        matches!((parts.next(), parts.next(), parts.next()), (Some("colorMaps"), Some(_), None))
            && html::explicit_format(raw_q).is_none()
            && accept_hdr.and_then(html::negotiate_colormap_accept).is_some()
    };
    let as_page = (offers_page
        && !colormap_prefers_image
        && html::wants_html(raw_q, accept_hdr))
    .then(|| (req.uri().path().to_string(), raw_q.to_string()));
    // Prevent a cached JSON ETag in `If-None-Match` from short-circuiting an HTML page request to 304.
    if as_page.is_some() {
        req.headers_mut().remove(IF_NONE_MATCH);
    }
    let mut resp = Box::pin(handle_request_inner(engine.clone(), req, is_head)).await?;
    if resp.headers_mut().remove("x-cogzilla-retry").is_some()
        && let Some((m, u, h)) = retry_head
    {
        engine.sync_tile_cache_epoch();
        let mut retry_req = Request::new(Full::new(Bytes::new()));
        *retry_req.method_mut() = m;
        *retry_req.uri_mut() = u;
        *retry_req.headers_mut() = h;
        resp = Box::pin(handle_request_inner(engine.clone(), retry_req, is_head)).await?;
        resp.headers_mut().remove("x-cogzilla-retry");
    }
    if let Some(permit) = aimd_permit.as_mut()
        && (resp.status().is_client_error()
            || resp.status().is_server_error()
            || resp.status() == StatusCode::NOT_MODIFIED
            || resp.status() == StatusCode::NO_CONTENT)
    {
        permit.ignore_rtt();
    }
    drop(aimd_permit);
    if let Some((path, query)) = as_page
        && resp.status() == StatusCode::OK
        && resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("application/json"))
    {
        let (mut parts, body) = resp.into_parts();
        let raw = body.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
        resp = match serde_json::from_slice::<serde_json::Value>(&raw) {
            Ok(doc) => {
                let page = html::page(&path, &query, &doc);
                parts.headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
                parts.headers.remove(hyper::header::CONTENT_LENGTH);
                parts.headers.remove(ETAG);
                Response::from_parts(parts, Full::new(Bytes::from(page)))
            }
            Err(_) => Response::from_parts(parts, Full::new(raw)),
        };
    }
    if offers_page && (resp.status().is_success() || resp.status() == StatusCode::NOT_MODIFIED) {
        html::append_vary_accept(resp.headers_mut());
    }
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if let Ok(hv) = HeaderValue::from_str(&format!("total;dur={dur_ms:.2}")) {
        resp.headers_mut().insert(
            hyper::header::HeaderName::from_static("server-timing"),
            hv,
        );
    }
    let status = resp.status();
    if status.is_server_error() {
        tracing::warn!(
            request_id = req_id.to_str().unwrap_or(""),
            method = %req_method,
            path = %req_uri.path(),
            status = status.as_u16(),
            dur_ms = format_args!("{dur_ms:.2}"),
            "http request failed"
        );
    } else if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!(
            request_id = req_id.to_str().unwrap_or(""),
            method = %req_method,
            path = %req_uri.path(),
            status = status.as_u16(),
            dur_ms = format_args!("{dur_ms:.2}"),
            "http request"
        );
    }
    resp.headers_mut().insert(X_REQUEST_ID, req_id);
    if !resp
        .headers()
        .contains_key(hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS)
    {
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS,
            HeaderValue::from_static("X-Request-Id"),
        );
    }
    if is_post {
        // RFC 9110: POST responses depend on the request body and must not carry public Cache-Control/ETag.
        resp.headers_mut().remove(CACHE_CONTROL);
        resp.headers_mut().remove(ETAG);
    } else if is_get_or_head
        && resp.status() == StatusCode::OK
        && !is_health_or_metrics
        && !resp.headers().contains_key(CACHE_CONTROL)
    {
        resp.headers_mut()
            .insert(CACHE_CONTROL, cache_control_header().clone());
    }

    let compressible = resp.status() == StatusCode::OK
        && resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(is_compressible_content_type);
    if compressible {
        if let Some(existing) = resp.headers().get(hyper::header::VARY).and_then(|v| v.to_str().ok()) {
            if !existing.to_ascii_lowercase().contains("accept-encoding")
                && let Ok(hv) = HeaderValue::from_str(&format!("{existing}, Accept-Encoding")) {
                    resp.headers_mut().insert(hyper::header::VARY, hv);
                }
        } else {
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept-Encoding"));
        }
        if preferred_enc != PreferredEncoding::None
            && !is_head
            && !resp.headers().contains_key(hyper::header::CONTENT_ENCODING)
        {
            let (mut parts, body) = resp.into_parts();
            let raw = body
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            let compressed = if raw.len() >= 500 {
                match preferred_enc {
                    PreferredEncoding::Brotli => {
                        cogzilla_codec::compress_brotli(&raw).ok().map(|b| (b, "br"))
                    }
                    PreferredEncoding::Gzip => {
                        cogzilla_codec::compress_gzip(&raw).ok().map(|b| (b, "gzip"))
                    }
                    PreferredEncoding::None => None,
                }
            } else {
                None
            };
            if let Some((enc_bytes, enc_name)) = compressed
                && enc_bytes.len() < raw.len()
            {
                parts.headers.insert(
                    hyper::header::CONTENT_ENCODING,
                    HeaderValue::from_static(enc_name),
                );
                if let Ok(hv) = HeaderValue::from_str(&enc_bytes.len().to_string()) {
                    parts.headers.insert(hyper::header::CONTENT_LENGTH, hv);
                }
                resp = Response::from_parts(parts, Full::new(Bytes::from(enc_bytes)));
            } else {
                resp = Response::from_parts(parts, Full::new(raw));
            }
        }
    }

    if is_head {
        let (mut parts, body) = resp.into_parts();
        if parts.status != StatusCode::NOT_MODIFIED && parts.status != StatusCode::NO_CONTENT
            && let Some(len) = hyper::body::Body::size_hint(&body).exact()
                && let Ok(hv) = HeaderValue::from_str(&len.to_string()) {
                    parts.headers.insert(hyper::header::CONTENT_LENGTH, hv);
                }
        return Ok(Response::from_parts(parts, Full::new(Bytes::new())));
    }
    Ok(resp)
}

async fn handle_request_inner<B>(
    engine: Arc<TileEngine>,
    req: Request<B>,
    is_head: bool,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let proto = req
        .headers()
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
        .map(|s| s.to_ascii_lowercase())
        .or_else(|| {
            req.uri()
                .scheme_str()
                .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
                .map(|s| s.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "http".to_string());
    let valid_host = |s: &&str| {
        !s.is_empty()
            && !s.contains('@')
            && s.parse::<hyper::http::uri::Authority>().is_ok()
    };
    let host_raw = req
        .headers()
        .get("x-forwarded-host")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(valid_host)
        .or_else(|| {
            req.headers()
                .get("host")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim())
                .filter(valid_host)
        })
        .or_else(|| req.uri().authority().map(|a| a.as_str()))
        .unwrap_or("127.0.0.1:8080");
    let forwarded_prefix = req
        .headers()
        .get("x-forwarded-prefix")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().trim_end_matches('/'))
        .filter(|s| {
            !s.is_empty()
                && s.starts_with('/')
                && !s.contains("//")
                && !s.contains("..")
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.' | b'~'))
        })
        .map(|s| s.to_string())
        .unwrap_or_default();
    let host = format!("{proto}://{host_raw}{forwarded_prefix}");
    let if_none_match = req
        .headers()
        .get(IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let method = if is_head {
        Method::GET
    } else {
        req.method().clone()
    };
    let is_post = method == Method::POST;
    let if_none_match = if is_post { None } else { if_none_match };
    let path = req.uri().path().to_string();
    let raw_query = req.uri().query().unwrap_or("").to_string();
    let raw_query = if path.starts_with("/searches/") || path.starts_with("/collections/") {
        pgstac::plain_numbers(&raw_query)
    } else {
        raw_query
    };

    if method == Method::OPTIONS {
        let mut resp = plain_resp(StatusCode::NO_CONTENT, Bytes::new(), "text/plain");
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, HEAD, OPTIONS"),
        );
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Content-Type, If-None-Match, Authorization, X-Request-Id"),
        );
        return Ok(resp);
    }

    if method == Method::GET && path == "/healthz" {
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain"));
    }

    if method == Method::GET && path == "/readyz" {
        if let Some(pg) = engine.pgstac() {
            match tokio::time::timeout(std::time::Duration::from_secs(3), pg.version()).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "readiness check failed: pgSTAC unreachable");
                    let mut resp = plain_resp(
                        StatusCode::SERVICE_UNAVAILABLE,
                        Bytes::from_static(b"not ready: pgSTAC unreachable"),
                        "text/plain",
                    );
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
                    return Ok(resp);
                }
                Err(_) => {
                    tracing::warn!("readiness check failed: pgSTAC timed out");
                    let mut resp = plain_resp(
                        StatusCode::SERVICE_UNAVAILABLE,
                        Bytes::from_static(b"not ready: pgSTAC timed out"),
                        "text/plain",
                    );
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
                    return Ok(resp);
                }
            }
        }
        let mut resp = plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain");
        resp.headers_mut()
            .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        return Ok(resp);
    }

    if method == Method::GET && (path == "/conformance" || path == "/conformance/") {
        let conf = serde_json::json!({
            "conformsTo": [
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/landing-page",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/oas30",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/json",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/html",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tileset",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tilesets-list",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/dataset-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/geodata-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/crs",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/scaling",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/spatial-subsetting",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/json-tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixsetlimits"
            ]
        });
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(conf.to_string()),
            "application/json",
        ));
    }

    if method == Method::GET && (path == "/" || path == "/index.html") {
        let wants_json = raw_query.split('&').any(|kv| kv == "f=json")
            || (path == "/"
                && req
                    .headers()
                    .get("accept")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|a| a.contains("application/json") && !a.contains("text/html")));
        if wants_json {
            let landing = serde_json::json!({
                "title": "cogzilla",
                "description": "Zero-GDAL Rust + SIMD drop-in TiTiler COG, STAC & MosaicJSON tile server",
                "links": [
                    { "href": format!("{host}/"), "rel": "self", "type": "application/json", "title": "Landing Page" },
                    { "href": format!("{host}/api"), "rel": "service-desc", "type": "application/vnd.oai.openapi+json;version=3.0", "title": "OpenAPI definition" },
                    { "href": format!("{host}/api.html"), "rel": "service-doc", "type": "text/html", "title": "Swagger UI" },
                    { "href": format!("{host}/conformance"), "rel": "conformance", "type": "application/json", "title": "OGC API conformance classes" },
                    { "href": format!("{host}/tileMatrixSets"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-schemes", "type": "application/json", "title": "TileMatrixSets" },
                    { "href": format!("{host}/colorMaps"), "rel": "data", "type": "application/json", "title": "ColorMaps" },
                    { "href": format!("{host}/algorithms"), "rel": "data", "type": "application/json", "title": "Algorithms" }
                ]
            });
            let mut resp = plain_resp(
                StatusCode::OK,
                Bytes::from(landing.to_string()),
                "application/json",
            );
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept"));
            return Ok(resp);
        }
        let base_path_js = serde_json::to_string(&forwarded_prefix)
            .unwrap_or_else(|_| "\"\"".to_string())
            .replace("</", "<\\/");
        let html = include_str!("landing.html")
            .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
            .replace("{{BASE_PATH}}", &forwarded_prefix)
            .replace("{{BASE_PATH_JS}}", &base_path_js);
        let mut resp = plain_resp(StatusCode::OK, Bytes::from(html), "text/html; charset=utf-8");
        if path == "/" {
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept"));
        }
        return Ok(resp);
    }

    if method == Method::GET && (path == "/openapi.json" || path == "/api") {
        let ct = if path == "/api" {
            "application/vnd.oai.openapi+json;version=3.1"
        } else {
            "application/json"
        };
        // The shared document, with the pgSTAC routes' paths beside its own.
        let mut body = crate::openapi::openapi_json_bytes();
        if let Ok(mut spec) = serde_json::from_slice::<serde_json::Value>(&body)
            && let (Some(paths), serde_json::Value::Object(extra)) =
                (spec["paths"].as_object_mut(), pgstac::openapi_paths())
        {
            paths.extend(extra);
            body = Bytes::from(spec.to_string());
        }
        return Ok(plain_resp(StatusCode::OK, body, ct));
    }

    if method == Method::GET && (path == "/docs" || path == "/api.html") {
        let spec_url_js = serde_json::to_string(&format!("{forwarded_prefix}/openapi.json"))
            .unwrap_or_else(|_| "\"/openapi.json\"".to_string())
            .replace("</", "<\\/");
        let html = format!(
            r#"<!DOCTYPE html>
<html><head><title>cogzilla API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({{url:{spec_url_js},dom_id:'#swagger-ui'}});</script>
</body></html>"#
        );
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(html),
            "text/html; charset=utf-8",
        ));
    }

    if method == Method::GET && path == "/stats" {
        let body = metrics::render_stats_json(&engine);
        return Ok(plain_resp(StatusCode::OK, Bytes::from(body), "application/json"));
    }

    if method == Method::GET && path == "/metrics" {
        let prom = metrics::render_prometheus_metrics(&engine);
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(prom),
            "text/plain; version=0.0.4; charset=utf-8",
        ));
    }

    if method == Method::POST && path == "/admin/clear-cache" {
        engine.store.clear_open_uris();
        engine.store.cache.clear();
        engine.tile_cache.clear();
        engine.clear_gpu_caches();
        engine.stac_cache.clear();
        engine.mosaicjson_cache.clear();
        zarr::clear_zarr_cache();
        if let Some(pg) = engine.pgstac() {
            pg.clear_caches();
        }
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"cleared"), "text/plain"));
    }

    let mut pq = parse_query(if raw_query.is_empty() {
        None
    } else {
        Some(&raw_query)
    });

    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();

    // titiler-pgstac's item routes, /collections/{c}/items/{i}/..., are the /stac router over
    // an item read from pgSTAC instead of from `url`.
    let mut pgstac_item: Option<Arc<StacItem>> = None;
    let mut pgstac_item_extent: Option<pgstac::ItemExtent> = None;
    let mut stac_prefix = String::from("stac");
    // An item's statistics whose assets carry band math of their own, rewritten
    // as the request's expression (`pgstac::item_asset_math`).
    let mut item_asset_math = false;
    let item_preview: Option<String>;
    let item_rest: Option<Vec<&str>> = match segs.as_slice() {
        ["collections", c, "items", i, rest @ ..] if !rest.is_empty() => {
            let ctx = pgstac::Ctx {
                engine: &engine,
                host: &host,
                forwarded_prefix: &forwarded_prefix,
                path: &path,
                raw_query: &raw_query,
                if_none_match: if_none_match.as_deref(),
                only: None,
            };
            // A feature posted for an item's statistics is read here: those are worked out by
            // the search routes, from the item alone, and never reach `/stac`.
            if method == Method::POST && rest == ["statistics"] {
                let body = match read_limited_body(req.into_body()).await {
                    Ok(body) => body,
                    Err(e) => return Ok(pgstac::engine_error(&e)),
                };
                return Ok(match Box::pin(pgstac::item_route(&ctx, &method, c, i, rest, &pq, Some(body))).await {
                    pgstac::ItemRoute::Done(resp) => resp,
                    pgstac::ItemRoute::Stac(..) => pgstac::bad_request("an item's statistics of a posted feature are worked out from the item alone".into()),
                });
            }
            match Box::pin(pgstac::item_route(&ctx, &method, c, i, rest, &pq, None)).await {
                pgstac::ItemRoute::Done(resp) => return Ok(pgstac_cacheable(resp, &method)),
                pgstac::ItemRoute::Stac(item, extent) => {
                    pgstac_item = Some(item);
                    pgstac_item_extent = extent;
                    // An item is one image: titiler-pgstac's item routes have no
                    // `pixel_selection`, and pass over any value of it.
                    pq.keys.retain(|k| k != "pixel_selection");
                    pq.errors.retain(|e| !e.starts_with("unsupported pixel_selection"));
                    // What the route does not read, titiler-pgstac passes over: not checked,
                    // and not seen by the routes' own checks of what each takes.
                    pq.errors.retain(|e| !pgstac::item_passes_over_error(&method, rest, e));
                    pq.keys.retain(|k| !pgstac::item_passes_over(&method, rest, k));
                    if pgstac::item_passes_over(&method, rest, "algorithm") {
                        (pq.algorithm, pq.algorithm_error, pq.algorithm_params_unread) = (None, None, None);
                    }
                    pq.pixel_selection = Selection::First;
                    // Band math on an item's point is applied to the `/stac` answer by
                    // `pgstac::item_fixup`, with TiTiler's names: `/stac` must not apply it first.
                    // An expression is over every band read, so the request's `bidx` does
                    // not narrow them first; each asset's own selection still applies.
                    match pgstac::item_asset_options(rest, &raw_query) {
                        Err(resp) => return Ok(resp),
                        Ok(Some((per_asset, own))) => {
                            pq.bidx = own;
                            pq.asset_bidx = per_asset;
                            pq.forget_asset_entries();
                        }
                        Ok(None) => {}
                    }
                    // Each asset's band math on a preview or the statistics: one expression
                    // over the stacked bands, read whole.
                    let pairs = pgstac::query_pairs(&raw_query);
                    let has_asset_math = pgstac::asset_math_asked(&pairs);
                    let math = if has_asset_math {
                        pgstac::item_asset_math(rest, &pairs)
                    } else {
                        Ok(None)
                    };
                    match math {
                        Err(message) => return Ok(pgstac::bad_request(message)),
                        Ok(Some((expression, _))) => {
                            item_asset_math = true;
                            pq.asset_math_stages = pgstac::item_asset_math_stages(rest, &pairs);
                            if let Some(names) = pgstac::asset_entry_names(&raw_query) {
                                pq.assets = names;
                            }
                            pq.expr_raw = Some(expression);
                            pq.asset_exprs.clear();
                            pq.asset_bidx.clear();
                            pq.forget_asset_entries();
                            pq.bidx.clear();
                        }
                        Ok(None) => {}
                    }
                    // Each asset's band math on a point is applied by `pgstac::item_fixup` to
                    // every band of the asset, named as such; `asset_as_band` names them there.
                    // A preview's stack is planned over the `assets=` entries, each with its own
                    // band math and bands (`expand_stac_asset_bidx`).
                    // titiler-pgstac reads the listed assets only, not one `asset_bidx` adds.
                    if pgstac::item_preview_route(rest) && pq.asset_entries.iter().any(|e| e.listed) {
                        pq.asset_entries.retain(|e| e.listed);
                        pq.assets = pq.asset_entries.iter().map(|e| e.name.clone()).collect();
                    }
                    if matches!(rest, ["point", _]) && has_asset_math {
                        pgstac::point_asset_math_reads(&mut pq, &raw_query);
                    }
                    if matches!(rest, ["point", _]) && pq.expr_raw.take().is_some() {
                        pq.bidx.clear();
                    }
                    stac_prefix = format!("collections/{c}/items/{i}");
                    let mut as_stac = vec!["stac"];
                    // A preview with a side of zero in its path is the preview sized from
                    // the query, which is what works such a side out.
                    item_preview = pgstac::item_preview_by_query(rest, &mut pq);
                    match &item_preview {
                        Some(preview) => as_stac.push(preview),
                        None => as_stac.extend_from_slice(rest),
                    }
                    Some(as_stac)
                }
            }
        }
        _ => None,
    };
    let segs = item_rest.unwrap_or(segs);

    if let Err(msg) = validate_query(&segs, &pq) {
        if pgstac_item.is_none() {
            let body = serde_json::json!({
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["query"],
                        "msg": msg,
                    }
                ]
            })
            .to_string();
            return Ok(plain_resp(
                StatusCode::UNPROCESSABLE_ENTITY,
                Bytes::from(body),
                "application/json",
            ));
        }
        // Item routes answer in JSON, and do band math on point values themselves.
        let point_math = segs.get(1) == Some(&"point") && msg.contains("'expression'");
        if !point_math {
            return Ok(pgstac::bad_request(msg));
        }
    }

    // titiler-pgstac's routes: /searches/... and /collections/...
    if matches!(segs.first(), Some(&("searches" | "collections"))) {
        let ctx = pgstac::Ctx {
            engine: &engine,
            host: &host,
            forwarded_prefix: &forwarded_prefix,
            path: &path,
            raw_query: &raw_query,
            if_none_match: if_none_match.as_deref(),
            only: None,
        };
        let list = matches!(segs[..], ["searches"] | ["searches", "" | "list"]);
        let cacheable = method == Method::GET && !list;
        let mut resp = Box::pin(pgstac::handle(ctx, &method, &segs, pq, req)).await;
        if list && method == Method::GET && resp.status() == StatusCode::OK {
            // Prevent handle_request's default GET Cache-Control from caching live search listings.
            resp.headers_mut()
                .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        }
        return Ok(if cacheable { pgstac_cacheable(resp, &Method::GET) } else { resp });
    }
    if pq.keys.iter().any(|k| k == "pixel_selection") && !route_supports_pixel_selection(&segs) {
        return Ok(json_error_resp(
            StatusCode::BAD_REQUEST,
            "parameter 'pixel_selection' is only supported on mosaic tiles, bbox, feature, and statistics routes",
        ));
    }
    // Parameters the algorithm's model refuses, as TiTiler answers them, on the routes that
    // apply an algorithm.
    if route_applies_algorithm(&segs) {
        if let Some(input) = &pq.algorithm_params_unread {
            let body = serde_json::json!({"detail": [{
                "type": "value_error",
                "loc": ["query", "algorithm_params"],
                "msg": "Value error, invalid JSON content",
                "input": input,
            }]});
            return Ok(plain_resp(StatusCode::UNPROCESSABLE_ENTITY, Bytes::from(body.to_string()), "application/json"));
        }
        if let Some(msg) = pq.algorithm_error.as_deref() {
            return Ok(json_error_resp(StatusCode::BAD_REQUEST, msg));
        }
    }

    // ------------------------------------------------------------------------
    // Dynamic registration of spatial catalogs (MosaicJSON, STAC FeatureCollection,
    // JSON array, or live STAC-API POST /search):
    //   POST /mosaic/register
    // ------------------------------------------------------------------------
    if method == Method::POST && path == "/mosaic/register" {
        let res: Result<(Bytes, &'static str), EngineError> = async {
            let body_bytes = read_limited_body(req.into_body()).await?;
            let spec: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid JSON body: {e}")))?;
            let name = spec
                .get("name")
                .and_then(|v| v.as_str())
                .filter(|s| {
                    !s.is_empty()
                        && *s != "tiles"
                        && *s != "register"
                        && !s.contains('/')
                        && !s.contains('?')
                        && !s.contains('#')
                        && !s.chars().any(|c| c.is_whitespace())
                })
                .ok_or_else(|| {
                    EngineError::BadRequest("missing valid 'name' for mosaic catalog".into())
                })?
                .to_string();

            let catalog: Arc<dyn SceneCatalog> = if let Some(cat_val) = spec.get("catalog") {
                load_catalog_json_str(&cat_val.to_string()).map_err(EngineError::BadRequest)?
            } else if let Some(search_url) = spec.get("stac_api_url").and_then(|v| v.as_str()) {
                cogzilla_io::check_uri_allowed(search_url)?;
                let mut search_body = serde_json::Map::new();
                for k in ["collections", "bbox", "datetime", "limit", "query", "filter"] {
                    if let Some(v) = spec.get(k) {
                        search_body.insert(k.to_string(), v.clone());
                    }
                }
                let resp = engine
                    .store
                    .client
                    .post(search_url)
                    .header("content-type", "application/json")
                    .body(serde_json::Value::Object(search_body).to_string())
                    .send()
                    .await
                    .map_err(|e| match cogzilla_io::io_error_from_reqwest(e) {
                        io_err if io_err.is_fatal() => EngineError::Io(io_err),
                        other => EngineError::BadRequest(format!("STAC API search failed: {other}")),
                    })?;
                if !resp.status().is_success() {
                    return Err(EngineError::BadRequest(format!(
                        "STAC API search returned HTTP {}",
                        resp.status()
                    )));
                }
                let body_bytes = cogzilla_io::read_response_capped(resp, MAX_POST_BODY_BYTES)
                    .await
                    .map_err(|e| {
                        EngineError::BadRequest(format!("STAC API response read failed: {e}"))
                    })?;
                let text = String::from_utf8_lossy(&body_bytes);
                load_catalog_json_str(&text).map_err(EngineError::BadRequest)?
            } else {
                return Err(EngineError::BadRequest(
                    "expected 'catalog' or 'stac_api_url' in /mosaic/register body".into(),
                ));
            };

            let count = catalog.len();
            if engine.catalog_count() >= 256 && engine.get_catalog(&name).is_none() {
                return Err(EngineError::BadRequest(
                    "maximum registered mosaic catalogs (256) reached".into(),
                ));
            }
            engine.register_catalog(name.clone(), catalog);
            engine.tile_cache.clear();
            let body = serde_json::json!({
                "name": name,
                "scenes": count,
                "tiles_url": format!("{forwarded_prefix}/mosaic/{name}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"),
            })
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await;
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler MosaicJSON Router:
    //   GET  /mosaicjson/info?url=...
    //   GET  /mosaicjson/bounds?url=...
    //   GET  /mosaicjson/tiles?url=...
    //   GET  /mosaicjson/tiles/{tms}?url=...
    //   GET  /mosaicjson/point/{lon},{lat}?url=...
    //   GET  /mosaicjson/{tms}/tilejson.json?url=...
    //   GET  /mosaicjson/{tms}/WMTSCapabilities.xml?url=...
    //   GET  /mosaicjson/{tms}/map.html?url=...
    //   GET  /mosaicjson/tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}]?url=...
    //   POST /mosaicjson/validate
    // ------------------------------------------------------------------------
    if method == Method::POST && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let mut x_assets: Vec<String> = Vec::new();
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson", "validate"] => async {
                let body_bytes = read_limited_body(req.into_body()).await?;
                let text = std::str::from_utf8(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid UTF-8 body: {e}")))?;
                let mj = MosaicJsonCatalog::from_json_str(text).map_err(EngineError::BadRequest)?;
                let clean = strip_null_object_fields(mj.raw_json.clone());
                Ok((Bytes::from(clean.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "statistics"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let stats_opts = build_stats_options(&pq, false)?;
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let fc_crs = match pq.coord_crs.as_ref() {
                    Some(c) => Some(c.clone()),
                    None => crate::titiler::extract_geojson_crs(&geojson)?,
                };
                let pq_shared = Arc::new(pq.clone());
                let compute_for_mask = {
                    let eng_m = engine.clone();
                    let mj_m = mj.clone();
                    let opts_m = stats_opts.clone();
                    let pq_m = pq_shared.clone();
                    Arc::new(move |mask: PolygonMask, bbox: [f64; 4]| {
                        let eng = eng_m.clone();
                        let mj_c = mj_m.clone();
                        let opts = opts_m.clone();
                        let pq_c = pq_m.clone();
                        async move {
                            if mask.is_empty() {
                                if let Some(first_url) = mj_c.first_asset() {
                                    match eng.store.open_uri(first_url).await.map(|c| eng.store.with_nodata(&c, pq_c.nodata)) {
                                        Ok(cog) => {
                                            let stats = compute_statistics_with_options(
                                                &eng,
                                                &cog,
                                                pq_c.max_size,
                                                Some(&mask),
                                                pq_c.unscale,
                                                pq_c.nodata,
                                                &opts,
                                            )
                                            .await?;
                                            return Ok((stats, Vec::new()));
                                        }
                                        Err(e)
                                            if e.is_fatal() =>
                                        {
                                            return Err(EngineError::Io(e));
                                        }
                                        Err(_) => {}
                                    }
                                }
                                let mut empty = compute_f32_band_stats(&[], "b1", &opts);
                                if let Some(o) = empty.as_object_mut() {
                                    o.insert("area".into(), serde_json::Value::Null);
                                }
                                return Ok((serde_json::json!({"b1": empty}), Vec::new()));
                            }
                            let urls = query_mosaicjson_wgs84_bbox(&mj_c, bbox);
                            compute_mosaic_statistics_with_fold(
                                &eng,
                                &urls,
                                bbox,
                                Some(&mask),
                                &pq_c,
                                &opts,
                            )
                            .await
                        }
                    })
                };
                if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                    let used_batches = run_feature_collection_stats(
                        &mut geojson,
                        fc_crs.as_ref(),
                        move |mask, bbox| compute_for_mask((*mask).clone(), bbox),
                    )
                    .await?;
                    let mut seen_assets = std::collections::HashSet::new();
                    for used in used_batches {
                        for a in used {
                            if seen_assets.insert(a.clone()) {
                                x_assets.push(a);
                            }
                        }
                    }
                    Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                } else {
                    let (mask, bbox) = parse_feature_mask_or_null(&geojson, fc_crs.as_ref())?;
                    let (stats, used) = compute_for_mask(mask, bbox).await?;
                    x_assets = used;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                        if let Some(props) =
                            geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                        {
                            props.insert("statistics".into(), stats);
                        } else if let Some(obj) = geojson.as_object_mut() {
                            obj.insert(
                                "properties".into(),
                                serde_json::json!({"statistics": stats}),
                            );
                        }
                        Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                    } else {
                        let feat = serde_json::json!({
                            "type": "Feature",
                            "geometry": geojson,
                            "properties": { "statistics": stats },
                        });
                        Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                    }
                }
            }
            .await,
            ["mosaicjson", feat_ext]
                if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
            async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, bbox);
                let ext = feat_ext
                    .strip_prefix("feature.")
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let (data, used) = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                let data = mosaic_array_in_its_type(
                    &engine,
                    array_asked(&pq, segs.last()),
                    &used,
                    pq.pixel_selection,
                    data,
                )
                .await;
                x_assets = used;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "feature", dims_ext] => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, bbox);
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
                let mask_opt = pq.shape_mask.then_some(&mask);
                let (data, used) = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                let data = mosaic_array_in_its_type(
                    &engine,
                    array_asked(&pq, segs.last()),
                    &used,
                    pq.pixel_selection,
                    data,
                )
                .await;
                x_assets = used;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            _ => Err(EngineError::NotFound(path.clone())),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        attach_x_assets(&mut resp, &x_assets);
        return Ok(resp);
    }

    if method == Method::GET && segs.first() == Some(&"mosaicjson") {
        let mj_url = pq.urls.first().cloned();
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let mut x_assets: Vec<String> = Vec::new();
        let res: Result<(Bytes, &'static str), EngineError> = match segs.as_slice() {
            ["mosaicjson"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let clean = strip_null_object_fields(mj.raw_json.clone());
                Ok((Bytes::from(clean.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "viewer"] => {
                let tj_url = format!("{forwarded_prefix}/mosaicjson/WebMercatorQuad/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            ["mosaicjson", "info"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let mut info = mj.info_json();
                apply_info_bounds_crs(&mut info, mj.bounds, pq.bounds_crs.as_ref())?;
                Ok((Bytes::from(info.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "info.geojson"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let mut props = mj.info_json();
                apply_info_bounds_crs(&mut props, mj.bounds, pq.bounds_crs.as_ref())?;
                let [w, s, e, n] = mj.bounds;
                let feat = serde_json::json!({
                    "type": "Feature",
                    "bbox": mj.bounds,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                    },
                    "properties": props
                });
                Ok((Bytes::from(feat.to_string()), "application/geo+json"))
            }
            .await,
            ["mosaicjson", "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let assets: Vec<String> = if let Some(b) = pq.bbox_param {
                    let b = resolve_input_bbox(b, pq.coord_crs.as_ref())?;
                    let wgs_bbox = bbox_to_wgs84(b, pq.coord_crs.as_ref())?;
                    query_mosaicjson_wgs84_bbox(&mj, wgs_bbox)
                } else {
                    let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
                    mj.query_bbox_3857(mx0, my0, mx1, my1)
                };
                Ok((Bytes::from(serde_json::json!(assets).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bounds"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(mj.bounds, dst)?
                } else {
                    mj.bounds
                };
                let body = serde_json::json!({ "bounds": bounds }).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", "statistics"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
                let urls = mj.query_bbox_3857(mx0, my0, mx1, my1);
                let stats_opts = build_stats_options(&pq, false)?;
                let (json, used) = compute_mosaic_statistics_with_fold(
                    &engine, &urls, mj.bounds, None, &pq, &stats_opts,
                )
                .await?;
                x_assets = used;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg] => async {
                let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or({
                    if pq.coord_crs_3857 {
                        (cogzilla_proj::CrsModel::Epsg3857, 3857)
                    } else {
                        (cogzilla_proj::CrsModel::Geographic, 4326)
                    }
                });
                let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let (data, used) = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                let data = mosaic_array_in_its_type(
                    &engine,
                    array_asked(&pq, segs.last()),
                    &used,
                    pq.pixel_selection,
                    data,
                )
                .await;
                x_assets = used;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg, "assets"] => async {
                let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "bbox", bbox_seg, dims_ext] => async {
                let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or({
                    if pq.coord_crs_3857 {
                        (cogzilla_proj::CrsModel::Epsg3857, 3857)
                    } else {
                        (cogzilla_proj::CrsModel::Geographic, 4326)
                    }
                });
                let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let (data, used) = render_mosaic_bbox_composite(
                    &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
                )
                .await?;
                let data = mosaic_array_in_its_type(
                    &engine,
                    array_asked(&pq, segs.last()),
                    &used,
                    pq.pixel_selection,
                    data,
                )
                .await;
                x_assets = used;
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            ["mosaicjson", "tiles"] => async {
                let bounds = if let Some(ref u) = mj_url {
                    crate::engine::tolerate_asset_failure(load_mosaicjson(&engine, u).await)?
                        .map(|m| m.bounds)
                } else {
                    None
                };
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "mosaicjson",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", tms] => async {
                if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                    let (bounds, minz, maxz) = if let Some(ref u) = mj_url {
                        match crate::engine::tolerate_asset_failure(
                            load_mosaicjson(&engine, u).await,
                        )? {
                            Some(m) => (
                                Some(m.bounds),
                                pq.minzoom.or(Some(m.minzoom)),
                                pq.maxzoom.or(Some(m.maxzoom)),
                            ),
                            None => (None, pq.minzoom, pq.maxzoom),
                        }
                    } else {
                        (None, pq.minzoom, pq.maxzoom)
                    };
                    let body = build_ogc_tileset_json(
                        &host,
                        &forwarded_prefix,
                        "mosaicjson",
                        tms_def,
                        &raw_query,
                        bounds,
                        minz,
                        maxz,
                    )
                    .to_string();
                    Ok((Bytes::from(body), "application/json"))
                } else {
                    Err(EngineError::NotFound(format!(
                        "unknown TileMatrixSet '{tms}'"
                    )))
                }
            }
            .await,
            ["mosaicjson", "point", coord] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let Some((x, y)) = parse_point_coord(coord) else {
                    return Err(EngineError::BadRequest(
                        "expected /mosaicjson/point/{lon},{lat}".into(),
                    ));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let (mx, my) = if pq.coord_crs_3857 {
                    (x, y)
                } else {
                    lonlat_deg_to_mercator(lon, lat)
                };
                let urls = mj.query_bbox_3857(mx, my, mx, my);
                let mut assets_out = Vec::new();
                let mut opened_any = false;
                let mut last_open_err: Option<EngineError> = None;
                for asset_url in urls.into_iter().take(16) {
                    let cog = match engine.store.open_uri(&asset_url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                        Ok(c) => {
                            opened_any = true;
                            c
                        }
                        Err(e) if e.is_fatal() => {
                            return Err(EngineError::Io(e));
                        }
                        Err(e) => {
                            last_open_err = Some(crate::engine::asset_fetch_error(&asset_url, e));
                            continue;
                        }
                    };
                    match query_point(&engine, &cog, lon, lat, pq.unscale, pq.nodata).await {
                        Ok(mut pt) => {
                            apply_point_filters(&mut pt, pq.expr_raw.as_deref(), &pq.bidx)?;
                            let vals = pt["values"].clone();
                            let names = pt["band_names"].clone();
                            let descs = pt
                                .get("band_descriptions")
                                .cloned()
                                .unwrap_or_else(|| names.clone());
                            assets_out.push(serde_json::json!({
                                "name": asset_url,
                                "values": vals,
                                "band_names": names,
                                "band_descriptions": descs,
                            }));
                        }
                        Err(EngineError::NotFound(_)) => continue,
                        Err(e) => return Err(e),
                    }
                }
                if assets_out.is_empty() {
                    if !opened_any
                        && let Some(err) = last_open_err
                    {
                        return Err(err);
                    }
                    return Err(EngineError::NotFound(format!(
                        "No asset covers point ({lon}, {lat})"
                    )));
                }
                let body = serde_json::json!({
                    "coordinates": [x, y],
                    "assets": assets_out,
                })
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            ["mosaicjson", _, "WMTSCapabilities.xml"] | ["mosaicjson", "WMTSCapabilities.xml"] => {
                async {
                    let explicit_tms = (segs.len() == 3).then_some(segs[1]);
                    if let Some(tms) = explicit_tms
                        && crate::tms::lookup_tms(tms).is_none()
                    {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                    let mj = load_mosaicjson(&engine, &u).await?;
                    let (minz, maxz) =
                        resolve_zoom_range(mj.minzoom, mj.maxzoom, pq.minzoom, pq.maxzoom);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = if let Some(tms) = explicit_tms {
                        build_wmts_capabilities(
                            &host,
                            "mosaicjson",
                            tms,
                            "MosaicJSON",
                            mj.bounds,
                            minz,
                            maxz,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    } else {
                        let entries: Vec<(&str, u8, u8)> = crate::tms::ALL_TMS
                            .iter()
                            .map(|t| (t.id, minz, maxz))
                            .collect();
                        build_wmts_capabilities_multi(
                            &host,
                            "mosaicjson",
                            &entries,
                            None,
                            "MosaicJSON",
                            mj.bounds,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    };
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
            }
            ["mosaicjson", _, "tilejson.json"] | ["mosaicjson", "tilejson.json"] => async {
                let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let center = serde_json::json!([
                    0.5 * (mj.bounds[0] + mj.bounds[2]),
                    0.5 * (mj.bounds[1] + mj.bounds[3]),
                    mj.minzoom,
                ]);
                let mut tj = serde_json::json!({
                    "tilejson": "3.0.0",
                    "version": mj.doc_version,
                    "scheme": "xyz",
                    "tiles": [],
                    "minzoom": mj.minzoom,
                    "maxzoom": mj.maxzoom,
                    "bounds": mj.bounds,
                    "center": center,
                    "raster_layers": {},
                });
                apply_tilejson_overrides(&mut tj, &pq, &host, "mosaicjson", tms, &raw_query);
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", _, "map.html"] | ["mosaicjson", "map.html"] => {
                let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
                let tj_url =
                    format!("{forwarded_prefix}/mosaicjson/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            ["mosaicjson", "point", coord, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let Some((x, y)) = parse_point_coord(coord) else {
                    return Err(EngineError::BadRequest("expected /mosaicjson/point/{lon},{lat}/assets".into()));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let (mx, my) = lonlat_deg_to_mercator(lon, lat);
                let eps = 1e-6;
                let urls = mj.query_bbox_3857(mx - eps, my - eps, mx + eps, my + eps);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", tms, zs, xs, ys, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let (Some(z), Some(x), Some(y)) = (parse_u8_like(zs), parse_u32_like(xs), parse_u32_like(ys)) else {
                    return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                };
                if z > 30 {
                    return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
                }
                let urls = if *tms == "WebMercatorQuad" {
                    let max_xy = 1u64 << (z as u32);
                    if (x as u64) >= max_xy || (y as u64) >= max_xy {
                        return Err(EngineError::BadRequest(format!(
                            "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                        )));
                    }
                    let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
                    mj.query_bbox_3857(b[0], b[1], b[2], b[3])
                } else {
                    let Some(tms_def) = crate::tms::lookup_tms(tms) else {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    };
                    let Some(tb) = tms_def.tile_bounds(z, x, y, pq.buffer, 256) else {
                        return Err(EngineError::BadRequest(format!(
                            "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                        )));
                    };
                    let dst_crs = tms_def.crs_model()?;
                    let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                    query_mosaicjson_wgs84_bbox(&mj, wgs)
                };
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", zs, xs, ys, "assets"]
            | ["mosaicjson", zs, xs, ys, "assets"] => async {
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let (Some(z), Some(x), Some(y)) = (parse_u8_like(zs), parse_u32_like(xs), parse_u32_like(ys)) else {
                    return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                };
                if z > 30 {
                    return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
                }
                let max_xy = 1u64 << (z as u32);
                if (x as u64) >= max_xy || (y as u64) >= max_xy {
                    return Err(EngineError::BadRequest(format!(
                        "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                    )));
                }
                let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
                let urls = mj.query_bbox_3857(b[0], b[1], b[2], b[3]);
                Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
            }
            .await,
            ["mosaicjson", "tiles", _, zs, xs, ys]
            | ["mosaicjson", "tiles", zs, xs, ys] => {
                let tms = if segs.len() == 6 { segs[2] } else { "WebMercatorQuad" };
                let cache_key = format!("{path}?{raw_query}");
                if let (Some(u), Some(z), Some(x), Some((y, _, _))) = (
                    mj_url.as_deref(),
                    parse_u8_like(zs),
                    parse_u32_like(xs),
                    parse_y_scale_ext(ys),
                ) && let Some(mj) = engine.mosaicjson_cache.get(u)
                    && let Some(tms_def) = crate::tms::lookup_tms(tms)
                    && let Some(b) = tms_def.tile_bounds(z, x, y, pq.buffer, 256)
                {
                    let urls = if tms_def.id == "WebMercatorQuad" {
                        mj.query_bbox_3857(b[0], b[1], b[2], b[3])
                    } else {
                        mj.query_bbox_3857(-20_037_508.34, -20_037_508.34, 20_037_508.34, 20_037_508.34)
                    };
                    let _ = engine.store.ensure_uris_fresh(&urls).await;
                    engine.sync_tile_cache_epoch();
                }
                let failed_assets = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                let failed_assets_inner = failed_assets.clone();
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let u = mj_url
                            .ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                        let mj = load_mosaicjson(&engine, &u).await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            parse_u8_like(zs),
                            parse_u32_like(xs),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest(
                                "invalid /mosaicjson/tiles coordinate path".into(),
                            ));
                        };
                        if z > 30 {
                            return Err(EngineError::BadRequest(format!(
                                "invalid zoom level z={z} (expected 0..=30)"
                            )));
                        }
                        let tile = TileCoord { z, x, y };
                        let auto_fmt = ext.is_none() && pq.format_param.is_none();
                        if tms != "WebMercatorQuad" {
                            let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'"))
                            })?;
                            let query_ts = match sc {
                                Some(s) if (1..=4).contains(&s) => s * 256,
                                _ => pq.tile_size,
                            };
                            let tb = tms_def
                                .tile_bounds(z, x, y, pq.buffer, query_ts)
                                .ok_or_else(|| {
                                    EngineError::BadRequest(format!(
                                        "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                                    ))
                                })?;
                            let dst_crs = tms_def.crs_model()?;
                            let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                            let urls = query_mosaicjson_wgs84_bbox(&mj, wgs);
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (mut treq, ct) = build_tile_request_for_tms(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                tms,
                                None,
                                true,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            if auto_fmt {
                                treq.format = OutputFormat::RawRgba;
                            }
                            if urls.is_empty() {
                                if !wgs84_bboxes_intersect(wgs, mj.bounds) {
                                    return Err(EngineError::NotFound(format!(
                                        "Tile {tms}/{z}/{x}/{y} is outside mosaic bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                        mj.bounds[0], mj.bounds[1], mj.bounds[2], mj.bounds[3]
                                    )));
                                }
                                return Err(EngineError::NoContent);
                            }
                            let (data, used) = render_mosaic_bbox_composite(
                                &engine,
                                &urls,
                                tb,
                                &dst_crs,
                                out_dim,
                                out_dim,
                                None,
                                &pq,
                                treq.format,
                                if auto_fmt { "application/octet-stream" } else { ct },
                                tms_def.epsg,
                            )
                            .await?;
                            if auto_fmt {
                                return Ok(encode_auto_image(
                                    out_dim as usize,
                                    out_dim as usize,
                                    &data,
                                    pq.jpeg_quality,
                                    pq.return_mask,
                                ));
                            }
                            let data = mosaic_array_in_its_type(
                                &engine,
                                array_asked(&pq, segs.last()),
                                &used,
                                pq.pixel_selection,
                                data,
                            )
                            .await;
                            return Ok((data, ct));
                        }
                        let query_ts = match sc {
                            Some(s) if (1..=4).contains(&s) => s * 256,
                            _ => pq.tile_size,
                        };
                        let buf_bbox =
                            web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                        let urls = mj.query_bbox_3857(
                            buf_bbox[0],
                            buf_bbox[1],
                            buf_bbox[2],
                            buf_bbox[3],
                        );
                        if urls.is_empty() {
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (treq, _) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let (lo0, la0) = mercator_to_lonlat_deg(buf_bbox[0], buf_bbox[1]);
                            let (lo1, la1) = mercator_to_lonlat_deg(buf_bbox[2], buf_bbox[3]);
                            let tile_wgs = [lo0.min(lo1), la0.min(la1), lo0.max(lo1), la0.max(la1)];
                            if !wgs84_bboxes_intersect(tile_wgs, mj.bounds) {
                                Err(EngineError::NotFound(format!(
                                    "Tile WebMercatorQuad/{z}/{x}/{y} is outside mosaic bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                    mj.bounds[0], mj.bounds[1], mj.bounds[2], mj.bounds[3]
                                )))
                            } else {
                                Err(EngineError::NoContent)
                            }
                        } else {
                            let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (mut treq, ct) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let ts = treq.tile_size;
                            if auto_fmt {
                                treq.format = OutputFormat::RawRgba;
                            }
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let mut failed = Vec::new();
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                                &mut failed,
                            )
                            .await?;
                            if !failed.is_empty() {
                                *failed_assets_inner.lock().unwrap_or_else(|e| e.into_inner()) = failed;
                            }
                            let (data, ct) = finalize_image_output(
                                b,
                                ct,
                                fmt,
                                auto_fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.jpeg_quality,
                                pq.return_mask,
                            );
                            let data = mosaic_array_in_its_type(
                                &engine,
                                array_asked(&pq, segs.last()),
                                &urls,
                                pq.pixel_selection,
                                data,
                            )
                            .await;
                            Ok((data, ct))
                        }
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                let failed = failed_assets.lock().unwrap_or_else(|e| e.into_inner()).clone();
                attach_x_failed_assets(&mut resp, &failed);
                if let (Some(z), Some(x), Some((y, sc, _))) =
                    (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
                }
                return Ok(resp);
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        attach_x_assets(&mut resp, &x_assets);
        return Ok(resp);
    }

    // ------------------------------------------------------------------------
    // Dynamic early-exit COG mosaic:
    //   GET /mosaic/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
    //   GET /mosaic/tiles/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
    // Indexed spatial-catalog mosaic:
    //   GET /mosaic/{name}/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]
    //   GET /mosaic/{name}/tiles/{z}/{x}/{y}[@scale][.{ext}]
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(1) == Some(&"tiles") {
        let coords = match segs.as_slice() {
            ["mosaic", "tiles", "WebMercatorQuad", zs, xs, ys]
            | ["mosaic", "tiles", zs, xs, ys] => (
                parse_u8_like(zs),
                parse_u32_like(xs),
                parse_y_scale_ext(ys),
            ),
            _ => (None, None, None),
        };
        let cache_key = format!("{path}?{raw_query}");
        let res = match coords {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if z > 30 {
                    Err(EngineError::BadRequest(format!(
                        "invalid zoom level z={z} (expected 0..=30)"
                    )))
                } else if pq.urls.is_empty() {
                    Err(EngineError::BadRequest(
                        "missing required ?url= parameter(s) for /mosaic/tiles".into(),
                    ))
                } else if pq.urls.len() > 16 {
                    Err(EngineError::BadRequest(format!(
                        "too many ?url= parameters ({}; max 16)",
                        pq.urls.len()
                    )))
                } else {
                    if let Err(e) = engine.store.ensure_uris_fresh(&pq.urls).await {
                        engine.sync_tile_cache_epoch();
                        if !matches!(e.root(), cogzilla_io::IoError::Fs(io) if io.kind() == std::io::ErrorKind::NotFound) {
                            return Ok(to_http_response(
                                &engine,
                                Err(EngineError::Io(e)),
                                if_none_match.as_deref(),
                            ));
                        }
                    } else {
                        engine.sync_tile_cache_epoch();
                    }
                    let failed_assets = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                    let failed_assets_inner = failed_assets.clone();
                    let cached = engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let first_cog = open_first_mosaic_cog(&engine, &pq.urls).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (treq, ct) = build_tile_request(
                                Some(&engine),
                                TileCoord { z, x, y },
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let ts = treq.tile_size;
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let mut failed = Vec::new();
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &pq.urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                                &mut failed,
                            )
                            .await?;
                            if !failed.is_empty() {
                                *failed_assets_inner.lock().unwrap_or_else(|e| e.into_inner()) = failed;
                            }
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        })
                        .await;
                    let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                    let failed = failed_assets.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    attach_x_failed_assets(&mut resp, &failed);
                    attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                    return Ok(resp);
                }
            }
            _ => Err(EngineError::BadRequest(
                "invalid /mosaic/tiles coordinate path".into(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(2) == Some(&"tiles") {
        let cat_name = segs[1];
        let coords = match segs.as_slice() {
            ["mosaic", _, "tiles", "WebMercatorQuad", zs, xs, ys]
            | ["mosaic", _, "tiles", zs, xs, ys] => (
                parse_u8_like(zs),
                parse_u32_like(xs),
                parse_y_scale_ext(ys),
            ),
            _ => (None, None, None),
        };
        let res = match (engine.get_catalog(cat_name), coords) {
            (None, _) => Err(EngineError::NotFound(format!("mosaic catalog '{cat_name}'"))),
            (Some(catalog), (Some(z), Some(x), Some((y, sc, ext)))) => {
                let cache_key = format!("{path}?{raw_query}");
                if z <= 30 {
                    let (min_x, min_y, max_x, max_y) =
                        cogzilla_proj::tile_bounds_3857(TileCoord { z, x, y });
                    let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
                    let _ = engine.store.ensure_uris_fresh(&urls).await;
                    engine.sync_tile_cache_epoch();
                }
                let failed_assets = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                let failed_assets_c = failed_assets.clone();
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        if z > 30 {
                            return Err(EngineError::BadRequest(format!(
                                "invalid zoom level z={z} (expected 0..=30)"
                            )));
                        }
                        let tile = TileCoord { z, x, y };
                        let query_ts = match sc {
                            Some(s) if (1..=4).contains(&s) => s * 256,
                            _ => pq.tile_size,
                        };
                        let buf_bbox =
                            web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                        let urls = catalog.query_bbox_3857(
                            buf_bbox[0],
                            buf_bbox[1],
                            buf_bbox[2],
                            buf_bbox[3],
                        );
                        if urls.is_empty() {
                            let empty_default_bands = if pq.bidx.is_empty()
                                && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                            {
                                1
                            } else {
                                3
                            };
                            let (treq, ct) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                empty_default_bands,
                                cogzilla_tiff::SampleType::Uint8,
                                false,
                                None,
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            let nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
                            let b = if treq.format == OutputFormat::Npy {
                                let raw = cogzilla_kernel::empty_tile_bytes(
                                    out_dim,
                                    OutputFormat::RawF32,
                                    nb,
                                );
                                cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                                    raw,
                                    nb,
                                    out_dim,
                                    out_dim,
                                    pq.return_mask,
                                )
                            } else {
                                cogzilla_kernel::empty_tile_bytes(out_dim, treq.format, nb)
                            };
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                treq.format,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        } else {
                            let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
                            let db = first_cog.index.full_res().samples_per_pixel;
                            let st = first_cog.index.full_res().sample_type;
                            let (treq, ct) = build_tile_request(
                                Some(&engine),
                                tile,
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&first_cog),
                            )?;
                            validate_mosaic_tile_params(&pq, treq.tile_size)?;
                            let ts = treq.tile_size;
                            let fmt = treq.format;
                            let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                            let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                            let mut failed = Vec::new();
                            let b = render_mosaic_tile_styled(
                                &engine,
                                &urls,
                                treq,
                                &pq,
                                first_cog.index.color_map.is_some(),
                                &mut failed,
                            )
                            .await?;
                            if !failed.is_empty()
                                && let Ok(mut guard) = failed_assets_c.lock()
                            {
                                *guard = failed;
                            }
                            let b = wrap_geotiff_fmt(
                                b,
                                ct,
                                fmt,
                                out_dim,
                                out_dim,
                                buf_bbox,
                                3857,
                                pq.return_mask,
                            );
                            Ok((b, ct))
                        }
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                if let Ok(guard) = failed_assets.lock() {
                    attach_x_failed_assets(&mut resp, &guard);
                }
                return Ok(resp);
            }
            _ => Err(EngineError::BadRequest(
                "invalid /mosaic/{name}/tiles coordinate path".into(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Pre-registered fast path: GET /tiles/{dataset_or_scene}/{z}/{x}/{y}[@scale][.{ext}]
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tiles") && segs.len() == 5 {
        let ds = segs[1];
        let parsed = (
            parse_u8_like(segs[2]),
            parse_u32_like(segs[3]),
            parse_y_scale_ext(segs[4]),
        );
        let cache_key = format!("{path}?{raw_query}");
        let res = match parsed {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if z > 30 {
                    Err(EngineError::BadRequest(format!(
                        "invalid zoom level z={z} (expected 0..=30)"
                    )))
                } else {
                    let cached = engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let cogs: Vec<Arc<cogzilla_io::OpenedCog>> =
                                if let Some(scene) = engine.get_scene(ds) {
                                    scene
                                } else if let Some(cog) = engine.store.get_registered(ds) {
                                    vec![cog]
                                } else {
                                    return Err(EngineError::NotFound(ds.to_string()));
                                };
                            let db = if cogs.len() > 1 {
                                cogs.len() as u16
                            } else {
                                cogs[0].index.full_res().samples_per_pixel
                            };
                            let st = widest_sample_type(&cogs);
                            let (mut treq, ct) = build_tile_request(
                                Some(&engine),
                                TileCoord { z, x, y },
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&cogs[0]),
                            )?;
                            if pq.unscale {
                                apply_unscale_cogs(&mut treq.bands, &cogs, None);
                            }
                            let needs_general_path = pq.nodata.is_some()
                                || pq.custom_colormap_json.is_some()
                                || pq.buffer > 0.0
                                || (!pq.return_mask
                                && matches!(treq.format, OutputFormat::Npy | OutputFormat::NpyF64))
                                || (pq.color_formula.is_some()
                                    && !matches!(
                                        treq.bands,
                                        BandRequest::Rgb { .. }
                                            | BandRequest::SarRgb { .. }
                                            | BandRequest::CustomExprRgb { .. }
                                            | BandRequest::Terrain { .. }
                                    ))
                                || (treq.tile_size != 256 && treq.tile_size != 512)
                                || matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        )
                                || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                                || !crate::engine::cogs_share_base_grid(&cogs);
                            if needs_general_path {
                                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                                let cf = pq.parse_color_formula()?;
                                let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                let data = render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?;
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    treq.format,
                                    out_dim,
                                    out_dim,
                                    bbox_3857,
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((data, ct))
                            } else if (pq.color_formula.is_some() || !pq.return_mask)
                                && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                            {
                                let cf = pq.parse_color_formula()?;
                                let orig_format = treq.format;
                                treq.format = OutputFormat::RawRgba;
                                let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let dim = treq.tile_size as usize;
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                                };
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    orig_format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((data, ct))
                            } else {
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let b = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let b = wrap_geotiff_fmt(
                                    b,
                                    ct,
                                    treq.format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((b, ct))
                            }
                        })
                        .await;
                    let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                    attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                    return Ok(resp);
                }
            }
            _ => Err(EngineError::BadRequest(
                "expected /tiles/{dataset}/{z}/{x}/{y}.{png|webp|f32}".to_string(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler ColorMaps Router: GET /colorMaps, GET /colorMaps/{colorMapId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"colorMaps") {
        let res = match segs.as_slice() {
            ["colorMaps"] => Ok((
                Bytes::from(list_colormaps_json(&host).to_string()),
                "application/json",
            )),
            ["colorMaps", id] => {
                let negotiated_fmt: Option<&str> = pq.format_param.as_deref().or_else(|| {
                    req.headers()
                        .get(hyper::header::ACCEPT)
                        .and_then(|v| v.to_str().ok())
                        .and_then(html::negotiate_colormap_accept)
                });
                match negotiated_fmt {
                    None | Some("json") | Some("html") => {
                        get_colormap_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
                    }
                    Some("npy") => get_colormap_image(
                        id,
                        OutputFormat::Npy,
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "application/x-binary")),
                    Some("png") => get_colormap_image(
                        id,
                        OutputFormat::Png,
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/png")),
                    Some("jpg" | "jpeg") => get_colormap_image(
                        id,
                        OutputFormat::Jpeg { quality: pq.jpeg_quality },
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/jpeg")),
                    Some("webp") => get_colormap_image(
                        id,
                        OutputFormat::Webp { quality: pq.webp_quality },
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/webp")),
                    Some(other) => Err(EngineError::BadRequest(format!(
                        "unsupported colormap format '{other}' (use json, html, npy, png, jpg, or webp)"
                    ))),
                }
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler OGC TileMatrixSets Router: /tileMatrixSets, /tileMatrixSets/{id}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tileMatrixSets") {
        let res = match segs.as_slice() {
            ["tileMatrixSets"] => Ok((
                Bytes::from(list_tms_json(&host).to_string()),
                "application/json",
            )),
            ["tileMatrixSets", id] => {
                get_tms_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler AlgorithmFactory Router: /algorithms, /algorithms/{algorithmId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"algorithms") {
        let res = match segs.as_slice() {
            ["algorithms"] => Ok((
                Bytes::from(list_algorithms_json(&host).to_string()),
                "application/json",
            )),
            ["algorithms", id] => {
                get_algorithm_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Full TiTiler STAC Item Router: /stac/* (MultiBaseTilerFactory compatible!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"stac") {
        let sub = &segs[1..];
        let stac_x_assets = pq.assets.clone();
        let stac_url_hdr = pq.urls.first().cloned();
        // An item read from pgSTAC is held here; one named by `url` is in the cache by then.
        let stac_item_hdr = pgstac_item.clone();
        let stac_req_assets = pq.assets.clone();
        let stac_req_expr = pq.expr_raw.clone();
        let stac_array = array_asked(&pq, sub.last());
        let attach_resolved_stac_assets = |resp: &mut Response<Full<Bytes>>| {
            if !resp.status().is_success() {
                return;
            }
            let item = stac_item_hdr
                .clone()
                .or_else(|| stac_url_hdr.as_ref().and_then(|u| engine.stac_cache.get(u).map(|item| item.value().clone())));
            if let Some(item) = item
                && let Ok(names) = crate::stac::select_stac_asset_names(
                    &item,
                    &stac_req_assets,
                    stac_req_expr.as_deref(),
                )
            {
                attach_x_assets(resp, &names);
            }
        };
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let stac_source = match pgstac_item.take() {
            Some(item) => StacSource::Item(item),
            None => StacSource::Url(pq.urls.first().cloned()),
        };
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // GET /stac/assets?url=item.json
            (Method::GET, ["assets"]) => async {
                let item = stac_source.load(&engine).await?;
                let keys: Vec<&String> = item.assets.keys().collect();
                Ok((Bytes::from(serde_json::json!(keys).to_string()), "application/json"))
            }
            .await,

            // GET /stac/bounds?url=item.json
            (Method::GET, ["bounds"]) => async {
                let item = stac_source.load(&engine).await?;
                let bounds = if let Some(b) = item.bbox {
                    b
                } else {
                    let (_, cogs, _) =
                        resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
                    let mut acc = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                    for cog in &cogs {
                        let crs = engine.get_or_build_crs(cog)?;
                        let full = cog.index.full_res();
                        let b = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        acc[0] = acc[0].min(b[0]);
                        acc[1] = acc[1].min(b[1]);
                        acc[2] = acc[2].max(b[2]);
                        acc[3] = acc[3].max(b[3]);
                    }
                    if !acc.iter().all(|v| v.is_finite()) {
                        return Err(EngineError::BadRequest("STAC item has no valid bounds".into()));
                    }
                    acc
                };
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(bounds, dst)?
                } else {
                    bounds
                };
                let body = serde_json::json!({"bounds": bounds}).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/info?url=item.json&assets=...
            (Method::GET, ["info"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
                let mut map = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs) {
                    let mut info = build_cog_info(&engine, &cog)?;
                    if pq.bounds_crs.is_some() {
                        let crs = engine.get_or_build_crs(&cog)?;
                        let full = cog.index.full_res();
                        let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                    }
                    map.insert(name, info);
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,

            // GET /stac/info.geojson?url=item.json&assets=...
            (Method::GET, ["info.geojson"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
                let mut props = serde_json::Map::new();
                for (name, cog) in names.into_iter().zip(cogs) {
                    let mut info = build_cog_info(&engine, &cog)?;
                    if pq.bounds_crs.is_some() {
                        let crs = engine.get_or_build_crs(&cog)?;
                        let full = cog.index.full_res();
                        let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                        apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                    }
                    props.insert(name, info);
                }
                let geom = item.geometry.clone().unwrap_or_else(|| {
                    let b = item.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                    serde_json::json!({
                        "type": "Polygon",
                        "coordinates": [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]
                    })
                });
                let feat = serde_json::json!({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props,
                });
                Ok((Bytes::from(feat.to_string()), "application/geo+json"))
            }
            .await,

            // GET /stac/asset_statistics (per-asset dict: {asset: {b1: ...}})
            (Method::GET, ["asset_statistics"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None, pq.nodata).await?;
                let n = names.len();
                let stats_opts = build_stats_options(&pq, false)?;
                let entry_opts = stac_entry_options(&pq, &names);
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let mut opts_c = stats_opts.clone();
                    if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                        let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                        opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                    } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                        opts_c.bidx = Some(sel.to_vec());
                    }
                    set.spawn(async move {
                        let st = compute_statistics_with_options(
                            &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                        )
                        .await?;
                        Ok::<_, EngineError>((idx, st))
                    });
                }
                let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some(st);
                }
                let mut map = serde_json::Map::new();
                for (name, st) in names.into_iter().zip(ordered) {
                    if let Some(mut st) = st {
                        if let Some(sel) = stac_asset_selected_bands(&pq, &name)
                            && let Some(obj) = st.as_object() {
                                let mut filtered = serde_json::Map::new();
                                for &b in sel {
                                    let k = format!("b{b}");
                                    if let Some(v) = obj.get(&k) {
                                        filtered.insert(k, v.clone());
                                    }
                                }
                                st = serde_json::Value::Object(filtered);
                            }
                        map.insert(name, st);
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            // GET /stac/statistics (merged band dict: {b1: ..., b2: ...})
            (Method::GET, ["statistics"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                // An item's band math is over `b1..bn` only, and may be one made of its assets'
                // own (`pgstac::item_asset_math`): an asset named `b2` is not read for `b2`.
                let rewritten_expr = if stac_prefix != "stac" { pq.expr_raw.clone() } else { rewritten_expr };
                if item_asset_math {
                    pgstac::check_item_asset_math(&raw_query, &names, &cogs)?;
                } else if stac_prefix != "stac"
                    && let Some(expression) = pq.expr_raw.as_deref()
                {
                    let bands: Vec<usize> = cogs.iter().map(|c| usize::from(c.index.full_res().samples_per_pixel)).collect();
                    let selected = !pq.bidx.is_empty() || pq.assets.iter().any(|a| stac_asset_selected_bands(&pq, a).is_some());
                    pgstac::check_item_statistics_expression(expression, &bands, pq.asset_as_band, selected)?;
                }
                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    crate::algorithms::lookup_algorithm(a)
                        .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
                });
                if is_dem_alg {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                    let mut bands = build_band_request(&pq, db, full.sample_type)?;
                    if pq.unscale {
                        apply_unscale_cogs(&mut bands, &cogs, None);
                    }
                    let stats_opts = build_stats_options(&pq, false)?;
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    if matches!(
                        bands,
                        BandRequest::Single { .. }
                            | BandRequest::Dem { .. }
                            | BandRequest::CustomExpr { .. }
                            | BandRequest::Spectral1 { .. }
                            | BandRequest::Spectral2 { .. }
                            | BandRequest::Spectral3 { .. }
                    ) {
                        let stat = render_native_preview_stats(
                            &engine,
                            &cogs,
                            bbox,
                            out_w,
                            out_h,
                            &bands,
                            pq.resample,
                            pq.nodata,
                            pq.unscale,
                            alg_name,
                            &stats_opts,
                        )
                        .await?;
                        let mut map = serde_json::Map::new();
                        map.insert("b1".into(), stat);
                        return Ok((
                            Bytes::from(serde_json::Value::Object(map).to_string()),
                            "application/json",
                        ));
                    }
                    let raw = render_native_preview(
                        &engine,
                        &cogs,
                        bbox,
                        out_w,
                        out_h,
                        &bands,
                        pq.resample,
                        OutputFormat::RawF32,
                        pq.nodata,
                        None,
                        None,
                        true,
                        pq.unscale,
                    )
                    .await?;
                    let n_px = (out_w as usize) * (out_h as usize);
                    let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                        (raw.len() / (n_px * 4)).clamp(1, 16)
                    } else {
                        1
                    };
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    let mut map = serde_json::Map::new();
                    for b in 0..nb {
                        let start = b * n_px * 4;
                        let end = (start + n_px * 4).min(raw.len());
                        let plane: Vec<f32> = raw[start..end]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let desc = if nb == 1 {
                            alg_name.to_string()
                        } else {
                            format!("{alg_name}_b{}", b + 1)
                        };
                        map.insert(
                            format!("b{}", b + 1),
                            compute_f32_band_stats_with_engine(&engine, &plane, &desc, &stats_opts),
                        );
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let mut expr_progs: Option<Vec<(String, ExprProgram)>> = None;
                if let Some(ref expr_str) = rewritten_expr {
                    let parts: Vec<&str> = expr_str
                        .split(';')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    if parts.len() > 16 {
                        return Err(EngineError::BadRequest(
                            "expression exceeds max 16 semicolon-separated outputs".into(),
                        ));
                    }
                    let raw_parts: Vec<&str> = pq
                        .expr_raw
                        .as_deref()
                        .unwrap_or(expr_str)
                        .split(';')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect();
                    let mut compiled = Vec::with_capacity(parts.len());
                    for (idx, part) in parts.iter().enumerate() {
                        let prog = ExprProgram::compile(part).map_err(EngineError::BadRequest)?;
                        let desc = raw_parts.get(idx).copied().unwrap_or(part).to_string();
                        compiled.push((desc, prog));
                    }
                    expr_progs = Some(compiled);
                } else if pq.algorithm.is_some() {
                    let db = if cogs.len() > 1 {
                        cogs.len() as u16
                    } else {
                        data_band_count(cogs[0].index.full_res())
                    };
                    expr_progs = build_stats_options_with_bands(&pq, true, db)?.expressions;
                }
                if let Some(compiled) = expr_progs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let stats_opts = build_stats_options(&pq, false)?;
                    let mut map = serde_json::Map::new();
                    for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                        let mut bands = BandRequest::CustomExpr {
                            prog,
                            rescale: (0.0, 1.0),
                            colormap: ColormapName::Grayscale,
                        };
                        if pq.unscale {
                            apply_unscale_cogs(&mut bands, &cogs, None);
                        }
                        let stat = render_native_preview_stats(
                            &engine,
                            &cogs,
                            bbox,
                            out_w,
                            out_h,
                            &bands,
                            pq.resample,
                            pq.nodata,
                            pq.unscale,
                            &desc,
                            &stats_opts,
                        )
                        .await?;
                        map.insert(format!("b{}", idx + 1), stat);
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let n = cogs.len();
                let stats_opts = build_stats_options(&pq, false)?;
                let entry_opts = stac_entry_options(&pq, &names);
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    let mut opts_c = stats_opts.clone();
                    if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                        let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                        opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                    } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                        opts_c.bidx = Some(sel.to_vec());
                    }
                    set.spawn(async move {
                        let nb = cog.index.full_res().samples_per_pixel as usize;
                        let st = compute_statistics_with_options(
                            &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                        )
                        .await?;
                        Ok::<_, EngineError>((idx, nb, st))
                    });
                }
                let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, nb, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some((nb, st));
                }
                let mut map = serde_json::Map::new();
                let mut band_idx = 1usize;
                for ((name, slot), (own, sel)) in names.iter().zip(ordered).zip(&entry_opts) {
                    let Some((nb, st)) = slot else { continue };
                    if let Some(obj) = st.as_object() {
                        let bands_iter: Vec<usize> = if own.is_some() {
                            vec![1]
                        } else {
                            match sel.as_deref() {
                                Some(sel) => sel.iter().map(|&b| b as usize).collect(),
                                None if pq.asset_as_band => vec![1],
                                None if !pq.bidx.is_empty() => {
                                    pq.bidx.iter().map(|&b| b as usize).collect()
                                }
                                None => (1..=nb).collect(),
                            }
                        };
                        for i in bands_iter {
                            if let Some(v) = obj.get(&format!("b{i}")) {
                                let mut band_val = v.clone();
                                if let Some(bobj) = band_val.as_object_mut() {
                                    let desc = if pq.asset_as_band {
                                        name.clone()
                                    } else {
                                        format!("{name}_b{i}")
                                    };
                                    bobj.insert(
                                        "description".to_string(),
                                        serde_json::json!(desc),
                                    );
                                }
                                map.insert(format!("b{band_idx}"), band_val);
                                band_idx += 1;
                            }
                        }
                    }
                }
                Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                // (An item's posted statistics are the search routes', and never come here.)
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;

                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    crate::algorithms::lookup_algorithm(a)
                        .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
                });
                let post_expr_progs: Option<Vec<(String, ExprProgram)>> = if let Some(ref expr_str) = rewritten_expr {
                    let parts: Vec<&str> = expr_str.split(';').map(str::trim).filter(|s| !s.is_empty()).collect();
                    if parts.len() > 16 {
                        return Err(EngineError::BadRequest(
                            "expression exceeds max 16 semicolon-separated outputs".into(),
                        ));
                    }
                    let mut compiled = Vec::with_capacity(parts.len());
                    for part in parts {
                        compiled.push((part.to_string(), ExprProgram::compile(part).map_err(EngineError::BadRequest)?));
                    }
                    Some(compiled)
                } else if pq.algorithm.is_some() && !is_dem_alg {
                    let db = if cogs.len() > 1 {
                        cogs.len() as u16
                    } else {
                        data_band_count(cogs[0].index.full_res())
                    };
                    build_stats_options_with_bands(&pq, true, db)?.expressions
                } else {
                    None
                };
                let stats_opts_post = build_stats_options(&pq, false)?;
                let eng_stac = engine.clone();
                let pq_stac = pq.clone();
                let compute_stac_merged_stats = move |mask: Arc<PolygonMask>, poly_bbox: [f64; 4]| {
                    let cogs_c = cogs.clone();
                    let names_c = names.clone();
                    let eng = eng_stac.clone();
                    let max_size = pq_stac.max_size;
                    let width_opt = pq_stac.width;
                    let height_opt = pq_stac.height;
                    let unscale = pq_stac.unscale;
                    let nodata = pq_stac.nodata;
                    let resample = pq_stac.resample;
                    let stats_opts = stats_opts_post.clone();
                    let pq_bidx_c = pq_stac.bidx.clone();
                    let asset_as_band = pq_stac.asset_as_band;
                    let expr_progs_c = post_expr_progs.clone();
                    let alg_c = pq_stac.algorithm.clone();
                    let pq_clone = pq_stac.clone();
                    let n = names_c.len();
                    async move {
                        if is_dem_alg || expr_progs_c.is_some() {
                            if is_dem_alg {
                                let full = cogs_c[0].index.full_res();
                                let db = if cogs_c.len() > 1 { cogs_c.len() as u16 } else { data_band_count(full) };
                                let mut bands = build_band_request(&pq_clone, db, full.sample_type)?;
                                if mask.is_empty() {
                                    let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                                    let nb = crate::algorithms::lookup_algorithm(alg_name)
                                        .and_then(|d| d.out_bands)
                                        .unwrap_or(1) as usize;
                                    let mut map = serde_json::Map::new();
                                    for b in 0..nb {
                                        let desc = if nb == 1 {
                                            alg_name.to_string()
                                        } else {
                                            format!("{alg_name}_b{}", b + 1)
                                        };
                                        let mut empty = compute_f32_band_stats(&[], &desc, &stats_opts);
                                        if let Some(o) = empty.as_object_mut() {
                                            o.insert("area".into(), serde_json::Value::Null);
                                        }
                                        map.insert(format!("b{}", b + 1), empty);
                                    }
                                    return Ok::<_, EngineError>(serde_json::Value::Object(map));
                                }
                                let (def_w, def_h) = aspect_preserving_dims(poly_bbox, max_size);
                                let (out_w, out_h) = resolve_query_dims(def_w, def_h, width_opt, height_opt);
                                let n_px = (out_w as usize) * (out_h as usize);
                                if unscale {
                                    apply_unscale_cogs(&mut bands, &cogs_c, None);
                                }
                                let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                                if matches!(
                                    bands,
                                    BandRequest::Single { .. }
                                        | BandRequest::Dem { .. }
                                        | BandRequest::CustomExpr { .. }
                                        | BandRequest::Spectral1 { .. }
                                        | BandRequest::Spectral2 { .. }
                                        | BandRequest::Spectral3 { .. }
                                ) {
                                    let stat = render_wgs84_bbox_stats(
                                        &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                        resample, nodata, unscale, alg_name, &stats_opts,
                                    )
                                    .await?;
                                    let mut map = serde_json::Map::new();
                                    map.insert("b1".into(), stat);
                                    return Ok::<_, EngineError>(serde_json::Value::Object(map));
                                }
                                let raw = render_wgs84_bbox(
                                    &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                    resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                )
                                .await?;
                                let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                                    (raw.len() / (n_px * 4)).clamp(1, 16)
                                } else {
                                    1
                                };
                                let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                                let mut map = serde_json::Map::new();
                                for b in 0..nb {
                                    let start = b * n_px * 4;
                                    let end = (start + n_px * 4).min(raw.len());
                                    let plane: Vec<f32> = raw[start..end]
                                        .chunks_exact(4)
                                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                        .collect();
                                    let desc = if nb == 1 { alg_name.to_string() } else { format!("{alg_name}_b{}", b + 1) };
                                    map.insert(format!("b{}", b + 1), compute_f32_band_stats_with_engine(&eng, &plane, &desc, &stats_opts));
                                }
                                return Ok::<_, EngineError>(serde_json::Value::Object(map));
                            }
                            let compiled = expr_progs_c.unwrap();
                            if mask.is_empty() {
                                let mut map = serde_json::Map::new();
                                for (idx, (desc, _)) in compiled.into_iter().enumerate() {
                                    let mut empty = compute_f32_band_stats(&[], &desc, &stats_opts);
                                    if let Some(o) = empty.as_object_mut() {
                                        o.insert("area".into(), serde_json::Value::Null);
                                    }
                                    map.insert(format!("b{}", idx + 1), empty);
                                }
                                return Ok::<_, EngineError>(serde_json::Value::Object(map));
                            }
                            let (def_w, def_h) = aspect_preserving_dims(poly_bbox, max_size);
                            let (out_w, out_h) = resolve_query_dims(def_w, def_h, width_opt, height_opt);
                            // Fast exact analytical coverage + ellipsoidal area-weighted native-grid path
                            // (shared with POST /cog/statistics) whenever assets share the same pixel grid
                            // and no explicit output width/height resampling was requested.
                            if width_opt.is_none()
                                && height_opt.is_none()
                                && let Some(exact) = crate::zonal::compute_native_exact_multi_cog_expr_stats(
                                    &eng,
                                    &cogs_c,
                                    &mask,
                                    max_size,
                                    unscale,
                                    nodata,
                                    &compiled,
                                    &stats_opts,
                                )
                                .await?
                            {
                                return Ok::<_, EngineError>(exact);
                            }
                            let mut eff_bbox = poly_bbox;
                            if eff_bbox[2] <= eff_bbox[0] || eff_bbox[3] <= eff_bbox[1] {
                                let (px_dx, px_dy) = if let Some(first) = cogs_c.first()
                                    && let Ok(crs) = eng.get_or_build_crs(first)
                                {
                                    let f = first.index.full_res();
                                    let b = crs.wgs84_bounds(&f.pixel_to_crs, f.width, f.height);
                                    (
                                        ((b[2] - b[0]).abs() / f.width.max(1) as f64).max(1e-6),
                                        ((b[3] - b[1]).abs() / f.height.max(1) as f64).max(1e-6),
                                    )
                                } else {
                                    (1e-5, 1e-5)
                                };
                                if eff_bbox[2] <= eff_bbox[0] {
                                    eff_bbox[0] -= 0.5 * px_dx;
                                    eff_bbox[2] += 0.5 * px_dx;
                                }
                                if eff_bbox[3] <= eff_bbox[1] {
                                    eff_bbox[1] -= 0.5 * px_dy;
                                    eff_bbox[3] += 0.5 * px_dy;
                                }
                            }
                            let mut rendered: Vec<(String, String, Vec<f32>)> =
                                Vec::with_capacity(compiled.len());
                            for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                                let mut bands = BandRequest::CustomExpr {
                                    prog,
                                    rescale: (0.0, 1.0),
                                    colormap: ColormapName::Grayscale,
                                };
                                if unscale {
                                    apply_unscale_cogs(&mut bands, &cogs_c, None);
                                }
                                let raw = render_wgs84_bbox(
                                    &eng, &cogs_c, eff_bbox, out_w, out_h, None, &bands,
                                    resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                )
                                .await?;
                                let pixels: Vec<f32> = raw
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect();
                                rendered.push((format!("b{}", idx + 1), desc, pixels));
                            }
                            let plane_refs: Vec<(&str, &str, &[f32])> = rendered
                                .iter()
                                .map(|(k, d, p)| (k.as_str(), d.as_str(), p.as_slice()))
                                .collect();
                            return Ok::<_, EngineError>(
                                crate::zonal::compute_exact_grid_polygon_stats(
                                    &plane_refs,
                                    eff_bbox,
                                    out_w as usize,
                                    out_h as usize,
                                    &mask,
                                    &stats_opts,
                                ),
                            );
                        }
                        let entry_opts = stac_entry_options(&pq_clone, &names_c);
                        let mut set = tokio::task::JoinSet::new();
                        for (idx, cog) in cogs_c.into_iter().enumerate() {
                            let eng_c = eng.clone();
                            let mask_c = mask.clone();
                            let mut opts_c = stats_opts.clone();
                            if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                                let prog = ExprProgram::compile(expr_str)
                                    .map_err(EngineError::BadRequest)?;
                                opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                            } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                                opts_c.bidx = Some(sel.to_vec());
                            }
                            let nb = cog.index.full_res().samples_per_pixel as usize;
                            set.spawn(async move {
                                let st = compute_statistics_with_options(
                                    &eng_c,
                                    &cog,
                                    max_size,
                                    Some(&mask_c),
                                    unscale,
                                    nodata,
                                    &opts_c,
                                )
                                .await?;
                                Ok::<_, EngineError>((idx, nb, st))
                            });
                        }
                        let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
                        while let Some(res) = set.join_next().await {
                            let (idx, nb, st) = res
                                .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                            ordered[idx] = Some((nb, st));
                        }
                        // Merge bands in numeric order 1..=nb across assets to match TiTiler's POST /stac/statistics.
                        let mut merged = serde_json::Map::new();
                        let mut b_counter = 1usize;
                        for ((name, slot), (own, sel)) in names_c.iter().zip(ordered).zip(&entry_opts) {
                            let Some((nb, st)) = slot else { continue };
                            if let Some(obj) = st.as_object() {
                                let bands_iter: Vec<usize> =
                                    if own.is_some() {
                                        vec![1]
                                    } else {
                                        match sel.as_deref() {
                                            Some(sel) => {
                                                sel.iter().map(|&b| b as usize).collect()
                                            }
                                            None if asset_as_band => vec![1],
                                            None if !pq_bidx_c.is_empty() => {
                                                pq_bidx_c.iter().map(|&b| b as usize).collect()
                                            }
                                            None => (1..=nb).collect(),
                                        }
                                    };
                                for b in bands_iter {
                                    if let Some(v) = obj.get(&format!("b{b}")) {
                                        let mut band_val = v.clone();
                                        if let Some(bobj) = band_val.as_object_mut() {
                                            let desc = if asset_as_band {
                                                name.clone()
                                            } else {
                                                format!("{name}_b{b}")
                                            };
                                            bobj.insert(
                                                "description".to_string(),
                                                serde_json::json!(desc),
                                            );
                                        }
                                        merged.insert(format!("b{b_counter}"), band_val);
                                        b_counter += 1;
                                    }
                                }
                            }
                        }
                        Ok::<_, EngineError>(serde_json::Value::Object(merged))
                    }
                };

                if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                    let fc_crs = match pq.coord_crs.as_ref() {
                        Some(c) => Some(c.clone()),
                        None => crate::titiler::extract_geojson_crs(&geojson)?,
                    };
                    let compute_shared = Arc::new(compute_stac_merged_stats);
                    run_feature_collection_stats(
                        &mut geojson,
                        fc_crs.as_ref(),
                        move |mask, poly_bbox| {
                            let compute_c = compute_shared.clone();
                            async move { Ok((compute_c(mask, poly_bbox).await?, ())) }
                        },
                    )
                    .await?;
                    Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                } else {
                    let (mask, poly_bbox) =
                        parse_feature_mask_or_null(&geojson, pq.coord_crs.as_ref())?;
                    let stats = compute_stac_merged_stats(Arc::new(mask), poly_bbox).await?;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                        if let Some(props) =
                            geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                        {
                            props.insert("statistics".into(), stats);
                        } else if let Some(obj) = geojson.as_object_mut() {
                            obj.insert(
                                "properties".into(),
                                serde_json::json!({"statistics": stats}),
                            );
                        }
                        Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                    } else {
                        let feat = serde_json::json!({
                            "type": "Feature",
                            "geometry": geojson,
                            "properties": { "statistics": stats },
                        });
                        Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                    }
                }
            }
            .await,

            // GET /stac/tiles & /stac/tiles/{tms} (OGC TilesetList & Tileset, matching /cog/tiles*)
            (Method::GET, ["tiles"]) => async {
                let bounds = crate::engine::tolerate_asset_failure(stac_source.load(&engine).await)?
                    .and_then(|item| item.bbox);
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "stac",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,
            (Method::GET, ["tiles", tms]) => async {
                let tms_def = crate::tms::lookup_tms(tms)
                    .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
                let bounds = crate::engine::tolerate_asset_failure(stac_source.load(&engine).await)?
                    .and_then(|item| item.bbox);
                let body = build_ogc_tileset_json(
                    &host,
                    &forwarded_prefix,
                    "stac",
                    tms_def,
                    &raw_query,
                    bounds,
                    pq.minzoom,
                    pq.maxzoom,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /stac/{tms}/tilejson.json, /stac/tilejson.json, /stac/{tms}/map.html, /stac/map.html
            (Method::GET, [_, "tilejson.json"]) | (Method::GET, ["tilejson.json"]) => async {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let item = stac_source.load(&engine).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), None).await?;
                let mut tj = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                apply_tilejson_overrides(&mut tj, &pq, &host, &stac_prefix, tms, &raw_query);
                tj["name"] = serde_json::json!(item.id);
                if !item.renders.is_empty() {
                    tj["raster_layers"] = serde_json::json!(&item.renders);
                }
                Ok((Bytes::from(tj.to_string()), "application/json"))
            }
            .await,
            // titiler-pgstac has no viewer on an item.
            (Method::GET, ["viewer"]) if stac_prefix != "stac" => {
                Err(EngineError::NotFound("Not Found".into()))
            }
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url =
                    format!("{forwarded_prefix}/{stac_prefix}/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            (Method::GET, ["renders"]) => async {
                // The links name the item's address, which only an item given by `url` has.
                let u = match &stac_source {
                    StacSource::Url(Some(u)) => u.clone(),
                    _ => return Err(EngineError::BadRequest("missing ?url=".into())),
                };
                let item = stac_source.load(&engine).await?;
                let suffix = if !forwarded_prefix.is_empty() && !host.ends_with(&forwarded_prefix) {
                    forwarded_prefix.as_str()
                } else {
                    ""
                };
                let origin = if host.starts_with("http://") || host.starts_with("https://") {
                    format!("{host}{suffix}")
                } else {
                    format!("http://{host}{suffix}")
                };
                let mut renders_map = serde_json::Map::new();
                for (rid, rval) in &item.renders {
                    renders_map.insert(
                        rid.clone(),
                        build_stac_render_entry(&host, &forwarded_prefix, &u, rid, rval),
                    );
                }
                let out = serde_json::json!({
                    "renders": serde_json::Value::Object(renders_map),
                    "links": [
                        {
                            "rel": "self",
                            "type": "application/json",
                            "href": format!("{origin}/stac/renders?url={}", percent_encode_component(&u)),
                        }
                    ],
                });
                Ok((Bytes::from(out.to_string()), "application/json"))
            }
            .await,
            (Method::GET, ["renders", render_id]) => async {
                let u = match &stac_source {
                    StacSource::Url(Some(u)) => u.clone(),
                    _ => return Err(EngineError::BadRequest("missing ?url=".into())),
                };
                let item = stac_source.load(&engine).await?;
                let dec_id = percent_decode_opts(render_id, false);
                let render = item
                    .renders
                    .get(&dec_id)
                    .or_else(|| item.renders.get(*render_id))
                    .ok_or_else(|| EngineError::NotFound(format!("render '{dec_id}' not found")))?;
                let out = build_stac_render_entry(&host, &forwarded_prefix, &u, &dec_id, render);
                Ok((Bytes::from(out.to_string()), "application/json"))
            }
            .await,
            // GET /stac/map (OGC API - Maps)
            (Method::GET, ["map"]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let (dst_crs, dst_epsg) = pq_stac
                    .dst_crs
                    .clone()
                    .or_else(|| pq_stac.bounds_crs.clone())
                    .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
                let out_bbox = if let Some(raw_bbox) = pq_stac.bbox_param {
                    let raw_bbox =
                        resolve_input_bbox(raw_bbox, pq_stac.bbox_crs.as_ref().map(|(c, _)| c))?;
                    if pq_stac.bbox_crs.as_ref().map(|(_, e)| *e) == Some(dst_epsg)
                        || (pq_stac.bbox_crs.is_none() && dst_epsg == 4326)
                    {
                        normalize_render_bbox(raw_bbox, &dst_crs)?
                    } else {
                        let wb = bbox_to_wgs84(raw_bbox, pq_stac.bbox_crs.as_ref().map(|(c, _)| c))?;
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    }
                } else {
                    let wb = item.bbox.unwrap_or_else(|| {
                        crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height)
                    });
                    normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                };
                let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq_stac.max_size);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = pq_stac.format_param.as_deref().unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands =
                    build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                );
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let data = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), out_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, dst_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(dst_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let explicit_tms = (sub.len() == 2).then_some(sub[0]);
                    if let Some(tms) = explicit_tms
                        && crate::tms::lookup_tms(tms).is_none()
                    {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let item = stac_source.load(&engine).await?;
                    let (_, cogs, _) =
                        resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), None)
                            .await?;
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bounds = item
                        .bbox
                        .unwrap_or_else(|| crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height));
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = if let Some(tms) = explicit_tms {
                        let (def_min, def_max) = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                            tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
                        } else {
                            crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                        };
                        let (minz, maxz) =
                            resolve_zoom_range(def_min, def_max, pq.minzoom, pq.maxzoom);
                        build_wmts_capabilities(
                            &host,
                            "stac",
                            tms,
                            &item.id,
                            bounds,
                            minz,
                            maxz,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    } else {
                        let entries: Vec<(&str, u8, u8)> = crate::tms::ALL_TMS
                            .iter()
                            .map(|t| {
                                let (dmin, dmax) = t.zoom_range_for_cog(
                                    &crs,
                                    &full.pixel_to_crs,
                                    full.width,
                                    full.height,
                                    ov_factor,
                                );
                                let (z0, z1) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                                (t.id, z0, z1)
                            })
                            .collect();
                        build_wmts_capabilities_multi(
                            &host,
                            "stac",
                            &entries,
                            None,
                            &item.id,
                            bounds,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    };
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
            }

            // GET /stac/tiles/{tms}/{z}/{x}/{y}[@scale][.{format}]
            // and alias GET /stac/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
                let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
                // An item read from the database can change under the same path: its tile
                // is of the assets it has now.
                let cache_key = match &stac_source {
                    StacSource::Item(item) => {
                        format!("{path}?{raw_query}#{:x}", pgstac::source_identity(&item.assets))
                    }
                    StacSource::Url(_) => format!("{path}?{raw_query}"),
                };
                if let StacSource::Url(Some(u)) = &stac_source
                    && let Some(item) = engine.stac_cache.get(u)
                {
                    let asset_urls: Vec<String> = item.assets.values().cloned().collect();
                    let _ = engine.store.ensure_uris_fresh(&asset_urls).await;
                    engine.sync_tile_cache_epoch();
                }
                let stac_tile_geo = TileGeoOpts::from_pq(&pq);
                let stac_tile_source = stac_source.clone();
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || stac_tile_in_its_type(&engine, stac_array.clone(), stac_tile_source, async {
                        let item = stac_source.load(&engine).await?;
                        let (names, cogs, rewritten_expr) = resolve_stac_assets(
                            &engine,
                            &item,
                            &pq.assets,
                            pq.expr_raw.as_deref(),
                            pq.nodata,
                        )
                        .await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            parse_u8_like(zs),
                            parse_u32_like(xs),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                        };
                        let mut pq_stac = pq;
                        let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                        let db = if cogs.len() > 1 {
                            cogs.len() as u16
                        } else {
                            cogs[0].index.full_res().samples_per_pixel
                        };
                        let st = widest_sample_type(&cogs);
                        let (mut treq, ct) = build_tile_request_for_tms(
                            Some(&engine),
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq_stac,
                            db,
                            st,
                            tms,
                            Some(&cogs[0]),
                            true,
                        )?;
                        // titiler-pgstac's item routes name the media type after the extension
                        // and, without one, choose the format from the pixels.
                        let item_route = stac_prefix != "stac";
                        let ct = if item_route && ext == Some("jpg") { "image/jpg" } else { ct };
                        let item_auto_format =
                            item_route && ext.is_none() && pq_stac.format_param.is_none();
                        if pq_stac.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs, slot_bands.as_deref());
                        }
                        if cogs.len() == 1
                            && slot_bands.is_none()
                            && matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                        {
                            treq.format = geotiff_render_format(
                                treq.format,
                                ct,
                                &pq_stac,
                                Some(&cogs[0]),
                                &treq.bands,
                                st,
                                true,
                            );
                        }
                        let mut any_intersects = false;
                        for c in &cogs {
                            if tile_intersects_cog(
                                &engine,
                                c,
                                tms,
                                TileCoord { z, x, y },
                                pq_stac.buffer,
                                treq.tile_size,
                            )? {
                                any_intersects = true;
                                break;
                            }
                        }
                        if !any_intersects {
                            let full = cogs[0].index.full_res();
                            let b = engine
                                .get_or_build_crs(&cogs[0])
                                .map(|c| c.wgs84_bounds(&full.pixel_to_crs, full.width, full.height))
                                .unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                            return Err(EngineError::NotFound(format!(
                                "Tile {tms}/{z}/{x}/{y} is outside dataset bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                b[0], b[1], b[2], b[3]
                            )));
                        }
                        let needs_general = tms != "WebMercatorQuad"
                            || slot_bands.is_some()
                            || pq_stac.buffer > 0.0
                            || pq_stac.nodata.is_some()
                            || pq_stac.custom_colormap_json.is_some()
                            || (!pq_stac.return_mask
                                && matches!(treq.format, OutputFormat::Npy | OutputFormat::NpyF64))
                            || (pq_stac.color_formula.is_some()
                                && !matches!(
                                    treq.bands,
                                    BandRequest::Rgb { .. }
                                        | BandRequest::SarRgb { .. }
                                        | BandRequest::CustomExprRgb { .. }
                                        | BandRequest::Terrain { .. }
                                ))
                            || matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        )
                            || (treq.tile_size != 256 && treq.tile_size != 512)
                            || has_embedded_palette(&pq_stac, &cogs[0], &treq.bands)
                            || !crate::engine::cogs_share_base_grid(&cogs);
                        let auto_fmt = ext.is_none() && pq_stac.format_param.is_none();
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        } else if wants_u16_png(treq.format, ct, &pq_stac, Some(&cogs[0]), &treq.bands, st) {
                            treq.format = OutputFormat::RawF32;
                        }
                        if let Some(ref cf_str) = pq_stac.color_formula {
                            ColorFormula::parse(cf_str).map_err(EngineError::BadRequest)?;
                        }
                        if needs_general {
                            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &treq.bands)?;
                            let cf = pq_stac.parse_color_formula()?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq_stac.buffer).round() as u32;
                            let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                                let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                    EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                                })?;
                                let bbox = tms_def
                                    .tile_bounds(z, x, y, pq_stac.buffer, treq.tile_size)
                                    .ok_or_else(|| {
                                        EngineError::NotFound(format!(
                                            "Tile {z}/{x}/{y} is outside {tms} matrix"
                                        ))
                                    })?;
                                let dst_crs = tms_def.crs_model()?;
                                let d = render_stac_bbox_opt(
                                    &engine,
                                    &cogs,
                                    slot_bands.as_deref(),
                                    bbox,
                                    &dst_crs,
                                    false,
                                    out_dim,
                                    out_dim,
                                    None,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq_stac.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq_stac.return_mask,
                                    pq_stac.unscale,
                                )
                                .await?;
                                (d, bbox, tms_def.epsg)
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq_stac.buffer,
                                    treq.tile_size,
                                );
                                let d = render_stac_bbox_opt(
                                    &engine,
                                    &cogs,
                                    slot_bands.as_deref(),
                                    bbox_3857,
                                    &cogzilla_proj::CrsModel::Epsg3857,
                                    false,
                                    out_dim,
                                    out_dim,
                                    None,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq_stac.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq_stac.return_mask,
                                    pq_stac.unscale,
                                )
                                .await?;
                                (d, bbox_3857, 3857u16)
                            };
                            // An item's picture with pixels missing is a PNG whether or not
                            // the mask was asked for: the format is chosen from the pixels.
                            if item_auto_format
                                && !pq_stac.return_mask
                                && data.len() == (out_dim as usize).pow(2) * 4
                                && data.chunks_exact(4).any(|p| p[3] != 255)
                            {
                                let (w, h) = (out_dim as usize, out_dim as usize);
                                return Ok((encode_rgba_output(w, h, &data, OutputFormat::Png, false), "image/png"));
                            }
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask))
                        } else if item_auto_format {
                            pgstac::item_tile_auto_format(&engine, &cogs, treq, &pq_stac).await
                        } else if (pq_stac.color_formula.is_some() || !pq_stac.return_mask)
                            && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                        {
                            let cf = pq_stac.parse_color_formula()?;
                            let orig_format = treq.format;
                            treq.format = OutputFormat::RawRgba;
                            let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let dim = treq.tile_size as usize;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            if auto_fmt {
                                let buf = if let Some(cf) = cf {
                                    let mut b = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut b, 4);
                                    Bytes::from(b)
                                } else {
                                    raw
                                };
                                Ok(encode_auto_image(dim, dim, &buf, pq_stac.jpeg_quality, pq_stac.return_mask))
                            } else {
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq_stac.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq_stac.return_mask)
                                };
                                let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.return_mask);
                                Ok((data, ct))
                            }
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.jpeg_quality, pq_stac.return_mask))
                        }
                    }))
                    .await;
                if stac_prefix != "stac"
                    && let Err(e) = &cached
                {
                    return Ok(pgstac::engine_error(e));
                }
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                attach_resolved_stac_assets(&mut resp);
                if let (Some(z), Some(x), Some((y, sc, _))) =
                    (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers_opts(&mut resp, tms, z, x, y, sc, stac_tile_geo);
                }
                // An item's tile says where it lies as titiler-pgstac writes it.
                if stac_prefix != "stac" {
                    resp = pgstac::item_placed(resp, &segs[1..], &raw_query);
                }
                return Ok(resp);
            }

            // GET /stac/point/{lon},{lat}
            (Method::GET, ["point", coord]) => async {
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let Some((x, y)) = parse_point_coord(coord) else {
                    return Err(EngineError::BadRequest("expected /stac/point/{lon},{lat}".into()));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let n = names.len();
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    set.spawn(async move {
                        let pt = query_point(&eng, &cog, lon, lat, unscale, nodata).await?;
                        Ok::<_, EngineError>((idx, pt))
                    });
                }
                let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
                while let Some(res) = set.join_next().await {
                    let (idx, pt) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                    ordered[idx] = Some(pt);
                }
                let mut all_vals = Vec::new();
                let mut all_names = Vec::new();
                let single_asset_expr = names.len() == 1 && rewritten_expr.is_some();
                let single_band_per_asset = (names.len() > 1 && rewritten_expr.is_some()) || pq.asset_as_band;
                let mut used_asset_bidx = false;
                let entry_opts = stac_entry_options(&pq, &names);
                for ((name, pt), (own, sel)) in names.into_iter().zip(ordered).zip(entry_opts) {
                    if let Some(vs) = pt.as_ref().and_then(|p| p["values"].as_array()) {
                        if !single_asset_expr
                            && let Some(expr_str) = own.as_deref()
                        {
                            let parts: Vec<&str> = expr_str
                                .split(';')
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .collect();
                            for expr in parts {
                                let prog =
                                    ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
                                let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
                                for &b in &prog.used_bands {
                                    if b == 0 || (b as usize) > vs.len() {
                                        return Err(EngineError::BadRequest(format!(
                                            "expression references b{b} out of range for asset '{name}' (dataset has {} bands)",
                                            vs.len()
                                        )));
                                    }
                                    let f = vs[(b - 1) as usize]
                                        .as_f64()
                                        .map(|x| x as f32)
                                        .unwrap_or(f32::NAN);
                                    slot_vals.push(f);
                                }
                                let res = prog.eval(&slot_vals);
                                all_vals.push(if res.is_finite() {
                                    serde_json::json!(res)
                                } else {
                                    serde_json::Value::Null
                                });
                                all_names.push(format!("{name}_{expr}"));
                            }
                        } else if !single_asset_expr
                            && let Some(sel) = sel.as_deref()
                        {
                            used_asset_bidx = true;
                            let iter_sel = if single_band_per_asset {
                                &sel[..sel.len().min(1)]
                            } else {
                                sel
                            };
                            for &b in iter_sel {
                                if b == 0 || (b as usize) > vs.len() {
                                    return Err(EngineError::BadRequest(format!(
                                        "band {b} out of range for asset '{name}' (dataset has {} bands)",
                                        vs.len()
                                    )));
                                }
                                all_vals.push(vs[(b as usize) - 1].clone());
                                all_names.push(if pq.asset_as_band {
                                    name.clone()
                                } else {
                                    format!("{name}_b{b}")
                                });
                            }
                        } else if single_band_per_asset {
                            if let Some(v) = vs.first() {
                                all_vals.push(v.clone());
                                all_names.push(if pq.asset_as_band {
                                    name.clone()
                                } else {
                                    format!("{name}_b1")
                                });
                            }
                        } else {
                            for (i, v) in vs.iter().enumerate() {
                                all_vals.push(v.clone());
                                all_names.push(format!("{name}_b{}", i + 1));
                            }
                        }
                    }
                }
                let band_names: Vec<String> =
                    (1..=all_vals.len()).map(|i| format!("b{i}")).collect();
                let mut pt_json = serde_json::json!({
                    "coordinates": [x, y],
                    "values": all_vals,
                    "band_names": band_names,
                    "band_descriptions": all_names,
                });
                let point_bidx: &[u16] = if used_asset_bidx && rewritten_expr.is_none() {
                    &[]
                } else {
                    &pq.bidx
                };
                apply_point_filters(&mut pt_json, rewritten_expr.as_deref(), point_bidx)?;
                Ok((Bytes::from(pt_json.to_string()), "application/json"))
            }
            .await,

            // GET /stac/preview[/{w}x{h}][.{ext}]
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext_opt = p.strip_prefix("preview.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                // An item's band math is over `b1..bn`: its asset names are not rewritten.
                let rewritten_expr = if stac_prefix != "stac" { pq_stac.expr_raw.clone() } else { rewritten_expr };
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, stac_prefix == "stac")?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(
                        fmt,
                        ct,
                        &pq_stac,
                        Some(&cogs[0]),
                        &bands,
                        st,
                        cogs.len() == 1 && slot_bands.is_none(),
                    )
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, pq_stac.max_size);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let scale = (pq_stac.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                            rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, bbox, out_w, out_h, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                            pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, epsg)
                    } else {
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                            pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                    }
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                // An item's band math is over `b1..bn`: its asset names are not rewritten.
                let rewritten_expr = if stac_prefix != "stac" { pq_stac.expr_raw.clone() } else { rewritten_expr };
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, stac_prefix == "stac")?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                );
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                            rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, bbox, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                            pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, epsg)
                    } else {
                        let d = render_stac_bbox_opt(
                            &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                            pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                        )
                        .await?;
                        (d, nat_bbox, nat_epsg.unwrap_or(4326))
                    }
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            // GET /stac/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png"),
                    pq_stac.webp_quality,
                    pq_stac.jpeg_quality,
                )?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(
                        fmt,
                        ct,
                        &pq_stac,
                        Some(&cogs[0]),
                        &bands,
                        st,
                        cogs.len() == 1 && slot_bands.is_none(),
                    )
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let wgs_bbox_src = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                let bbox_max = effective_bbox_max_size(&pq_stac);
                let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &dst_crs, wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if pq_stac.coord_crs_3857 {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Epsg3857, wgs_bbox_src, bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 3857u16)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, wgs_bbox_src, wgs_bbox_src, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), wgs_bbox_src, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, wgs_bbox_src, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
                let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let item = stac_source.load(&engine).await?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                );
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                    let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if pq_stac.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, 3857u16)
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                    let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), wgs_bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, wgs_bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,

            // POST /stac/feature/{w}x{h}[.{ext}] and POST /stac/feature[.{ext}]
            (Method::POST, ["feature", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                );
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let eff_dst = resolve_bbox_dst_crs(&pq_stac);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
                let mask_opt = pq_stac.shape_mask.then_some(&mask);
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::POST, [feat_ext])
                if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
            async {
                let ext_opt = feat_ext.strip_prefix("feature.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let s = effective_bbox_max_size(&pq);
                let item = stac_source.load(&engine).await?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let (names, cogs, rewritten_expr) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
                let mut pq_stac = pq;
                let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
                let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(
                        fmt,
                        ct,
                        &pq_stac,
                        Some(&cogs[0]),
                        &bands,
                        st,
                        cogs.len() == 1 && slot_bands.is_none(),
                    )
                };
                if pq_stac.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                }
                let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
                let cf = pq_stac.parse_color_formula()?;
                let eff_dst = resolve_bbox_dst_crs(&pq_stac);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
                let mask_opt = pq_stac.shape_mask.then_some(&mask);
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, s);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, bbox, bbox, s);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
            (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
                Err(EngineError::MethodNotAllowed("POST"))
            }
            (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
            (_, ["info"] | ["info.geojson"] | ["bounds"] | ["assets"] | ["asset_statistics"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["viewer"] | ["renders"] | ["renders", _] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
                Err(EngineError::MethodNotAllowed("GET, HEAD"))
            }

            _ => Err(EngineError::NotFound(path.clone())),
        };
        let res = match res {
            Ok((data, ct)) if stac_array.is_some() && may_be_f32_array(&data) => {
                let source = match &stac_item_hdr {
                    Some(item) => StacSource::Item(item.clone()),
                    None => StacSource::Url(stac_url_hdr.clone()),
                };
                match source.load(&engine).await {
                    Ok(item) => Ok((stac_array_in_its_type(&engine, stac_array, &item, data).await, ct)),
                    Err(_) => Ok((data, ct)),
                }
            }
            res => res,
        };
        // titiler-pgstac answers errors as JSON `{"detail": ...}`.
        if stac_prefix != "stac" {
            return Ok(match res {
                Err(e) => pgstac::engine_error(&e),
                Ok((body, ct)) => {
                    match pgstac::item_fixup(
                        &segs[1..],
                        pgstac_item_extent.as_ref(),
                        &raw_query,
                        body,
                        ct,
                    ) {
                        Ok(fixed) => {
                            let mut resp = to_http_response(&engine, Ok(fixed), if_none_match.as_deref());
                            attach_x_assets(&mut resp, &stac_x_assets);
                            let resp = if method == Method::GET {
                                pgstac::item_placed(resp, &segs[1..], &raw_query)
                            } else {
                                resp
                            };
                            let resp = pgstac::item_drawn_placed(resp, &segs[1..], geo_meta);
                            pgstac_cacheable(resp, &method)
                        }
                        Err(resp) => resp,
                    }
                }
            });
        }
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if !matches!(
            sub,
            ["assets"]
                | ["tiles"]
                | ["tiles", _]
                | [_, "map.html"]
                | ["map.html"]
                | ["viewer"]
                | ["renders"]
                | ["renders", _]
        ) {
            attach_resolved_stac_assets(&mut resp);
        }
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
    }

    // ------------------------------------------------------------------------
    // Zarr API: /zarr/* (metadata, introspection, raster, and POST zonal routes)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"zarr") {
        let sub = &segs[1..];
        let body_bytes = if method == Method::POST {
            match read_limited_body(req.into_body()).await {
                Ok(b) => Some(b),
                Err(e) => return Ok(error_to_http_response(&e)),
            }
        } else {
            None
        };
        let resp = Box::pin(zarr::handle_zarr_request(
            &engine,
            method,
            sub,
            &raw_query,
            &host,
            if_none_match.as_deref(),
            body_bytes,
        ))
        .await;
        return Ok(resp);
    }

    // ------------------------------------------------------------------------
    // Full TiTiler Drop-In API: /cog/* (all 19 TiTiler TilerFactory routes!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"cog") {
        let sub = &segs[1..];
        let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
        let cog_array = array_asked(&pq, sub.last());
        let cog_urls = pq.urls.clone();
        let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
            // 1. GET /cog/info
            (Method::GET, ["info"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let mut json = build_cog_info(&engine, &cogs[0])?;
                if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let wgs = crs.wgs84_bounds_with_gcp(
                        &full.pixel_to_crs,
                        full.gcp_poly.as_deref(),
                        full.width,
                        full.height,
                    );
                    let rep = reproject_wgs84_bounds(wgs, dst)?;
                    json["bounds"] = serde_json::json!(rep);
                    json["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
                }
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 2. GET /cog/info.geojson
            (Method::GET, ["info.geojson"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let mut json = build_cog_info_geojson(&engine, &cogs[0])?;
                if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let wgs = crs.wgs84_bounds_with_gcp(
                        &full.pixel_to_crs,
                        full.gcp_poly.as_deref(),
                        full.width,
                        full.height,
                    );
                    let rep = reproject_wgs84_bounds(wgs, dst)?;
                    json["properties"]["bounds"] = serde_json::json!(rep);
                    json["properties"]["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
                }
                Ok((Bytes::from(json.to_string()), "application/geo+json"))
            }
            .await,

            // Alias: GET /cog/bounds
            (Method::GET, ["bounds"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bounds = crs.wgs84_bounds_with_gcp(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                    reproject_wgs84_bounds(bounds, dst)?
                } else {
                    bounds
                };
                let body = serde_json::json!({"bounds": bounds}).to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // GET /cog/validate?url=...&strict=false
            (Method::GET, ["validate"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let url = pq.urls.first().map(String::as_str).unwrap_or("");
                let json = build_cog_validate(&engine, &cogs[0], url, pq.strict)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 3. GET / POST /cog/statistics (POST supports GeoJSON Polygon clipping!)
            (Method::GET, ["statistics"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let db = data_band_count(full);
                let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                    crate::algorithms::lookup_algorithm(a)
                        .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
                });
                if is_dem_alg {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let mut bands = build_band_request(&pq, db, full.sample_type)?;
                    if pq.unscale {
                        apply_unscale_cogs(&mut bands, &cogs, None);
                    }
                    let stats_opts = build_stats_options(&pq, false)?;
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    if matches!(
                        bands,
                        BandRequest::Single { .. }
                            | BandRequest::Dem { .. }
                            | BandRequest::CustomExpr { .. }
                            | BandRequest::Spectral1 { .. }
                            | BandRequest::Spectral2 { .. }
                            | BandRequest::Spectral3 { .. }
                    ) {
                        let stat = render_native_preview_stats(
                            &engine,
                            &cogs,
                            bbox,
                            out_w,
                            out_h,
                            &bands,
                            pq.resample,
                            pq.nodata,
                            pq.unscale,
                            alg_name,
                            &stats_opts,
                        )
                        .await?;
                        let mut map = serde_json::Map::new();
                        map.insert("b1".into(), stat);
                        return Ok((
                            Bytes::from(serde_json::Value::Object(map).to_string()),
                            "application/json",
                        ));
                    }
                    let raw = render_native_preview(
                        &engine,
                        &cogs,
                        bbox,
                        out_w,
                        out_h,
                        &bands,
                        pq.resample,
                        OutputFormat::RawF32,
                        pq.nodata,
                        None,
                        None,
                        true,
                        pq.unscale,
                    )
                    .await?;
                    let n_px = (out_w as usize) * (out_h as usize);
                    let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                        (raw.len() / (n_px * 4)).clamp(1, 16)
                    } else {
                        1
                    };
                    let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                    let mut map = serde_json::Map::new();
                    for b in 0..nb {
                        let start = b * n_px * 4;
                        let end = (start + n_px * 4).min(raw.len());
                        let plane: Vec<f32> = raw[start..end]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let desc = if nb == 1 {
                            alg_name.to_string()
                        } else {
                            format!("{alg_name}_b{}", b + 1)
                        };
                        map.insert(
                            format!("b{}", b + 1),
                            compute_f32_band_stats_with_engine(&engine, &plane, &desc, &stats_opts),
                        );
                    }
                    return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
                }
                let stats_opts = build_stats_options_with_bands(&pq, true, db)?;
                let json = compute_statistics_with_options(
                    &engine,
                    &cogs[0],
                    pq.max_size,
                    None,
                    pq.unscale,
                    pq.nodata,
                    &stats_opts,
                )
                .await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::POST, ["statistics"]) => async {
                let body_bytes = read_limited_body(req.into_body()).await?;
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let db = data_band_count(cogs[0].index.full_res());
                let stats_opts = build_stats_options_with_bands(&pq, true, db)?;
                if body_bytes.is_empty() {
                    let json = compute_statistics_with_options(
                        &engine,
                        &cogs[0],
                        pq.max_size,
                        None,
                        pq.unscale,
                        pq.nodata,
                        &stats_opts,
                    )
                    .await?;
                    Ok((Bytes::from(json.to_string()), "application/json"))
                } else {
                    let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                        .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                    if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                        let fc_crs = match pq.coord_crs.as_ref() {
                            Some(c) => Some(c.clone()),
                            None => crate::titiler::extract_geojson_crs(&geojson)?,
                        };
                        let eng_c = engine.clone();
                        let cog_c = cogs[0].clone();
                        let opts_c = stats_opts.clone();
                        let max_size = pq.max_size;
                        let unscale = pq.unscale;
                        let nodata = pq.nodata;
                        run_feature_collection_stats(
                            &mut geojson,
                            fc_crs.as_ref(),
                            move |mask, _| {
                                let eng_c = eng_c.clone();
                                let cog_c = cog_c.clone();
                                let opts_c = opts_c.clone();
                                async move {
                                    let stats = compute_statistics_with_options(
                                        &eng_c,
                                        &cog_c,
                                        max_size,
                                        Some(&mask),
                                        unscale,
                                        nodata,
                                        &opts_c,
                                    )
                                    .await?;
                                    Ok((stats, ()))
                                }
                            },
                        )
                        .await?;
                        Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                    } else {
                        let (mask, _) =
                            parse_feature_mask_or_null(&geojson, pq.coord_crs.as_ref())?;
                        let stats = compute_statistics_with_options(
                            &engine,
                            &cogs[0],
                            pq.max_size,
                            Some(&mask),
                            pq.unscale,
                            pq.nodata,
                            &stats_opts,
                        )
                        .await?;
                        if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                            if let Some(props) =
                                geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                            {
                                props.insert("statistics".into(), stats);
                            } else if let Some(obj) = geojson.as_object_mut() {
                                obj.insert(
                                    "properties".into(),
                                    serde_json::json!({"statistics": stats}),
                                );
                            }
                            Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                        } else {
                            let feat = serde_json::json!({
                                "type": "Feature",
                                "geometry": geojson,
                                "properties": { "statistics": stats },
                            });
                            Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                        }
                    }
                }
            }
            .await,

            // 4. GET /cog/tiles (OGC Tiles list)
            (Method::GET, ["tiles"]) => async {
                let bounds = if !pq.urls.is_empty() {
                    crate::engine::tolerate_asset_failure(open_query_urls(&engine, &pq.urls, None).await)?
                        .and_then(|cogs| {
                            engine.get_or_build_crs(&cogs[0]).ok().map(|crs| {
                                let full = cogs[0].index.full_res();
                                crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height)
                            })
                        })
                } else {
                    None
                };
                let body = build_ogc_tilesets_list(
                    &host,
                    &forwarded_prefix,
                    "cog",
                    &raw_query,
                    bounds,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // 5. GET /cog/tiles/{tileMatrixSetId}
            (Method::GET, ["tiles", tms]) => async {
                let tms_def = crate::tms::lookup_tms(tms)
                    .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
                let (bounds, minz, maxz) = if !pq.urls.is_empty() {
                    if let Some(cogs) =
                        crate::engine::tolerate_asset_failure(open_query_urls(&engine, &pq.urls, None).await)?
                    {
                        if let Ok(crs) = engine.get_or_build_crs(&cogs[0]) {
                            let full = cogs[0].index.full_res();
                            let coarsest = cogs[0].index.ifds.last().unwrap();
                            let ov = (full.width as f64 / coarsest.width.max(1) as f64)
                                .max(full.height as f64 / coarsest.height.max(1) as f64);
                            let (dmin, dmax) = tms_def.zoom_range_for_cog(
                                &crs,
                                &full.pixel_to_crs,
                                full.width,
                                full.height,
                                ov,
                            );
                            let (mn, mx) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                            (
                                Some(crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height)),
                                Some(mn),
                                Some(mx),
                            )
                        } else {
                            (None, pq.minzoom, pq.maxzoom)
                        }
                    } else {
                        (None, pq.minzoom, pq.maxzoom)
                    }
                } else {
                    (None, pq.minzoom, pq.maxzoom)
                };
                let body = build_ogc_tileset_json(
                    &host,
                    &forwarded_prefix,
                    "cog",
                    tms_def,
                    &raw_query,
                    bounds,
                    minz,
                    maxz,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            }
            .await,

            // 6 & 7. GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{format}]
            //        and alias GET /cog/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
            (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
                let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
                let cache_key = format!("{path}?{raw_query}");
                if let Err(e) = engine.store.ensure_uris_fresh(&pq.urls).await {
                    engine.sync_tile_cache_epoch();
                    return Ok(error_to_http_response(&EngineError::Io(e)));
                }
                engine.sync_tile_cache_epoch();
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || cog_tile_in_its_type(&engine, cog_array.clone(), &cog_urls, async {
                        let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                        let (Some(z), Some(x), Some((y, sc, ext))) = (
                            parse_u8_like(zs),
                            parse_u32_like(xs),
                            parse_y_scale_ext(ys),
                        ) else {
                            return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                        };
                        let db = if cogs.len() > 1 {
                            cogs.len() as u16
                        } else {
                            cogs[0].index.full_res().samples_per_pixel
                        };
                        let st = widest_sample_type(&cogs);
                        let (mut treq, ct) = build_tile_request_for_tms(
                            Some(&engine),
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            tms,
                            Some(&cogs[0]),
                            true,
                        )?;
                        if pq.unscale {
                            apply_unscale_cogs(&mut treq.bands, &cogs, None);
                        }
                        if cogs.len() == 1
                            && matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                        {
                            treq.format = geotiff_render_format(
                                treq.format,
                                ct,
                                &pq,
                                Some(&cogs[0]),
                                &treq.bands,
                                st,
                                true,
                            );
                        }

                        let mut any_intersects = false;
                        for c in &cogs {
                            if tile_intersects_cog(
                                &engine,
                                c,
                                tms,
                                TileCoord { z, x, y },
                                pq.buffer,
                                treq.tile_size,
                            )? {
                                any_intersects = true;
                                break;
                            }
                        }
                        if !any_intersects {
                            let full = cogs[0].index.full_res();
                            let b = engine
                                .get_or_build_crs(&cogs[0])
                                .map(|c| c.wgs84_bounds(&full.pixel_to_crs, full.width, full.height))
                                .unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                            return Err(EngineError::NotFound(format!(
                                "Tile {tms}/{z}/{x}/{y} is outside dataset bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                b[0], b[1], b[2], b[3]
                            )));
                        }
                        let needs_general_path = tms != "WebMercatorQuad"
                            || pq.buffer > 0.0
                            || pq.nodata.is_some()
                            || pq.custom_colormap_json.is_some()
                            || (!pq.return_mask
                                && matches!(treq.format, OutputFormat::Npy | OutputFormat::NpyF64))
                            || (pq.color_formula.is_some()
                                && !matches!(
                                    treq.bands,
                                    BandRequest::Rgb { .. }
                                        | BandRequest::SarRgb { .. }
                                        | BandRequest::CustomExprRgb { .. }
                                        | BandRequest::Terrain { .. }
                                ))
                            || matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        )
                            || (treq.tile_size != 256 && treq.tile_size != 512)
                            || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                            || !crate::engine::cogs_share_base_grid(&cogs);

                        let auto_fmt = ext.is_none() && pq.format_param.is_none();
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        } else if wants_u16_png(treq.format, ct, &pq, Some(&cogs[0]), &treq.bands, st) {
                            treq.format = OutputFormat::RawF32;
                        }
                        if let Some(ref cf_str) = pq.color_formula {
                            ColorFormula::parse(cf_str).map_err(EngineError::BadRequest)?;
                        }

                        if needs_general_path {
                            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                            let cf = pq.parse_color_formula()?;
                            let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                            let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                                let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                    EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                                })?;
                                let bbox = tms_def
                                    .tile_bounds(z, x, y, pq.buffer, treq.tile_size)
                                    .ok_or_else(|| {
                                        EngineError::NotFound(format!(
                                            "Tile {z}/{x}/{y} is outside {tms} matrix"
                                        ))
                                    })?;
                                let dst_crs = tms_def.crs_model()?;
                                let d = render_crs_bbox(
                                    &engine,
                                    &cogs,
                                    bbox,
                                    &dst_crs,
                                    out_dim,
                                    out_dim,
                                    None,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?;
                                (d, bbox, tms_def.epsg)
                            } else {
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                let d = render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?;
                                (d, bbox_3857, 3857u16)
                            };
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq.jpeg_quality, pq.return_mask))
                        } else if (pq.color_formula.is_some() || !pq.return_mask)
                            && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                        {
                            let cf = pq.parse_color_formula()?;
                            let orig_format = treq.format;
                            treq.format = OutputFormat::RawRgba;
                            let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let dim = treq.tile_size as usize;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            if auto_fmt {
                                let buf = if let Some(cf) = cf {
                                    let mut b = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut b, 4);
                                    Bytes::from(b)
                                } else {
                                    raw
                                };
                                Ok(encode_auto_image(dim, dim, &buf, pq.jpeg_quality, pq.return_mask))
                            } else {
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                                };
                                let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq.return_mask);
                                Ok((data, ct))
                            }
                        } else {
                            let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                            let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                            Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq.jpeg_quality, pq.return_mask))
                        }
                    }))
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                if let (Some(z), Some(x), Some((y, sc, _))) =
                    (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
                {
                    attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
                }
                return Ok(resp);
            }

            // 8. GET /cog/{tileMatrixSetId}/tilejson.json & /cog/tilejson.json
            (Method::GET, [tms, "tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let mut json = build_tilejson(&engine, &cogs[0], &host, tms, &raw_query)?;
                apply_tilejson_overrides(&mut json, &pq, &host, "cog", tms, &raw_query);
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,
            (Method::GET, ["tilejson.json"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let mut json =
                    build_tilejson(&engine, &cogs[0], &host, "WebMercatorQuad", &raw_query)?;
                apply_tilejson_overrides(&mut json, &pq, &host, "cog", "WebMercatorQuad", &raw_query);
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 9. GET /cog/{tileMatrixSetId}/map.html & /cog/map.html
            (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
                let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
                let tj_url = format!("{forwarded_prefix}/cog/{tms}/tilejson.json?{raw_query}");
                Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
            }
            // GET /cog/stac — build a STAC 1.1.0 Item JSON from a COG (matching TiTiler / rio-stac)
            (Method::GET, ["stac"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let cog = &cogs[0];
                let (crs, native_epsg, native_bounds) =
                    dataset_native_crs_and_bounds(&engine, cog)?;
                let full = cog.index.full_res();
                let wgs_bbox = crs.wgs84_bounds_with_gcp(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                let [w, s, e, n] = wgs_bbox;
                let [nw, ns, ne, nn] = native_bounds;
                let mut q_id: Option<String> = None;
                let mut q_collection: Option<String> = None;
                let mut q_collection_url: Option<String> = None;
                let mut q_asset_name: Option<String> = None;
                let mut q_asset_href: Option<String> = None;
                let mut q_asset_media_type: Option<String> = None;
                let mut q_asset_roles: Vec<String> = Vec::new();
                let mut q_extensions: Vec<String> = Vec::new();
                let mut q_datetime: Option<String> = None;
                let mut with_proj = true;
                let mut with_raster = true;
                let mut with_eo = true;
                for pair in raw_query.split('&').filter(|p| !p.is_empty()) {
                    if let Some((k, v)) = pair.split_once('=') {
                        let dec = percent_decode(v);
                        match query_key(k).as_ref() {
                            "id" if !dec.is_empty() => q_id = Some(dec),
                            "collection" if !dec.is_empty() => q_collection = Some(dec),
                            "collection_url" if !dec.is_empty() => q_collection_url = Some(dec),
                            "asset_name" if !dec.is_empty() => q_asset_name = Some(dec),
                            "asset_href" if !dec.is_empty() => q_asset_href = Some(dec),
                            "asset_media_type" if !dec.is_empty() => q_asset_media_type = Some(dec),
                            "asset_roles" => {
                                for r in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                    q_asset_roles.push(r.to_string());
                                }
                            }
                            "extensions" => {
                                for ext in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                    q_extensions.push(ext.to_string());
                                }
                            }
                            "datetime" if !dec.is_empty() => q_datetime = Some(dec),
                            "with_proj" => match parse_bool_param(v) {
                                Some(b) => with_proj = b,
                                None => return Err(EngineError::BadRequest(format!("invalid with_proj={dec}: expected boolean"))),
                            },
                            "with_raster" => match parse_bool_param(v) {
                                Some(b) => with_raster = b,
                                None => return Err(EngineError::BadRequest(format!("invalid with_raster={dec}: expected boolean"))),
                            },
                            "with_eo" => match parse_bool_param(v) {
                                Some(b) => with_eo = b,
                                None => return Err(EngineError::BadRequest(format!("invalid with_eo={dec}: expected boolean"))),
                            },
                            _ => {}
                        }
                    }
                }
                let item_id = q_id.unwrap_or_else(|| {
                    std::path::Path::new(&cog.uri)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("cog-item")
                        .to_string()
                });
                let asset_name = q_asset_name.unwrap_or_else(|| "data".to_string());
                let asset_href = q_asset_href.unwrap_or_else(|| cog.uri.clone());
                let media_type = match q_asset_media_type.as_deref() {
                    None | Some("auto") => "image/tiff; application=geotiff".to_string(),
                    Some("cog") => "image/tiff; application=geotiff; profile=cloud-optimized".to_string(),
                    Some(other) => other.to_string(),
                };
                let mut extensions: Vec<String> = Vec::new();
                let mut push_ext = |uri: &str| {
                    if !extensions.iter().any(|e| e == uri) {
                        extensions.push(uri.to_string());
                    }
                };
                let mut props = serde_json::Map::new();
                props.insert(
                    "datetime".into(),
                    serde_json::json!(q_datetime.unwrap_or_else(utc_now_rfc3339)),
                );
                if with_proj {
                    push_ext("https://stac-extensions.github.io/projection/v2.0.0/schema.json");
                    props.insert("proj:epsg".into(), serde_json::json!(native_epsg));
                    props.insert("proj:shape".into(), serde_json::json!([full.height, full.width]));
                    props.insert(
                        "proj:transform".into(),
                        serde_json::json!([
                            full.pixel_to_crs.a,
                            full.pixel_to_crs.b,
                            full.pixel_to_crs.c,
                            full.pixel_to_crs.d,
                            full.pixel_to_crs.e,
                            full.pixel_to_crs.f,
                            0.0,
                            0.0,
                            1.0
                        ]),
                    );
                    props.insert("proj:bbox".into(), serde_json::json!(native_bounds));
                    props.insert(
                        "proj:geometry".into(),
                        serde_json::json!({
                            "type": "Polygon",
                            "coordinates": [[[nw, ns], [ne, ns], [ne, nn], [nw, nn], [nw, ns]]]
                        }),
                    );
                }
                let count = full.samples_per_pixel as usize;
                let mut asset_obj = serde_json::json!({
                    "href": asset_href,
                    "type": media_type,
                    "roles": q_asset_roles,
                });
                if with_raster {
                    push_ext("https://stac-extensions.github.io/raster/v1.1.0/schema.json");
                    let sampling = if cog.index.geo_keys.raster_type == Some(2) {
                        "point"
                    } else {
                        "area"
                    };
                    let stats_json = compute_statistics_with_options(
                        &engine,
                        cog,
                        128,
                        None,
                        false,
                        None,
                        &StatsOptions::default(),
                    )
                    .await
                    .ok();
                    let rb: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            let mut b = serde_json::Map::new();
                            b.insert("data_type".into(), serde_json::json!(full.sample_type.gdal_name()));
                            b.insert("scale".into(), serde_json::json!(cog.index.scales_f64.get(i).copied().unwrap_or(1.0)));
                            b.insert("offset".into(), serde_json::json!(cog.index.offsets_f64.get(i).copied().unwrap_or(0.0)));
                            b.insert("sampling".into(), serde_json::json!(sampling));
                            if let Some(nd) = cog.index.nodata_f64.or_else(|| cog.index.nodata.map(|v| v as f64)) {
                                if nd.is_nan() {
                                    b.insert("nodata".into(), serde_json::json!("nan"));
                                } else {
                                    b.insert("nodata".into(), serde_json::json!(nd));
                                }
                            }
                            if let Some(bs) = stats_json.as_ref().and_then(|s| s.get(format!("b{}", i + 1))) {
                                b.insert(
                                    "statistics".into(),
                                    serde_json::json!({
                                        "mean": bs["mean"],
                                        "minimum": bs["min"],
                                        "maximum": bs["max"],
                                        "stddev": bs["std"],
                                        "valid_percent": bs["valid_percent"],
                                    }),
                                );
                                if let Some(h) = bs.get("histogram").and_then(|h| h.as_array())
                                    && h.len() == 2
                                    && let (Some(counts), Some(edges)) = (h[0].as_array(), h[1].as_array())
                                    && edges.len() >= 2
                                {
                                    b.insert(
                                        "histogram".into(),
                                        serde_json::json!({
                                            "count": counts.len() + 1,
                                            "min": edges.first(),
                                            "max": edges.last(),
                                            "buckets": counts,
                                        }),
                                    );
                                }
                            }
                            serde_json::Value::Object(b)
                        })
                        .collect();
                    asset_obj["raster:bands"] = serde_json::Value::Array(rb);
                }
                if with_eo {
                    push_ext("https://stac-extensions.github.io/eo/v1.1.0/schema.json");
                    let eb: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            let name = format!("b{}", i + 1);
                            let desc = cog
                                .index
                                .band_descriptions
                                .get(i)
                                .filter(|s| !s.is_empty())
                                .cloned()
                                .unwrap_or_else(|| name.clone());
                            serde_json::json!({ "name": name, "description": desc })
                        })
                        .collect();
                    asset_obj["eo:bands"] = serde_json::Value::Array(eb);
                }
                for ext in &q_extensions {
                    push_ext(ext);
                }
                let mut links: Vec<serde_json::Value> = Vec::new();
                if let Some(col_url) = q_collection_url {
                    links.push(serde_json::json!({
                        "rel": "collection",
                        "type": "application/json",
                        "href": col_url,
                    }));
                }
                let mut assets_map = serde_json::Map::new();
                assets_map.insert(asset_name, asset_obj);
                let mut item = serde_json::json!({
                    "type": "Feature",
                    "stac_version": "1.1.0",
                    "stac_extensions": extensions,
                    "id": item_id,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                    },
                    "bbox": wgs_bbox,
                    "properties": serde_json::Value::Object(props),
                    "links": links,
                    "assets": serde_json::Value::Object(assets_map),
                });
                if let Some(col) = q_collection {
                    item["collection"] = serde_json::json!(col);
                }
                Ok((Bytes::from(item.to_string()), "application/geo+json"))
            }
            .await,
            // GET /cog/map (OGC API - Maps)
            (Method::GET, ["map"]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let (native_crs, native_epsg, native_bounds) =
                    dataset_native_crs_and_bounds(&engine, &cogs[0])?;
                let full = cogs[0].index.full_res();
                let ext = pq.format_param.as_deref().unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let explicit_crs = pq.dst_crs.clone().or_else(|| pq.bounds_crs.clone());
                let (dst_crs, dst_epsg, used_native_crs) = match explicit_crs {
                    Some((c, e)) => (c, Some(e), native_epsg == Some(e)),
                    None if ct == GEOTIFF_CONTENT_TYPE && native_epsg.is_none() => {
                        // User-defined projected COGs have no standard EPSG code for GeoTIFF output;
                        // reproject to WGS84 rather than tagging metre coordinates as EPSG:4326.
                        (cogzilla_proj::CrsModel::Geographic, Some(4326), false)
                    }
                    None => (native_crs.clone(), native_epsg, true),
                };
                let (wgs_bbox, out_bbox) = if let Some(raw_bbox) = pq.bbox_param {
                    let raw_bbox =
                        resolve_input_bbox(raw_bbox, pq.bbox_crs.as_ref().map(|(c, _)| c))?;
                    let wb = bbox_to_wgs84(
                        raw_bbox,
                        pq.bbox_crs.as_ref().map(|(c, _)| c),
                    )?;
                    let ob = if (dst_epsg.is_some()
                        && pq.bbox_crs.as_ref().map(|(_, e)| *e) == dst_epsg)
                        || (pq.bbox_crs.is_none() && dst_epsg == Some(4326))
                    {
                        normalize_render_bbox(raw_bbox, &dst_crs)?
                    } else {
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    };
                    (wb, ob)
                } else {
                    let wb = native_crs.wgs84_bounds_with_gcp(
                        &full.pixel_to_crs,
                        full.gcp_poly.as_deref(),
                        full.width,
                        full.height,
                    );
                    let ob = if !used_native_crs {
                        normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                    } else {
                        native_bounds
                    };
                    (wb, ob)
                };
                let (def_w, def_h) = if used_native_crs {
                    let px_x = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d).max(1e-12);
                    let px_y = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e).max(1e-12);
                    let nat_w = ((out_bbox[2] - out_bbox[0]).abs() / px_x)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    let nat_h = ((out_bbox[3] - out_bbox[1]).abs() / px_y)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    let cap = pq.max_size.clamp(1, 4096);
                    if nat_w.max(nat_h) <= cap {
                        (nat_w, nat_h)
                    } else if nat_w >= nat_h {
                        (
                            cap,
                            ((nat_h as f64 / nat_w as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                        )
                    } else {
                        (
                            ((nat_w as f64 / nat_h as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                            cap,
                        )
                    }
                } else {
                    capped_native_bbox_dims(
                        &engine,
                        &cogs[0],
                        &dst_crs,
                        wgs_bbox,
                        out_bbox,
                        pq.max_size,
                    )
                };
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg.unwrap_or(0));
                let data = render_crs_bbox(
                    &engine, &cogs, out_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                let data = wrap_geotiff_fmt(
                    data,
                    ct,
                    fmt,
                    out_w,
                    out_h,
                    out_bbox,
                    dst_epsg.unwrap_or(4326),
                    pq.return_mask,
                );
                geo_meta = Some((out_bbox, dst_epsg, false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["wms"]) => {
                handle_cog_wms(&engine, &host, &raw_query, &pq, &mut geo_meta).await
            }
            (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
                async {
                    let explicit_tms = (sub.len() == 2).then_some(sub[0]);
                    if let Some(tms) = explicit_tms
                        && crate::tms::lookup_tms(tms).is_none()
                    {
                        return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                    }
                    let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let full = cogs[0].index.full_res();
                    let bounds = crs.wgs84_bounds_with_gcp(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                    let coarsest = cogs[0].index.ifds.last().unwrap();
                    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                        .max(full.height as f64 / coarsest.height.max(1) as f64);
                    let fmt = pq.tile_format.as_deref().unwrap_or("png");
                    let scale = pq.tile_scale.unwrap_or(1);
                    let tile_q = filter_tile_template_query(&raw_query);
                    let xml = if let Some(tms) = explicit_tms {
                        let (def_min, def_max) = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                            tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
                        } else {
                            crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                        };
                        let (minz, maxz) =
                            resolve_zoom_range(def_min, def_max, pq.minzoom, pq.maxzoom);
                        build_wmts_capabilities(
                            &host,
                            "cog",
                            tms,
                            "Cloud Optimized GeoTIFF",
                            bounds,
                            minz,
                            maxz,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    } else {
                        let entries: Vec<(&str, u8, u8)> = crate::tms::ALL_TMS
                            .iter()
                            .map(|t| {
                                let (dmin, dmax) = t.zoom_range_for_cog(
                                    &crs,
                                    &full.pixel_to_crs,
                                    full.width,
                                    full.height,
                                    ov_factor,
                                );
                                let (z0, z1) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                                (t.id, z0, z1)
                            })
                            .collect();
                        build_wmts_capabilities_multi(
                            &host,
                            "cog",
                            &entries,
                            None,
                            "Cloud Optimized GeoTIFF",
                            bounds,
                            fmt,
                            scale,
                            &tile_q,
                            &raw_query,
                            pq.use_epsg,
                        )
                    };
                    Ok((Bytes::from(xml), "application/xml"))
                }
                .await
            }

            // 10. GET /cog/point/{lon},{lat} (supports coord_crs=EPSG:3857)
            (Method::GET, ["point", coord]) => async {
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let Some((x, y)) = parse_point_coord(coord) else {
                    return Err(EngineError::BadRequest("expected /cog/point/{lon},{lat}".into()));
                };
                let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
                let mut json =
                    query_point(&engine, &cogs[0], lon, lat, pq.unscale, pq.nodata).await?;
                if pq.coord_crs.is_some() {
                    json["coordinates"] = serde_json::json!([x, y]);
                }
                apply_point_filters(&mut json, pq.expr_raw.as_deref(), &pq.bidx)?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            }
            .await,

            // 11. GET /cog/preview, /cog/preview.{format}, /cog/preview/{w}x{h}.{format}
            (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
                let ext_opt = p.strip_prefix("preview.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let ext = ext_opt
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let is_georef = cogs[0].index.is_georeferenced;
                if !is_georef && pq.dst_crs.is_some() {
                    let _ = engine.get_or_build_crs(&cogs[0])?;
                }
                let (crs_opt, bbox) = if is_georef {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let b = crs.wgs84_bounds_with_gcp(
                        &full.pixel_to_crs,
                        full.gcp_poly.as_deref(),
                        full.width,
                        full.height,
                    );
                    (Some(crs), b)
                } else {
                    (None, [0.0, 0.0, full.width as f64, full.height as f64])
                };
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, pq.max_size);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                    let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if !is_georef {
                        if ct == GEOTIFF_CONTENT_TYPE {
                            return Err(EngineError::BadRequest(
                                "GeoTIFF output requires a georeferenced dataset; use .png, .jpg, .webp, or .npy".into(),
                            ));
                        }
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, out_w, out_h, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                        let d = render_wgs84_bbox(
                            &engine, &cogs, bbox, out_w, out_h, None, &bands,
                            rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, out_w, out_h, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_crs_bbox(
                            &engine, &cogs, nat_bbox, crs_opt.as_ref().unwrap(), out_w, out_h, None, &bands,
                            pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, epsg)
                    } else {
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                    }
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
                if is_georef {
                    geo_meta = Some((out_bbox, Some(out_epsg), false));
                }
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["preview", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let is_georef = cogs[0].index.is_georeferenced;
                if !is_georef && pq.dst_crs.is_some() {
                    let _ = engine.get_or_build_crs(&cogs[0])?;
                }
                let (crs_opt, bbox) = if is_georef {
                    let crs = engine.get_or_build_crs(&cogs[0])?;
                    let b = crs.wgs84_bounds_with_gcp(
                        &full.pixel_to_crs,
                        full.gcp_poly.as_deref(),
                        full.width,
                        full.height,
                    );
                    (Some(crs), b)
                } else {
                    (None, [0.0, 0.0, full.width as f64, full.height as f64])
                };
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                    let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                        && full.pixel_to_crs.a > 0.0
                        && full.pixel_to_crs.e < 0.0;
                    if !is_georef {
                        if ct == GEOTIFF_CONTENT_TYPE {
                            return Err(EngineError::BadRequest(
                                "GeoTIFF output requires a georeferenced dataset; use .png, .jpg, .webp, or .npy".into(),
                            ));
                        }
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                        let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                        let d = render_wgs84_bbox(
                            &engine, &cogs, bbox, out_w, out_h, None, &bands,
                            rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, bbox, 4326)
                    } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                        let epsg = nat_epsg.unwrap_or(4326);
                        let d = render_crs_bbox(
                            &engine, &cogs, nat_bbox, crs_opt.as_ref().unwrap(), out_w, out_h, None, &bands,
                            pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                            pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, epsg)
                    } else {
                        let d = render_native_preview(
                            &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                            pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                        )
                        .await?;
                        (d, nat_bbox, nat_epsg.unwrap_or(4326))
                    }
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
                if is_georef {
                    geo_meta = Some((out_bbox, Some(out_epsg), false));
                }
                Ok((data, ct))
            }
            .await,

            // 12. GET /cog/bbox/{minx},{miny},{maxx},{maxy}[.{format}] & /cog/bbox/{...}/{w}x{h}.{format}
            (Method::GET, ["bbox", bbox_seg]) => async {
                let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("expected /cog/bbox/{minx},{miny},{maxx},{maxy}.{ext}".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(
                    ext_opt.or(pq.format_param.as_deref()).unwrap_or("png"),
                    pq.webp_quality,
                    pq.jpeg_quality,
                )?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let wgs_bbox_src = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let bbox_max = effective_bbox_max_size(&pq);
                let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &dst_crs, wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, wgs_bbox_src, dst_bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else if pq.coord_crs_3857 {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Epsg3857, wgs_bbox_src, bbox, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                    let d = render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 3857u16)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, wgs_bbox_src, wgs_bbox_src, bbox_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, wgs_bbox_src, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, wgs_bbox_src, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq.jpeg_quality, pq.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
                let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                    return Err(EngineError::BadRequest("invalid bbox coordinates".into()));
                };
                let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        bbox
                    } else {
                        let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                        reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                    };
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else if pq.coord_crs_3857 {
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                    let d = render_mercator_bbox(
                        &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 3857u16)
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                    let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, wgs_bbox, out_w, out_h, None, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, wgs_bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(epsg), false));
                Ok((data, ct))
            }
            .await,

            // 13. POST /cog/feature, /cog/feature.{format}, /cog/feature/{w}x{h}.{format}
            (Method::POST, [f]) if *f == "feature" || f.starts_with("feature.") => async {
                let ext_opt = f.strip_prefix("feature.");
                let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
                let ext = ext_opt
                    .or(pq.format_param.as_deref())
                    .unwrap_or("png");
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = if auto_fmt {
                    OutputFormat::RawRgba
                } else {
                    render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
                };
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let eff_dst = resolve_bbox_dst_crs(&pq);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let feat_max = effective_bbox_max_size(&pq);
                let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        mask.raw_bbox
                    } else if let Some(b) = mask.bbox_in_crs(dst_crs) {
                        b
                    } else {
                        reproject_wgs84_bounds(bbox, dst_crs)?
                    };
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, feat_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, out_w, out_h, dst_epsg)
                } else {
                    let (def_w, def_h) = capped_native_bbox_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, bbox, bbox, feat_max);
                    let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326u16)
                };
                let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,
            (Method::POST, ["feature", dims_ext]) => async {
                let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
                let body_bytes = read_limited_body(req.into_body()).await?;
                let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
                let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                let full = cogs[0].index.full_res();
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
                let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
                let st = widest_sample_type(&cogs);
                let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
                let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
                let cf = pq.parse_color_formula()?;
                let eff_dst = resolve_bbox_dst_crs(&pq);
                let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
                let mask_opt = pq.shape_mask.then_some(&mask);
                let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                    let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                        mask.raw_bbox
                    } else if let Some(b) = mask.bbox_in_crs(dst_crs) {
                        b
                    } else {
                        reproject_wgs84_bounds(bbox, dst_crs)?
                    };
                    let d = render_crs_bbox(
                        &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, dst_bbox, dst_epsg)
                } else {
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 4326u16)
                };
                let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
                geo_meta = Some((out_bbox, Some(out_epsg), false));
                Ok((data, ct))
            }
            .await,

            (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
            (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
                Err(EngineError::MethodNotAllowed("POST"))
            }
            (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
            (_, ["info"] | ["info.geojson"] | ["bounds"] | ["validate"] | ["stac"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["wms"] | ["viewer"] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
                Err(EngineError::MethodNotAllowed("GET, HEAD"))
            }

            _ => Err(EngineError::NotFound(path)),
        };
        let res = match res {
            Ok((data, ct)) => Ok((cog_array_in_its_type(&engine, cog_array, &cog_urls, data).await, ct)),
            Err(e) => Err(e),
        };
        let mut resp = to_http_response(&engine, res, if_none_match.as_deref());
        if let Some((bbox, epsg, is_crs84)) = geo_meta {
            attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
        }
        return Ok(resp);
    }

    // 405 Method Not Allowed when a known route is requested with an unsupported HTTP verb
    if matches!(
        path.as_str(),
        "/" | "/index.html"
            | "/healthz"
            | "/readyz"
            | "/metrics"
            | "/conformance"
            | "/conformance/"
            | "/openapi.json"
            | "/api"
            | "/docs"
            | "/api.html"
    ) || matches!(
        segs.first().copied(),
        Some("algorithms" | "colorMaps" | "tileMatrixSets" | "mosaicjson")
    ) {
        let allow = if matches!(segs.as_slice(), ["mosaicjson", "validate" | "statistics" | "feature", ..]) {
            "POST"
        } else {
            "GET, HEAD"
        };
        return Ok(method_not_allowed_resp(allow));
    }

    Ok(json_error_resp(StatusCode::NOT_FOUND, &path))
}

/// A pgSTAC route's answer with the `Cache-Control` the tiles carry, as titiler-pgstac
/// puts one on what it serves: on what a `GET` found, never on an error, and not where
/// one is already set.
fn pgstac_cacheable(mut resp: Response<Full<Bytes>>, method: &Method) -> Response<Full<Bytes>> {
    if *method == Method::GET && resp.status().is_success() && !resp.headers().contains_key(CACHE_CONTROL) {
        resp.headers_mut().insert(CACHE_CONTROL, cache_control_header().clone());
    }
    resp
}

fn cache_control_header() -> &'static HeaderValue {
    crate::config::Config::global().cache_control_header()
}

fn etag_matches(if_none_match: &str, etag: &str) -> bool {
    let trimmed = if_none_match.trim();
    if trimmed == "*" {
        return true;
    }
    let target = etag.strip_prefix("W/").unwrap_or(etag);
    for part in trimmed.split(',') {
        let cand = part.trim().strip_prefix("W/").unwrap_or(part.trim());
        if cand == target {
            return true;
        }
    }
    false
}

fn attach_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    bbox: [f64; 4],
    epsg: Option<u16>,
    is_crs84: bool,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if let Ok(hv) = HeaderValue::from_str(&format!(
        "{},{},{},{}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )) {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-bbox"), hv);
    }
    let crs_hv = if is_crs84 {
        Some(HeaderValue::from_static(
            "<http://www.opengis.net/def/crs/OGC/0/CRS84>",
        ))
    } else if let Some(code) = epsg {
        match code {
            3857 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/3857>",
            )),
            4326 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/4326>",
            )),
            other => HeaderValue::from_str(&format!(
                "<http://www.opengis.net/def/crs/EPSG/0/{other}>"
            ))
            .ok(),
        }
    } else {
        None
    };
    if let Some(hv) = crs_hv {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-crs"), hv);
    }
}

#[derive(Clone, Copy)]
struct TileGeoOpts {
    buffer: f32,
    tile_size: u32,
    has_explicit_ts: bool,
}

impl TileGeoOpts {
    fn from_pq(pq: &ParsedQuery) -> Self {
        let has_explicit_ts = pq
            .keys
            .iter()
            .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
        Self {
            buffer: pq.buffer,
            tile_size: pq.tile_size,
            has_explicit_ts,
        }
    }
}

fn attach_tile_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    pq: &ParsedQuery,
) {
    attach_tile_geo_headers_opts(resp, tms, z, x, y, scale_opt, TileGeoOpts::from_pq(pq));
}

fn attach_tile_geo_headers_opts(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    opts: TileGeoOpts,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if tms == "WebMercatorQuad" {
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(256))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { 256 });
        let bbox = web_mercator_buffered_3857_bbox(z, x, y, opts.buffer, ts);
        attach_geo_headers(resp, bbox, Some(3857), false);
    } else if let Some(tms_def) = crate::tms::lookup_tms(tms)
        && let Some(m) = tms_def.matrix(z)
    {
        let base_ts = m.tile_width.clamp(64, 4096);
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(base_ts))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { base_ts });
        if let Some(bbox) = tms_def.tile_bounds(z, x, y, opts.buffer, ts) {
            let is_crs84 = tms == "WorldCRS84Quad";
            attach_geo_headers(resp, bbox, Some(tms_def.epsg), is_crs84);
        }
    }
}

fn to_http_response(
    engine: &TileEngine,
    res: Result<(Bytes, &'static str), EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok((bytes, content_type)) => {
            let is_binary_tile = content_type.starts_with("image/")
                || content_type == "application/octet-stream"
                || content_type == "application/x-binary";
            let etag = is_binary_tile.then(|| crate::tile_cache::compute_tile_etag(&bytes));
            binary_or_plain_response(engine, bytes, content_type, etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn to_http_response_cached(
    engine: &TileEngine,
    res: Result<crate::tile_cache::CachedTile, EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok(tile) => {
            binary_or_plain_response(engine, tile.bytes, tile.content_type, tile.etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn binary_or_plain_response(
    engine: &TileEngine,
    bytes: Bytes,
    content_type: &'static str,
    etag: Option<HeaderValue>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    if let Some(etag_hv) = etag {
        if let Some(inm) = if_none_match
            && let Ok(etag_str) = etag_hv.to_str()
                && etag_matches(inm, etag_str) {
                    engine
                        .stats
                        .http_304_not_modified
                        .fetch_add(1, Ordering::Relaxed);
                    let mut resp = Response::new(Full::new(Bytes::new()));
                    *resp.status_mut() = StatusCode::NOT_MODIFIED;
                    resp.headers_mut().insert(ETAG, etag_hv);
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, cache_control_header().clone());
                    resp.headers_mut()
                        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
                    return resp;
                }
        let mut resp = plain_resp(StatusCode::OK, bytes, content_type);
        resp.headers_mut().insert(ETAG, etag_hv);
        resp.headers_mut()
            .insert(CACHE_CONTROL, cache_control_header().clone());
        resp
    } else {
        plain_resp(StatusCode::OK, bytes, content_type)
    }
}

fn json_error_resp(status: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": msg }).to_string();
    plain_resp(status, Bytes::from(body), "application/json")
}

fn error_to_http_response(err: &EngineError) -> Response<Full<Bytes>> {
    match err {
        EngineError::Shared(inner) => error_to_http_response(inner),
        EngineError::NoContent => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("access-control-allow-origin", "*")
            .body(Full::new(Bytes::new()))
            .unwrap(),
        EngineError::MethodNotAllowed(allow) => method_not_allowed_resp(allow),
        EngineError::NotFound(msg) => json_error_resp(StatusCode::NOT_FOUND, msg),
        EngineError::BadGateway(msg) => json_error_resp(StatusCode::BAD_GATEWAY, msg),
        EngineError::BadRequest(msg) => json_error_resp(StatusCode::BAD_REQUEST, msg),
        EngineError::PayloadTooLarge(msg) => json_error_resp(StatusCode::PAYLOAD_TOO_LARGE, msg),
        EngineError::RequestTimeout(msg) => json_error_resp(StatusCode::REQUEST_TIMEOUT, msg),
        // The file is in a coordinate system this server cannot place. That is a
        // property of the requested source, not a server fault.
        EngineError::Proj(e) => json_error_resp(StatusCode::BAD_REQUEST, &e.to_string()),
        EngineError::Io(io_e) => match io_e.root() {
            e @ cogzilla_io::IoError::InvalidUri(_) => {
                json_error_resp(StatusCode::BAD_REQUEST, &e.to_string())
            }
            e @ cogzilla_io::IoError::Tiff(_) => {
                json_error_resp(StatusCode::BAD_REQUEST, &e.to_string())
            }
            e @ cogzilla_io::IoError::Forbidden(_) => {
                json_error_resp(StatusCode::FORBIDDEN, &e.to_string())
            }
            cogzilla_io::IoError::HttpStatus {
                status: status @ (401 | 403),
                url,
            } => json_error_resp(
                StatusCode::FORBIDDEN,
                &format!("origin refused access (HTTP {status}) fetching {url}"),
            ),
            e @ cogzilla_io::IoError::HttpStatus { status: 404, .. } => {
                json_error_resp(StatusCode::NOT_FOUND, &e.to_string())
            }
            e @ cogzilla_io::IoError::HttpStatus {
                status: 408 | 504, ..
            } => json_error_resp(StatusCode::GATEWAY_TIMEOUT, &e.to_string()),
            e @ cogzilla_io::IoError::HttpStatus { .. } => {
                json_error_resp(StatusCode::BAD_GATEWAY, &e.to_string())
            }
            e @ cogzilla_io::IoError::RangeNotSupported { .. } => {
                json_error_resp(StatusCode::BAD_GATEWAY, &e.to_string())
            }
            e @ cogzilla_io::IoError::SourceChanged { .. } => {
                let mut r = json_error_resp(StatusCode::BAD_GATEWAY, &e.to_string());
                r.headers_mut()
                    .insert("x-cogzilla-retry", HeaderValue::from_static("1"));
                r
            }
            e @ cogzilla_io::IoError::ResponseTooLarge { .. } => {
                json_error_resp(StatusCode::BAD_GATEWAY, &e.to_string())
            }
            e @ cogzilla_io::IoError::Http(req_err) => {
                let mut src: Option<&(dyn std::error::Error + 'static)> =
                    std::error::Error::source(req_err);
                while let Some(s) = src {
                    if let Some(cogzilla_io::IoError::Forbidden(u)) =
                        s.downcast_ref::<cogzilla_io::IoError>()
                    {
                        return json_error_resp(
                            StatusCode::FORBIDDEN,
                            &cogzilla_io::IoError::Forbidden(u.clone()).to_string(),
                        );
                    }
                    src = s.source();
                }
                if req_err.is_timeout() {
                    json_error_resp(
                        StatusCode::GATEWAY_TIMEOUT,
                        &format!("upstream timeout: {e}"),
                    )
                } else {
                    json_error_resp(StatusCode::BAD_GATEWAY, &e.to_string())
                }
            }
            e @ cogzilla_io::IoError::Fs(io_err)
                if io_err.kind() == std::io::ErrorKind::NotFound =>
            {
                json_error_resp(StatusCode::NOT_FOUND, &e.to_string())
            }
            e => {
                tracing::warn!(error = %e, "request failed with 500");
                json_error_resp(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn single(rescale: (f32, f32), colormap: ColormapName) -> BandRequest {
        BandRequest::Single { band_1based: 1, rescale, colormap }
    }

    fn selected(bands: usize, values: &[f32]) -> Selected {
        Selected { bands, values: values.to_vec() }
    }

    #[test]
    fn a_pixel_selection_is_painted_by_rio_tilers_rules() {
        let pq = parse_query(None);
        let nan = f32::NAN;
        // One band without a colour map: the byte is the grey.
        let grey = single((0.0, 255.0), ColormapName::Grayscale);
        let rgba = paint_selected(selected(1, &[33.0, 244.0, nan]), 3, &grey, &pq, None).unwrap();
        assert_eq!(rgba, [33, 33, 33, 255, 244, 244, 244, 255, 0, 0, 0, 0]);
        // The byte is `linear_rescale`'s: the top of the range is 255.
        let rgba = paint_selected(selected(1, &[0.3]), 1, &single((0.0, 0.3), ColormapName::Grayscale), &pq, None).unwrap();
        assert_eq!(rgba, [255, 255, 255, 255]);
        // Red, green and blue: there where any band has a value, with 0 for one that has not.
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
        let rgba = paint_selected(selected(3, &[10.0, nan, nan, 20.0, 20.0, nan, 30.0, nan, nan]), 3, &rgb, &pq, None).unwrap();
        assert_eq!(rgba, [10, 20, 30, 255, 0, 20, 0, 255, 0, 0, 0, 0]);
        // An infinity is a value, clipped to the range.
        let rgba = paint_selected(selected(1, &[f32::INFINITY, f32::NEG_INFINITY]), 2, &grey, &pq, None).unwrap();
        assert_eq!(rgba, [255, 255, 255, 255, 0, 0, 0, 255]);
        // A range with no width is 0 throughout.
        let rgba = paint_selected(selected(1, &[2.0, 3.0, 4.0]), 3, &single((3.0, 3.0), ColormapName::Grayscale), &pq, None).unwrap();
        assert_eq!(rgba, [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn a_pixel_selection_keeps_the_colour_maps_a_raster_or_a_request_brings() {
        let pq = parse_query(None);
        let grey = single((0.0, 255.0), ColormapName::Grayscale);
        let values = selected(1, &[1.0, 2.0, 7.0, f32::NAN, 300.0]);
        // A raster's own palette: the class's colour and its transparency, by the value.
        let mut palette = vec![[0u8; 4]; 256];
        palette[1] = [10, 20, 30, 255];
        palette[2] = [40, 50, 60, 128];
        let own = CustomColormap::from_rgba_palette(&palette);
        let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&own)).unwrap();
        assert_eq!(rgba[..8], [10, 20, 30, 255, 40, 50, 60, 128]);
        assert_eq!(rgba[8..], [0; 12]);
        // A colour map given as JSON, classes and intervals.
        let classes = CustomColormap::from_json_str(r##"{"1": [1, 2, 3, 200], "300": "#ff0000"}"##, false).unwrap();
        let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&classes)).unwrap();
        assert_eq!(rgba[..4], [1, 2, 3, 200]);
        assert_eq!(rgba[4..16], [0; 12]);
        assert_eq!(rgba[16..], [255, 0, 0, 255]);
        let spans = CustomColormap::from_json_str("[[[0, 5], [9, 9, 9, 255]], [[5, 10], [7, 7, 7, 255]]]", false).unwrap();
        let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&spans)).unwrap();
        assert_eq!(rgba[..12], [9, 9, 9, 255, 9, 9, 9, 255, 7, 7, 7, 255]);
        assert_eq!(rgba[12..], [0; 8]);
        // A ramp between stops, over the rescaled byte.
        let ramp = CustomColormap::from_json_str_with_type(r#"{"0": [0, 0, 0, 255], "255": [255, 255, 255, 255]}"#, true, true).unwrap();
        let rgba = paint_selected(selected(1, &[0.0, 50.0, 100.0]), 3, &single((0.0, 100.0), ColormapName::Grayscale), &pq, Some(&ramp)).unwrap();
        assert_eq!([rgba[0], rgba[8]], [0, 255]);
        assert!((126..=128).contains(&rgba[4]), "{}", rgba[4]);
        // Three bands are a picture already: no colour map is looked at.
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
        let rgba = paint_selected(selected(3, &[1.0, 2.0, 3.0]), 1, &rgb, &pq, Some(&own)).unwrap();
        assert_eq!(rgba, [1, 2, 3, 255]);
    }

    #[test]
    fn an_array_asked_with_a_rescale_is_the_pictures_bytes() {
        // One band of 50 and a pixel that is not there, with its mask, as the renderer writes it.
        let f32_npy = |values: &[f32], mask: bool| {
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(Bytes::from(raw), values.len() / 2, 1, 2, mask)
        };
        let body = |npy: &[u8]| npy[10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]))..].to_vec();
        let shown = |q: &str, npy: &[u8]| shown_array(&parse_query(Some(q)), None, npy).unwrap();
        let grey = shown("rescale=0,100", &f32_npy(&[50.0, f32::NAN], true));
        assert!(std::str::from_utf8(&grey[10..60]).unwrap().contains("'descr': '|u1'"));
        assert!(std::str::from_utf8(&grey[10..80]).unwrap().contains("'shape': (2, 1, 2)"));
        // 50 of 0..100 is 127, as rio-tiler makes it; the mask is 255 where the pixel is.
        assert_eq!(body(&grey), [127, 0, 255, 0]);
        // A colour map is red, green and blue.
        let mapped = shown("rescale=0,100&colormap_name=viridis", &f32_npy(&[50.0, f32::NAN], true));
        let rgb = ColormapName::Viridis.lut().rgb[127];
        assert_eq!(body(&mapped), [rgb[0], 0, rgb[1], 0, rgb[2], 0, 255, 0]);
        // Without the mask, the band alone.
        let bare = shown("rescale=0,100&return_mask=false", &f32_npy(&[50.0, 100.0], false));
        assert_eq!(body(&bare), [127, 255]);
        // 2-band and 4-band rescaled arrays preserve every band and the any-band-valid mask.
        let two = shown("rescale=0,100&rescale=0,200", &f32_npy(&[50.0, f32::NAN, 100.0, 200.0], true));
        assert!(std::str::from_utf8(&two[10..80]).unwrap().contains("'shape': (3, 1, 2)"));
        assert_eq!(body(&two), [127, 0, 127, 255, 255, 255]);
        let four = shown("rescale=0,100", &f32_npy(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], true));
        assert!(std::str::from_utf8(&four[10..80]).unwrap().contains("'shape': (5, 1, 2)"));
        assert_eq!(body(&four).len(), 10);
        // STAC per-asset expression with rescale still asks for painted bytes.
        let pq_ae = parse_query(Some("assets=red|expression=b1*2&rescale=0,100"));
        assert!(matches!(array_asked(&pq_ae, Some(&"tile.npy")), Some(ArrayAsked::Shown(_))));
        // What is not a float32 array is not painted.
        assert!(shown_array(&parse_query(Some("rescale=0,100")), None, b"not an array").is_none());
    }

    #[test]
    fn a_pixel_selection_keeps_a_range_given_upside_down() {
        let pq = parse_query(None);
        let rgba = paint_selected(selected(1, &[0.0, 51.0, 255.0, f32::NAN]), 4, &single((255.0, 0.0), ColormapName::Grayscale), &pq, None).unwrap();
        assert_eq!(rgba, [255, 255, 255, 255, 204, 204, 204, 255, 0, 0, 0, 255, 0, 0, 0, 0]);
        let table = ColormapName::Viridis.lut();
        let rgba = paint_selected(selected(1, &[0.0]), 1, &single((255.0, 0.0), ColormapName::Viridis), &pq, None).unwrap();
        assert_eq!(rgba[..3], table.rgb[255]);
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0, 100.0, 0.0], rescale_max: [100.0, 0.0, 100.0] };
        let rgba = paint_selected(selected(3, &[20.0, 20.0, 20.0]), 1, &rgb, &pq, None).unwrap();
        assert_eq!(rgba, [51, 204, 51, 255]);
    }

    #[test]
    fn a_pixel_selection_paints_whatever_bands_the_request_leaves() {
        let pq = parse_query(None);
        // Band math for red, green and blue keeps its own ranges.
        let prog = || ExprProgram::compile("b1").unwrap();
        let math = BandRequest::CustomExprRgb { progs: [prog(), prog(), prog()], rescale_min: [0.0; 3], rescale_max: [100.0, 200.0, 400.0] };
        let rgba = paint_selected(selected(3, &[100.0, 100.0, 100.0]), 1, &math, &pq, None).unwrap();
        assert_eq!(rgba, [255, 127, 63, 255]);
        // Bands with no range of their own are bytes as they are: a terrain encoding's
        // three, and the first three of however many were asked for.
        let terrain = BandRequest::Terrain { band_1based: 1, encoding: TerrainEncoding::Terrarium };
        let rgba = paint_selected(selected(3, &[128.0, 7.0, 255.0]), 1, &terrain, &pq, None).unwrap();
        assert_eq!(rgba, [128, 7, 255, 255]);
        let four = BandRequest::MultiBand { bands_1based: vec![1, 2, 3, 4] };
        let rgba = paint_selected(selected(4, &[1.0, 2.0, 3.0, 4.0]), 1, &four, &pq, None).unwrap();
        assert_eq!(rgba, [1, 2, 3, 255]);
        // Two bands: the first, under the request's colour map.
        let two = BandRequest::MultiBand { bands_1based: vec![1, 2] };
        let mapped = parse_query(Some("colormap_name=viridis"));
        let table = ColormapName::Viridis.lut();
        let rgba = paint_selected(selected(2, &[64.0, 200.0]), 1, &two, &mapped, None).unwrap();
        assert_eq!(rgba[..3], table.rgb[64]);
        // One band left of a request for three (a count): bytes as they are, not the
        // request's ranges.
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [4000.0; 3] };
        let rgba = paint_selected(selected(1, &[2.0]), 1, &rgb, &pq, None).unwrap();
        assert_eq!(rgba, [2, 2, 2, 255]);
        // A request for one band that three came of: red, green and blue as they are.
        let rgba = paint_selected(selected(3, &[1.0, 2.0, 3.0]), 1, &single((0.0, 4000.0), ColormapName::Viridis), &pq, None).unwrap();
        assert_eq!(rgba, [1, 2, 3, 255]);
    }

    #[test]
    fn a_pixel_selection_takes_a_colour_formula_before_the_colour_map() {
        let pq = parse_query(Some("color_formula=gamma%20rgb%202"));
        let formula = ColorFormula::parse("gamma rgb 2").unwrap();
        let levels = paint::formula_levels(&formula);
        assert_ne!(levels[64], 64);
        // One band: the formula on its byte, then the map, or the grey that byte is.
        let table = ColormapName::Viridis.lut();
        let rgba = paint_selected(selected(1, &[64.0, f32::NAN]), 2, &single((0.0, 255.0), ColormapName::Viridis), &pq, None).unwrap();
        assert_eq!(rgba[..3], table.rgb[usize::from(levels[64])]);
        assert_eq!(rgba[3..], [255, 0, 0, 0, 0]);
        let rgba = paint_selected(selected(1, &[64.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &pq, None).unwrap();
        assert_eq!(rgba, [levels[64], levels[64], levels[64], 255]);
        // Three bands: the formula on the finished picture.
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
        let rgba = paint_selected(selected(3, &[64.0, 0.0, 255.0]), 1, &rgb, &pq, None).unwrap();
        assert_eq!(rgba, [levels[64], 0, 255, 255]);
        // Under a colour map of the raster's or the request's own it is for the finished picture.
        let classes = CustomColormap::from_json_str(r#"{"64": [1, 2, 3, 255]}"#, false).unwrap();
        let rgba = paint_selected(selected(1, &[64.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &pq, Some(&classes)).unwrap();
        let mut mapped = [1u8, 2, 3, 255];
        formula.apply_buffer_in_place(&mut mapped, 4);
        assert_ne!(mapped, [1, 2, 3, 255]);
        assert_eq!(rgba, mapped);
        // A formula that does not parse is the request's fault.
        let bad = parse_query(Some("color_formula=nonsense"));
        assert!(matches!(
            paint_selected(selected(1, &[1.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &bad, None),
            Err(EngineError::BadRequest(_))
        ));
    }

    #[test]
    fn a_painted_pixel_selection_is_encoded_as_any_other_picture() {
        let pq = parse_query(None);
        let grey = single((0.0, 255.0), ColormapName::Grayscale);
        let rgba = paint_selected(selected(1, &[33.0, f32::NAN, 200.0, 90.0]), 4, &grey, &pq, None).unwrap();
        // The bare pixels, with and without the mask.
        assert_eq!(encode_rgba_output(2, 2, &rgba, OutputFormat::RawRgba, true).as_ref(), rgba.as_slice());
        let png = encode_rgba_output(2, 2, &rgba, OutputFormat::Png, true);
        assert_eq!(png[..8], [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let unmasked = encode_rgba_output(2, 2, &rgba, OutputFormat::Png, false);
        assert_eq!(unmasked[..8], png[..8]);
        assert_ne!(unmasked, png, "return_mask=false leaves the mask out");
        let jpeg = encode_rgba_output(2, 2, &rgba, OutputFormat::Jpeg { quality: 75 }, true);
        assert_eq!(jpeg[..2], [0xff, 0xd8]);
    }

    #[test]
    fn unrescaled_uint16_png_emits_16_bit_png_and_clamps_overshoot() {
        let pq_plain = parse_query(None);
        let pq_rescaled = parse_query(Some("rescale=0,1000"));
        let pq_cmap = parse_query(Some("colormap_name=viridis"));
        let single = BandRequest::Single { band_1based: 1, rescale: (0.0, 65535.0), colormap: ColormapName::Grayscale };
        assert!(wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &single, cogzilla_tiff::SampleType::Uint16));
        assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &single, cogzilla_tiff::SampleType::Uint8));
        assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_rescaled, None, &single, cogzilla_tiff::SampleType::Uint16));
        assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_cmap, None, &single, cogzilla_tiff::SampleType::Uint16));

        // 4x1 single-band f32 buffer testing clamp (-1.0 -> 0, 70000.0 -> 65535),
        // rounding (123.6 -> 124), and NaN (0, alpha=0) in 16-bit Grey+Alpha PNG.
        let mut raw = Vec::new();
        for v in [-1.0f32, 123.6, 70000.0, f32::NAN] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let png = wrap_geotiff_fmt(Bytes::from(raw), "image/png", OutputFormat::RawF32, 4, 1, [0.0; 4], 4326, true);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        // IHDR: byte 24 is bit depth (16), byte 25 is colour type (4 = Greyscale+Alpha).
        assert_eq!(png[24], 16, "expected 16-bit PNG");
        assert_eq!(png[25], 4, "expected Greyscale+Alpha colour type");
        // Extract and inflate IDAT payload to verify decoded 16-bit BE samples & alpha.
        let mut idat = Vec::new();
        let mut pos = 8usize;
        while pos + 12 <= png.len() {
            let len = u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]) as usize;
            let chunk_type = &png[pos + 4..pos + 8];
            if chunk_type == b"IDAT" {
                idat.extend_from_slice(&png[pos + 8..pos + 8 + len]);
            }
            pos += 12 + len;
        }
        let mut scanline = vec![0u8; 17];
        let n = libdeflater::Decompressor::new()
            .zlib_decompress(&idat, &mut scanline)
            .expect("valid zlib IDAT");
        // Filter byte (0) + 4 pixels * (u16 value + u16 alpha) = 1 + 16 = 17 bytes.
        assert_eq!(n, 17);
        assert_eq!(scanline[0], 0);
        let decoded: Vec<(u16, u16)> = scanline[1..]
            .chunks_exact(4)
            .map(|c| (u16::from_be_bytes([c[0], c[1]]), u16::from_be_bytes([c[2], c[3]])))
            .collect();
        assert_eq!(decoded, vec![(0, 65535), (124, 65535), (65535, 65535), (0, 0)]);

        // 2-band Uint16 `.png` tile request is accepted by build_tile_request_for_tms.
        let pq_2b = parse_query(Some("bidx=1&bidx=2"));
        let res = build_tile_request_for_tms(
            None,
            TileCoord { z: 0, x: 0, y: 0 },
            None,
            Some("png"),
            &pq_2b,
            2,
            cogzilla_tiff::SampleType::Uint16,
            "WebMercatorQuad",
            None,
            false,
        );
        assert!(res.is_ok(), "2-band Uint16 .png tile should be accepted: {:?}", res.err());
    }

    #[test]
    fn pixel_selection_names_keep_their_aliases_and_any_case() {
        for (name, method) in [
            ("first", Selection::First),
            ("highest", Selection::Highest),
            ("lowest", Selection::Lowest),
            ("mean", Selection::Mean),
            ("median", Selection::Median),
            ("stdev", Selection::Stdev),
            ("std", Selection::Stdev),
            ("STD", Selection::Stdev),
            ("StDev", Selection::Stdev),
            ("count", Selection::Count),
            ("Highest", Selection::Highest),
            ("LastBandLow", Selection::LastBandLow),
            ("LASTBANDHIGH", Selection::LastBandHigh),
        ] {
            assert_eq!(parse_pixel_selection(name), Some(method), "{name}");
            let pq = parse_query(Some(&format!("pixel_selection={name}")));
            assert_eq!(pq.pixel_selection, method, "{name}");
            assert!(pq.errors.is_empty(), "{name}: {:?}", pq.errors);
        }
        assert_eq!(parse_query(None).pixel_selection, Selection::First);
        for name in ["sum", "", "stddev", "last"] {
            assert_eq!(parse_pixel_selection(name), None, "{name}");
        }
        let pq = parse_query(Some("pixel_selection=sum"));
        assert_eq!(
            pq.errors,
            ["unsupported pixel_selection 'sum' (expected first, highest, lowest, mean, median, stdev, count, lastbandlow, or lastbandhigh)"]
        );
    }

    #[test]
    fn mean_and_median_are_cast_back_only_for_a_raster_s_own_whole_numbers() {
        use cogzilla_tiff::SampleType as T;
        let single = BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Grayscale };
        let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
        let multi = BandRequest::MultiBand { bands_1based: vec![1, 2] };
        for bands in [&single, &rgb, &multi] {
            for kind in [T::Uint8, T::Int8, T::Uint16, T::Int16, T::Uint32, T::Int32] {
                assert!(reads_whole_numbers(false, bands, kind), "{bands:?} {kind:?}");
                assert!(!reads_whole_numbers(true, bands, kind), "unscaled {bands:?} {kind:?}");
            }
            for kind in [T::Float32, T::Float64] {
                assert!(!reads_whole_numbers(false, bands, kind), "{bands:?} {kind:?}");
            }
        }
        let pq = parse_query(Some("expression=b1/2"));
        let math = build_band_request(&pq, 1, T::Uint8).expect("a band math request");
        assert!(!reads_whole_numbers(false, &math, T::Uint8), "{math:?}");
    }

    #[test]
    fn the_median_budget_is_64_mib_of_held_scenes() {
        assert_eq!(MAX_MEDIAN_BUFFER_BYTES, 64 * 1024 * 1024);
        let (bands, pixels) = (3, 256 * 256);
        // The 64 scenes a request may name fit at the default tile size.
        assert!(Fold::<f32>::held_bytes(Selection::Median, 64, bands, pixels) <= MAX_MEDIAN_BUFFER_BYTES);
        // At 2048 px a side, one three-band scene fits and the second does not.
        let pixels = 2048 * 2048;
        assert!(Fold::<f32>::held_bytes(Selection::Median, 1, bands, pixels) <= MAX_MEDIAN_BUFFER_BYTES);
        assert!(Fold::<f32>::held_bytes(Selection::Median, 2, bands, pixels) > MAX_MEDIAN_BUFFER_BYTES);
        // No other method holds its scenes, so none is bounded by their number.
        assert_eq!(Fold::<f32>::held_bytes(Selection::Mean, 64, bands, pixels), 0);
    }

    #[test]
    fn colormap_param_sets_explicit_for_hillshade_and_stac_bbox_accepts_coord_crs() {
        let pq = parse_query(Some("algorithm=hillshade&colormap=terrain"));
        assert!(pq.colormap_explicit);
        let bands = build_band_request(&pq, 1, cogzilla_tiff::SampleType::Float32)
            .expect("valid hillshade request");
        match bands {
            BandRequest::Dem { colormap, .. } => {
                assert_ne!(colormap, ColormapName::Grayscale);
                assert_eq!(colormap, ColormapName::from_str_opt("terrain").unwrap());
            }
            other => panic!("expected BandRequest::Dem, got {other:?}"),
        }

        // /stac/bbox must accept coord_crs=EPSG:3857 (matching /cog/bbox)
        let pq_bbox = parse_query(Some("url=foo.json&coord_crs=EPSG:3857"));
        let res = validate_query(&["stac", "bbox", "0,0,1000,1000.png"], &pq_bbox);
        assert!(res.is_ok(), "unexpected error: {res:?}");
    }

    #[test]
    fn xyz_bounds_and_http_error_status_mapping() {
        let pq = parse_query(None);
        assert_eq!(pq.resample, ResampleMode::Nearest);
        assert_eq!(pq.reproject, ResampleMode::Nearest);
        let u8t = cogzilla_tiff::SampleType::Uint8;
        // WebMercatorQuad at z=2 has 4x4 tiles (0..=3)
        assert!(build_tile_request(None, TileCoord { z: 2, x: 3, y: 3 }, None, Some("png"), &pq, 3, u8t, false, None).is_ok());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 4, y: 0 }, None, Some("png"), &pq, 3, u8t, false, None).is_err());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, false, None).is_err());
        // WorldCRS84Quad at z=2 has 8x4 tiles (x: 0..=7, y: 0..=3)
        assert!(build_tile_request(None, TileCoord { z: 2, x: 7, y: 3 }, None, Some("png"), &pq, 3, u8t, true, None).is_ok());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 8, y: 0 }, None, Some("png"), &pq, 3, u8t, true, None).is_err());
        assert!(build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, true, None).is_err());

        // ?reproject= is accepted and parsed independently from ?resampling=
        let pq_rp = parse_query(Some("resampling=bilinear&reproject=cubic"));
        assert!(pq_rp.errors.is_empty());
        assert_eq!(pq_rp.resample, ResampleMode::Bilinear);
        assert_eq!(pq_rp.reproject, ResampleMode::Cubic);
        assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "0", "0", "0.png"], &pq_rp).is_ok());

        // Single-band default colormap is Grayscale (matching TiTiler)
        assert_eq!(pq.colormap, ColormapName::Grayscale);

        // Form-encoded ?expression= with spaces (`+`), `%2B`, `%2A`, `%5E`, `%28`/`%29` decodes cleanly, while raw `(b1+b2)` still works
        let pq_form = parse_query(Some("expression=%28b1+-+b2%29+%2F+%28b1+%2B+b2%29"));
        assert_eq!(pq_form.expr_raw.as_deref(), Some("(b1 - b2) / (b1 + b2)"));
        let pq_mul = parse_query(Some("expression=10+%2A+log10%28b1%29"));
        assert_eq!(pq_mul.expr_raw.as_deref(), Some("10 * log10(b1)"));
        let pq_raw = parse_query(Some("expression=(b1-b2)/(b1+b2)"));
        assert_eq!(pq_raw.expr_raw.as_deref(), Some("(b1-b2)/(b1+b2)"));

        // ?format= applies on format-less tile routes parsed via parse_y_scale_ext
        let (y_parsed, sc_parsed, ext_opt) = parse_y_scale_ext("1@2x").unwrap();
        assert_eq!((y_parsed, sc_parsed, ext_opt), (1, Some(2), None));
        let pq_fmt = parse_query(Some("format=webp"));
        let (treq, ct) =
            build_tile_request(None, TileCoord { z: 2, x: 1, y: y_parsed }, sc_parsed, ext_opt, &pq_fmt, 3, u8t, false, None)
                .unwrap();
        assert_eq!(ct, "image/webp");
        assert!(matches!(treq.format, OutputFormat::Webp { .. }));

        // Invalid minzoom > maxzoom rejected on tilejson.json
        let pq_bad_z = parse_query(Some("minzoom=14&maxzoom=8"));
        assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_bad_z).is_err());

        // Default dtype rescale ranges & aspect-preserving dims & hex colormaps
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Uint16, false), (0.0, 65535.0));
        assert_eq!(default_rescale_for_dtype(cogzilla_tiff::SampleType::Int16, true), (-32768.0, 32767.0));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 2.0, 1.0], 512), (512, 256));
        assert_eq!(aspect_preserving_dims([0.0, 0.0, 1.0, 4.0], 512), (128, 512));
        let hex_lut = cogzilla_kernel::ColormapLut::from_json_str(r##"{"0":"#112233","255":"#ff000080"}"##).unwrap();
        assert_eq!(hex_lut.rgb[0], [0x11, 0x22, 0x33]);
        assert_eq!(hex_lut.rgba_u32[255].to_le_bytes(), [0xff, 0x00, 0x00, 0x80]);

        // TiTiler query param parity: nodata=nan/inf/-inf, z_exaggeration, [0,1000] rescale,
        // colormap_name precedence over inline colormap, and tilejson minzoom/maxzoom/tile_format.
        let pq_nd = parse_query(Some("nodata=nan&rescale=[0,%201000]&algorithm_params=%7B%22z_exaggeration%22:2.5%7D"));
        assert!(pq_nd.errors.is_empty());
        assert!(pq_nd.nodata.unwrap().is_nan());
        assert_eq!(pq_nd.rescale, vec![(0.0, 1000.0)]);
        assert!((pq_nd.z_factor - 2.5).abs() < 1e-6);

        let pq_cm = parse_query(Some("colormap=%7B%220%22:[0,0,0,255]%7D&colormap_name=viridis"));
        assert!(pq_cm.errors.is_empty());
        assert!(pq_cm.custom_colormap_json.is_none());
        assert_eq!(pq_cm.colormap, ColormapName::from_str_opt("viridis").unwrap());

        let pq_tj = parse_query(Some("url=a.tif&minzoom=4&maxzoom=14&tile_format=webp"));
        assert!(pq_tj.errors.is_empty());
        assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_tj).is_ok());
        assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "4", "0", "0.png"], &pq_tj).is_err());
        let mut tj = serde_json::json!({"minzoom": 0, "maxzoom": 10, "center": [0.0, 0.0, 0], "tiles": []});
        apply_tilejson_overrides(&mut tj, &pq_tj, "http://host", "cog", "WebMercatorQuad", "url=a.tif&minzoom=4&maxzoom=14&tile_format=webp");
        assert_eq!(tj["minzoom"], 4);
        assert_eq!(tj["maxzoom"], 14);
        assert_eq!(tj["center"][2], 4);
        assert_eq!(tj["tiles"][0], "http://host/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.webp?url=a.tif");

        // HTTP status mapping for upstream 404, local NotFound, and invalid TIFF
        let r404 = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::HttpStatus {
            status: 404,
            url: "https://example.com/missing.tif".into(),
        }));
        assert_eq!(r404.status(), StatusCode::NOT_FOUND);
        let r_enoent = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Fs(
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        )));
        assert_eq!(r_enoent.status(), StatusCode::NOT_FOUND);
        let r_tiff = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Tiff(
            cogzilla_tiff::TiffError::Corrupt("not a tiff"),
        )));
        assert_eq!(r_tiff.status(), StatusCode::BAD_REQUEST);

        // AlgorithmFactory JSON responses (/algorithms and /algorithms/{id})
        let algs = list_algorithms_json("http://localhost:8080");
        let list = algs["algorithms"].as_array().unwrap();
        assert_eq!(list.len(), 19);
        assert_eq!(list[0]["id"], "hillshade");
        assert_eq!(get_algorithm_json("contours").unwrap()["title"], "Contours");
        assert_eq!(
            list[0]["links"][0]["href"],
            "http://localhost:8080/algorithms/hillshade"
        );
        let hs = get_algorithm_json("hillshade").unwrap();
        assert_eq!(hs["title"], "Hillshade");
        assert_eq!(hs["inputs"]["nbands"], 1);
        assert_eq!(hs["outputs"]["dtype"], "uint8");
        assert_eq!(get_algorithm_json("normalizedIndex").unwrap()["title"], "Normalized Difference Index");
        assert_eq!(get_algorithm_json("grayscale").unwrap()["title"], "Transform a RGB Image to Grayscale");
        assert_eq!(get_algorithm_json("bitonal").unwrap()["title"], "Transform an Image to Bitonal");
        assert!(matches!(
            get_algorithm_json("nonexistent"),
            Err(EngineError::NotFound(_))
        ));

        // WMTSCapabilities.xml generation & XML escaping
        let xml = build_wmts_capabilities(
            "http://localhost:8080",
            "cog",
            "WebMercatorQuad",
            "Test & <Layer>",
            [-122.7, 37.7, -122.5, 37.8],
            8,
            10,
            "webp",
            2,
            "url=https://ex.com/a.tif&rescale=0,100",
            "url=https://ex.com/a.tif&rescale=0,100&tile_format=webp&tile_scale=2",
            false,
        );
        assert!(xml.contains("xmlns=\"http://www.opengis.net/wmts/1.0\""));
        assert!(xml.contains("Test &amp; &lt;Layer&gt;"));
        assert!(xml.contains(
            "http://localhost:8080/cog/tiles/WebMercatorQuad/{TileMatrix}/{TileCol}/{TileRow}@2x.webp?url=https://ex.com/a.tif&amp;rescale=0,100"
        ));
        assert!(xml.contains("<TileWidth>512</TileWidth>"));
        assert!(xml.contains("<ows:Identifier>8</ows:Identifier>"));
        assert!(xml.contains("<ows:Identifier>10</ows:Identifier>"));

        // Vertical colormap PNG (20x256) + orientation/width/height query validation
        let vpng = get_colormap_image("viridis", OutputFormat::Png, true, None, None).unwrap();
        let sized = get_colormap_image("viridis", OutputFormat::Png, false, Some(128), Some(16)).unwrap();
        assert_eq!(&sized[16..20], &128u32.to_be_bytes());
        assert_eq!(&sized[20..24], &16u32.to_be_bytes());
        assert_eq!(&vpng[0..8], b"\x89PNG\r\n\x1a\n");
        let w = u32::from_be_bytes([vpng[16], vpng[17], vpng[18], vpng[19]]);
        let h = u32::from_be_bytes([vpng[20], vpng[21], vpng[22], vpng[23]]);
        assert_eq!((w, h), (20, 256));

        let pq_wh = parse_query(Some("url=a.tif&width=320&height=240"));
        assert!(validate_query(&["cog", "preview.png"], &pq_wh).is_ok());
        assert!(validate_query(&["cog", "preview", "256x256.png"], &pq_wh).is_err());
        assert_eq!(resolve_query_dims(512, 256, Some(200), None), (200, 100));
        assert_eq!(resolve_query_dims(512, 256, None, Some(100)), (200, 100));

        let pq_ori = parse_query(Some("f=png&orientation=vertical"));
        assert!(validate_query(&["colorMaps", "viridis"], &pq_ori).is_ok());
        assert!(validate_query(&["cog", "info"], &pq_ori).is_err());

        let pq_wmts = parse_query(Some("url=a.tif&minzoom=4&maxzoom=12&tile_format=png&tile_scale=2"));
        assert!(validate_query(&["cog", "WebMercatorQuad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
        let pq_wms = parse_query(Some("url=https://example.com/a.tif&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=CRS:84&BBOX=-123,37,-122,38&WIDTH=256&HEIGHT=256&FORMAT=image/png"));
        assert!(pq_wms.errors.is_empty(), "wms errors: {:?}", pq_wms.errors);
        assert!(validate_query(&["cog", "wms"], &pq_wms).is_ok());
        assert_eq!(parse_coord_crs_with_epsg("CRS:84").unwrap().1, 4326);
        assert!(parse_wms_format(Some("image/png"), None, 85).is_ok());
        assert!(parse_wms_format(Some("image/tiff"), None, 85).is_ok());
        assert!(parse_wms_format(Some("image/gif"), None, 85).is_err());
        let cap = build_wms_capabilities("https://example.com", "url=https://example.com/a.tif&SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0", [-123.0, 37.0, -122.0, 38.0], Some(32610));
        assert!(cap.contains("<WMS_Capabilities version=\"1.3.0\""));
        assert!(cap.contains("<CRS>EPSG:32610</CRS>"));
        assert!(cap.contains("<BoundingBox CRS=\"EPSG:4326\" minx=\"37\" miny=\"-123\" maxx=\"38\" maxy=\"-122\"/>"));
        assert!(cap.contains("xlink:href=\"https://example.com/cog/wms?url=https://example.com/a.tif&amp;\""), "OnlineResource should strip WMS protocol params and end with &: {cap}");
        assert!(!cap.contains("REQUEST=GetCapabilities"));
        assert!(parse_wms_format(Some("IMAGE/PNG"), None, 85).is_ok());
        assert!(validate_query(&["mosaicjson", "WorldCRS84Quad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
        assert!(validate_query(&["mosaicjson", "NotATms", "WMTSCapabilities.xml"], &pq_wmts).is_err());
    }

    #[test]
    fn geospatial_bounds_reproject_and_user_defined_crs_parity() {
        // 1. reproject_wgs84_bounds on global COG into UTM31 (EPSG:32631), NZTM (EPSG:2193),
        //    and LAEA (EPSG:3035) must include interior points (Rome/Berlin/NZ/Europe), not just edges.
        let global = [-180.0, -85.0, 180.0, 85.0];
        let utm31 = cogzilla_proj::CrsModel::from_epsg(32631).unwrap();
        let utm_b = reproject_wgs84_bounds(global, &utm31).unwrap();
        let (rome_x, rome_y) = utm31.from_lonlat_deg(12.5, 41.9);
        assert!(
            rome_x >= utm_b[0] && rome_x <= utm_b[2] && rome_y >= utm_b[1] && rome_y <= utm_b[3],
            "Rome ({rome_x}, {rome_y}) outside projected UTM31 bounds {utm_b:?}"
        );

        let nztm = cogzilla_proj::CrsModel::from_epsg(2193).unwrap();
        let nztm_b = reproject_wgs84_bounds(global, &nztm).unwrap();
        let (nz_x, nz_y) = nztm.from_lonlat_deg(170.5, -43.5);
        assert!(
            nz_x >= nztm_b[0] && nz_x <= nztm_b[2] && nz_y >= nztm_b[1] && nz_y <= nztm_b[3],
            "NZ ({nz_x}, {nz_y}) outside projected NZTM bounds {nztm_b:?}"
        );

        let laea = cogzilla_proj::CrsModel::from_epsg(3035).unwrap();
        let laea_b = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &laea).unwrap();
        let (ber_x, ber_y) = laea.from_lonlat_deg(13.4, 52.5);
        assert!(
            ber_x >= laea_b[0] && ber_x <= laea_b[2] && ber_y >= laea_b[1] && ber_y <= laea_b[3],
            "Berlin ({ber_x}, {ber_y}) outside projected LAEA bounds {laea_b:?}"
        );

        // 2. cog_standard_epsg returns None for user-defined projected CRS (32767) even with geographic_type=4326
        let mut gk = cogzilla_tiff::GeoKeys {
            model_type: Some(1),
            projected_cs_type: Some(32767),
            geographic_type: Some(4326),
            ..cogzilla_tiff::GeoKeys::default()
        };
        assert_eq!(cog_standard_epsg(&gk), None);

        // Standard projected EPSG returns Some(32610)
        gk.projected_cs_type = Some(32610);
        assert_eq!(cog_standard_epsg(&gk), Some(32610));

        // Pure geographic returns Some(4326)
        let gk_geo = cogzilla_tiff::GeoKeys {
            model_type: Some(2),
            geographic_type: Some(4326),
            ..cogzilla_tiff::GeoKeys::default()
        };
        assert_eq!(cog_standard_epsg(&gk_geo), Some(4326));

        // 3. GEOTIFF_CONTENT_TYPE matches parse_ext_format("tif", ...)
        let (_, tif_ct) = parse_ext_format("tif", None, 85).unwrap();
        assert_eq!(tif_ct, GEOTIFF_CONTENT_TYPE);

        // 4. PolygonMask::check_work_budget rejects O(W*H*V) DoS payloads
        let ring: Vec<serde_json::Value> = (0..10_000)
            .map(|i| {
                let a = (i as f64) * std::f64::consts::TAU / 10_000.0;
                serde_json::json!([a.cos(), a.sin()])
            })
            .collect();
        let poly_json = serde_json::json!({
            "type": "Polygon",
            "coordinates": [ring],
        });
        let (pm, _) = PolygonMask::from_geojson(&poly_json).unwrap();
        assert!(pm.check_work_budget(256 * 256).is_err());
        assert!(pm.check_work_budget(64 * 64).is_ok());

        // 5. ?assets=visual|bidx=2&assets=nir stores per-asset bidx without leaking into global bidx
        let pq_ab = parse_query(Some("assets=visual|bidx=2&assets=nir&asset_as_band=true"));
        assert_eq!(pq_ab.assets, vec!["visual", "nir"]);
        assert!(pq_ab.bidx.is_empty());
        assert_eq!(pq_ab.asset_bidx, vec![("visual".to_string(), vec![2])]);
        assert!(pq_ab.asset_as_band);

        // 6. CrsModel::linear_unit_metres returns US survey foot for EPSG:2227 and 1.0 for EPSG:32610
        let ca_ft = cogzilla_proj::CrsModel::from_epsg(2227).unwrap();
        assert!((ca_ft.linear_unit_metres() - 1200.0 / 3937.0).abs() < 1e-6);
        let utm10 = cogzilla_proj::CrsModel::from_epsg(32610).unwrap();
        assert!((utm10.linear_unit_metres() - 1.0).abs() < 1e-9);

        // 7. resolve_bbox_dst_crs defaults to coord_crs when dst_crs is omitted
        let pq_utm = parse_query(Some("url=a.tif&coord_crs=EPSG:32610"));
        assert_eq!(resolve_bbox_dst_crs(&pq_utm).map(|(_, e)| e), Some(32610));

        // 8. Form-encoded comparison & semicolon expressions decode `+` as space
        let pq_cmp = parse_query(Some("expression=b1+%3E+100"));
        assert_eq!(pq_cmp.expr_raw.as_deref(), Some("b1 > 100"));
        let pq_semi = parse_query(Some("expression=b1%3B+b2%3B+b3"));
        assert_eq!(pq_semi.expr_raw.as_deref(), Some("b1; b2; b3"));

        // 9. resolve_zoom_range honors ?maxzoom < def_min and WMTS tif MIME type
        assert_eq!(resolve_zoom_range(10, 15, None, Some(5)), (5, 5));
        let tif_xml = build_wmts_capabilities(
            "http://localhost:8080",
            "cog",
            "WebMercatorQuad",
            "Test",
            [-122.7, 37.7, -122.5, 37.8],
            5,
            5,
            "tif",
            1,
            "url=a.tif",
            "url=a.tif&tile_format=tif",
            false,
        );
        assert!(tif_xml.contains("<Format>image/tiff; application=geotiff</Format>"));

        // 10. /{family}/map.html must reject OGC Maps-only width/dst_crs params
        let pq_map_html = parse_query(Some("url=a.tif&width=256"));
        assert!(validate_query(&["cog", "map.html"], &pq_map_html).is_err());
        assert!(validate_query(&["cog", "map"], &pq_map_html).is_ok());

        // 11. Antimeridian-crossing WGS84 bbox (w > e, Fiji) projects across 180°, not 0°
        let utm01s = cogzilla_proj::CrsModel::from_epsg(32701).unwrap();
        let fiji_b = reproject_wgs84_bounds([177.0, -20.0, -178.0, -16.0], &utm01s).unwrap();
        let (fiji_x, fiji_y) = utm01s.from_lonlat_deg(179.5, -18.0);
        assert!(
            fiji_x >= fiji_b[0] && fiji_x <= fiji_b[2] && fiji_y >= fiji_b[1] && fiji_y <= fiji_b[3],
            "Fiji ({fiji_x}, {fiji_y}) outside antimeridian projected bounds {fiji_b:?}"
        );
        assert!(
            (fiji_b[2] - fiji_b[0]) < 1_000_000.0,
            "antimeridian bounds swept wrong hemisphere: {fiji_b:?}"
        );

        // 12. normalize_render_bbox unwraps antimeridian Geographic bboxes & rejects inverted boxes
        let fiji_geo = normalize_render_bbox(
            [177.0, -20.0, -178.0, -16.0],
            &cogzilla_proj::CrsModel::Geographic,
        )
        .unwrap();
        assert_eq!(fiji_geo, [177.0, -20.0, 182.0, -16.0]);
        assert_eq!(aspect_preserving_dims(fiji_geo, 512), (512, 410));
        assert!(normalize_render_bbox(
            [0.0, 10.0, 1.0, -10.0],
            &cogzilla_proj::CrsModel::Geographic,
        )
        .is_err());
        assert!(normalize_render_bbox(
            [10.0, 0.0, 5.0, 1.0],
            &cogzilla_proj::CrsModel::Epsg3857,
        )
        .is_err());

        // 13. bbox_to_wgs84 unwraps antimeridian-crossing projected bboxes instead of inverting
        let utm60n = cogzilla_proj::CrsModel::from_epsg(32660).unwrap();
        let am_wgs = bbox_to_wgs84([800_000.0, 0.0, 900_000.0, 100_000.0], Some(&utm60n)).unwrap();
        assert!(
            am_wgs[0] > 179.0 && am_wgs[2] > 180.0 && (am_wgs[2] - am_wgs[0]) < 2.0,
            "antimeridian UTM60N bbox inverted to complement: {am_wgs:?}"
        );

        // 14. reproject_wgs84_bounds into UTM on a global [-180,-90,180,90] COG covers >67.5°N
        let utm31_global = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &utm31).unwrap();
        let (_, sval_y) = utm31.from_lonlat_deg(3.0, 78.0);
        assert!(
            sval_y <= utm31_global[3],
            "global UTM31 bounds truncated below 78°N ({sval_y} > {})",
            utm31_global[3]
        );

        // 15. Buffered CDB1GlobalGrid @4x tile exceeding 4096 px must return 400 (not silent clamp)
        let pq_cdb_buf = parse_query(Some("url=a.tif&buffer=64"));
        assert!(build_tile_request_for_tms(
            None,
            TileCoord { z: 0, x: 0, y: 0 },
            Some(4),
            Some("png"),
            &pq_cdb_buf,
            1,
            cogzilla_tiff::SampleType::Uint8,
            "CDB1GlobalGrid",
            None,
            true,
        )
        .is_err());

        // 16. bbox_to_wgs84 preserves full [-180, 180] on global EPSG:3857 and polar EPSG:3413,
        // and reproject_wgs84_bounds rejects polar-cap bboxes wholly outside TM's ±89.35° domain
        let m = 20_037_508.342_789_244;
        let glob_3857 =
            bbox_to_wgs84([-m, -m, m, m], Some(&cogzilla_proj::CrsModel::Epsg3857)).unwrap();
        assert!(
            (glob_3857[0] + 180.0).abs() < 1e-6 && (glob_3857[2] - 180.0).abs() < 1e-6,
            "global EPSG:3857 bbox must span [-180, 180], got {glob_3857:?}"
        );
        let ps_north = cogzilla_proj::CrsModel::from_epsg(3413).unwrap();
        let polar_wgs =
            bbox_to_wgs84([-2_000_000.0, -2_000_000.0, 2_000_000.0, 2_000_000.0], Some(&ps_north))
                .unwrap();
        assert_eq!(polar_wgs[0], -180.0);
        assert_eq!(polar_wgs[2], 180.0);
        assert_eq!(polar_wgs[3], 90.0);
        assert!(reproject_wgs84_bounds([0.0, 89.5, 10.0, 90.0], &utm31).is_err());

        // 17. Narrow polar strip in Cylindrical Equal Area (EPSG:6933) must NOT expand to [-180, 180]
        let cea = cogzilla_proj::CrsModel::from_epsg(6933).unwrap();
        let cea_strip = bbox_to_wgs84([-100_000.0, 7_000_000.0, 100_000.0, 7_400_000.0], Some(&cea))
            .unwrap();
        assert!(
            cea_strip[2] - cea_strip[0] < 10.0,
            "narrow EPSG:6933 strip must not expand to [-180, 180], got {cea_strip:?}"
        );

        // 18. wgs84_bboxes_intersect handles regional 0..360 and -360..0 bboxes without false positives
        assert!(!wgs84_bboxes_intersect(
            [235.0, 32.0, 245.0, 42.0],
            [-170.0, 35.0, -160.0, 40.0]
        ));
        assert!(wgs84_bboxes_intersect(
            [235.0, 32.0, 245.0, 42.0],
            [-122.0, 35.0, -118.0, 40.0]
        ));
        assert!(!wgs84_bboxes_intersect(
            [-245.0, 32.0, -235.0, 42.0],
            [160.0, 35.0, 170.0, 40.0]
        ));
        assert!(wgs84_bboxes_intersect(
            [-245.0, 32.0, -235.0, 42.0],
            [118.0, 35.0, 122.0, 40.0]
        ));

        // 19. OGC tileset & STAC render links must never double X-Forwarded-Prefix
        let tms = crate::tms::lookup_tms("WebMercatorQuad").unwrap();
        let ts_json = build_ogc_tileset_json(
            "https://api.example.com/prod",
            "/prod",
            "cog",
            tms,
            "url=a.tif",
            Some([-122.5, 37.5, -122.0, 38.0]),
            Some(0),
            Some(12),
        );
        let ts_str = ts_json.to_string();
        assert!(
            !ts_str.contains("/prod/prod"),
            "OGC tileset links doubled X-Forwarded-Prefix: {ts_str}"
        );
        assert!(ts_str.contains("https://api.example.com/prod/cog/"));
    }

    #[tokio::test]
    async fn sweep22_body_limits_timeout_and_feature_collection_cap() {
        // BB1: default FeatureCollection limit allows thousands of features (not capped at 64)
        assert!(check_feature_collection_len(200).is_ok());
        assert!(check_feature_collection_len(1_000).is_ok());
        assert!(check_feature_collection_len(crate::config::DEFAULT_MAX_STATISTICS_FEATURES).is_ok());
        assert!(check_feature_collection_len(crate::config::DEFAULT_MAX_STATISTICS_FEATURES + 1).is_err());

        // BB2: oversized POST body returns 413 Content Too Large with exact message
        let big = Bytes::from(vec![b'a'; MAX_POST_BODY_BYTES + 1]);
        let err = read_limited_body_with_timeout(http_body_util::Full::new(big), 5)
            .await
            .unwrap_err();
        let resp = error_to_http_response(&err);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let msg = err.to_string();
        assert!(
            msg.contains("request body exceeds 16 MiB limit"),
            "unexpected 413 message: {msg}"
        );

        // BB3: stalled POST body returns 408 Request Timeout
        struct NeverBody;
        impl hyper::body::Body for NeverBody {
            type Data = Bytes;
            type Error = std::io::Error;
            fn poll_frame(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>>
            {
                std::task::Poll::Pending
            }
        }
        let err_to = read_limited_body_with_timeout(NeverBody, 1)
            .await
            .unwrap_err();
        let resp_to = error_to_http_response(&err_to);
        assert_eq!(resp_to.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn ogc_tileset_filters_template_query_and_skips_negative_cdb_matrices_and_stac_renders_encodes() {
        // 1. OGC tileset strips minzoom/maxzoom/crs from item href and skips CDB1GlobalGrid negative matrices.
        let cdb = crate::tms::lookup_tms("CDB1GlobalGrid").unwrap();
        let ts = build_ogc_tileset_json(
            "example.com",
            "",
            "cog",
            cdb,
            "url=a.tif&minzoom=0&maxzoom=2&crs=EPSG:4326&bidx=1",
            Some([-10.0, -10.0, 10.0, 10.0]),
            Some(0),
            Some(2),
        );
        let limits = ts["tileMatrixSetLimits"].as_array().unwrap();
        assert!(
            limits.iter().all(|l| !l["tileMatrix"].as_str().unwrap().starts_with('-')),
            "CDB1GlobalGrid must not advertise negative matrix IDs: {limits:?}"
        );
        let item_href = ts["links"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["rel"] == "item")
            .unwrap()["href"]
            .as_str()
            .unwrap();
        assert!(item_href.contains("url=a.tif") && item_href.contains("bidx=1"));
        assert!(!item_href.contains("minzoom=") && !item_href.contains("maxzoom=") && !item_href.contains("crs="));

        // 2. STAC render links percent-encode presigned URLs, render IDs, and inline object colormaps.
        let render = serde_json::json!({
            "assets": ["red"],
            "colormap": {"1": [255, 0, 0, 255]}
        });
        let entry = build_stac_render_entry(
            "example.com",
            "",
            "https://s3.example/item.json?a=1&b=2",
            "true color",
            &render,
        );
        let links = entry["links"].as_array().unwrap();
        let self_href = links[0]["href"].as_str().unwrap();
        let tj_href = links[1]["href"].as_str().unwrap();
        assert!(self_href.contains("/stac/renders/true%20color?url=https%3A%2F%2Fs3.example%2Fitem.json%3Fa%3D1%26b%3D2"), "{self_href}");
        assert!(tj_href.contains("colormap=%7B%221%22%3A%5B255%2C0%2C0%2C255%5D%7D"), "{tj_href}");
    }

    #[test]
    fn audit87_header_accepts_gzip_rfc9110_precedence_and_nan() {
        assert_eq!(negotiate_encoding(["gzip;q=0, *"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["*, gzip;q=0"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["x-gzip;q=0, *;q=1"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["gzip;q=NaN"]), PreferredEncoding::None);
        assert_eq!(negotiate_encoding(["gzip;q=inf"]), PreferredEncoding::None);
        assert_eq!(negotiate_encoding(["gzip;q=-0.5"]), PreferredEncoding::None);
        assert_eq!(negotiate_encoding(["br, deflate"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["br, *;q=0.5"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["x-gzip;q=0.1"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["gzip, deflate"]), PreferredEncoding::Gzip);
    }

    #[test]
    fn audit88_attach_x_failed_assets_redacts_credentials_and_presigned_queries() {
        let mut resp = Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::new()))
            .unwrap();
        attach_x_failed_assets(
            &mut resp,
            &[
                "https://user:secret@s3.example.com/bucket/a.tif?X-Amz-Signature=deadbeef".into(),
                "/local/ok.tif".into(),
            ],
        );
        let hv = resp.headers().get("x-failed-assets").unwrap().to_str().unwrap();
        assert!(!hv.contains("secret") && !hv.contains("deadbeef"), "leaked creds in x-failed-assets: {hv}");
        assert!(hv.contains("https://s3.example.com/bucket/a.tif") && hv.contains("/local/ok.tif"), "{hv}");
    }

    #[test]
    fn audit91_native_bbox_dims_aspect_antimeridian_and_0_360() {
        let make_ifd = |w: u32, h: u32, a: f64, e: f64, c: f64, f: f64| {
            let p2c = cogzilla_tiff::Affine2D { a, b: 0.0, c, d: 0.0, e, f };
            cogzilla_tiff::IfdMeta {
                width: w,
                height: h,
                tile_width: 256,
                tile_height: 256,
                tiles_across: w.div_ceil(256),
                tiles_down: h.div_ceil(256),
                sample_type: cogzilla_tiff::SampleType::Uint8,
                samples_per_pixel: 1,
                planar_config: cogzilla_tiff::PlanarConfig::Chunky,
                compression: cogzilla_tiff::Compression::None,
                predictor: cogzilla_tiff::Predictor::None,
                photometric: 1,
                extra_samples: Vec::new(),
                jpeg_tables: None,
                mask_ifd: None,
                is_stripped: false,
                pixel_to_crs: p2c,
                crs_to_pixel: p2c.invert().unwrap(),
                gcp_poly: None,
                tile_offsets: Box::default(),
                tile_byte_counts: Box::default(),
            }
        };
        let geo = cogzilla_proj::CrsModel::Geographic;
        let merc = cogzilla_proj::CrsModel::Epsg3857;

        // 1. Large 2:1 global COG preserves 2:1 aspect ratio at the 4096 cap (4096x2048, not 4096x4096).
        let ifd_2to1 = make_ifd(21600, 10800, 360.0 / 21600.0, -180.0 / 10800.0, -180.0, 90.0);
        let dims = native_bbox_dims_in_crs(&ifd_2to1, &geo, &geo, [-180.0, -90.0, 180.0, 90.0], [-180.0, -90.0, 180.0, 90.0]);
        assert_eq!(dims, (4096, 2048));

        // 2. Global 0..360° Geographic COG (1440x720) keeps 1440x720 in Geographic and does not collapse ddx in 3857.
        let ifd_360 = make_ifd(1440, 720, 0.25, -0.25, 0.0, 90.0);
        assert_eq!(
            native_bbox_dims_in_crs(&ifd_360, &geo, &geo, [-180.0, -90.0, 180.0, 90.0], [-180.0, -90.0, 180.0, 90.0]),
            (1440, 720)
        );
        let merc_bb = [-20_037_508.342_789_244, -20_037_508.342_789_244, 20_037_508.342_789_244, 20_037_508.342_789_244];
        let (mw, mh) = native_bbox_dims_in_crs(&ifd_360, &geo, &merc, [-180.0, -85.05, 180.0, 85.05], merc_bb);
        assert!((1100..=1500).contains(&mw) && (1100..=1500).contains(&mh), "got ({mw}, {mh})");

        // 3. Global 1024x1024 EPSG:3857 -> EPSG:4326 does NOT falsely wrap ddx (stays ~1309x619, not 3065x1448).
        let span = 2.0 * 20_037_508.342_789_244;
        let ifd_merc = make_ifd(1024, 1024, span / 1024.0, -span / 1024.0, -20_037_508.342_789_244, 20_037_508.342_789_244);
        let (gw, gh) = native_bbox_dims_in_crs(&ifd_merc, &merc, &geo, [-180.0, -85.05, 180.0, 85.05], [-180.0, -85.05, 180.0, 85.05]);
        assert!((1200..=1400).contains(&gw) && (550..=700).contains(&gh), "got ({gw}, {gh})");

        // 4. Antimeridian-crossing UTM 60N (179°E..179°W) -> EPSG:4326 wraps ddx properly (~512x512, not 4x4).
        let utm60 = cogzilla_proj::CrsModel::from_epsg(32660).unwrap();
        let (x0, y0) = utm60.from_lonlat_deg(179.0, 1.0);
        let (x1, y1) = utm60.from_lonlat_deg(-179.0, -1.0);
        let ifd_am = make_ifd(512, 512, (x1 - x0) / 512.0, (y1 - y0) / 512.0, x0, y0);
        let (aw, ah) = native_bbox_dims_in_crs(&ifd_am, &utm60, &geo, [179.0, -1.0, 181.0, 1.0], [179.0, -1.0, 181.0, 1.0]);
        assert!((450..=580).contains(&aw) && (450..=580).contains(&ah), "got ({aw}, {ah})");
    }

    #[test]
    fn attach_x_assets_redacts_secrets_and_preserves_plain_names() {
        let mut resp = Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::new()))
            .unwrap();
        attach_x_assets(
            &mut resp,
            &[
                "https://user:secret@example.com/a.tif?X-Amz-Signature=deadbeef&sig=123".to_string(),
                "red".to_string(),
                "B04".to_string(),
            ],
        );
        let hv = resp.headers().get("x-assets").unwrap().to_str().unwrap();
        assert_eq!(hv, "https://example.com/a.tif,red,B04");
        assert!(!hv.contains("secret") && !hv.contains("deadbeef"));
    }

    #[tokio::test]
    async fn forwarded_prefix_rejects_protocol_relative_and_invalid_host() {
        let engine = Arc::new(TileEngine::new(4 * 1024 * 1024));
        let req = Request::builder()
            .method(Method::GET)
            .uri("/docs")
            .header("x-forwarded-prefix", "//evil.com")
            .header("x-forwarded-host", "attacker@evil.com/bad")
            .header("host", "127.0.0.1:8080")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = http_body_util::BodyExt::collect(resp.into_body())
            .await
            .unwrap()
            .to_bytes();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(!html.contains("//evil.com"), "protocol-relative prefix leaked into HTML: {html}");
        assert!(!html.contains("attacker@"), "invalid forwarded host leaked into HTML: {html}");
    }

    #[tokio::test]
    async fn sweep45_qq1_colormaps_html_npy_and_rr1_accept_negotiation_vary() {
        let engine = Arc::new(TileEngine::new(4 * 1024 * 1024));
        let browser_accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";

        // QQ1: /colorMaps/viridis?f=html -> 200 text/html + Vary: Accept
        let req = Request::builder()
            .method(Method::GET)
            .uri("/colorMaps/viridis?f=html")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("text/html"));
        let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
        assert!(vary.contains("accept"), "missing Vary: Accept: {vary}");

        // QQ1: /colorMaps/viridis?f=npy -> 200 (4, 20, 256) uint8 NumPy array
        let req = Request::builder()
            .method(Method::GET)
            .uri("/colorMaps/viridis?f=npy")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes();
        assert_eq!(&body[0..6], b"\x93NUMPY");
        let hlen = u16::from_le_bytes([body[8], body[9]]) as usize;
        let hdr = std::str::from_utf8(&body[10..10 + hlen]).unwrap();
        assert!(hdr.contains("'|u1'") && hdr.contains("(4, 20, 256)"), "unexpected npy header: {hdr}");
        assert_eq!(body.len(), 10 + hlen + 4 * 20 * 256);

        // RR1: browser Accept -> text/html + Vary: Accept on all offers_html routes
        for uri in [
            "/tileMatrixSets",
            "/tileMatrixSets/WebMercatorQuad",
            "/conformance",
            "/algorithms",
            "/colorMaps/viridis",
        ] {
            let req = Request::builder()
                .method(Method::GET)
                .uri(uri)
                .header(hyper::header::ACCEPT, browser_accept)
                .body(Full::new(Bytes::new()))
                .unwrap();
            let resp = handle_request(engine.clone(), req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
            assert!(
                resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("text/html"),
                "expected text/html for {uri}"
            );
            let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
            assert!(vary.contains("accept"), "missing Vary: Accept for {uri}: {vary}");
        }

        // RR1: explicit ?f=json wins over browser Accept, and still carries Vary: Accept
        let req = Request::builder()
            .method(Method::GET)
            .uri("/tileMatrixSets?f=json")
            .header(hyper::header::ACCEPT, browser_accept)
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("application/json"));
        let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
        assert!(vary.contains("accept"), "missing Vary: Accept on ?f=json: {vary}");

        // RR1: API client (no Accept or Accept: */*) gets application/json + Vary: Accept
        let req = Request::builder()
            .method(Method::GET)
            .uri("/tileMatrixSets")
            .header(hyper::header::ACCEPT, "*/*")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("application/json"));
        let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
        assert!(vary.contains("accept"), "missing Vary: Accept on */*: {vary}");

        // Strict boolean query parameter validation (rejects bogus booleans with 422).
        for q in [
            "return_mask=bogus",
            "unscale=maybe",
            "strict=nope",
            "categorical=2",
            "area_weighted=invalid",
            "use_epsg=nah",
            "asset_as_band=wat",
        ] {
            let pq = parse_query(Some(q));
            assert!(!pq.errors.is_empty(), "expected validation error for {q}");
        }
        assert!(parse_query(Some("return_mask=false&unscale=1&categorical=yes")).errors.is_empty());
    }

    #[tokio::test]
    async fn readyz_and_graceful_shutdown_drain() {
        let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
        let req = Request::builder()
            .method(Method::GET)
            .uri("/readyz")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
            Some("no-store")
        );

        // HEAD /readyz -> 200 with no-store and empty body; POST /readyz -> 405.
        let head_req = Request::builder()
            .method(Method::HEAD)
            .uri("/readyz")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let head_resp = handle_request(engine.clone(), head_req).await.unwrap();
        assert_eq!(head_resp.status(), StatusCode::OK);
        assert_eq!(
            head_resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
            Some("no-store")
        );
        let post_req = Request::builder()
            .method(Method::POST)
            .uri("/readyz")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let post_resp = handle_request(engine.clone(), post_req).await.unwrap();
        assert_eq!(post_resp.status(), StatusCode::METHOD_NOT_ALLOWED);

        // Unreachable pgSTAC -> 503 with Cache-Control: no-store.
        let dead_port = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
        let pg_cfg = cogzilla_pgstac::PgStacConfig::from_connection_string(&format!(
            "postgresql://u:p@127.0.0.1:{dead_port}/db?connect_timeout=1"
        ))
        .unwrap();
        let pg = Arc::new(cogzilla_pgstac::PgStac::new(pg_cfg).unwrap());
        let pg_engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
        pg_engine.set_pgstac(pg);
        let unready_req = Request::builder()
            .method(Method::GET)
            .uri("/readyz")
            .body(Full::new(Bytes::new()))
            .unwrap();
        let unready_resp = handle_request(pg_engine, unready_req).await.unwrap();
        assert_eq!(unready_resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            unready_resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
            Some("no-store")
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let srv = tokio::spawn(async move {
            serve_listener_with_shutdown(engine, listener, async move {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        let client = reqwest::Client::new();
        let r = client.get(format!("http://{addr}/readyz")).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
        assert_eq!(r.text().await.unwrap(), "ok");

        let _ = tx.send(());
        tokio::time::timeout(std::time::Duration::from_secs(5), srv)
            .await
            .expect("server should drain and exit promptly")
            .unwrap();
    }

    #[test]
    fn negotiate_encoding_prefers_brotli_and_honours_rfc9110_q_values() {
        assert_eq!(negotiate_encoding(["gzip, deflate, br"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["br"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["gzip, deflate"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["br;q=0, gzip"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["gzip;q=1.0, br;q=0.5"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["gzip;q=0.5, br;q=0.8"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["gzip;q=0.5, *"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["br;q=0.5, *"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["*"]), PreferredEncoding::Gzip);
        assert_eq!(negotiate_encoding(["gzip;q=0, *"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["gzip;q=0, br;q=0, *"]), PreferredEncoding::None);
        assert_eq!(negotiate_encoding(["identity;q=1.0, br;q=0.5"]), PreferredEncoding::None);
        assert_eq!(negotiate_encoding(["br;q=NaN"]), PreferredEncoding::None);
        // Multi-header HTTP/1.1 & HTTP/2 Accept-Encoding lines
        assert_eq!(negotiate_encoding(["gzip", "br"]), PreferredEncoding::Brotli);
        assert_eq!(negotiate_encoding(["*", "gzip;q=0", "br;q=0"]), PreferredEncoding::None);
    }

    #[tokio::test]
    async fn x_request_id_echoes_valid_generates_when_absent_or_invalid_and_on_429() {
        let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));

        // Absent -> generated 21-byte <8-hex>-<12-hex>, unique across calls.
        let r1 = handle_request(
            engine.clone(),
            Request::builder().uri("/healthz").body(Full::new(Bytes::new())).unwrap(),
        )
        .await
        .unwrap();
        let r2 = handle_request(
            engine.clone(),
            Request::builder().uri("/healthz").body(Full::new(Bytes::new())).unwrap(),
        )
        .await
        .unwrap();
        let id1 = r1.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
        let id2 = r2.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
        assert_eq!(id1.len(), 21);
        assert_eq!(id1.as_bytes()[8], b'-');
        assert!(id1[..8].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert!(id1[9..].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert_ne!(id1, id2);
        assert_eq!(
            r1.headers()
                .get(hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS)
                .and_then(|v| v.to_str().ok()),
            Some("X-Request-Id")
        );

        // Valid caller-supplied X-Request-Id (including exact 128-byte boundary) is echoed verbatim.
        let exact_128 = "a".repeat(128);
        for valid in ["trace-abc-123_XYZ", exact_128.as_str()] {
            let r_echo = handle_request(
                engine.clone(),
                Request::builder()
                    .uri("/healthz")
                    .header("x-request-id", valid)
                    .body(Full::new(Bytes::new()))
                    .unwrap(),
            )
            .await
            .unwrap();
            assert_eq!(
                r_echo.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap(),
                valid
            );
        }

        // Invalid (empty, contains whitespace/tab, or >128 bytes) -> replaced with fresh generated ID.
        for invalid in ["", "bad id with spaces", "bad\tid"] {
            let r_bad = handle_request(
                engine.clone(),
                Request::builder()
                    .uri("/healthz")
                    .header("x-request-id", invalid)
                    .body(Full::new(Bytes::new()))
                    .unwrap(),
            )
            .await
            .unwrap();
            let bad_out = r_bad.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
            assert_ne!(bad_out, invalid);
            assert_eq!(bad_out.len(), 21);
        }

        let too_long = "a".repeat(129);
        let r_long = handle_request(
            engine.clone(),
            Request::builder()
                .uri("/healthz")
                .header("x-request-id", &too_long)
                .body(Full::new(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(r_long.headers().get(&X_REQUEST_ID).unwrap().len(), 21);

        // AIMD 429 shedding still carries X-Request-Id.
        let tight = Arc::new(
            TileEngine::new(8 * 1024 * 1024)
                .with_aimd(crate::aimd::AimdLimiter::new(true, 1, 1, 1, 150)),
        );
        let _held = tight.aimd.try_acquire().unwrap();
        let r_429 = handle_request(
            tight.clone(),
            Request::builder()
                .uri("/cog/info")
                .header("x-request-id", "shed-req-42")
                .body(Full::new(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(r_429.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            r_429.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap(),
            "shed-req-42"
        );
    }
}
