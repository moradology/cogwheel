use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use cogzilla_io::OpenedCog;
use cogzilla_kernel::ColormapName;
use cogzilla_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row};
use dashmap::DashMap;

use crate::engine::{EngineError, TileEngine};

/// Parsed & cached STAC Item holding resolved raster asset hrefs.
#[derive(Debug, Clone)]
pub struct StacItem {
    pub id: String,
    pub bbox: Option<[f64; 4]>,
    pub geometry: Option<serde_json::Value>,
    /// Ordered map of raster asset key -> resolved absolute path or HTTP/S3 URL.
    pub assets: BTreeMap<String, String>,
    /// Optional STAC Render Extension (`renders`) definitions.
    pub renders: BTreeMap<String, serde_json::Value>,
}

fn is_raster_asset(key: &str, asset: &serde_json::Value) -> bool {
    let href = asset.get("href").and_then(|v| v.as_str()).unwrap_or("");
    if href.is_empty() {
        return false;
    }
    if let Some(ty) = asset.get("type").and_then(|v| v.as_str()) {
        let lower = ty.to_ascii_lowercase();
        if lower.contains("tiff") || lower.contains("geotiff") {
            return true;
        }
        if lower.contains("json")
            || lower.contains("xml")
            || lower.contains("png")
            || lower.contains("jpeg")
            || lower.contains("text")
        {
            return false;
        }
    }
    let lower_href = href.to_ascii_lowercase();
    lower_href.ends_with(".tif")
        || lower_href.ends_with(".tiff")
        || lower_href.ends_with(".cog")
        || (!key.contains("thumbnail") && !key.contains("metadata") && lower_href.contains(".tif"))
}

/// Split a remote URI (including GDAL `/vsicurl/`, `/vsis3/`, `/vsigs/`, `/vsiaz/`)
/// into `(origin_prefix, path_part)` with query/fragment stripped from `item_uri`.
/// Returns `None` for local filesystem paths.
fn split_remote_origin_and_path(item_uri: &str) -> Option<(&str, &str)> {
    let path_only = item_uri.split(['?', '#']).next().unwrap_or(item_uri);
    if let Some(rest) = path_only.strip_prefix("/vsicurl/") {
        let scheme_end = rest.find("://")?;
        let after_scheme = "/vsicurl/".len() + scheme_end + 3;
        let origin_end = path_only[after_scheme..]
            .find('/')
            .map(|p| after_scheme + p)
            .unwrap_or(path_only.len());
        return Some((&path_only[..origin_end], &path_only[origin_end..]));
    }
    for vsi in ["/vsis3/", "/vsigs/", "/vsiaz/"] {
        if let Some(rest) = path_only.strip_prefix(vsi) {
            let bucket_end = vsi.len() + rest.find('/').unwrap_or(rest.len());
            return Some((&path_only[..bucket_end], &path_only[bucket_end..]));
        }
    }
    if path_only.starts_with("http://")
        || path_only.starts_with("https://")
        || path_only.starts_with("s3://")
        || path_only.starts_with("gs://")
        || path_only.starts_with("az://")
        || path_only.starts_with("abfs://")
        || path_only.starts_with("abfss://")
    {
        let scheme_end = path_only.find("://")?;
        let after_scheme = scheme_end + 3;
        let origin_end = path_only[after_scheme..]
            .find('/')
            .map(|p| after_scheme + p)
            .unwrap_or(path_only.len());
        return Some((&path_only[..origin_end], &path_only[origin_end..]));
    }
    None
}

/// Normalize `.` and `..` segments in a remote URI path, clamped at the origin/bucket root,
/// while preserving any `?query` or `#fragment` suffix on `href`.
fn normalize_remote_path(base_dir: &str, href: &str) -> String {
    let (href_path, suffix) = match href.find(['?', '#']) {
        Some(idx) => (&href[..idx], &href[idx..]),
        None => (href, ""),
    };
    let mut segs: Vec<&str> = Vec::new();
    let iter = if href_path.starts_with('/') {
        itertools_none_or_chain("", href_path)
    } else {
        itertools_none_or_chain(base_dir, href_path)
    };
    for seg in iter {
        match seg {
            "" | "." => {}
            ".." => {
                segs.pop();
            }
            s => segs.push(s),
        }
    }
    if segs.is_empty() {
        format!("/{suffix}")
    } else {
        format!("/{}{suffix}", segs.join("/"))
    }
}

fn itertools_none_or_chain<'a>(a: &'a str, b: &'a str) -> impl Iterator<Item = &'a str> {
    a.split('/').chain(b.split('/'))
}

fn resolve_href(item_uri: &str, href: &str) -> String {
    if href.starts_with("http://")
        || href.starts_with("https://")
        || href.starts_with("s3://")
        || href.starts_with("gs://")
        || href.starts_with("az://")
        || href.starts_with("abfs://")
        || href.starts_with("abfss://")
        || href.starts_with("file://")
        || href.starts_with("/vsicurl/")
        || href.starts_with("/vsis3/")
        || href.starts_with("/vsigs/")
        || href.starts_with("/vsiaz/")
    {
        return href.to_string();
    }
    if let Some((origin, item_path)) = split_remote_origin_and_path(item_uri) {
        let base_dir = match item_path.rfind('/') {
            Some(idx) => &item_path[..=idx],
            None => "/",
        };
        let norm = normalize_remote_path(base_dir, href);
        format!("{origin}{norm}")
    } else if href.starts_with('/') {
        href.to_string()
    } else {
        let local = match cogzilla_io::strip_file_uri_prefix(item_uri) {
            Ok(Some(path)) => path,
            _ => item_uri,
        };
        let parent = Path::new(local).parent().unwrap_or_else(|| Path::new("."));
        parent
            .join(href)
            .to_string_lossy()
            .into_owned()
    }
}

/// Maximum STAC Item JSON size (16 MiB) to prevent OOM on hostile/oversized metadata URLs.
pub const MAX_STAC_JSON_BYTES: usize = 16 * 1024 * 1024;

/// Load (or fetch from lock-free cache) a STAC Item JSON from local disk, HTTP(S), S3, GCS, or Azure.
pub async fn load_stac_item(
    cache: &DashMap<String, Arc<StacItem>, ahash::RandomState>,
    store: &cogzilla_io::CogStore,
    uri: &str,
) -> Result<Arc<StacItem>, EngineError> {
    cogzilla_io::check_uri_allowed(uri)?;
    if let Some(hit) = cache.get(uri) {
        return Ok(hit.clone());
    }

    let raw_bytes = store.fetch_uri_bytes(uri, MAX_STAC_JSON_BYTES).await?;

    let val: serde_json::Value = serde_json::from_slice(&raw_bytes)
        .map_err(|e| EngineError::BadRequest(format!("invalid STAC JSON: {e}")))?;

    let id = val
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("stac-item")
        .to_string();
    let bbox = val.get("bbox").and_then(|v| v.as_array()).and_then(|a| {
        let coords = match a.len() {
            4 => [a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?, a[3].as_f64()?],
            6 => [a[0].as_f64()?, a[1].as_f64()?, a[3].as_f64()?, a[4].as_f64()?],
            _ => return None,
        };
        coords.iter().all(|x| x.is_finite()).then_some(coords)
    });
    let geometry = val.get("geometry").cloned();

    let mut assets = BTreeMap::new();
    if let Some(obj) = val.get("assets").and_then(|v| v.as_object()) {
        for (k, a) in obj {
            if is_raster_asset(k, a) {
                if let Some(href) = a.get("href").and_then(|v| v.as_str()) {
                    assets.insert(k.clone(), resolve_href(uri, href));
                }
            }
        }
    }

    let mut renders = BTreeMap::new();
    let renders_obj = val
        .get("renders")
        .or_else(|| val.get("properties").and_then(|p| p.get("renders")))
        .and_then(|v| v.as_object());
    if let Some(robj) = renders_obj {
        for (k, v) in robj {
            renders.insert(k.clone(), v.clone());
        }
    }

    let item = Arc::new(StacItem {
        id,
        bbox,
        geometry,
        assets,
        renders,
    });
    if cache.len() >= 1024 {
        cache.clear();
    }
    cache.insert(uri.to_string(), item.clone());
    Ok(item)
}

/// Resolve requested STAC asset names (expanding `:all:`, extracting asset names referenced in
/// `?expression=`, or defaulting to up to 3 raster assets) without opening COGs.
pub fn select_stac_asset_names(
    item: &StacItem,
    requested_assets: &[String],
    expr_raw: Option<&str>,
) -> Result<Vec<String>, EngineError> {
    let mut names: Vec<String> = if requested_assets.iter().any(|a| a == ":all:") {
        item.assets.keys().take(16).cloned().collect()
    } else {
        requested_assets.to_vec()
    };

    // If no explicit `?assets=` given, extract exact identifier tokens referenced in `?expression=`
    // using longest-prefix matching against known asset names (supporting both unspaced
    // subtraction like `nir-red` and hyphenated asset names like `SR-B5-SR-B4`), otherwise
    // default to all raster assets (up to 3).
    if names.is_empty() {
        if let Some(expr) = expr_raw {
            let mut known: Vec<&str> = item.assets.keys().map(|s| s.as_str()).collect();
            known.sort_by(|a, b| b.len().cmp(&a.len()));
            let mut seen = std::collections::HashSet::new();
            let bytes = expr.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                let prev_is_ident =
                    i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                if !prev_is_ident && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
                    let rest = &expr[i..];
                    if let Some(&matched) = known.iter().find(|&&name| {
                        if !rest.starts_with(name) {
                            return false;
                        }
                        let after = i + name.len();
                        if !bytes
                            .get(after)
                            .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
                        {
                            return true;
                        }
                        // Also match TiTiler's default `{asset}_b{k}` band syntax (e.g. `nir_b1`).
                        if bytes.get(after) == Some(&b'_') && bytes.get(after + 1) == Some(&b'b') {
                            let mut j = after + 2;
                            while j < bytes.len() && bytes[j].is_ascii_digit() {
                                j += 1;
                            }
                            if j > after + 2
                                && !bytes
                                    .get(j)
                                    .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
                            {
                                return true;
                            }
                        }
                        false
                    }) {
                        if seen.insert(matched) {
                            names.push(matched.to_string());
                        }
                        i += matched.len();
                        if bytes.get(i) == Some(&b'_') && bytes.get(i + 1) == Some(&b'b') {
                            let mut j = i + 2;
                            while j < bytes.len() && bytes[j].is_ascii_digit() {
                                j += 1;
                            }
                            if j > i + 2
                                && !bytes
                                    .get(j)
                                    .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
                            {
                                i = j;
                            }
                        }
                        continue;
                    }
                }
                i += 1;
            }
        }
        if names.is_empty() {
            return Err(EngineError::BadRequest(
                "missing ?assets= (or asset names in ?expression=)".into(),
            ));
        }
    }
    if names.is_empty() {
        return Err(EngineError::BadRequest(
            "STAC item has no raster COG assets".into(),
        ));
    }
    if names.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "too many STAC assets ({}; max 16)",
            names.len()
        )));
    }
    Ok(names)
}

/// Resolve requested STAC asset names (or extract them from a STAC `?expression=`) and open all
/// referenced COGs concurrently via `CogStore`.
/// Also rewrites asset-name identifiers in `expr_raw` (e.g. `(red-green)/(red+green)` -> `(b1-b2)/(b1+b2)`).
pub async fn resolve_stac_assets(
    engine: &TileEngine,
    item: &StacItem,
    requested_assets: &[String],
    expr_raw: Option<&str>,
) -> Result<(Vec<String>, Vec<Arc<OpenedCog>>, Option<String>), EngineError> {
    let names = select_stac_asset_names(item, requested_assets, expr_raw)?;

    let mut hrefs = Vec::with_capacity(names.len());
    for name in &names {
        let href = item.assets.get(name).ok_or_else(|| {
            EngineError::NotFound(format!(
                "STAC asset '{name}' not found (available: {:?})",
                item.assets.keys().collect::<Vec<_>>()
            ))
        })?;
        hrefs.push(href.clone());
    }

    // Open all STAC asset COGs concurrently!
    let mut js = tokio::task::JoinSet::new();
    for (idx, href) in hrefs.into_iter().enumerate() {
        let store = engine.store.clone();
        js.spawn(async move {
            let cog = store
                .open_uri(&href)
                .await
                .map_err(|e| crate::engine::asset_fetch_error(&href, e))?;
            Ok::<_, EngineError>((idx, cog))
        });
    }
    let mut opened: Vec<Option<Arc<OpenedCog>>> = vec![None; names.len()];
    while let Some(res) = js.join_next().await {
        let (idx, cog) = res
            .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
        opened[idx] = Some(cog);
    }
    let cogs: Vec<Arc<OpenedCog>> = opened.into_iter().map(|o| o.unwrap()).collect();

    // Single-pass identifier-boundary rewrite (e.g. `(B4-B3)/(B4+B3)` -> `(b1-b2)/(b1+b2)`),
    // preventing sequential double-replacement bugs when asset names overlap (`?assets=b2,b1`).
    let rewritten_expr = expr_raw
        .map(|expr| rewrite_stac_expr(expr, &names))
        .transpose()?;

    Ok((names, cogs, rewritten_expr))
}

/// Single-pass longest-prefix identifier-boundary replacement of STAC asset names with
/// `b1`, `b2`, ... (handles TiTiler's default `{asset}_b{i}` syntax, bare `{asset}` when
/// `asset_as_band=true`, unspaced subtraction `nir-red`, and hyphenated names `SR-B5-SR-B4`).
pub(crate) fn rewrite_stac_expr(expr: &str, names: &[String]) -> Result<String, EngineError> {
    let mut ordered: Vec<(usize, &str)> = names
        .iter()
        .enumerate()
        .map(|(idx, n)| (idx + 1, n.as_str()))
        .collect();
    ordered.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let single_asset = names.len() == 1;
    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len());
    let mut i = 0;
    while i < bytes.len() {
        let prev_is_ident =
            i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !prev_is_ident && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            let rest = &expr[i..];
            // 1. Bare `{asset}` first (longest-first, matching `select_stac_asset_names`), so an
            //    asset literally named `red_b1` beats prefix `red` + `_b1`.
            if let Some(&(band_1based, matched)) = ordered.iter().find(|&&(_, name)| {
                rest.starts_with(name)
                    && !bytes
                        .get(i + name.len())
                        .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
            }) {
                out.push('b');
                out.push_str(&band_1based.to_string());
                i += matched.len();
                continue;
            }

            // 2. TiTiler default `{asset}_b{k}` (e.g. `nir_b1`, `visual_b3`).
            let mut matched_band_suffix: Option<(usize, usize)> = None;
            for &(asset_1based, name) in &ordered {
                if !rest.starts_with(name) {
                    continue;
                }
                let after = i + name.len();
                if bytes.get(after) == Some(&b'_') && bytes.get(after + 1) == Some(&b'b') {
                    let mut j = after + 2;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > after + 2
                        && !bytes
                            .get(j)
                            .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
                    {
                        if let Ok(band_k) = expr[after + 2..j].parse::<usize>() {
                            if !single_asset && band_k != 1 {
                                return Err(EngineError::BadRequest(format!(
                                    "multi-asset STAC expression '{name}_b{band_k}' with band != 1 is not supported; use band 1 ({name}_b1) or a single multi-band asset"
                                )));
                            }
                            let target_band = if single_asset { band_k } else { asset_1based };
                            matched_band_suffix = Some((target_band, j - i));
                            break;
                        }
                    }
                }
            }
            if let Some((target_band, consumed)) = matched_band_suffix {
                out.push('b');
                out.push_str(&target_band.to_string());
                i += consumed;
                continue;
            }
        }
        let ch = expr[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

// ============================================================================
// /colorMaps & /colorMaps/{colorMapId}
// ============================================================================

pub const AVAILABLE_COLORMAPS: &[&str] = &[
    "gray",
    "viridis",
    "magma",
    "plasma",
    "inferno",
    "turbo",
    "terrain",
    "coolwarm",
    "rdylgn",
    "ndvi",
];

pub fn list_colormaps_json(host: &str) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let items: Vec<serde_json::Value> = cogzilla_kernel::colormap_catalog::COLORMAP_CATALOG_NAMES
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "links": [{
                    "href": format!("{origin}/colorMaps/{id}"),
                    "rel": "item",
                    "type": "application/json",
                    "title": format!("Definition of {id} ColorMap")
                }]
            })
        })
        .collect();
    serde_json::json!({ "colormaps": items })
}

pub fn get_colormap_json(id: &str) -> Result<serde_json::Value, EngineError> {
    let cm = ColormapName::from_str_opt(id)
        .ok_or_else(|| EngineError::NotFound(format!("unknown colormap '{id}'")))?;
    let lut = cm.lut();
    let mut map = serde_json::Map::with_capacity(256);
    for i in 0..256 {
        let [r, g, b, a] = lut.rgba_u32[i].to_le_bytes();
        map.insert(i.to_string(), serde_json::json!([r, g, b, a]));
    }
    Ok(serde_json::Value::Object(map))
}

pub fn get_colormap_png(
    id: &str,
    vertical: bool,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Bytes, EngineError> {
    let cm = ColormapName::from_str_opt(id)
        .ok_or_else(|| EngineError::NotFound(format!("unknown colormap '{id}'")))?;
    let lut = cm.lut();
    let (def_w, def_h) = if vertical { (20u32, 256u32) } else { (256u32, 20u32) };
    let w = width.unwrap_or(def_w).clamp(1, 4096) as usize;
    let h = height.unwrap_or(def_h).clamp(1, 4096) as usize;
    let stride = 1 + w * 3;
    let mut filtered = vec![0u8; h * stride];
    let mut row = vec![0u8; w * 3];
    if vertical {
        let denom = (h.saturating_sub(1)).max(1);
        for y in 0..h {
            let idx = ((y * 255 + denom / 2) / denom).min(255);
            let c = lut.rgb[idx];
            for x in 0..w {
                row[x * 3] = c[0];
                row[x * 3 + 1] = c[1];
                row[x * 3 + 2] = c[2];
            }
            filter_sub_rgb_row(&row, &mut filtered[y * stride..(y + 1) * stride]);
        }
    } else {
        let denom = (w.saturating_sub(1)).max(1);
        for x in 0..w {
            let idx = ((x * 255 + denom / 2) / denom).min(255);
            let c = lut.rgb[idx];
            row[x * 3] = c[0];
            row[x * 3 + 1] = c[1];
            row[x * 3 + 2] = c[2];
        }
        for y in 0..h {
            filter_sub_rgb_row(&row, &mut filtered[y * stride..(y + 1) * stride]);
        }
    }
    Ok(encode_filtered_png(w as u32, h as u32, 2, &filtered))
}

/// Render a colormap colorbar in PNG, JPEG, or WebP with optional custom width/height.
pub fn get_colormap_image(
    id: &str,
    format: cogzilla_kernel::OutputFormat,
    vertical: bool,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Bytes, EngineError> {
    if matches!(format, cogzilla_kernel::OutputFormat::Png) {
        return get_colormap_png(id, vertical, width, height);
    }
    let cm = ColormapName::from_str_opt(id)
        .ok_or_else(|| EngineError::NotFound(format!("unknown colormap '{id}'")))?;
    let lut = cm.lut();
    let (def_w, def_h) = if vertical { (20u32, 256u32) } else { (256u32, 20u32) };
    let w = width.unwrap_or(def_w).clamp(1, 4096) as usize;
    let h = height.unwrap_or(def_h).clamp(1, 4096) as usize;
    let mut rgba = vec![0u8; w * h * 4];
    let denom = if vertical {
        (h.saturating_sub(1)).max(1)
    } else {
        (w.saturating_sub(1)).max(1)
    };
    for y in 0..h {
        let row_off = y * w * 4;
        for x in 0..w {
            let pos = if vertical { y } else { x };
            let idx = ((pos * 255 + denom / 2) / denom).min(255);
            let [r, g, b, a] = lut.rgba_u32[idx].to_le_bytes();
            let o = row_off + x * 4;
            rgba[o] = r;
            rgba[o + 1] = g;
            rgba[o + 2] = b;
            rgba[o + 3] = a;
        }
    }
    if matches!(format, cogzilla_kernel::OutputFormat::Npy) {
        let mut dict = format!(
            "{{'descr': '|u1', 'fortran_order': False, 'shape': (4, {h}, {w}), }}"
        );
        while (10 + dict.len() + 1) % 64 != 0 {
            dict.push(' ');
        }
        dict.push('\n');
        let hlen = dict.len() as u16;
        let pixels = w * h;
        let mut out = Vec::with_capacity(10 + dict.len() + 4 * pixels);
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&hlen.to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        for ch in 0..4usize {
            for p in 0..pixels {
                out.push(rgba[p * 4 + ch]);
            }
        }
        return Ok(Bytes::from(out));
    }
    Ok(crate::titiler::encode_rgba_output(w, h, &rgba, format, false))
}

// ============================================================================
// /tileMatrixSets & /tileMatrixSets/{tileMatrixSetId}
// ============================================================================

pub fn list_tms_json(host: &str) -> serde_json::Value {
    crate::tms::list_all_tms_json(host)
}

pub fn get_tms_json(id: &str) -> Result<serde_json::Value, EngineError> {
    crate::tms::get_tms_def_json(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stac_expr_rewriter_is_single_pass_and_respects_boundaries() {
        // Overlapping asset names (`b2`, `b1`) must NOT double-replace and collapse!
        let names = vec!["b2".to_string(), "b1".to_string()];
        assert_eq!(
            rewrite_stac_expr("(b2-b1)/(b2+b1)", &names).unwrap(),
            "(b1-b2)/(b1+b2)"
        );

        // Short asset names (`r`, `g`, `b`) must NOT clobber function names (`abs`, `sqrt`)
        let rgb = vec!["r".to_string(), "g".to_string(), "b".to_string()];
        assert_eq!(
            rewrite_stac_expr("sqrt(abs(r - b)) + g", &rgb).unwrap(),
            "sqrt(abs(b1 - b3)) + b2"
        );

        // Unspaced subtraction with hyphenated asset names (`SR-B5-SR-B4`) must rewrite cleanly
        let hyphenated = vec!["SR-B5".to_string(), "SR-B4".to_string()];
        assert_eq!(
            rewrite_stac_expr("(SR-B5-SR-B4)/(SR-B5+SR-B4)", &hyphenated).unwrap(),
            "(b1-b2)/(b1+b2)"
        );

        // TiTiler default `{asset}_b{i}` syntax for both multi-asset and single multi-band assets
        let nir_red = vec!["nir".to_string(), "red".to_string()];
        assert_eq!(
            rewrite_stac_expr("(nir_b1-red_b1)/(nir_b1+red_b1)", &nir_red).unwrap(),
            "(b1-b2)/(b1+b2)"
        );
        // Multi-asset `{asset}_b{k}` with k != 1 must fail loud instead of silently mapping to band 1
        assert!(rewrite_stac_expr("(nir_b2-red_b1)/(nir_b2+red_b1)", &nir_red).is_err());
        assert!(rewrite_stac_expr("nir_b0 + red_b1", &nir_red).is_err());

        let visual = vec!["visual".to_string()];
        assert_eq!(
            rewrite_stac_expr("(visual_b3-visual_b1)/(visual_b3+visual_b1)", &visual).unwrap(),
            "(b3-b1)/(b3+b1)"
        );

        // resolve_href must ignore slashes inside signed query strings, handle /vsi* prefixes
        // without escaping to local disk, and normalize `..` segments clamped at origin/bucket root
        assert_eq!(
            resolve_href(
                "https://host.example/items/scene.json?sig=a/b/c",
                "./B10.tif"
            ),
            "https://host.example/items/B10.tif"
        );
        assert_eq!(
            resolve_href("https://host.example/stac/items/scene.json", "/data/B10.tif"),
            "https://host.example/data/B10.tif"
        );
        assert_eq!(
            resolve_href("s3://bucket/stac/items/scene.json", "../cogs/B04.tif"),
            "s3://bucket/stac/cogs/B04.tif"
        );
        assert_eq!(
            resolve_href("s3://bucket/stac/scene.json", "../../../../cogs/B04.tif"),
            "s3://bucket/cogs/B04.tif"
        );
        assert_eq!(
            resolve_href(
                "/vsicurl/https://host.example/stac/item.json?sig=a/b",
                "/etc/passwd"
            ),
            "/vsicurl/https://host.example/etc/passwd"
        );
        assert_eq!(
            resolve_href(
                "/vsicurl/https://host.example/stac/items/item.json?sig=a/b",
                "../cogs/B04.tif"
            ),
            "/vsicurl/https://host.example/stac/cogs/B04.tif"
        );
        assert_eq!(
            resolve_href("/vsis3/mybucket/stac/items/item.json", "/etc/passwd"),
            "/vsis3/mybucket/etc/passwd"
        );
        assert_eq!(
            resolve_href("/vsis3/mybucket/stac/items/item.json", "../cogs/B04.tif"),
            "/vsis3/mybucket/stac/cogs/B04.tif"
        );
        assert_eq!(
            resolve_href("file://localhost/allowed/items/item.json", "B04.tif"),
            "/allowed/items/B04.tif"
        );
        assert_eq!(
            resolve_href("file:///allowed/items/item.json", "B04.tif"),
            "/allowed/items/B04.tif"
        );
    }

    #[test]
    fn rewrite_stac_expr_prefers_exact_asset_name_over_shorter_band_suffix() {
        // Audit #70 #7: assets ["red", "red_b1"] with expr "red_b1-red" -> "b2-b1" (not "b1-b1")
        let names = vec!["red".to_string(), "red_b1".to_string()];
        assert_eq!(rewrite_stac_expr("red_b1-red", &names).unwrap(), "b2-b1");
        let names2 = vec!["red".to_string(), "red_b2".to_string()];
        assert_eq!(rewrite_stac_expr("red_b2-red", &names2).unwrap(), "b2-b1");
        // Single-asset band suffix still works
        let single = vec!["visual".to_string()];
        assert_eq!(rewrite_stac_expr("visual_b3-visual_b1", &single).unwrap(), "b3-b1");
    }
}
