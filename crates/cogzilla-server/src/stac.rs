use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use cogzilla_io::OpenedCog;
use cogzilla_kernel::ColormapName;
use cogzilla_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row};
use cogzilla_proj::ORIGIN_SHIFT;
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

fn resolve_href(item_uri: &str, href: &str) -> String {
    if href.starts_with("http://")
        || href.starts_with("https://")
        || href.starts_with("s3://")
        || href.starts_with("gs://")
        || href.starts_with("az://")
        || href.starts_with("abfs://")
        || href.starts_with("abfss://")
        || href.starts_with("file://")
    {
        return href.to_string();
    }
    let is_remote_item = item_uri.starts_with("http://")
        || item_uri.starts_with("https://")
        || item_uri.starts_with("s3://")
        || item_uri.starts_with("gs://")
        || item_uri.starts_with("az://")
        || item_uri.starts_with("abfs://")
        || item_uri.starts_with("abfss://");
    if is_remote_item {
        let path_only = item_uri.split(['?', '#']).next().unwrap_or(item_uri);
        if href.starts_with('/') {
            // Root-relative href against remote scheme://authority origin
            if let Some(scheme_end) = path_only.find("://") {
                let after_scheme = scheme_end + 3;
                let origin_end = path_only[after_scheme..]
                    .find('/')
                    .map(|p| after_scheme + p)
                    .unwrap_or(path_only.len());
                return format!("{}{}", &path_only[..origin_end], href);
            }
        }
        let base = match path_only.rfind('/') {
            Some(idx) => &path_only[..=idx],
            None => path_only,
        };
        format!("{base}{}", href.trim_start_matches("./"))
    } else if href.starts_with('/') {
        href.to_string()
    } else {
        let local = item_uri.strip_prefix("file://").unwrap_or(item_uri);
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

    let raw_bytes = store
        .fetch_uri_bytes(uri, MAX_STAC_JSON_BYTES)
        .await
        .map_err(|e| match e {
            cogzilla_io::IoError::Forbidden(u) => EngineError::Io(cogzilla_io::IoError::Forbidden(u)),
            cogzilla_io::IoError::HttpStatus { status: 404, url } => {
                EngineError::NotFound(format!("STAC item HTTP 404: {url}"))
            }
            cogzilla_io::IoError::Fs(ioe) if ioe.kind() == std::io::ErrorKind::NotFound => {
                let safe_uri = cogzilla_io::redact_http_url(uri);
                EngineError::NotFound(format!("cannot read STAC item '{safe_uri}': {ioe}"))
            }
            other => {
                let safe_uri = cogzilla_io::redact_http_url(uri);
                EngineError::BadRequest(format!("failed to fetch STAC item '{safe_uri}': {other}"))
            }
        })?;

    let val: serde_json::Value = serde_json::from_slice(&raw_bytes)
        .map_err(|e| EngineError::BadRequest(format!("invalid STAC JSON: {e}")))?;

    let id = val
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("stac-item")
        .to_string();
    let bbox = val.get("bbox").and_then(|v| v.as_array()).and_then(|a| {
        if a.len() >= 4 {
            Some([
                a[0].as_f64()?,
                a[1].as_f64()?,
                a[2].as_f64()?,
                a[3].as_f64()?,
            ])
        } else {
            None
        }
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

    let item = Arc::new(StacItem {
        id,
        bbox,
        geometry,
        assets,
    });
    if cache.len() >= 1024 {
        cache.clear();
    }
    cache.insert(uri.to_string(), item.clone());
    Ok(item)
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
    let mut names: Vec<String> = requested_assets.to_vec();

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
                        rest.starts_with(name)
                            && !bytes
                                .get(i + name.len())
                                .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_')
                    }) {
                        if seen.insert(matched) {
                            names.push(matched.to_string());
                        }
                        i += matched.len();
                        continue;
                    }
                }
                i += 1;
            }
        }
        if names.is_empty() {
            names = item.assets.keys().take(3).cloned().collect();
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
            let cog = store.open_uri(&href).await?;
            Ok::<_, cogzilla_io::IoError>((idx, cog))
        });
    }
    let mut opened: Vec<Option<Arc<OpenedCog>>> = vec![None; names.len()];
    while let Some(res) = js.join_next().await {
        let (idx, cog) = res
            .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))?
            .map_err(EngineError::Io)?;
        opened[idx] = Some(cog);
    }
    let cogs: Vec<Arc<OpenedCog>> = opened.into_iter().map(|o| o.unwrap()).collect();

    // Single-pass identifier-boundary rewrite (e.g. `(B4-B3)/(B4+B3)` -> `(b1-b2)/(b1+b2)`),
    // preventing sequential double-replacement bugs when asset names overlap (`?assets=b2,b1`).
    let rewritten_expr = expr_raw.map(|expr| rewrite_stac_expr(expr, &names));

    Ok((names, cogs, rewritten_expr))
}

/// Single-pass longest-prefix identifier-boundary replacement of STAC asset names with
/// `b1`, `b2`, ... (handles both unspaced subtraction `nir-red` and hyphenated names `SR-B5-SR-B4`).
pub(crate) fn rewrite_stac_expr(expr: &str, names: &[String]) -> String {
    let mut ordered: Vec<(usize, &str)> = names
        .iter()
        .enumerate()
        .map(|(idx, n)| (idx + 1, n.as_str()))
        .collect();
    ordered.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len());
    let mut i = 0;
    while i < bytes.len() {
        let prev_is_ident =
            i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !prev_is_ident && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            let rest = &expr[i..];
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
        }
        let ch = expr[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
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

pub fn get_colormap_png(id: &str) -> Result<Bytes, EngineError> {
    let cm = ColormapName::from_str_opt(id)
        .ok_or_else(|| EngineError::NotFound(format!("unknown colormap '{id}'")))?;
    let lut = cm.lut();
    let w = 256usize;
    let h = 20usize;
    let mut row = vec![0u8; w * 3];
    for x in 0..w {
        let c = lut.rgb[x];
        row[x * 3] = c[0];
        row[x * 3 + 1] = c[1];
        row[x * 3 + 2] = c[2];
    }
    let stride = 1 + w * 3;
    let mut filtered = vec![0u8; h * stride];
    for y in 0..h {
        filter_sub_rgb_row(&row, &mut filtered[y * stride..(y + 1) * stride]);
    }
    Ok(encode_filtered_png(w as u32, h as u32, 2, &filtered))
}

// ============================================================================
// /tileMatrixSets & /tileMatrixSets/{tileMatrixSetId}
// ============================================================================

pub fn list_tms_json(host: &str) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    serde_json::json!({
        "tileMatrixSets": [
            {
                "id": "WebMercatorQuad",
                "title": "Google Maps Compatible for the World",
                "links": [{
                    "href": format!("{origin}/tileMatrixSets/WebMercatorQuad"),
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                    "type": "application/json"
                }]
            },
            {
                "id": "WorldCRS84Quad",
                "title": "CRS84 for the World",
                "links": [{
                    "href": format!("{origin}/tileMatrixSets/WorldCRS84Quad"),
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                    "type": "application/json"
                }]
            }
        ]
    })
}

pub fn get_tms_json(id: &str) -> Result<serde_json::Value, EngineError> {
    match id {
        "WebMercatorQuad" => {
            let mut matrices = Vec::with_capacity(25);
            let world_span = 2.0 * ORIGIN_SHIFT;
            for z in 0..=24u32 {
                let n = 1u64 << z;
                let cell_size = world_span / ((n * 256) as f64);
                let scale_denom = cell_size / 0.00028;
                matrices.push(serde_json::json!({
                    "id": z.to_string(),
                    "scaleDenominator": scale_denom,
                    "cellSize": cell_size,
                    "cornerOfOrigin": "topLeft",
                    "pointOfOrigin": [-ORIGIN_SHIFT, ORIGIN_SHIFT],
                    "tileWidth": 256,
                    "tileHeight": 256,
                    "matrixWidth": n,
                    "matrixHeight": n,
                }));
            }
            Ok(serde_json::json!({
                "id": "WebMercatorQuad",
                "title": "Google Maps Compatible for the World",
                "crs": "http://www.opengis.net/def/crs/EPSG/0/3857",
                "orderedAxes": ["X", "Y"],
                "wellKnownScaleSet": "http://www.opengis.net/def/wkss/OGC/1.0/GoogleMapsCompatible",
                "tileMatrices": matrices,
            }))
        }
        "WorldCRS84Quad" => {
            let mut matrices = Vec::with_capacity(23);
            for z in 0..=22u32 {
                let rows = 1u64 << z;
                let cols = rows * 2;
                let cell_size = 180.0 / ((rows * 256) as f64);
                let scale_denom = (cell_size * 111_319.49079327358) / 0.00028;
                matrices.push(serde_json::json!({
                    "id": z.to_string(),
                    "scaleDenominator": scale_denom,
                    "cellSize": cell_size,
                    "cornerOfOrigin": "topLeft",
                    "pointOfOrigin": [-180.0, 90.0],
                    "tileWidth": 256,
                    "tileHeight": 256,
                    "matrixWidth": cols,
                    "matrixHeight": rows,
                }));
            }
            Ok(serde_json::json!({
                "id": "WorldCRS84Quad",
                "title": "CRS84 for the World",
                "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                "orderedAxes": ["Lon", "Lat"],
                "wellKnownScaleSet": "http://www.opengis.net/def/wkss/OGC/1.0/GoogleCRS84Quad",
                "tileMatrices": matrices,
            }))
        }
        other => Err(EngineError::NotFound(format!(
            "unknown TileMatrixSet '{other}' (supported: WebMercatorQuad, WorldCRS84Quad)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stac_expr_rewriter_is_single_pass_and_respects_boundaries() {
        // Overlapping asset names (`b2`, `b1`) must NOT double-replace and collapse!
        let names = vec!["b2".to_string(), "b1".to_string()];
        assert_eq!(rewrite_stac_expr("(b2-b1)/(b2+b1)", &names), "(b1-b2)/(b1+b2)");

        // Short asset names (`r`, `g`, `b`) must NOT clobber function names (`abs`, `sqrt`)
        let rgb = vec!["r".to_string(), "g".to_string(), "b".to_string()];
        assert_eq!(
            rewrite_stac_expr("sqrt(abs(r - b)) + g", &rgb),
            "sqrt(abs(b1 - b3)) + b2"
        );

        // Unspaced subtraction with hyphenated asset names (`SR-B5-SR-B4`) must rewrite cleanly
        let hyphenated = vec!["SR-B5".to_string(), "SR-B4".to_string()];
        assert_eq!(
            rewrite_stac_expr("(SR-B5-SR-B4)/(SR-B5+SR-B4)", &hyphenated),
            "(b1-b2)/(b1+b2)"
        );

        // resolve_href must ignore slashes inside signed query strings and resolve root-relative hrefs
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
    }
}
