use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use cogwheel_io::OpenedCog;
use cogwheel_kernel::ColormapName;
use cogwheel_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row};
use cogwheel_proj::ORIGIN_SHIFT;
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
        || href.starts_with("file://")
        || href.starts_with('/')
    {
        return href.to_string();
    }
    if item_uri.starts_with("http://")
        || item_uri.starts_with("https://")
        || item_uri.starts_with("s3://")
    {
        let base = match item_uri.rfind('/') {
            Some(idx) => &item_uri[..=idx],
            None => item_uri,
        };
        format!("{base}{}", href.trim_start_matches("./"))
    } else {
        let local = item_uri.strip_prefix("file://").unwrap_or(item_uri);
        let parent = Path::new(local).parent().unwrap_or_else(|| Path::new("."));
        parent
            .join(href)
            .to_string_lossy()
            .into_owned()
    }
}

/// Load (or fetch from lock-free cache) a STAC Item JSON from local disk or HTTP(S).
pub async fn load_stac_item(
    cache: &DashMap<String, Arc<StacItem>, ahash::RandomState>,
    client: &reqwest::Client,
    uri: &str,
) -> Result<Arc<StacItem>, EngineError> {
    if let Some(hit) = cache.get(uri) {
        return Ok(hit.clone());
    }

    let raw_bytes = if let Some((bucket, key)) = cogwheel_io::parse_s3_uri(uri) {
        let cfg = cogwheel_io::S3Config::from_env_full();
        cogwheel_io::fetch_s3_object(client, bucket, key, &cfg, None, None)
            .await
            .map_err(|e| EngineError::BadRequest(format!("failed to fetch S3 STAC item: {e}")))?
            .to_vec()
    } else if uri.starts_with("http://") || uri.starts_with("https://") {
        let resp = client
            .get(uri)
            .send()
            .await
            .map_err(|e| EngineError::BadRequest(format!("failed to fetch STAC item: {e}")))?;
        if !resp.status().is_success() {
            return Err(EngineError::NotFound(format!(
                "STAC item HTTP {}: {uri}",
                resp.status().as_u16()
            )));
        }
        resp.bytes()
            .await
            .map_err(|e| EngineError::BadRequest(format!("STAC body error: {e}")))?
            .to_vec()
    } else {
        let local = uri.strip_prefix("file://").unwrap_or(uri);
        std::fs::read(local)
            .map_err(|e| EngineError::NotFound(format!("cannot read STAC item '{local}': {e}")))?
    };

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

    // If no explicit `?assets=` given, try extracting asset names referenced in `?expression=`,
    // otherwise default to all raster assets (up to 3).
    if names.is_empty() {
        if let Some(expr) = expr_raw {
            for k in item.assets.keys() {
                if expr.contains(k.as_str()) {
                    names.push(k.clone());
                }
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
            Ok::<_, cogwheel_io::IoError>((idx, cog))
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

    // Rewrite asset names in expression (e.g. `(B4-B3)/(B4+B3)` -> `(b1-b2)/(b1+b2)`)
    let rewritten_expr = expr_raw.map(|expr| {
        // Sort by descending key length so `B10` is replaced before `B1`
        let mut pairs: Vec<(usize, &String)> = names.iter().enumerate().collect();
        pairs.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        let mut out = expr.to_string();
        for (i, name) in pairs {
            if !name.eq_ignore_ascii_case(&format!("b{}", i + 1)) {
                out = out.replace(name.as_str(), &format!("b{}", i + 1));
            }
        }
        out
    });

    Ok((names, cogs, rewritten_expr))
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
    let items: Vec<serde_json::Value> = AVAILABLE_COLORMAPS
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "links": [{
                    "href": format!("http://{host}/colorMaps/{id}"),
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
        let c = lut.rgb[i];
        map.insert(i.to_string(), serde_json::json!([c[0], c[1], c[2], 255]));
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
    serde_json::json!({
        "tileMatrixSets": [
            {
                "id": "WebMercatorQuad",
                "title": "Google Maps Compatible for the World",
                "links": [{
                    "href": format!("http://{host}/tileMatrixSets/WebMercatorQuad"),
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                    "type": "application/json"
                }]
            },
            {
                "id": "WorldCRS84Quad",
                "title": "CRS84 for the World",
                "links": [{
                    "href": format!("http://{host}/tileMatrixSets/WorldCRS84Quad"),
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
