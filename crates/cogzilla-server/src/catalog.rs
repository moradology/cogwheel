use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use cogzilla_proj::{ORIGIN_SHIFT, WEB_MERCATOR_R};

const FINE_DIM: usize = 256;
const COARSE_DIM: usize = 32;
const WORLD_SPAN: f64 = ORIGIN_SHIFT * 2.0;

/// A single scene footprint in a spatial mosaic catalog.
#[derive(Debug, Clone)]
pub struct SceneEntry {
    pub uri: String,
    /// `[min_x, min_y, max_x, max_y]` in EPSG:3857 meters.
    pub bbox_3857: [f64; 4],
    /// Higher priority is composited first (front-to-back).
    pub priority: i64,
}

/// Pluggable spatial index in front of `/mosaic/{name}/tiles`.
///
/// Implementations can back this with an in-memory spatial grid (`InMemoryCatalog`),
/// an R-tree, a memory-mapped file, or an external STAC / GeoParquet index.
pub trait SceneCatalog: Send + Sync {
    /// Return scene URIs whose bounding box intersects `[min_x, min_y, max_x, max_y]`
    /// (in EPSG:3857 meters), ordered front-to-back (descending priority).
    fn query_bbox_3857(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<String>;

    /// Total number of indexed scenes.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Convert WGS84 `[min_lon, min_lat, max_lon, max_lat]` (degrees) to EPSG:3857 meters.
pub fn wgs84_bbox_to_3857(bbox: [f64; 4]) -> [f64; 4] {
    let lon_to_x = |lon: f64| -> f64 {
        (lon.clamp(-180.0, 180.0).to_radians() * WEB_MERCATOR_R)
            .clamp(-ORIGIN_SHIFT, ORIGIN_SHIFT)
    };
    let lat_to_y = |lat: f64| -> f64 {
        let clamped = lat.clamp(-85.051_128_779_806_59, 85.051_128_779_806_59).to_radians();
        ((std::f64::consts::FRAC_PI_4 + clamped * 0.5).tan().ln() * WEB_MERCATOR_R)
            .clamp(-ORIGIN_SHIFT, ORIGIN_SHIFT)
    };
    let (min_x, max_x) = if bbox[0] > bbox[2] || bbox[0] < -180.0 || bbox[2] > 180.0 {
        // Antimeridian-crossing bbox: cover full WebMercator X span conservatively.
        (-ORIGIN_SHIFT, ORIGIN_SHIFT)
    } else {
        (lon_to_x(bbox[0]), lon_to_x(bbox[2]))
    };
    let min_y = lat_to_y(bbox[1]);
    let max_y = lat_to_y(bbox[3]);
    [min_x.min(max_x), min_y.min(max_y), min_x.max(max_x), min_y.max(max_y)]
}

#[inline]
fn cell_range(min_v: f64, max_v: f64, dim: usize) -> (usize, usize) {
    let scale = dim as f64 / WORLD_SPAN;
    let c0 = ((min_v + ORIGIN_SHIFT) * scale).floor() as isize;
    let c1 = ((max_v + ORIGIN_SHIFT) * scale).floor() as isize;
    let max_idx = (dim - 1) as isize;
    (c0.clamp(0, max_idx) as usize, c1.clamp(0, max_idx) as usize)
}

#[inline]
fn bboxes_intersect(a: &[f64; 4], min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> bool {
    a[0] <= max_x && a[2] >= min_x && a[1] <= max_y && a[3] >= min_y
}

struct FlatGrid {
    dim: usize,
    offsets: Vec<u32>,
    entries: Vec<u32>,
}

impl FlatGrid {
    fn build(dim: usize, items: &[(u32, [f64; 4])]) -> Self {
        let n_cells = dim * dim;
        let mut counts = vec![0u32; n_cells];
        for &(_, bb) in items {
            let (x0, x1) = cell_range(bb[0], bb[2], dim);
            let (y0, y1) = cell_range(bb[1], bb[3], dim);
            for cy in y0..=y1 {
                let row = cy * dim;
                for cx in x0..=x1 {
                    counts[row + cx] += 1;
                }
            }
        }
        let mut offsets = Vec::with_capacity(n_cells + 1);
        let mut acc = 0u32;
        for &c in &counts {
            offsets.push(acc);
            acc += c;
        }
        offsets.push(acc);

        let mut entries = vec![0u32; acc as usize];
        let mut cursor = offsets[..n_cells].to_vec();
        for &(idx, bb) in items {
            let (x0, x1) = cell_range(bb[0], bb[2], dim);
            let (y0, y1) = cell_range(bb[1], bb[3], dim);
            for cy in y0..=y1 {
                let row = cy * dim;
                for cx in x0..=x1 {
                    let cell = row + cx;
                    let pos = cursor[cell] as usize;
                    entries[pos] = idx;
                    cursor[cell] = (pos + 1) as u32;
                }
            }
        }
        Self {
            dim,
            offsets,
            entries,
        }
    }

    #[inline]
    fn cell_slice(&self, cx: usize, cy: usize) -> &[u32] {
        let cell = cy * self.dim + cx;
        let start = self.offsets[cell] as usize;
        let end = self.offsets[cell + 1] as usize;
        &self.entries[start..end]
    }
}

/// High-speed two-tier CSR spatial grid index in EPSG:3857.
///
/// Scenes are sorted once at construction time by descending `priority`, so scene index
/// `0..len` directly encodes priority order. Small/medium scenes go into a `256×256` fine
/// CSR grid, larger scenes into a `32×32` coarse CSR grid, and global scenes into an
/// overflow list. Typical tile queries touch 1 fine cell + 1 coarse cell in `< 1 µs`.
pub struct InMemoryCatalog {
    uris: Vec<String>,
    bboxes: Vec<[f64; 4]>,
    fine: FlatGrid,
    coarse: FlatGrid,
    overflow: Vec<u32>,
}

impl InMemoryCatalog {
    pub fn new(mut entries: Vec<SceneEntry>) -> Self {
        // Stable sort by descending priority so ties preserve input order.
        entries.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut uris = Vec::with_capacity(entries.len());
        let mut bboxes = Vec::with_capacity(entries.len());
        let mut fine_items = Vec::new();
        let mut coarse_items = Vec::new();
        let mut overflow = Vec::new();

        for (i, e) in entries.into_iter().enumerate() {
            let idx = i as u32;
            let bb = [
                e.bbox_3857[0].min(e.bbox_3857[2]),
                e.bbox_3857[1].min(e.bbox_3857[3]),
                e.bbox_3857[0].max(e.bbox_3857[2]),
                e.bbox_3857[1].max(e.bbox_3857[3]),
            ];
            uris.push(e.uri);
            bboxes.push(bb);

            let (fx0, fx1) = cell_range(bb[0], bb[2], FINE_DIM);
            let (fy0, fy1) = cell_range(bb[1], bb[3], FINE_DIM);
            if (fx1 - fx0 + 1) * (fy1 - fy0 + 1) <= 16 {
                fine_items.push((idx, bb));
            } else {
                let (cx0, cx1) = cell_range(bb[0], bb[2], COARSE_DIM);
                let (cy0, cy1) = cell_range(bb[1], bb[3], COARSE_DIM);
                if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) <= 16 {
                    coarse_items.push((idx, bb));
                } else {
                    overflow.push(idx);
                }
            }
        }

        let fine = FlatGrid::build(FINE_DIM, &fine_items);
        let coarse = FlatGrid::build(COARSE_DIM, &coarse_items);

        Self {
            uris,
            bboxes,
            fine,
            coarse,
            overflow,
        }
    }

    /// Load a catalog from either:
    /// - A STAC `FeatureCollection` JSON (`{"type": "FeatureCollection", "features": [...]}`), or
    /// - A JSON array of objects (`[{"url": "...", "bbox": [minlon, minlat, maxlon, maxlat], "priority": 10}]`).
    ///
    /// For STAC items, `bbox` is WGS84 degrees; if an entry specifies `"bbox_3857"`, it is used directly.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("invalid catalog JSON: {e}"))?;
        let mut entries = Vec::new();

        if let Some(arr) = val.as_array() {
            for (i, item) in arr.iter().enumerate() {
                let uri = item
                    .get("url")
                    .or_else(|| item.get("uri"))
                    .or_else(|| item.get("href"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("catalog entry {i} missing url/uri/href"))?
                    .to_string();
                let bbox_3857 = parse_entry_bbox(item)
                    .ok_or_else(|| format!("catalog entry {i} missing valid bbox or bbox_3857"))?;
                let priority = item
                    .get("priority")
                    .and_then(|v| v.as_i64())
                    .unwrap_or((arr.len() - i) as i64);
                entries.push(SceneEntry {
                    uri,
                    bbox_3857,
                    priority,
                });
            }
            return Ok(Self::new(entries));
        }

        if let Some(features) = val.get("features").and_then(|v| v.as_array()) {
            for (i, feat) in features.iter().enumerate() {
                let Some(bbox_3857) = parse_entry_bbox(feat) else {
                    continue;
                };
                let Some(uri) = extract_stac_feature_uri(feat) else {
                    continue;
                };
                let priority = feat
                    .get("properties")
                    .and_then(|p| p.get("priority").and_then(|v| v.as_i64()))
                    .or_else(|| {
                        feat.get("properties")
                            .and_then(|p| p.get("datetime").and_then(|v| v.as_str()))
                            .and_then(parse_iso_datetime_rank)
                    })
                    .unwrap_or((features.len() - i) as i64);
                entries.push(SceneEntry {
                    uri,
                    bbox_3857,
                    priority,
                });
            }
            return Ok(Self::new(entries));
        }

        Err("expected JSON array or STAC FeatureCollection with 'features'".into())
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("failed to read catalog {}: {e}", path.as_ref().display()))?;
        Self::from_json_str(&raw)
    }
}

impl SceneCatalog for InMemoryCatalog {
    fn query_bbox_3857(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<String> {
        let mut hits: Vec<u32> = Vec::new();

        let (fx0, fx1) = cell_range(min_x, max_x, FINE_DIM);
        let (fy0, fy1) = cell_range(min_y, max_y, FINE_DIM);
        for cy in fy0..=fy1 {
            for cx in fx0..=fx1 {
                for &idx in self.fine.cell_slice(cx, cy) {
                    if bboxes_intersect(&self.bboxes[idx as usize], min_x, min_y, max_x, max_y) {
                        hits.push(idx);
                    }
                }
            }
        }

        let (cx0, cx1) = cell_range(min_x, max_x, COARSE_DIM);
        let (cy0, cy1) = cell_range(min_y, max_y, COARSE_DIM);
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                for &idx in self.coarse.cell_slice(cx, cy) {
                    if bboxes_intersect(&self.bboxes[idx as usize], min_x, min_y, max_x, max_y) {
                        hits.push(idx);
                    }
                }
            }
        }

        for &idx in &self.overflow {
            if bboxes_intersect(&self.bboxes[idx as usize], min_x, min_y, max_x, max_y) {
                hits.push(idx);
            }
        }

        if hits.len() > 1 {
            hits.sort_unstable();
            hits.dedup();
        }
        hits.into_iter()
            .map(|idx| self.uris[idx as usize].clone())
            .collect()
    }

    fn len(&self) -> usize {
        self.uris.len()
    }
}

fn parse_bbox4_val(v: &serde_json::Value) -> Option<[f64; 4]> {
    let arr = v.as_array()?;
    let coords = match arr.len() {
        4 => [arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?, arr[3].as_f64()?],
        6 => [arr[0].as_f64()?, arr[1].as_f64()?, arr[3].as_f64()?, arr[4].as_f64()?],
        _ => return None,
    };
    coords.iter().all(|x| x.is_finite()).then_some(coords)
}

fn parse_entry_bbox(obj: &serde_json::Value) -> Option<[f64; 4]> {
    if let Some(bb) = obj.get("bbox_3857").and_then(parse_bbox4_val) {
        return Some(bb);
    }
    if let Some(bb) = obj.get("bbox").and_then(parse_bbox4_val) {
        return Some(wgs84_bbox_to_3857(bb));
    }
    None
}

fn extract_stac_feature_uri(feat: &serde_json::Value) -> Option<String> {
    if let Some(u) = feat
        .get("url")
        .or_else(|| feat.get("href"))
        .and_then(|v| v.as_str())
    {
        return Some(u.to_string());
    }
    let assets = feat.get("assets")?.as_object()?;
    for preferred in ["visual", "data", "image", "cog", "B04"] {
        if let Some(href) = assets
            .get(preferred)
            .and_then(|a| a.get("href"))
            .and_then(|v| v.as_str())
        {
            return Some(href.to_string());
        }
    }
    for asset in assets.values() {
        if let Some(href) = asset.get("href").and_then(|v| v.as_str()) {
            let lower = href.to_ascii_lowercase();
            if lower.ends_with(".tif") || lower.ends_with(".tiff") {
                return Some(href.to_string());
            }
        }
    }
    assets
        .values()
        .find_map(|a| a.get("href").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

/// Convert `YYYY-MM-DD[THH:MM:SS...]` into a monotonic integer rank so newer scenes sort first.
fn parse_iso_datetime_rank(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 10 {
        return None;
    }
    let y: i64 = s.get(0..4)?.parse().ok()?;
    let m: i64 = s.get(5..7)?.parse().ok()?;
    let d: i64 = s.get(8..10)?.parse().ok()?;
    let hh: i64 = s.get(11..13).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mm: i64 = s.get(14..16).and_then(|v| v.parse().ok()).unwrap_or(0);
    let ss: i64 = s.get(17..19).and_then(|v| v.parse().ok()).unwrap_or(0);
    Some((((((y * 100 + m) * 100 + d) * 100 + hh) * 100) + mm) * 100 + ss)
}

/// Encode an XYZ WebMercator tile `(z, x, y)` into a Bing/Mercantile quadkey string.
pub fn tile_to_quadkey(z: u8, x: u32, y: u32) -> String {
    let mut qk = String::with_capacity(z as usize);
    for i in (1..=z).rev() {
        let mask = 1u32 << (i - 1);
        let mut digit = b'0';
        if (x & mask) != 0 {
            digit += 1;
        }
        if (y & mask) != 0 {
            digit += 2;
        }
        qk.push(digit as char);
    }
    qk
}

/// MosaicJSON 0.0.2 / 0.0.3 quadkey-indexed spatial catalog.
///
/// Implements `SceneCatalog` so MosaicJSON documents plug straight into both
/// `/mosaic/{name}/tiles` (via `--mosaic`) and `/mosaicjson/tiles?...&url=...`.
pub struct MosaicJsonCatalog {
    pub version: String,
    pub doc_version: String,
    pub name: Option<String>,
    pub minzoom: u8,
    pub maxzoom: u8,
    pub quadkey_zoom: u8,
    pub bounds: [f64; 4],
    pub raw_json: serde_json::Value,
    tiles: HashMap<String, Vec<String>, ahash::RandomState>,
}

fn apply_asset_prefix(prefix: &str, asset: &str) -> String {
    if prefix.is_empty() || asset.starts_with('/') || asset.contains("://") {
        return asset.to_string();
    }
    if prefix.ends_with('/') {
        format!("{prefix}{asset}")
    } else {
        format!("{prefix}/{asset}")
    }
}

impl MosaicJsonCatalog {
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("invalid MosaicJSON: {e}"))?;
        let version = val
            .get("mosaicjson")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'mosaicjson' version field".to_string())?
            .to_string();
        let doc_version = val
            .get("version")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("1.0.0")
            .to_string();
        let name = val.get("name").and_then(|v| v.as_str()).map(String::from);
        let asset_prefix = val
            .get("asset_prefix")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let minzoom = val.get("minzoom").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        let maxzoom = val.get("maxzoom").and_then(|v| v.as_u64()).unwrap_or(22) as u8;
        let bounds = val
            .get("bounds")
            .and_then(parse_bbox4_val)
            .unwrap_or([-180.0, -85.05112878, 180.0, 85.05112878]);
        let tiles_obj = val
            .get("tiles")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "missing 'tiles' object in MosaicJSON".to_string())?;

        let mut inferred_qk_zoom: Option<usize> = None;
        let mut tiles: HashMap<String, Vec<String>, ahash::RandomState> =
            HashMap::with_capacity_and_hasher(tiles_obj.len(), ahash::RandomState::new());
        for (qk, arr) in tiles_obj {
            if inferred_qk_zoom.is_none() {
                inferred_qk_zoom = Some(qk.len());
            }
            if let Some(urls) = arr.as_array() {
                let list: Vec<String> = urls
                    .iter()
                    .filter_map(|u| u.as_str().map(|s| apply_asset_prefix(asset_prefix, s)))
                    .collect();
                if !list.is_empty() {
                    tiles.insert(qk.clone(), list);
                }
            }
        }
        let raw_qk_zoom = val
            .get("quadkey_zoom")
            .and_then(|v| v.as_u64())
            .or_else(|| inferred_qk_zoom.map(|z| z as u64))
            .unwrap_or(minzoom as u64);
        if raw_qk_zoom > 30 {
            return Err(format!(
                "MosaicJSON quadkey_zoom {raw_qk_zoom} exceeds maximum supported zoom 30"
            ));
        }
        let quadkey_zoom = raw_qk_zoom as u8;

        Ok(Self {
            version,
            doc_version,
            name,
            minzoom,
            maxzoom,
            quadkey_zoom,
            bounds,
            raw_json: val,
            tiles,
        })
    }

    pub fn first_asset(&self) -> Option<&str> {
        self.tiles
            .values()
            .flat_map(|v| v.iter())
            .next()
            .map(String::as_str)
    }

    pub fn info_json(&self) -> serde_json::Value {
        let mut quadkeys: Vec<&String> = self.tiles.keys().collect();
        quadkeys.sort();
        serde_json::json!({
            "bounds": self.bounds,
            "crs": "http://www.opengis.net/def/crs/EPSG/0/4326",
            "name": self.name.as_deref().unwrap_or("mosaic"),
            "quadkeys": quadkeys,
            "mosaic_tilematrixset": "WebMercatorQuad",
            "mosaic_minzoom": self.minzoom,
            "mosaic_maxzoom": self.maxzoom,
            "mosaicjson": self.version,
            "minzoom": self.minzoom,
            "maxzoom": self.maxzoom,
            "quadkey_zoom": self.quadkey_zoom,
            "tiles_count": self.tiles.len(),
        })
    }
}

impl SceneCatalog for MosaicJsonCatalog {
    fn query_bbox_3857(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<String> {
        if !min_x.is_finite()
            || !min_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
            || min_x > max_x
            || min_y > max_y
        {
            return Vec::new();
        }
        let qz = self.quadkey_zoom;
        let n = (1u64 << qz) as f64;
        let tile_span = WORLD_SPAN / n;
        // Shrink slightly inward so adjacent-tile shared edges don't pull neighboring quadkeys,
        // while still allowing degenerate point queries (`min_x == max_x && min_y == max_y`).
        let min_span = (max_x - min_x).min(max_y - min_y);
        let eps = if min_span > 0.0 {
            (min_span * 1e-6).max(1e-6).min(min_span * 0.25)
        } else {
            0.0
        };
        let x0 = (((min_x + eps) + ORIGIN_SHIFT) / tile_span).floor() as i64;
        let x1 = (((max_x - eps) + ORIGIN_SHIFT) / tile_span).floor() as i64;
        // WebMercator Y increases northwards while XYZ tile y increases southwards.
        let y0 = ((ORIGIN_SHIFT - (max_y - eps)) / tile_span).floor() as i64;
        let y1 = ((ORIGIN_SHIFT - (min_y + eps)) / tile_span).floor() as i64;
        let max_idx = (1i64 << qz) - 1;
        if x1 < 0 || x0 > max_idx || y1 < 0 || y0 > max_idx || x0 > x1 || y0 > y1 {
            return Vec::new();
        }
        let cx0 = x0.clamp(0, max_idx) as u32;
        let cx1 = x1.clamp(0, max_idx) as u32;
        let cy0 = y0.clamp(0, max_idx) as u32;
        let cy1 = y1.clamp(0, max_idx) as u32;
        if cx0 > cx1 || cy0 > cy1 {
            return Vec::new();
        }

        // Fast path: single quadkey hit (all requests at z >= quadkey_zoom)
        if cx0 == cx1 && cy0 == cy1 {
            let qk = tile_to_quadkey(qz, cx0, cy0);
            return self.tiles.get(&qk).cloned().unwrap_or_default();
        }

        // Low-zoom overview request: cap scan to avoid pathological zoom-0 fan-out
        let span_tiles = (cx1 - cx0 + 1) as u64 * (cy1 - cy0 + 1) as u64;
        let mut out = Vec::new();
        let mut seen: HashSet<&str, ahash::RandomState> = HashSet::with_hasher(ahash::RandomState::new());
        if span_tiles <= 4096 {
            for ty in cy0..=cy1 {
                for tx in cx0..=cx1 {
                    let qk = tile_to_quadkey(qz, tx, ty);
                    if let Some(urls) = self.tiles.get(&qk) {
                        for u in urls {
                            if seen.insert(u.as_str()) {
                                out.push(u.clone());
                            }
                        }
                    }
                }
            }
        } else {
            // Very zoomed-out request: iterate indexed quadkeys directly, filtering each key's
            // decoded (tx, ty) against (cx0..=cx1, cy0..=cy1) so out-of-bbox scenes are excluded.
            let mut matched: Vec<(&str, &Vec<String>)> = self
                .tiles
                .iter()
                .filter_map(|(qk, urls)| {
                    if qk.len() != qz as usize {
                        return None;
                    }
                    let (tx, ty) = quadkey_to_tile(qk)?;
                    if tx >= cx0 && tx <= cx1 && ty >= cy0 && ty <= cy1 {
                        Some((qk.as_str(), urls))
                    } else {
                        None
                    }
                })
                .collect();
            matched.sort_unstable_by_key(|(qk, _)| *qk);
            for (_, urls) in matched {
                for u in urls {
                    if seen.insert(u.as_str()) {
                        out.push(u.clone());
                    }
                }
            }
        }
        out
    }

    fn len(&self) -> usize {
        self.tiles.len()
    }
}

/// Decode a Bing/MosaicJSON quadkey string into `(tx, ty)`.
fn quadkey_to_tile(qk: &str) -> Option<(u32, u32)> {
    if qk.len() > 30 {
        return None;
    }
    let mut tx = 0u32;
    let mut ty = 0u32;
    for b in qk.bytes() {
        tx <<= 1;
        ty <<= 1;
        match b {
            b'0' => {}
            b'1' => tx |= 1,
            b'2' => ty |= 1,
            b'3' => {
                tx |= 1;
                ty |= 1;
            }
            _ => return None,
        }
    }
    Some((tx, ty))
}

/// Auto-detect and parse any supported catalog JSON format:
/// 1. MosaicJSON (`{"mosaicjson": ..., "tiles": {...}}`)
/// 2. STAC `FeatureCollection` (`{"features": [...]}`)
/// 3. Simple JSON array (`[{"url": ..., "bbox": [...]}]`)
pub fn load_catalog_json_str(json: &str) -> Result<Arc<dyn SceneCatalog>, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid catalog JSON: {e}"))?;
    if val.get("mosaicjson").is_some() && val.get("tiles").is_some() {
        return Ok(Arc::new(MosaicJsonCatalog::from_json_str(json)?));
    }
    Ok(Arc::new(InMemoryCatalog::from_json_str(json)?))
}

pub fn load_catalog_json_file(path: impl AsRef<Path>) -> Result<Arc<dyn SceneCatalog>, String> {
    let raw_bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("failed to read catalog {}: {e}", path.as_ref().display()))?;
    let json_bytes = if raw_bytes.len() >= 2 && raw_bytes[0] == 0x1f && raw_bytes[1] == 0x8b {
        cogzilla_codec::decompress_gzip_limited(&raw_bytes, 64 * 1024 * 1024)?
    } else {
        raw_bytes
    };
    let raw = String::from_utf8(json_bytes)
        .map_err(|e| format!("catalog {} is not valid UTF-8: {e}", path.as_ref().display()))?;
    load_catalog_json_str(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_proj::{TileCoord, tile_bounds_3857};
    use std::time::Instant;

    #[test]
    fn spatial_index_prunes_100k_scenes_in_microseconds_and_preserves_priority() {
        // Generate 100,000 scenes scattered uniformly across the globe (~20km x 20km footprints)
        // plus 3 overlapping target scenes at SF Bay with distinct priorities.
        let mut entries = Vec::with_capacity(100_003);
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let next_f64 = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for i in 0..100_000usize {
            let x = (next_f64(&mut seed) * 2.0 - 1.0) * (ORIGIN_SHIFT * 0.95);
            let y = (next_f64(&mut seed) * 2.0 - 1.0) * (ORIGIN_SHIFT * 0.95);
            entries.push(SceneEntry {
                uri: format!("s3://bucket/scene_{i}.tif"),
                bbox_3857: [x, y, x + 20_000.0, y + 20_000.0],
                priority: i as i64,
            });
        }
        // SF Bay tile z=12, x=655, y=1583
        let target_tile = TileCoord { z: 12, x: 655, y: 1583 };
        let (tx0, ty0, tx1, ty1) = tile_bounds_3857(target_tile);
        entries.push(SceneEntry {
            uri: "s3://bucket/sf_low.tif".into(),
            bbox_3857: [tx0 - 1000.0, ty0 - 1000.0, tx1 + 1000.0, ty1 + 1000.0],
            priority: 1_000_000,
        });
        entries.push(SceneEntry {
            uri: "s3://bucket/sf_high.tif".into(),
            bbox_3857: [tx0 - 500.0, ty0 - 500.0, tx1 + 500.0, ty1 + 500.0],
            priority: 3_000_000,
        });
        entries.push(SceneEntry {
            uri: "s3://bucket/sf_mid.tif".into(),
            bbox_3857: [tx0 - 2000.0, ty0 - 2000.0, tx1 + 2000.0, ty1 + 2000.0],
            priority: 2_000_000,
        });

        let cat = InMemoryCatalog::new(entries);
        assert_eq!(cat.len(), 100_003);

        let hits = cat.query_bbox_3857(tx0, ty0, tx1, ty1);
        assert!(hits.len() >= 3);
        assert_eq!(hits[0], "s3://bucket/sf_high.tif");
        assert_eq!(hits[1], "s3://bucket/sf_mid.tif");
        assert_eq!(hits[2], "s3://bucket/sf_low.tif");

        // Measure 1,000 queries to verify sub-microsecond / single-digit-microsecond latency.
        let t0 = Instant::now();
        let mut total_hits = 0usize;
        for _ in 0..1_000 {
            total_hits += cat.query_bbox_3857(tx0, ty0, tx1, ty1).len();
        }
        let elapsed = t0.elapsed();
        assert!(total_hits >= 3_000);
        let avg_us = elapsed.as_secs_f64() * 1e6 / 1_000.0;
        eprintln!("InMemoryCatalog 100k-scene query avg: {avg_us:.3} µs");
        assert!(avg_us < 50.0, "expected fast spatial query, got {avg_us:.2} µs");
    }

    #[test]
    fn loads_stac_feature_collection_and_orders_by_datetime() {
        let json = r#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "bbox": [-122.6, 37.6, -122.2, 37.9],
                    "properties": { "datetime": "2024-01-15T10:00:00Z" },
                    "assets": { "visual": { "href": "/old.tif" } }
                },
                {
                    "type": "Feature",
                    "bbox": [-122.6, 37.6, -122.2, 37.9],
                    "properties": { "datetime": "2025-06-20T18:30:00Z" },
                    "assets": { "visual": { "href": "/new.tif" } }
                }
            ]
        }"#;
        let cat = InMemoryCatalog::from_json_str(json).unwrap();
        assert_eq!(cat.len(), 2);
        let bb = wgs84_bbox_to_3857([-122.45, 37.75, -122.40, 37.78]);
        let hits = cat.query_bbox_3857(bb[0], bb[1], bb[2], bb[3]);
        assert_eq!(hits, vec!["/new.tif".to_string(), "/old.tif".to_string()]);
    }

    #[test]
    fn mosaicjson_quadkey_lookup_and_autodetect() {
        let tile = TileCoord { z: 12, x: 655, y: 1583 };
        let qk = tile_to_quadkey(12, 655, 1583);
        assert_eq!(qk.len(), 12);
        let json = serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 10,
            "maxzoom": 16,
            "quadkey_zoom": 12,
            "bounds": [-123.0, 37.0, -122.0, 38.0],
            "tiles": {
                qk: ["/front.tif", "/back.tif"]
            }
        })
        .to_string();

        let cat = load_catalog_json_str(&json).unwrap();
        assert_eq!(cat.len(), 1);
        // Exact z=12 match
        let (x0, y0, x1, y1) = tile_bounds_3857(tile);
        assert_eq!(
            cat.query_bbox_3857(x0, y0, x1, y1),
            vec!["/front.tif".to_string(), "/back.tif".to_string()]
        );
        // Child z=13 tile inside z=12 quadkey also hits!
        let (cx0, cy0, cx1, cy1) = tile_bounds_3857(TileCoord {
            z: 13,
            x: 655 * 2,
            y: 1583 * 2,
        });
        assert_eq!(
            cat.query_bbox_3857(cx0, cy0, cx1, cy1),
            vec!["/front.tif".to_string(), "/back.tif".to_string()]
        );
        // Unrelated tile returns empty
        let (ox0, oy0, ox1, oy1) = tile_bounds_3857(TileCoord { z: 12, x: 10, y: 10 });
        assert!(cat.query_bbox_3857(ox0, oy0, ox1, oy1).is_empty());
    }
}
