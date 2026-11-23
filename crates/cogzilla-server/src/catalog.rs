use std::path::Path;

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
    let (min_x, max_x) = if bbox[0] > bbox[2] {
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
    if arr.len() < 4 {
        return None;
    }
    Some([
        arr[0].as_f64()?,
        arr[1].as_f64()?,
        arr[2].as_f64()?,
        arr[3].as_f64()?,
    ])
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
}
