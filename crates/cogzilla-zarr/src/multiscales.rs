//! GeoZarr and OME-Zarr `multiscales` convention reader (#375).
//!
//! Kept in one isolated module so convention field names (`multiscales`, `datasets`, `layout`,
//! `path`, `asset`) do not leak into the rest of the reader as the GeoZarr proposal settles.

use serde_json::Value;

use crate::{ArrayMeta, GroupView, ZarrError, ZarrHierarchy};

/// One overview level inside a multiscale pyramid.
#[derive(Debug, Clone)]
pub struct MultiscaleLevel {
    /// Group path of this level relative to the hierarchy root (e.g. `"0"`, `"1"`).
    pub group_path: String,
    /// Variable name inside `group_path` (e.g. `"elevation"`).
    pub variable: String,
    /// Full store prefix for chunk keys (e.g. `"0/elevation"`).
    pub store_prefix: String,
    pub width: u32,
    pub height: u32,
    /// Relative downsampling factor compared to level 0 (`1.0` for finest level, `2.0` for 2x coarser).
    pub downsample_factor: f64,
}

/// Ordered multiscale pyramid (finest level first, coarsest level last).
#[derive(Debug, Clone)]
pub struct MultiscalePyramid {
    pub levels: Vec<MultiscaleLevel>,
}

impl MultiscalePyramid {
    /// Inspect `group`'s `multiscales` attribute (supporting both OME-Zarr/NGFF `datasets[].path`
    /// and GeoZarr Proposal `layout[].asset` / `datasets[].path`).
    ///
    /// Returns `Ok(None)` if the group does not declare `multiscales` or has `< 2` usable levels.
    pub fn discover(
        hierarchy: &ZarrHierarchy,
        group: &GroupView<'_>,
        requested_variable: Option<&str>,
    ) -> Result<Option<Self>, ZarrError> {
        let Some(ms_val) = group.attributes().get("multiscales") else {
            return Ok(None);
        };
        let paths = extract_dataset_paths(ms_val);
        if paths.len() < 2 {
            return Ok(None);
        }

        let mut raw_levels: Vec<(String, String, String, u32, u32)> = Vec::new();
        for rel_path in paths {
            let full_group = group.child_prefix(&rel_path);
            let Ok(level_group) = hierarchy.group(Some(&full_group)) else {
                continue;
            };
            let var_name = match requested_variable {
                Some(v) if level_group.array(v).is_some() => v.to_string(),
                _ => {
                    let vars = level_group.data_variables();
                    if vars.len() == 1 {
                        vars[0].clone()
                    } else {
                        continue;
                    }
                }
            };
            let Some(meta) = level_group.array(&var_name) else {
                continue;
            };
            if let Some((w, h)) = spatial_wh(meta) {
                let prefix = level_group.child_prefix(&var_name);
                raw_levels.push((full_group, var_name, prefix, w, h));
            }
        }

        if raw_levels.len() < 2 {
            return Ok(None);
        }

        // Sort finest (largest pixel count) first.
        raw_levels.sort_by(|a, b| {
            let area_a = (a.3 as u64) * (a.4 as u64);
            let area_b = (b.3 as u64) * (b.4 as u64);
            area_b.cmp(&area_a)
        });

        let base_w = raw_levels[0].3.max(1) as f64;
        let base_h = raw_levels[0].4.max(1) as f64;
        let levels = raw_levels
            .into_iter()
            .map(|(group_path, variable, store_prefix, width, height)| {
                let fx = base_w / (width.max(1) as f64);
                let fy = base_h / (height.max(1) as f64);
                MultiscaleLevel {
                    group_path,
                    variable,
                    store_prefix,
                    width,
                    height,
                    downsample_factor: fx.max(fy),
                }
            })
            .collect();

        Ok(Some(Self { levels }))
    }

    /// Select the coarsest overview level whose ground resolution is still at least as fine as
    /// `required_downsample` (with a 5% tolerance matching COG ground-resolution overview selection).
    ///
    /// `required_downsample = requested_ground_m_per_px / base_ground_m_per_px`:
    /// - `<= 1.0` (zoomed in at or beyond native resolution) -> level 0 (finest).
    /// - `>= 2.0` -> picks the 2x coarser overview level if available.
    pub fn select_for_downsample(&self, required_downsample: f64) -> &MultiscaleLevel {
        if !required_downsample.is_finite() || required_downsample <= 1.0 {
            return &self.levels[0];
        }
        let mut chosen = &self.levels[0];
        for lvl in &self.levels {
            if lvl.downsample_factor <= required_downsample * 1.05 {
                chosen = lvl;
            }
        }
        chosen
    }

    /// Maximum downsampling factor across the pyramid (analogous to COG `ov_factor`).
    pub fn max_downsample_factor(&self) -> f64 {
        self.levels
            .last()
            .map(|l| l.downsample_factor)
            .unwrap_or(1.0)
    }
}

fn spatial_wh(meta: &ArrayMeta) -> Option<(u32, u32)> {
    match meta.rank() {
        2 => Some((meta.shape[1] as u32, meta.shape[0] as u32)),
        3 => Some((meta.shape[2] as u32, meta.shape[1] as u32)),
        _ => None,
    }
}

fn extract_dataset_paths(ms_val: &Value) -> Vec<String> {
    // OME-Zarr / NGFF: multiscales is an array of objects, each with `datasets: [{path: ...}]`
    if let Some(arr) = ms_val.as_array()
        && let Some(first) = arr.first()
    {
        return extract_paths_from_obj(first);
    }
    // GeoZarr Proposal: multiscales is an object with `layout: [{asset: ...}]` or `datasets`
    extract_paths_from_obj(ms_val)
}

fn extract_paths_from_obj(obj: &Value) -> Vec<String> {
    if let Some(datasets) = obj.get("datasets").and_then(Value::as_array) {
        return datasets
            .iter()
            .filter_map(|d| d.get("path").or_else(|| d.get("asset")).and_then(Value::as_str))
            .map(|s| s.trim_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(layout) = obj.get("layout").and_then(Value::as_array) {
        return layout
            .iter()
            .filter_map(|d| d.get("asset").or_else(|| d.get("path")).and_then(Value::as_str))
            .map(|s| s.trim_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryStore;

    #[tokio::test]
    async fn discovers_geozarr_layout_and_selects_by_ground_resolution() {
        let mut store = MemoryStore::new();
        let root = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "multiscales": {
                    "layout": [
                        {"asset": "0"},
                        {"asset": "1"},
                        {"asset": "2"}
                    ]
                }
            },
            "consolidated_metadata": {
                "kind": "inline",
                "metadata": {
                    "0": {"zarr_format": 3, "node_type": "group", "attributes": {}},
                    "0/band": {
                        "zarr_format": 3, "node_type": "array", "shape": [512, 512],
                        "data_type": "float32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [256, 256]}},
                        "codecs": ["bytes"], "fill_value": "NaN"
                    },
                    "1": {"zarr_format": 3, "node_type": "group", "attributes": {}},
                    "1/band": {
                        "zarr_format": 3, "node_type": "array", "shape": [256, 256],
                        "data_type": "float32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [256, 256]}},
                        "codecs": ["bytes"], "fill_value": "NaN"
                    },
                    "2": {"zarr_format": 3, "node_type": "group", "attributes": {}},
                    "2/band": {
                        "zarr_format": 3, "node_type": "array", "shape": [128, 128],
                        "data_type": "float32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [128, 128]}},
                        "codecs": ["bytes"], "fill_value": "NaN"
                    }
                }
            }
        });
        store.insert("zarr.json", serde_json::to_vec(&root).unwrap());
        let h = ZarrHierarchy::open(&store).await.unwrap();
        let g = h.group(None).unwrap();
        let pyr = MultiscalePyramid::discover(&h, &g, None).unwrap().unwrap();
        assert_eq!(pyr.levels.len(), 3);
        assert_eq!(pyr.select_for_downsample(0.5).group_path, "0");
        assert_eq!(pyr.select_for_downsample(1.0).group_path, "0");
        assert_eq!(pyr.select_for_downsample(2.0).group_path, "1");
        assert_eq!(pyr.select_for_downsample(4.5).group_path, "2");
        assert_eq!(pyr.max_downsample_factor(), 4.0);
    }
}
