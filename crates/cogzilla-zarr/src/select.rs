//! Variable, group, rank, and dimension selection (#370).
//!
//! Mirrors `titiler.xarray`'s variable lookup, `?group=` scoping, 2-D/3-D rank enforcement,
//! spatial-last dimension arrangement (`_arrange_dims`), and `valid_min`/`valid_max` extraction
//! from `valid_range`.

use serde_json::Value;

use crate::{ArrayMeta, GroupView, ZarrError, ZarrHierarchy};

const X_NAMES: &[&str] = &["x", "X", "lon", "LON", "longitude", "LONGITUDE", "long"];
const Y_NAMES: &[&str] = &["y", "Y", "lat", "LAT", "latitude", "LATITUDE"];

/// Axis indices after arranging spatial dimensions last: `(y, x)` for 2-D or `(extra, y, x)` for 3-D.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrangedDims {
    pub y_axis: usize,
    pub x_axis: usize,
    pub extra_axis: Option<usize>,
    pub y_name: Option<String>,
    pub x_name: Option<String>,
    pub extra_name: Option<String>,
}

impl ArrangedDims {
    /// True when axes are already in canonical `(y, x)` or `(extra, y, x)` order.
    pub fn is_canonical(&self) -> bool {
        match self.extra_axis {
            None => self.y_axis == 0 && self.x_axis == 1,
            Some(e) => e == 0 && self.y_axis == 1 && self.x_axis == 2,
        }
    }
}

/// A resolved variable ready for georeferencing and chunk reading.
#[derive(Debug, Clone)]
pub struct SelectedVariable<'a> {
    pub name: String,
    pub store_prefix: String,
    pub meta: &'a ArrayMeta,
    pub dims: ArrangedDims,
    pub valid_min: Option<f64>,
    pub valid_max: Option<f64>,
}

/// Select a single renderable (2-D or 3-D) variable from `hierarchy` scoped by optional `group`.
///
/// - If `hierarchy` is a bare root array (no enclosing group) and `group`/`variable` are `None`,
///   selects the root array.
/// - If `variable` is `None` and the group has exactly one data variable, auto-selects it.
/// - Otherwise refuses with `VariableNotFound` or `AmbiguousVariable` listing all available
///   variables in the group.
pub fn select_variable<'a>(
    hierarchy: &'a ZarrHierarchy,
    group_path: Option<&str>,
    variable: Option<&str>,
) -> Result<SelectedVariable<'a>, ZarrError> {
    // Handle bare root array (v2 `.zarray` or v3 `node_type: "array"` at root)
    if group_path.is_none_or(|p| p.trim_matches('/').is_empty())
        && let Some(root_arr) = hierarchy.root_array()
    {
        if let Some(req) = variable
            && !req.is_empty()
        {
            return Err(ZarrError::VariableNotFound {
                requested: req.to_string(),
                available: vec!["".to_string()],
            });
        }
        return build_selected(String::new(), String::new(), root_arr);
    }

    let group = hierarchy.group(group_path)?;
    select_variable_in_group(&group, variable)
}

/// Select a single variable inside an already-resolved [`GroupView`].
pub fn select_variable_in_group<'a>(
    group: &GroupView<'a>,
    variable: Option<&str>,
) -> Result<SelectedVariable<'a>, ZarrError> {
    let available = group.data_variables();
    let name = match variable.map(str::trim).filter(|s| !s.is_empty()) {
        Some(v) => v.to_string(),
        None => {
            if available.len() == 1 {
                available[0].clone()
            } else {
                return Err(ZarrError::AmbiguousVariable { available });
            }
        }
    };

    let Some(meta) = group.array(&name) else {
        return Err(ZarrError::VariableNotFound {
            requested: name,
            available,
        });
    };

    let prefix = group.child_prefix(&name);
    build_selected(name, prefix, meta)
}

/// Select multiple variables (matching `?variables=a&variables=b` on multi-band routes).
pub fn select_variables<'a>(
    hierarchy: &'a ZarrHierarchy,
    group_path: Option<&str>,
    variables: &[String],
) -> Result<Vec<SelectedVariable<'a>>, ZarrError> {
    let group = hierarchy.group(group_path)?;
    if variables.is_empty() {
        return Err(ZarrError::AmbiguousVariable {
            available: group.data_variables(),
        });
    }
    let mut out = Vec::with_capacity(variables.len());
    for v in variables {
        out.push(select_variable_in_group(&group, Some(v))?);
    }
    Ok(out)
}

fn build_selected<'a>(
    name: String,
    store_prefix: String,
    meta: &'a ArrayMeta,
) -> Result<SelectedVariable<'a>, ZarrError> {
    meta.require_renderable_rank()?;
    let dims = arrange_dims(meta)?;
    let (valid_min, valid_max) = valid_min_max(meta);
    Ok(SelectedVariable {
        name,
        store_prefix,
        meta,
        dims,
        valid_min,
        valid_max,
    })
}

/// Identify `(y, x)` and optional `extra` axes, placing spatial dimensions last as
/// `titiler.xarray._arrange_dims` does.
pub fn arrange_dims(meta: &ArrayMeta) -> Result<ArrangedDims, ZarrError> {
    meta.require_renderable_rank()?;
    let rank = meta.rank();
    let dim_names: Vec<Option<String>> = meta
        .dimension_names
        .clone()
        .unwrap_or_else(|| vec![None; rank]);

    let find_axis = |candidates: &[&str]| -> Option<usize> {
        dim_names.iter().position(|opt| {
            opt.as_deref()
                .is_some_and(|n| candidates.contains(&n))
        })
    };

    let (y_axis, x_axis) = match (find_axis(Y_NAMES), find_axis(X_NAMES)) {
        (Some(y), Some(x)) if y != x => (y, x),
        _ => {
            // Default: last two axes are (y, x)
            if rank == 2 { (0, 1) } else { (1, 2) }
        }
    };

    let extra_axis = if rank == 3 {
        (0..3).find(|&ax| ax != y_axis && ax != x_axis)
    } else {
        None
    };

    Ok(ArrangedDims {
        y_axis,
        x_axis,
        extra_axis,
        y_name: dim_names.get(y_axis).cloned().flatten(),
        x_name: dim_names.get(x_axis).cloned().flatten(),
        extra_name: extra_axis.and_then(|e| dim_names.get(e).cloned().flatten()),
    })
}

/// Extract `(valid_min, valid_max)` from CF attributes, carrying them out of `valid_range`
/// `[min, max]` when only the range form is present.
pub fn valid_min_max(meta: &ArrayMeta) -> (Option<f64>, Option<f64>) {
    let mut vmin = meta.attributes.get("valid_min").and_then(Value::as_f64);
    let mut vmax = meta.attributes.get("valid_max").and_then(Value::as_f64);
    if (vmin.is_none() || vmax.is_none())
        && let Some(range) = meta.attributes.get("valid_range").and_then(Value::as_array)
        && range.len() == 2
    {
        if vmin.is_none() {
            vmin = range[0].as_f64();
        }
        if vmax.is_none() {
            vmax = range[1].as_f64();
        }
    }
    (vmin, vmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryStore;

    fn array_json(shape: &[u64], dims: &[&str], extra_attrs: serde_json::Value) -> serde_json::Value {
        let mut attrs = extra_attrs.as_object().cloned().unwrap_or_default();
        let _ = &mut attrs;
        serde_json::json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
            "codecs": ["bytes"],
            "fill_value": "NaN",
            "dimension_names": dims,
            "attributes": attrs,
        })
    }

    #[tokio::test]
    async fn selects_variables_refuses_ranks_and_extracts_valid_range() {
        let mut store = MemoryStore::new();
        let root = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
            "consolidated_metadata": {
                "kind": "inline",
                "metadata": {
                    "y": array_json(&[8], &["y"], serde_json::json!({})),
                    "x": array_json(&[8], &["x"], serde_json::json!({})),
                    "a2d": array_json(&[8, 8], &["y", "x"], serde_json::json!({"valid_range": [0.0, 100.0]})),
                    "b3d": array_json(&[8, 8, 3], &["y", "x", "band"], serde_json::json!({})),
                    "c4d": array_json(&[2, 2, 8, 8], &["e", "t", "y", "x"], serde_json::json!({})),
                }
            }
        });
        store.insert("zarr.json", serde_json::to_vec(&root).unwrap());
        let h = ZarrHierarchy::open(&store).await.unwrap();

        // Ambiguous when omitted and >1 data variable exists
        let amb = select_variable(&h, None, None).unwrap_err();
        assert!(matches!(&amb, ZarrError::AmbiguousVariable { available } if available.len() == 3));

        // Missing variable lists available ones
        let miss = select_variable(&h, None, Some("nope")).unwrap_err();
        assert!(matches!(&miss, ZarrError::VariableNotFound { requested, available } if requested == "nope" && available.contains(&"a2d".to_string())));

        // 2-D variable carries valid_min/valid_max out of valid_range
        let s2 = select_variable(&h, None, Some("a2d")).unwrap();
        assert!(s2.dims.is_canonical());
        assert_eq!((s2.valid_min, s2.valid_max), (Some(0.0), Some(100.0)));

        // 3-D variable with (y, x, band) identifies extra_axis = 2
        let s3 = select_variable(&h, None, Some("b3d")).unwrap();
        assert_eq!(s3.dims.y_axis, 0);
        assert_eq!(s3.dims.x_axis, 1);
        assert_eq!(s3.dims.extra_axis, Some(2));
        assert_eq!(s3.dims.extra_name.as_deref(), Some("band"));

        // 4-D variable refused by rank
        let err4 = select_variable(&h, None, Some("c4d")).unwrap_err();
        assert!(matches!(err4, ZarrError::UnsupportedRank(4)));
    }
}
