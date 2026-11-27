//! Georeferencing from 1-D Zarr coordinate arrays (#369):
//! - X / Y coordinate variable discovery (by dimension names, CF `axis`/`standard_name`, and
//!   TiTiler fallback name lists)
//! - 1-D coordinate array reading through the codec chain
//! - Strict regularity check (refusing irregular or curvilinear grids by name and index)
//! - Half-cell-center affine derivation (handling ascending & descending Y by sign)
//! - `0..360` longitude normalization to `-180..180` reusing `cogzilla_proj::normalize_geographic_lon`

use cogzilla_proj::normalize_geographic_lon;
use cogzilla_tiff::Affine2D;
use serde_json::Value;

use crate::crs::{ZarrCrs, resolve_crs};
use crate::{ArrayMeta, DataType, GroupView, ZarrError, ZarrStore, decode_chunk_bytes};

const X_NAMES: &[&str] = &["x", "X", "lon", "LON", "longitude", "LONGITUDE", "long"];
const Y_NAMES: &[&str] = &["y", "Y", "lat", "LAT", "latitude", "LATITUDE"];

const X_STD_NAMES: &[&str] = &["longitude", "projection_x_coordinate", "grid_longitude"];
const Y_STD_NAMES: &[&str] = &["latitude", "projection_y_coordinate", "grid_latitude"];

/// Complete georeferencing derived for a 2-D or 3-D Zarr array.
#[derive(Debug, Clone)]
pub struct ZarrGeo {
    pub crs: ZarrCrs,
    /// Pixel-to-CRS affine transform using the cell-center half-cell convention:
    /// `affine.forward(col + 0.5, row + 0.5) == (x[col], y[row])`.
    pub affine: Affine2D,
    /// Outer spatial bounds `[minx, miny, maxx, maxy]` in the array's CRS (half a cell outside
    /// the first and last coordinate centers).
    pub bounds: [f64; 4],
    /// Name of the X coordinate variable used.
    pub x_dim: String,
    /// Name of the Y coordinate variable used.
    pub y_dim: String,
    /// True when the source X coordinate array was in `0..360` geographic longitude and normalized
    /// to `[-180..180]`.
    pub lon_0_360: bool,
    /// Column shift when a global `0..360` array wraps across `180°`: logical column `c` maps to
    /// source column `(c + shift) % width`. `0` when no column wrap is needed.
    pub lon_wrap_shift: u32,
}

/// Resolve CRS + regular affine + bounds for `var_name` inside `group`.
pub async fn resolve_geo(
    store: &dyn ZarrStore,
    group: &GroupView<'_>,
    meta: &ArrayMeta,
) -> Result<ZarrGeo, ZarrError> {
    meta.require_renderable_rank()?;
    let crs = resolve_crs(group, meta)?;

    // Reject explicit curvilinear 2-D `coordinates` before looking for 1-D axes.
    check_not_curvilinear(group, meta)?;

    let (y_name, x_name) = find_xy_coord_names(group, meta)?;
    let y_meta = group.array(&y_name).ok_or_else(|| ZarrError::InvalidSelection {
        detail: format!("Y coordinate array {y_name:?} not found in group"),
    })?;
    let x_meta = group.array(&x_name).ok_or_else(|| ZarrError::InvalidSelection {
        detail: format!("X coordinate array {x_name:?} not found in group"),
    })?;

    if y_meta.rank() != 1 {
        return Err(ZarrError::IrregularGrid {
            dim: y_name,
            index: 0,
            detail: format!("curvilinear Y coordinate has rank {} (expected 1D)", y_meta.rank()),
        });
    }
    if x_meta.rank() != 1 {
        return Err(ZarrError::IrregularGrid {
            dim: x_name,
            index: 0,
            detail: format!("curvilinear X coordinate has rank {} (expected 1D)", x_meta.rank()),
        });
    }

    let y_vals = read_coord_1d(store, &group.child_prefix(&y_name), y_meta).await?;
    let x_vals = read_coord_1d(store, &group.child_prefix(&x_name), x_meta).await?;

    let (h, w) = if meta.rank() == 2 {
        (meta.shape[0] as usize, meta.shape[1] as usize)
    } else {
        (meta.shape[1] as usize, meta.shape[2] as usize)
    };
    if y_vals.len() != h {
        return Err(ZarrError::IrregularGrid {
            dim: y_name,
            index: 0,
            detail: format!("Y coordinate length {} != array height {h}", y_vals.len()),
        });
    }
    if x_vals.len() != w {
        return Err(ZarrError::IrregularGrid {
            dim: x_name,
            index: 0,
            detail: format!("X coordinate length {} != array width {w}", x_vals.len()),
        });
    }

    derive_regular_geo_with_dtypes(crs, x_name, &x_vals, x_meta.dtype, y_name, &y_vals, y_meta.dtype)
}

/// Derive a regular affine and bounds from 1-D cell-center coordinate arrays, enforcing uniform
/// spacing and normalizing `0..360` geographic longitudes.
pub fn derive_regular_geo(
    crs: ZarrCrs,
    x_dim: String,
    x_vals: &[f64],
    y_dim: String,
    y_vals: &[f64],
) -> Result<ZarrGeo, ZarrError> {
    derive_regular_geo_with_dtypes(
        crs,
        x_dim,
        x_vals,
        DataType::Float64,
        y_dim,
        y_vals,
        DataType::Float64,
    )
}

fn derive_regular_geo_with_dtypes(
    crs: ZarrCrs,
    x_dim: String,
    x_vals: &[f64],
    x_dtype: DataType,
    y_dim: String,
    y_vals: &[f64],
    y_dtype: DataType,
) -> Result<ZarrGeo, ZarrError> {
    let dx = check_regular_axis_with_dtype(&x_dim, x_vals, x_dtype)?;
    let dy = check_regular_axis_with_dtype(&y_dim, y_vals, y_dtype)?;

    let w = x_vals.len() as f64;
    let h = y_vals.len() as f64;

    // Half-cell convention: coordinate values are cell centers, so pixel (0, 0)'s outer corner is
    // half a step before center[0].
    let mut x_origin = x_vals[0] - 0.5 * dx;
    let y_origin = y_vals[0] - 0.5 * dy;

    let mut lon_0_360 = false;
    let mut lon_wrap_shift = 0u32;

    // 0..360 geographic longitude normalization.
    // - Regional [180..360] grids shift `x_origin` by -360° directly.
    // - Global grids spanning across 180° keep their true `0..360` pixel-to-CRS affine so
    //   `cogzilla_proj::CoordPlan` wraps western-hemisphere longitudes without column shifting,
    //   while reporting WGS84 bounds via `CrsModel::wgs84_bounds`.
    if crs.model.is_geographic() {
        let x_min = x_vals.first().copied().unwrap_or(0.0).min(x_vals.last().copied().unwrap_or(0.0));
        let x_max = x_vals.first().copied().unwrap_or(0.0).max(x_vals.last().copied().unwrap_or(0.0));
        if x_min >= -1e-6 && x_max > 180.0 && x_max <= 360.0 + dx.abs() {
            lon_0_360 = true;
            if x_min >= 180.0 - 1e-6 {
                // Entirely in [180..360]: shift origin by -360°, no column wrap needed.
                if let Some(norm_first) = normalize_geographic_lon(x_vals[0]) {
                    x_origin = norm_first - 0.5 * dx;
                }
            } else if dx > 0.0
                && let Some(split) = x_vals.iter().position(|&lon| lon >= 180.0)
            {
                lon_wrap_shift = split as u32;
            }
        }
    }

    let affine = Affine2D {
        a: dx,
        b: 0.0,
        c: x_origin,
        d: 0.0,
        e: dy,
        f: y_origin,
    };

    let bounds = if lon_0_360 && lon_wrap_shift > 0 {
        crs.model.wgs84_bounds(&affine, x_vals.len() as u32, y_vals.len() as u32)
    } else {
        let (x0, y0) = affine.forward(0.0, 0.0);
        let (x1, y1) = affine.forward(w, h);
        [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
    };

    Ok(ZarrGeo {
        crs,
        affine,
        bounds,
        x_dim,
        y_dim,
        lon_0_360,
        lon_wrap_shift,
    })
}

/// Check that a 1-D coordinate array has uniform spacing within tolerance, returning the step
/// `vals[1] - vals[0]` (or mean step for float32 coordinates). Refuses non-uniform spacing with
/// dimension name and offending index.
pub fn check_regular_axis(dim: &str, vals: &[f64]) -> Result<f64, ZarrError> {
    check_regular_axis_with_dtype(dim, vals, DataType::Float64)
}

fn check_regular_axis_with_dtype(
    dim: &str,
    vals: &[f64],
    dtype: DataType,
) -> Result<f64, ZarrError> {
    if vals.len() < 2 {
        return Err(ZarrError::IrregularGrid {
            dim: dim.to_string(),
            index: 0,
            detail: format!("need at least 2 coordinate values, got {}", vals.len()),
        });
    }
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(ZarrError::IrregularGrid {
                dim: dim.to_string(),
                index: i,
                detail: format!("non-finite coordinate value {v}"),
            });
        }
    }
    let step = if matches!(dtype, DataType::Float32) {
        (vals[vals.len() - 1] - vals[0]) / (vals.len() - 1) as f64
    } else {
        vals[1] - vals[0]
    };
    if !step.is_finite() || step == 0.0 {
        return Err(ZarrError::IrregularGrid {
            dim: dim.to_string(),
            index: 1,
            detail: format!("zero or invalid initial spacing ({} -> {})", vals[0], vals[1]),
        });
    }
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let base_tol = step.abs() * 1e-4 + 1e-9;
    let tol = if matches!(dtype, DataType::Float32) {
        base_tol.max((f32::EPSILON as f64) * max_abs.max(1.0) * 4.0)
    } else {
        base_tol
    };
    for i in 0..vals.len() - 1 {
        let d = vals[i + 1] - vals[i];
        if (d - step).abs() > tol {
            return Err(ZarrError::IrregularGrid {
                dim: dim.to_string(),
                index: i + 1,
                detail: format!(
                    "spacing {d} at index {} differs from expected spacing {step} (tolerance {tol:.2e})",
                    i + 1
                ),
            });
        }
        if matches!(dtype, DataType::Float32) {
            let expected = vals[0] + (i + 1) as f64 * step;
            if (vals[i + 1] - expected).abs() > tol {
                return Err(ZarrError::IrregularGrid {
                    dim: dim.to_string(),
                    index: i + 1,
                    detail: format!(
                        "coordinate {} at index {} drifts from regular grid {expected} (tolerance {tol:.2e})",
                        vals[i + 1],
                        i + 1
                    ),
                });
            }
        }
    }
    Ok(step)
}

/// Read all chunks of a 1-D Zarr coordinate array into `Vec<f64>`.
pub async fn read_coord_1d(
    store: &dyn ZarrStore,
    array_prefix: &str,
    meta: &ArrayMeta,
) -> Result<Vec<f64>, ZarrError> {
    if meta.rank() != 1 {
        return Err(ZarrError::InvalidSelection {
            detail: format!("expected 1D coordinate array, got rank {}", meta.rank()),
        });
    }
    let len = meta.shape[0] as usize;
    let c = meta.chunk_shape[0] as usize;
    if c == 0 {
        return Err(ZarrError::InvalidSelection {
            detail: "1D coordinate array has chunk_shape 0".into(),
        });
    }
    let n_chunks = len.div_ceil(c);
    let prefix = array_prefix.trim_matches('/');
    let mut out = Vec::with_capacity(len);

    for ci in 0..n_chunks {
        let rel = meta.chunk_key(&[ci as u64]);
        let full_key = if prefix.is_empty() {
            rel
        } else {
            format!("{prefix}/{rel}")
        };
        let valid_elems = (len - ci * c).min(c);
        let Some(raw) = store.get(&full_key).await? else {
            if let Some(fv) = meta.fill_value {
                out.extend(std::iter::repeat_n(fv, valid_elems));
                continue;
            }
            return Err(ZarrError::MissingChunk {
                chunk_key: full_key,
            });
        };
        let decoded = decode_chunk_bytes(meta, &raw, &full_key)?;
        append_native_as_f64(&decoded, meta.dtype, valid_elems, &mut out);
    }
    Ok(out)
}

fn append_native_as_f64(bytes: &[u8], dtype: DataType, count: usize, out: &mut Vec<f64>) {
    match dtype {
        DataType::Float64 => {
            for c in bytes[..count * 8].chunks_exact(8) {
                out.push(f64::from_ne_bytes(c.try_into().unwrap()));
            }
        }
        DataType::Float32 => {
            for c in bytes[..count * 4].chunks_exact(4) {
                out.push(f32::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::Int64 => {
            for c in bytes[..count * 8].chunks_exact(8) {
                out.push(i64::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::UInt64 => {
            for c in bytes[..count * 8].chunks_exact(8) {
                out.push(u64::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::Int32 => {
            for c in bytes[..count * 4].chunks_exact(4) {
                out.push(i32::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::UInt32 => {
            for c in bytes[..count * 4].chunks_exact(4) {
                out.push(u32::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::Int16 => {
            for c in bytes[..count * 2].chunks_exact(2) {
                out.push(i16::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::UInt16 => {
            for c in bytes[..count * 2].chunks_exact(2) {
                out.push(u16::from_ne_bytes(c.try_into().unwrap()) as f64);
            }
        }
        DataType::Int8 => {
            for &b in &bytes[..count] {
                out.push((b as i8) as f64);
            }
        }
        DataType::UInt8 | DataType::Bool => {
            for &b in &bytes[..count] {
                out.push(b as f64);
            }
        }
        DataType::Float16 => {
            for c in bytes[..count * 2].chunks_exact(2) {
                let bits = u16::from_ne_bytes([c[0], c[1]]);
                let sign = ((bits & 0x8000) as u32) << 16;
                let exp = ((bits >> 10) & 0x1F) as u32;
                let mant = (bits & 0x03FF) as u32;
                let f = if exp == 0 {
                    f32::from_bits(sign)
                } else if exp == 31 {
                    f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
                } else {
                    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
                };
                out.push(f as f64);
            }
        }
    }
}

fn check_not_curvilinear(group: &GroupView<'_>, meta: &ArrayMeta) -> Result<(), ZarrError> {
    if let Some(coords) = meta.attributes.get("coordinates").and_then(Value::as_str) {
        for name in coords.split_whitespace() {
            if let Some(arr) = group.array(name)
                && arr.rank() >= 2
            {
                return Err(ZarrError::IrregularGrid {
                    dim: name.to_string(),
                    index: 0,
                    detail: format!(
                        "curvilinear {}D coordinate array {name:?} is not supported",
                        arr.rank()
                    ),
                });
            }
        }
    }
    Ok(())
}

fn find_xy_coord_names(
    group: &GroupView<'_>,
    meta: &ArrayMeta,
) -> Result<(String, String), ZarrError> {
    // 1. Spatial dimension names on the array itself (last two dims: [..., y, x]).
    if let Some(names) = &meta.dimension_names
        && names.len() >= 2
    {
        let y_opt = names[names.len() - 2].as_deref();
        let x_opt = names[names.len() - 1].as_deref();
        if let (Some(y), Some(x)) = (y_opt, x_opt)
            && group.array(y).is_some()
            && group.array(x).is_some()
        {
            return Ok((y.to_string(), x.to_string()));
        }
    }

    // 2. CF `axis` ("X"/"Y") or `standard_name` attributes on 1-D arrays in the group.
    let mut found_x = None;
    let mut found_y = None;
    for (name, arr) in group.arrays() {
        if arr.rank() != 1 {
            continue;
        }
        let axis = arr.attributes.get("axis").and_then(Value::as_str);
        let std_name = arr.attributes.get("standard_name").and_then(Value::as_str);
        if found_x.is_none()
            && (axis == Some("X") || std_name.is_some_and(|s| X_STD_NAMES.contains(&s)))
        {
            found_x = Some(name.clone());
        }
        if found_y.is_none()
            && (axis == Some("Y") || std_name.is_some_and(|s| Y_STD_NAMES.contains(&s)))
        {
            found_y = Some(name);
        }
    }
    if let (Some(y), Some(x)) = (found_y, found_x) {
        return Ok((y, x));
    }

    // 3. Fallback to TiTiler's standard coordinate name lists.
    let y_fallback = Y_NAMES
        .iter()
        .find(|&&n| group.array(n).is_some())
        .map(|&s| s.to_string());
    let x_fallback = X_NAMES
        .iter()
        .find(|&&n| group.array(n).is_some())
        .map(|&s| s.to_string());
    match (y_fallback, x_fallback) {
        (Some(y), Some(x)) => Ok((y, x)),
        _ => Err(ZarrError::InvalidSelection {
            detail: "could not identify X and Y coordinate variables in group".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo_crs() -> ZarrCrs {
        ZarrCrs {
            model: cogzilla_proj::CrsModel::Geographic,
            epsg: Some(4326),
            description: "EPSG:4326".into(),
        }
    }

    #[test]
    fn half_cell_convention_round_trips_centers_for_desc_asc_and_nonsquare() {
        // Descending Y, non-square cells: dx = 0.5, dy = -0.25
        let x = [10.25, 10.75, 11.25, 11.75];
        let y = [50.875, 50.625, 50.375];
        let g = derive_regular_geo(geo_crs(), "x".into(), &x, "y".into(), &y).unwrap();
        assert_eq!(g.bounds, [10.0, 50.25, 12.0, 51.0]);
        for (c, &xc) in x.iter().enumerate() {
            for (r, &yc) in y.iter().enumerate() {
                let (fx, fy) = g.affine.forward(c as f64 + 0.5, r as f64 + 0.5);
                assert!((fx - xc).abs() < 1e-12);
                assert!((fy - yc).abs() < 1e-12);
            }
        }

        // Ascending Y (south-up): dy > 0, handled by sign without sorting data
        let y_asc = [50.375, 50.625, 50.875];
        let ga = derive_regular_geo(geo_crs(), "x".into(), &x, "y".into(), &y_asc).unwrap();
        assert!(ga.affine.e > 0.0);
        assert_eq!(ga.bounds, [10.0, 50.25, 12.0, 51.0]);
        let (_, fy0) = ga.affine.forward(0.5, 0.5);
        assert!((fy0 - 50.375).abs() < 1e-12);
    }

    #[test]
    fn irregular_spacing_names_dimension_and_offending_index() {
        let x = [0.0, 1.0, 2.0, 4.5, 5.5];
        let y = [10.0, 9.0, 8.0];
        let err = derive_regular_geo(geo_crs(), "lon".into(), &x, "lat".into(), &y).unwrap_err();
        match err {
            ZarrError::IrregularGrid { dim, index, .. } => {
                assert_eq!(dim, "lon");
                assert_eq!(index, 3);
            }
            other => panic!("expected IrregularGrid, got {other:?}"),
        }
    }

    #[test]
    fn lon_0_360_normalizes_global_and_regional() {
        // Global 0..360 with 5-degree cells (2.5 .. 357.5)
        let x: Vec<f64> = (0..72).map(|i| i as f64 * 5.0 + 2.5).collect();
        let y: Vec<f64> = (0..36).map(|i| 87.5 - i as f64 * 5.0).collect();
        let g = derive_regular_geo(geo_crs(), "x".into(), &x, "y".into(), &y).unwrap();
        assert!(g.lon_0_360);
        assert_eq!(g.lon_wrap_shift, 36);
        assert_eq!(g.bounds, [-180.0, -90.0, 180.0, 90.0]);

        // Regional 190..210 (Pacific)
        let xr = [190.5, 191.5, 192.5, 193.5];
        let yr = [20.5, 19.5];
        let gr = derive_regular_geo(geo_crs(), "x".into(), &xr, "y".into(), &yr).unwrap();
        assert!(gr.lon_0_360);
        assert_eq!(gr.lon_wrap_shift, 0);
        assert_eq!(gr.bounds, [-170.0, 19.0, -166.0, 21.0]);
    }

    #[test]
    fn float32_regular_coords_tolerate_f32_ulp_but_refuse_irregular() {
        // 0.01° grid quantized to f32 near -180° (MUR SST pattern)
        let x: Vec<f64> = (0..1000)
            .map(|i| (-179.99f32 + i as f32 * 0.01f32) as f64)
            .collect();
        let step = check_regular_axis_with_dtype("lon", &x, DataType::Float32).unwrap();
        assert!((step - 0.01).abs() < 1e-6);

        // Genuine irregularity (0.05° jump) is still refused even in Float32
        let mut bad = x.clone();
        bad[500] += 0.05;
        assert!(check_regular_axis_with_dtype("lon", &bad, DataType::Float32).is_err());
    }
}
