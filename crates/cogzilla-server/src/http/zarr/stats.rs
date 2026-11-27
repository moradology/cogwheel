//! Zarr `/zarr/statistics` (GET/POST) and `/zarr/feature` (POST) handlers.

use std::sync::Arc;

use bytes::Bytes;
use cogzilla_kernel::sample_general;
use cogzilla_proj::CrsModel;
use cogzilla_zarr::{
    ZarrCaps, ZarrError, ZarrGeo, read_chunk_window_for_band_with_caps, resolve_geo,
    select_variable,
};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Response, StatusCode};
use serde_json::{Value, json};

use crate::stats::{PolygonMask, StatsOptions, compute_f32_band_stats};
use crate::zonal::{
    PixelAreaModel, ZonalGeomKind, compute_weighted_band_stats_with_geom,
    densify_ring_lonlat_to_pixel, rasterize_exact_coverage,
};

use super::{
    ServerZarrStore, ZarrQuery, encode_zarr_plane, json_ok, open_store_and_hierarchy,
    parse_output_format, record_window_metrics, resolve_band_index,
};

pub(super) async fn handle_get_statistics(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        0,
        0,
        ifd_w,
        ifd_h,
        16,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mut pixels = Vec::with_capacity((ifd_w as usize) * (ifd_h as usize));
    for r in 0..ifd_h as i32 {
        for c in 0..ifd_w as i32 {
            pixels.push(src.sample_checked(c, r));
        }
    }
    let opts = StatsOptions::default();
    let b1 = compute_f32_band_stats(&pixels, &sel_var.name, &opts);
    record_window_metrics(&wb);
    Ok(json_ok(&json!({ "b1": b1 }), "application/json"))
}

// ============================================================================
// POST /zarr/statistics & POST /zarr/feature (#374) — reuses zonal.rs directly
// ============================================================================

pub(super) async fn handle_post_statistics(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    body: Bytes,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let geojson: Value = serde_json::from_slice(&body)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let coord_crs_model = match q.coord_crs {
        Some(code) => Some(CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{code}"),
            detail: e.to_string(),
        })?),
        None => None,
    };

    if geojson.get("type").and_then(Value::as_str) == Some("FeatureCollection") {
        let mut out = geojson.clone();
        if let Some(feats) = out.get_mut("features").and_then(Value::as_array_mut) {
            for feat in feats.iter_mut() {
                let stats = compute_zarr_feature_stats(
                    &store,
                    &sel_var,
                    &geo,
                    band_idx,
                    feat,
                    coord_crs_model.as_ref(),
                    q,
                )
                .await?;
                if !feat.get("properties").is_some_and(Value::is_object) {
                    feat["properties"] = json!({});
                }
                feat["properties"]["statistics"] = stats;
            }
        }
        return Ok(json_ok(&out, "application/geo+json"));
    }

    let stats = compute_zarr_feature_stats(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &geojson,
        coord_crs_model.as_ref(),
        q,
    )
    .await?;

    if geojson.get("type").and_then(Value::as_str) == Some("Feature") {
        let mut out = geojson;
        if !out.get("properties").is_some_and(Value::is_object) {
            out["properties"] = json!({});
        }
        out["properties"]["statistics"] = stats;
        Ok(json_ok(&out, "application/geo+json"))
    } else {
        let out = json!({
            "type": "Feature",
            "geometry": geojson,
            "properties": { "statistics": stats },
        });
        Ok(json_ok(&out, "application/geo+json"))
    }
}

async fn compute_zarr_feature_stats(
    store: &ServerZarrStore,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    geo: &ZarrGeo,
    band_idx: Option<u64>,
    geojson: &Value,
    coord_crs: Option<&CrsModel>,
    q: &ZarrQuery,
) -> Result<Value, ZarrError> {
    let (mask, _bbox) =
        PolygonMask::from_geojson_in_crs(geojson, coord_crs).map_err(|e| ZarrError::InvalidSelection {
            detail: e.to_string(),
        })?;
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as usize;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as usize;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    // Densify polygon rings directly into Zarr native pixel space
    let is_geo = geo.crs.model.is_geographic();
    let mut px_polys: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
    let mut min_c = f64::INFINITY;
    let mut max_c = f64::NEG_INFINITY;
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for poly in &mask.polygons {
        let mut rings = Vec::with_capacity(poly.len());
        for ring in poly {
            let densified = densify_ring_lonlat_to_pixel(
                ring,
                &geo.crs.model,
                &crs_to_pixel,
                None,
                is_geo,
                !is_geo,
                0.0,
                ring.len(),
            );
            for &(c, r) in &densified {
                if c.is_finite() && r.is_finite() {
                    min_c = min_c.min(c);
                    max_c = max_c.max(c);
                    min_r = min_r.min(r);
                    max_r = max_r.max(r);
                }
            }
            rings.push(densified);
        }
        px_polys.push(rings);
    }

    if !min_c.is_finite() || max_c < 0.0 || max_r < 0.0 || min_c >= ifd_w as f64 || min_r >= ifd_h as f64 {
        return Err(ZarrError::NotFound {
            key: "GeoJSON feature does not intersect Zarr dataset bounds".into(),
        });
    }

    let c0 = (min_c.floor().max(0.0) as usize).min(ifd_w.saturating_sub(1));
    let r0 = (min_r.floor().max(0.0) as usize).min(ifd_h.saturating_sub(1));
    let c1 = (max_c.ceil().max(0.0) as usize).min(ifd_w);
    let r1 = (max_r.ceil().max(0.0) as usize).min(ifd_h);
    let win_w = c1.saturating_sub(c0).max(1);
    let win_h = r1.saturating_sub(r0).max(1);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        c0 as u32,
        r0 as u32,
        (c0 + win_w) as u32,
        (r0 + win_h) as u32,
        16,
        &caps,
    )
    .await?;
    record_window_metrics(&wb);

    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mut pixels = Vec::with_capacity(win_w * win_h);
    for r in r0..r0 + win_h {
        for c in c0..c0 + win_w {
            pixels.push(src.sample_checked(c as i32, r as i32));
        }
    }

    let cov = rasterize_exact_coverage(&px_polys, &[], &[], c0, r0, win_w, win_h);
    let area_model = PixelAreaModel::build(&geo.crs.model, &geo.affine, c0, r0, win_w, win_h);
    let opts = StatsOptions::default();
    let b1 = compute_weighted_band_stats_with_geom(
        &pixels,
        &cov,
        win_w,
        win_h,
        &area_model,
        true,
        ZonalGeomKind::Polygon,
        None,
        None,
        &sel_var.name,
        &opts,
    );
    Ok(json!({ "b1": b1 }))
}

pub(super) async fn handle_post_feature(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    body: Bytes,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let geojson: Value = serde_json::from_slice(&body)?;
    let coord_crs_model = match q.coord_crs {
        Some(code) => Some(CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{code}"),
            detail: e.to_string(),
        })?),
        None => None,
    };
    let (mask, wgs_bbox) =
        PolygonMask::from_geojson_in_crs(&geojson, coord_crs_model.as_ref()).map_err(|e| {
            ZarrError::InvalidSelection {
                detail: e.to_string(),
            }
        })?;

    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (out_w, out_h) = dims.unwrap_or((256, 256));
    let dst_crs = CrsModel::Geographic;
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    let [min_x, min_y, max_x, max_y] = wgs_bbox;
    let step_x = (max_x - min_x) / (out_w as f64);
    let step_y = (max_y - min_y) / (out_h as f64);
    let n_px = (out_w as usize) * (out_h as usize);
    let mut u_coords = vec![f32::NAN; n_px];
    let mut v_coords = vec![f32::NAN; n_px];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    for py in 0..out_h as usize {
        let lat = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * (out_w as usize);
        for px in 0..out_w as usize {
            let lon = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = geo.crs.model.from_lonlat_deg(lon, lat);
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u_coords[row_off + px] = uu;
                v_coords[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }
    if !any_in {
        return Err(ZarrError::NotFound {
            key: "GeoJSON feature does not intersect Zarr dataset bounds".into(),
        });
    }

    let pad = 2u32;
    let wx0 = (min_u.floor().max(0.0) as u32).saturating_sub(pad);
    let wy0 = (min_v.floor().max(0.0) as u32).saturating_sub(pad);
    let wx1 = ((max_u.ceil().max(0.0) as u32) + 1 + pad).min(ifd_w);
    let wy1 = ((max_v.ceil().max(0.0) as u32) + 1 + pad).min(ifd_h);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        wx0,
        wy0,
        wx1,
        wy1,
        16,
        &caps,
    )
    .await?;
    record_window_metrics(&wb);
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mode = q.resample_mode();
    let poly_mask = mask.rasterize_all_touched(out_w as usize, out_h as usize, wgs_bbox, &dst_crs);
    let mut plane = vec![f32::NAN; n_px];
    for i in 0..n_px {
        if poly_mask[i] && u_coords[i].is_finite() && v_coords[i].is_finite() {
            plane[i] = sample_general(&src, u_coords[i], v_coords[i], mode);
        }
    }

    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let encoded = encode_zarr_plane(&plane, out_w, out_h, wgs_bbox, 4326, q, fmt);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static(content_type))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(encoded))
        .unwrap())
}
