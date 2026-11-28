//! Zarr metadata, dataset inspection, OGC Tilesets, TileJSON, WMTS, and Map HTML handlers.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use cogzilla_zarr::{
    ArrayMeta, CfTimeUnits, SampleType, OpenMode, ShardingConfig, ZarrError, ZarrGeo, ZarrVersion,
    parse_sel_params, read_coord_1d, resolve_geo, resolve_sel_indices, select_variable,
};
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CONTENT_TYPE, HeaderValue};
use hyper::{Response, StatusCode};
use serde_json::{Value, json};

use crate::stats::build_map_html;
use crate::tms::{TmsDef, lookup_tms};

use super::{ZarrQuery, json_ok, open_store_and_hierarchy};

fn dtype_str(dt: SampleType) -> &'static str {
    dt.numpy_name()
}

fn wgs84_bounds_for_geo(geo: &cogzilla_zarr::ZarrGeo, width: u32, height: u32) -> [f64; 4] {
    geo.crs.model.wgs84_bounds(&geo.affine, None, width, height)
}

pub(super) async fn handle_info(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery, as_geojson: bool) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;

    let (height, width, count) = if sel_var.meta.rank() == 2 {
        (sel_var.meta.shape[0] as u32, sel_var.meta.shape[1] as u32, 1u32)
    } else {
        let mut c = sel_var.meta.shape[sel_var.dims.extra_axis.unwrap_or(0)] as u32;
        if !q.sel.is_empty() {
            let specs = parse_sel_params(&q.sel)?;
            for spec in &specs {
                if sel_var.dims.extra_name.as_deref() == Some(spec.dim.as_str())
                    && let Some(coord_meta) = group.array(&spec.dim)
                {
                    let vals = read_coord_1d(&store, &group.child_prefix(&spec.dim), coord_meta).await?;
                    let cf = if q.decode_times {
                        CfTimeUnits::from_meta(coord_meta)?
                    } else {
                        None
                    };
                    let idxs = resolve_sel_indices(spec, &vals, cf.as_ref())?;
                    c = idxs.len() as u32;
                }
            }
        }
        (
            sel_var.meta.shape[sel_var.dims.y_axis] as u32,
            sel_var.meta.shape[sel_var.dims.x_axis] as u32,
            c,
        )
    };

    let wgs84 = wgs84_bounds_for_geo(&geo, width, height);
    let band_descriptions: Vec<(String, String)> = (1..=count)
        .map(|i| (format!("b{i}"), sel_var.name.clone()))
        .collect();
    let band_metadata: Vec<(String, Value)> = (1..=count)
        .map(|i| (format!("b{i}"), json!(sel_var.meta.attributes)))
        .collect();
    let colorinterp: Vec<&str> = (0..count).map(|_| "undefined").collect();

    let nodata_val = sel_var.meta.effective_nodata();
    let (nodata_type, nodata_json) = match nodata_val {
        None => ("None", Value::Null),
        Some(v) if v.is_nan() => ("Nodata", Value::String("nan".into())),
        Some(v) => ("Nodata", json!(v)),
    };

    let mut info = json!({
        "bounds": geo.bounds,
        "wgs84_bounds": wgs84,
        "crs": geo.crs.description,
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype_str(sel_var.meta.dtype),
        "driver": "Zarr",
        "nodata_type": nodata_type,
        "nodata_value": nodata_json,
        "band_metadata": band_metadata,
        "band_descriptions": band_descriptions,
        "colorinterp": colorinterp,
        "variable": sel_var.name,
        "group": q.group.as_deref().unwrap_or(""),
        "dimensions": sel_var.meta.dimension_names,
        "attributes": sel_var.meta.attributes,
    });

    if let Some(vmin) = sel_var.valid_min {
        info["valid_min"] = json!(vmin);
    }
    if let Some(vmax) = sel_var.valid_max {
        info["valid_max"] = json!(vmax);
    }

    if q.show_times
        && let Some(ref extra_dim) = sel_var.dims.extra_name
        && let Some(t_meta) = group.array(extra_dim)
    {
        let raw_vals = read_coord_1d(&store, &group.child_prefix(extra_dim), t_meta).await?;
        if q.decode_times
            && let Some(cf) = CfTimeUnits::from_meta(t_meta)?
        {
            let iso: Vec<String> = raw_vals.iter().map(|&v| cf.decode_to_iso(v)).collect();
            info["times"] = json!(iso);
        } else {
            info["times"] = json!(raw_vals);
        }
    }

    if as_geojson {
        let [minx, miny, maxx, maxy] = wgs84;
        let feature = json!({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny],
                ]],
            },
            "properties": info,
        });
        Ok(json_ok(&feature, "application/geo+json"))
    } else {
        Ok(json_ok(&info, "application/json"))
    }
}

pub(super) async fn handle_bounds(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let width = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let height = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, width, height);
    let body = json!({
        "bounds": wgs84,
        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
    });
    Ok(json_ok(&body, "application/json"))
}

pub(super) async fn handle_validate(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;

    let version = match hierarchy.version {
        ZarrVersion::V2 => 2,
        ZarrVersion::V3 => 3,
    };
    let open_mode = match hierarchy.open_mode {
        OpenMode::V3Consolidated => "v3_consolidated",
        OpenMode::V3Walked => "v3_walked",
        OpenMode::V3RootArray => "v3_root_array",
        OpenMode::V2Consolidated => "v2_consolidated",
        OpenMode::V2Walked => "v2_walked",
        OpenMode::V2RootArray => "v2_root_array",
    };
    let sharding = ShardingConfig::from_meta(sel_var.meta)?;
    let codecs: Vec<&str> = sel_var.meta.codecs.iter().map(|c| c.name.as_str()).collect();
    let geo_res = resolve_geo(&store, &group, sel_var.meta).await;
    let mut warnings = Vec::new();
    if !hierarchy.is_consolidated() {
        warnings.push("store has no consolidated metadata; opening requires walking groups");
    }

    let (valid, regular_grid, crs_desc, bounds_val, geo_error) = match geo_res {
        Ok(g) => (true, true, Some(g.crs.description), Some(g.bounds), None),
        Err(e) => (false, false, None, None, Some(e.to_string())),
    };

    let body = json!({
        "valid": valid,
        "version": version,
        "consolidated": hierarchy.is_consolidated(),
        "open_mode": open_mode,
        "group": q.group.as_deref().unwrap_or(""),
        "variable": sel_var.name,
        "shape": sel_var.meta.shape,
        "chunk_shape": sel_var.meta.chunk_shape,
        "inner_chunk_shape": sharding.as_ref().map(|s| &s.inner_chunk_shape),
        "sharded": sharding.is_some(),
        "dtype": dtype_str(sel_var.meta.dtype),
        "codecs": codecs,
        "regular_grid": regular_grid,
        "crs": crs_desc,
        "bounds": bounds_val,
        "geo_error": geo_error,
        "warnings": warnings,
    });
    Ok(json_ok(&body, "application/json"))
}

pub(super) async fn handle_dataset_keys(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let keys = group.data_variables();
    Ok(json_ok(&json!(keys), "application/json"))
}

pub(super) async fn handle_dataset_dimensions(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    Ok(json_ok(&json!(group.dimensions()), "application/json"))
}

pub(super) async fn handle_dataset_dict(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (_store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let dims = group.dimensions();
    let data_vars_set: std::collections::BTreeSet<String> =
        group.data_variables().into_iter().collect();

    let mut coords = BTreeMap::new();
    let mut data_vars = BTreeMap::new();
    for (name, meta) in group.arrays() {
        let entry = array_summary_json(meta);
        if data_vars_set.contains(&name) {
            data_vars.insert(name, entry);
        } else {
            coords.insert(name, entry);
        }
    }

    let body = json!({
        "dims": dims,
        "attrs": group.attributes(),
        "coords": coords,
        "data_vars": data_vars,
        "groups": group.subgroups(),
    });
    Ok(json_ok(&body, "application/json"))
}

fn array_summary_json(meta: &ArrayMeta) -> Value {
    json!({
        "dims": meta.dimension_names,
        "shape": meta.shape,
        "chunks": meta.chunk_shape,
        "dtype": dtype_str(meta.dtype),
        "attrs": meta.attributes,
    })
}

pub(super) async fn handle_dataset_coordinate(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    coord_name: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let group = hierarchy.group(q.group.as_deref())?;
    let meta = group
        .array(coord_name)
        .ok_or_else(|| ZarrError::VariableNotFound {
            requested: coord_name.to_string(),
            available: group.variable_names(),
        })?;
    let raw_vals = read_coord_1d(&store, &group.child_prefix(coord_name), meta).await?;
    let data_val = if q.decode_times
        && let Some(cf) = CfTimeUnits::from_meta(meta)?
    {
        let iso: Vec<String> = raw_vals.iter().map(|&v| cf.decode_to_iso(v)).collect();
        json!(iso)
    } else {
        json!(raw_vals)
    };
    let body = json!({
        "name": coord_name,
        "dims": meta.dimension_names,
        "dtype": dtype_str(meta.dtype),
        "attrs": meta.attributes,
        "data": data_val,
    });
    Ok(json_ok(&body, "application/json"))
}

pub(super) fn require_tms(tms_id: &str) -> Result<&'static TmsDef, ZarrError> {
    lookup_tms(tms_id).ok_or_else(|| ZarrError::InvalidSelection {
        detail: format!("unknown TileMatrixSet {tms_id:?}"),
    })
}

fn zoom_range_for_zarr(tms: &TmsDef, geo: &ZarrGeo, width: u32, height: u32) -> (u8, u8) {
    tms.zoom_range_for_cog(&geo.crs.model, &geo.affine, width, height, 1.0)
}

pub(super) async fn handle_tileset_list(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tilesets: Vec<Value> = crate::tms::ALL_TMS
        .iter()
        .map(|tms| {
            json!({
                "title": format!("Zarr tileset ({})", tms.id),
                "dataType": "map",
                "crs": tms.crs_uri,
                "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms.id),
                "boundingBox": {
                    "lowerLeft": [wgs84[0], wgs84[1]],
                    "upperRight": [wgs84[2], wgs84[3]],
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
                },
                "links": [
                    {
                        "href": format!("{host}/zarr/tiles/{}{qs}", tms.id),
                        "rel": "self",
                        "type": "application/json",
                    },
                    {
                        "href": format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id),
                        "rel": "item",
                        "templated": true,
                    },
                ],
            })
        })
        .collect();
    Ok(json_ok(&json!({ "tilesets": tilesets }), "application/json"))
}

pub(super) async fn handle_tileset(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let limits: Vec<Value> = (minz..=maxz)
        .filter_map(|z| {
            let m = tms.matrix(z)?;
            let (min_col, max_col, min_row, max_row) = tms.tile_limits_for_wgs84_bounds(m, wgs84);
            Some(json!({
                "tileMatrix": m.id,
                "minTileRow": min_row,
                "maxTileRow": max_row,
                "minTileCol": min_col,
                "maxTileCol": max_col,
            }))
        })
        .collect();
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let body = json!({
        "title": format!("Zarr tileset ({})", tms.id),
        "dataType": "map",
        "crs": tms.crs_uri,
        "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms.id),
        "tileMatrixSetLimits": limits,
        "boundingBox": {
            "lowerLeft": [wgs84[0], wgs84[1]],
            "upperRight": [wgs84[2], wgs84[3]],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
        "links": [
            {
                "href": format!("{host}/zarr/tiles/{}{qs}", tms.id),
                "rel": "self",
                "type": "application/json",
            },
            {
                "href": format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id),
                "rel": "item",
                "templated": true,
            },
        ],
    });
    Ok(json_ok(&body, "application/json"))
}

pub(super) async fn handle_tilejson(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tile_url = format!("{host}/zarr/tiles/{}/{{z}}/{{x}}/{{y}}{qs}", tms.id);
    let body = json!({
        "tilejson": "3.0.0",
        "name": sel_var.name,
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [tile_url],
        "minzoom": minz,
        "maxzoom": maxz,
        "bounds": wgs84,
        "center": [0.5 * (wgs84[0] + wgs84[2]), 0.5 * (wgs84[1] + wgs84[3]), minz],
    });
    Ok(json_ok(&body, "application/json"))
}

pub(super) fn handle_map_html(tms_id: &str, host: &str, raw_query: &str) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let qs = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tj_url = format!("{host}/zarr/{}/tilejson.json{qs}", tms.id);
    let html = build_map_html(&tj_url);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(html)))
        .unwrap())
}

pub(super) async fn handle_wmts(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    tms_id: &str,
    host: &str,
    raw_query: &str,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let wgs84 = wgs84_bounds_for_geo(&geo, w, h);
    let (minz, maxz) = zoom_range_for_zarr(tms, &geo, w, h);
    let qs_esc = raw_query.replace('&', "&amp;");
    let tile_href = format!(
        "{host}/zarr/tiles/{}/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.png?{qs_esc}",
        tms.id
    );
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1" version="1.0.0">
  <Contents>
    <Layer>
      <ows:Title>{title}</ows:Title>
      <ows:Identifier>{title}</ows:Identifier>
      <ows:WGS84BoundingBox>
        <ows:LowerCorner>{minx} {miny}</ows:LowerCorner>
        <ows:UpperCorner>{maxx} {maxy}</ows:UpperCorner>
      </ows:WGS84BoundingBox>
      <Format>image/png</Format>
      <TileMatrixSetLink><TileMatrixSet>{tms_id}</TileMatrixSet></TileMatrixSetLink>
      <ResourceURL format="image/png" resourceType="tile" template="{tile_href}"/>
    </Layer>
    <TileMatrixSet>
      <ows:Identifier>{tms_id}</ows:Identifier>
      <ows:SupportedCRS>{crs_uri}</ows:SupportedCRS>
      <!-- minzoom={minz} maxzoom={maxz} -->
    </TileMatrixSet>
  </Contents>
</Capabilities>"#,
        title = sel_var.name,
        minx = wgs84[0],
        miny = wgs84[1],
        maxx = wgs84[2],
        maxy = wgs84[3],
        tms_id = tms.id,
        crs_uri = tms.crs_uri,
    );
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static("application/xml"))
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(Bytes::from(xml)))
        .unwrap())
}
