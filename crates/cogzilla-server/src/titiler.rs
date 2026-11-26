use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_io::OpenedCog;
use cogzilla_kernel::jpeg_fast::encode_jpeg_rgba;
use cogzilla_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row, filter_sub_rgba_row};
use cogzilla_kernel::webp_fast::{encode_webp_rgb, encode_webp_rgba};
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapName, CustomColormap, OutputFormat, ResampleMode,
    fast_db_from_power,
};
use cogzilla_tiff::PlanarConfig;

use crate::engine::{BandRequest, EngineError, TileEngine};

fn infer_colorinterp(full: &cogzilla_tiff::IfdMeta, has_color_map: bool) -> Vec<&'static str> {
    let count = full.samples_per_pixel as usize;
    if has_color_map || full.photometric == 3 {
        return vec!["palette"; count];
    }
    let alpha_idx = cogzilla_codec::alpha_channel_index(count, &full.extra_samples);
    match full.photometric {
        // RGB (2) or YCbCr (6)
        2 | 6 if count >= 3 => (0..count)
            .map(|i| match i {
                0 => "red",
                1 => "green",
                2 => "blue",
                _ if Some(i) == alpha_idx => "alpha",
                _ => "undefined",
            })
            .collect(),
        // MinIsWhite (0) or MinIsBlack (1)
        0 | 1 => (0..count)
            .map(|i| {
                if Some(i) == alpha_idx {
                    "alpha"
                } else if i == 0 {
                    "gray"
                } else {
                    "undefined"
                }
            })
            .collect(),
        _ => match count {
            3 => vec!["red", "green", "blue"],
            4 => vec!["red", "green", "blue", "alpha"],
            _ => vec!["gray"; count],
        },
    }
}

/// TiTiler-compatible `/cog/info` JSON payload built directly from our `CogIndex`.
pub fn build_cog_info(engine: &TileEngine, cog: &OpenedCog) -> Result<serde_json::Value, EngineError> {
    let full = cog.index.full_res();
    let count = full.samples_per_pixel as usize;
    let band_metadata: Vec<_> = (0..count)
        .map(|i| {
            let mut m = serde_json::Map::new();
            if let Some(tags) = cog.index.band_metadata.get(i) {
                for (k, v) in tags {
                    m.insert(k.clone(), serde_json::json!(v));
                }
            }
            serde_json::json!([format!("b{}", i + 1), serde_json::Value::Object(m)])
        })
        .collect();
    let band_descriptions: Vec<_> = (0..count)
        .map(|i| {
            let bname = format!("b{}", i + 1);
            let desc = cog
                .index
                .band_descriptions
                .get(i)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| bname.clone());
            serde_json::json!([bname, desc])
        })
        .collect();
    let colorinterp = infer_colorinterp(full, cog.index.color_map.is_some());
    let overviews: Vec<u32> = cog
        .index
        .ifds
        .iter()
        .skip(1)
        .map(|ifd| (full.width as f64 / ifd.width.max(1) as f64).round() as u32)
        .collect();

    let has_alpha = cogzilla_codec::alpha_channel_index(count, &full.extra_samples).is_some();
    let (nodata_type, nodata_value) = match cog.index.nodata {
        Some(v) if v.is_nan() => ("Nodata", Some(serde_json::Value::Null)),
        Some(v) => ("Nodata", Some(serde_json::json!(v))),
        None if has_alpha => ("Alpha", None),
        None if full.mask_ifd.is_some() => ("Mask", None),
        None => ("None", None),
    };

    let scales: Vec<f64> = (0..count)
        .map(|i| cog.index.scales_f64.get(i).copied().unwrap_or(1.0))
        .collect();
    let offsets: Vec<f64> = (0..count)
        .map(|i| cog.index.offsets_f64.get(i).copied().unwrap_or(0.0))
        .collect();

    let mut info = if !cog.index.is_georeferenced {
        serde_json::json!({
            "bounds": serde_json::Value::Null,
            "crs": serde_json::Value::Null,
            "wgs84_bounds": serde_json::Value::Null,
            "minzoom": serde_json::Value::Null,
            "maxzoom": serde_json::Value::Null,
            "band_metadata": band_metadata,
            "band_descriptions": band_descriptions,
            "dtype": full.sample_type.gdal_name(),
            "nodata_type": nodata_type,
            "colorinterp": colorinterp,
            "scales": scales,
            "offsets": offsets,
            "driver": "GTiff",
            "count": count,
            "width": full.width,
            "height": full.height,
            "overviews": overviews,
        })
    } else {
        let crs = engine.get_or_build_crs(cog)?;
        let bounds = crs.wgs84_bounds_with_gcp(
            &full.pixel_to_crs,
            full.gcp_poly.as_deref(),
            full.width,
            full.height,
        );
        let coarsest = cog.index.ifds.last().unwrap();
        let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
            .max(full.height as f64 / coarsest.height.max(1) as f64);
        let (minzoom, maxzoom) =
            crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);

        let gk = &cog.index.geo_keys;
        let std_proj_epsg = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767);
        let is_user_defined_projected = std_proj_epsg.is_none()
            && (gk.model_type == Some(1)
                || gk.projected_cs_type == Some(32767)
                || gk.proj_coord_trans.is_some());
        let crs_uri = if let Some(epsg) = std_proj_epsg {
            format!("http://www.opengis.net/def/crs/EPSG/0/{epsg}")
        } else if !is_user_defined_projected
            && let Some(epsg) = gk.geographic_type.filter(|&c| c > 0 && c != 32767)
        {
            format!("http://www.opengis.net/def/crs/EPSG/0/{epsg}")
        } else {
            "http://www.opengis.net/def/crs/EPSG/0/4326".to_string()
        };

        let (x00, y00) = full.pixel_to_crs_pt(0.0, 0.0);
        let (x10, y10) = full.pixel_to_crs_pt(full.width as f64, 0.0);
        let (x01, y01) = full.pixel_to_crs_pt(0.0, full.height as f64);
        let (x11, y11) = full.pixel_to_crs_pt(full.width as f64, full.height as f64);
        let native_bounds = [
            x00.min(x10).min(x01).min(x11),
            y00.min(y10).min(y01).min(y11),
            x00.max(x10).max(x01).max(x11),
            y00.max(y10).max(y01).max(y11),
        ];
        let reported_bounds = if is_user_defined_projected {
            bounds
        } else {
            native_bounds
        };

        serde_json::json!({
            "bounds": reported_bounds,
            "crs": crs_uri,
            "wgs84_bounds": bounds,
            "minzoom": minzoom,
            "maxzoom": maxzoom,
            "band_metadata": band_metadata,
            "band_descriptions": band_descriptions,
            "dtype": full.sample_type.gdal_name(),
            "nodata_type": nodata_type,
            "colorinterp": colorinterp,
            "scales": scales,
            "offsets": offsets,
            "driver": "GTiff",
            "count": count,
            "width": full.width,
            "height": full.height,
            "overviews": overviews,
        })
    };
    if let Some(res) = cog.index.gcp_residual_px {
        info["gcp_residual_px"] = serde_json::json!(res);
    }
    if let Some(nv) = nodata_value {
        info["nodata_value"] = nv;
    }
    if let Some(ref pal) = cog.index.color_map {
        let nd_idx: Option<usize> = cog
            .index
            .nodata
            .filter(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0)
            .map(|v| v as usize);
        let mut cmap = serde_json::Map::with_capacity(pal.len());
        for (i, &[r, g, b, a]) in pal.iter().enumerate() {
            let eff_a = if nd_idx == Some(i) { 0 } else { a };
            cmap.insert(i.to_string(), serde_json::json!([r, g, b, eff_a]));
        }
        info["colormap"] = serde_json::Value::Object(cmap);
    }
    Ok(info)
}

/// TiTiler-compatible `/cog/info.geojson` Feature wrapping the raster footprint polygon.
pub fn build_cog_info_geojson(
    engine: &TileEngine,
    cog: &OpenedCog,
) -> Result<serde_json::Value, EngineError> {
    let _ = engine.get_or_build_crs(cog)?;
    let info = build_cog_info(engine, cog)?;
    let b = info["wgs84_bounds"]
        .as_array()
        .ok_or_else(|| EngineError::BadRequest("invalid wgs84_bounds".into()))?;
    let (Some(w), Some(s), Some(e), Some(n)) = (
        b.first().and_then(|v| v.as_f64()),
        b.get(1).and_then(|v| v.as_f64()),
        b.get(2).and_then(|v| v.as_f64()),
        b.get(3).and_then(|v| v.as_f64()),
    ) else {
        return Err(EngineError::BadRequest(
            "dataset has non-finite wgs84 bounds".into(),
        ));
    };
    Ok(serde_json::json!({
        "type": "Feature",
        "bbox": [w, s, e, n],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
        },
        "properties": info,
    }))
}

/// TiTiler / `rio-cogeo`-compatible `/cog/validate` JSON payload.
pub fn build_cog_validate(
    engine: &TileEngine,
    cog: &OpenedCog,
    url: &str,
    strict: bool,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    if full.is_stripped && (full.width > 512 || full.height > 512) {
        errors.push(
            "The file is not tiled (strip-organized); COGs must use tiled IFD layout".into(),
        );
    }
    if (full.width > 512 || full.height > 512) && cog.index.ifds.len() <= 1 {
        warnings.push(
            "The file is greater than 512x512 pixels, it is recommended to include internal overviews"
                .into(),
        );
    }
    for (idx, ifd) in cog.index.ifds.iter().enumerate().skip(1) {
        if ifd.is_stripped && (ifd.width > 512 || ifd.height > 512) {
            errors.push(format!("Overview IFD {idx} is not tiled"));
        }
    }
    // rio-cogeo COG block-offset order check: coarser overviews must be stored before finer levels.
    let first_block_offset = |ifd: &cogzilla_tiff::IfdMeta| -> Option<u64> {
        ifd.tile_offsets.iter().copied().find(|&o| o > 0)
    };
    if cog.index.ifds.len() > 1 {
        if let (Some(main_off), Some(ov1_off)) = (
            first_block_offset(&cog.index.ifds[0]),
            first_block_offset(&cog.index.ifds[1]),
        ) && main_off < ov1_off
        {
            errors.push(
                "The offset of the first block of the main resolution image should be after the one of the overview of index 1"
                    .into(),
            );
        }
        for idx in 1..(cog.index.ifds.len() - 1) {
            if let (Some(finer_off), Some(coarser_off)) = (
                first_block_offset(&cog.index.ifds[idx]),
                first_block_offset(&cog.index.ifds[idx + 1]),
            ) && finer_off < coarser_off
            {
                errors.push(format!(
                    "The offset of the first block of overview of index {idx} should be after the one of the overview of index {}",
                    idx + 1
                ));
            }
        }
    }

    let is_cog = errors.is_empty() && (!strict || warnings.is_empty());

    let compression = match full.compression {
        cogzilla_tiff::Compression::None => None,
        cogzilla_tiff::Compression::Lzw => Some("LZW"),
        cogzilla_tiff::Compression::Jpeg => Some("JPEG"),
        cogzilla_tiff::Compression::Deflate => Some("DEFLATE"),
        cogzilla_tiff::Compression::PackBits => Some("PACKBITS"),
        cogzilla_tiff::Compression::Zstd => Some("ZSTD"),
        cogzilla_tiff::Compression::Webp => Some("WEBP"),
        cogzilla_tiff::Compression::Lerc => Some("LERC"),
        cogzilla_tiff::Compression::LercDeflate => Some("LERC_DEFLATE"),
        cogzilla_tiff::Compression::LercZstd => Some("LERC_ZSTD"),
    };
    let color_space = match full.photometric {
        0 => "MinIsWhite",
        1 => "MinIsBlack",
        2 => "RGB",
        3 => "Palette",
        5 => "CMYK",
        6 => "YCbCr",
        _ => "Unknown",
    };
    let interleave = match full.planar_config {
        PlanarConfig::Chunky => "pixel",
        PlanarConfig::Planar => "band",
    };
    let count = full.samples_per_pixel as usize;
    let has_alpha = cogzilla_codec::alpha_channel_index(count, &full.extra_samples).is_some();
    let colorinterp = infer_colorinterp(full, cog.index.color_map.is_some());
    let scales: Vec<f64> = (0..count)
        .map(|i| cog.index.scales_f64.get(i).copied().unwrap_or(1.0))
        .collect();
    let offsets: Vec<f64> = (0..count)
        .map(|i| cog.index.offsets_f64.get(i).copied().unwrap_or(0.0))
        .collect();
    let nodata_json = match cog.index.nodata {
        Some(v) if v.is_nan() => serde_json::json!("nan"),
        Some(v) => serde_json::json!(v),
        None => serde_json::Value::Null,
    };

    let gk = &cog.index.geo_keys;
    let std_proj_epsg = gk.projected_cs_type.filter(|&c| c > 0 && c != 32767);
    let is_user_defined_projected = std_proj_epsg.is_none()
        && (gk.model_type == Some(1)
            || gk.projected_cs_type == Some(32767)
            || gk.proj_coord_trans.is_some());
    let crs_str = if let Some(epsg) = std_proj_epsg {
        format!("EPSG:{epsg}")
    } else if !is_user_defined_projected
        && let Some(epsg) = gk.geographic_type.filter(|&c| c > 0 && c != 32767)
    {
        format!("EPSG:{epsg}")
    } else {
        "EPSG:4326".to_string()
    };

    let (x00, y00) = full.pixel_to_crs.forward(0.0, 0.0);
    let (x10, y10) = full.pixel_to_crs.forward(full.width as f64, 0.0);
    let (x01, y01) = full.pixel_to_crs.forward(0.0, full.height as f64);
    let (x11, y11) = full
        .pixel_to_crs
        .forward(full.width as f64, full.height as f64);
    let native_bounds = if is_user_defined_projected {
        cogzilla_proj::CrsModel::from_geo_keys(gk)
            .map(|c| c.wgs84_bounds(&full.pixel_to_crs, full.width, full.height))
            .unwrap_or([
                x00.min(x10).min(x01).min(x11),
                y00.min(y10).min(y01).min(y11),
                x00.max(x10).max(x01).max(x11),
                y00.max(y10).max(y01).max(y11),
            ])
    } else {
        [
            x00.min(x10).min(x01).min(x11),
            y00.min(y10).min(y01).min(y11),
            x00.max(x10).max(x01).max(x11),
            y00.max(y10).max(y01).max(y11),
        ]
    };
    let (res_x, res_y) = full.pixel_resolution();

    let ifds: Vec<serde_json::Value> = cog
        .index
        .ifds
        .iter()
        .enumerate()
        .map(|(level, ifd)| {
            let decimation = if level == 0 {
                0
            } else {
                (full.width as f64 / ifd.width.max(1) as f64).round() as u32
            };
            serde_json::json!({
                "Level": level,
                "Width": ifd.width,
                "Height": ifd.height,
                "Blocksize": [ifd.tile_width, ifd.tile_height],
                "Decimation": decimation,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "Path": url,
        "Driver": "GTiff",
        "COG": is_cog,
        "Compression": compression,
        "ColorSpace": color_space,
        "COG_errors": errors,
        "COG_warnings": warnings,
        "Profile": {
            "Bands": count,
            "Width": full.width,
            "Height": full.height,
            "Tiled": !full.is_stripped,
            "Dtype": full.sample_type.gdal_name(),
            "Interleave": interleave,
            "AlphaBand": has_alpha,
            "InternalMask": full.mask_ifd.is_some(),
            "Nodata": nodata_json,
            "ColorInterp": colorinterp,
            "ColorMap": cog.index.color_map.is_some(),
            "Scales": scales,
            "Offsets": offsets,
        },
        "GEO": {
            "CRS": crs_str,
            "BoundingBox": native_bounds,
            "Origin": [full.pixel_to_crs.c, full.pixel_to_crs.f],
            "Resolution": [res_x, res_y],
            "MinZoom": minzoom,
            "MaxZoom": maxzoom,
        },
        "Tags": {},
        "IFD": ifds,
    }))
}

const SUPPORTED_ALGORITHMS: &[&str] = &[
    "hillshade",
    "slope",
    "aspect",
    "terrarium",
    "terrainrgb",
    "normalizedIndex",
    "cast",
    "ceil",
    "floor",
    "min",
    "max",
    "median",
    "mean",
    "std",
    "var",
    "sum",
    "grayscale",
    "bitonal",
    "contours",
];

/// TiTiler 2.4 `AlgorithmFactory` `GET /algorithms` response.
pub fn list_algorithms_json(host: &str) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let algorithms: Vec<serde_json::Value> = SUPPORTED_ALGORITHMS
        .iter()
        .map(|&id| {
            serde_json::json!({
                "id": id,
                "links": [
                    {
                        "href": format!("{origin}/algorithms/{id}"),
                        "rel": "item",
                        "type": "application/json",
                        "title": format!("Definition of {id} Algorithm"),
                    }
                ]
            })
        })
        .collect();
    serde_json::json!({ "algorithms": algorithms })
}

/// TiTiler 2.4 `AlgorithmFactory` `GET /algorithms/{algorithmId}` response.
pub fn get_algorithm_json(id: &str) -> Result<serde_json::Value, EngineError> {
    let meta = match id {
        "hillshade" => serde_json::json!({
            "title": "Hillshade",
            "description": "Create hillshade from DEM dataset.",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 1, "dtype": "uint8", "min": null, "max": null },
            "parameters": {
                "azimuth": { "default": 45.0, "maximum": 360.0, "minimum": 0.0, "title": "Azimuth", "type": "number" },
                "angle_altitude": { "default": 45.0, "maximum": 90.0, "minimum": -90.0, "title": "Angle Altitude", "type": "number" },
                "buffer": { "default": 3, "maximum": 99, "minimum": 0, "title": "Buffer", "type": "integer" },
                "z_exaggeration": { "default": 1.0, "maximum": 1000000.0, "minimum": 1e-6, "title": "Z Exaggeration", "type": "number" }
            }
        }),
        "slope" => serde_json::json!({
            "title": "Slope",
            "description": "Calculate degrees of slope from DEM dataset.",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 1, "dtype": "float32", "min": [0.0], "max": [90.0] },
            "parameters": {
                "buffer": { "default": 0, "description": "Buffer size for edge effects", "maximum": 99, "minimum": 0, "title": "Buffer", "type": "integer" },
                "z_exaggeration": { "default": 1.0, "maximum": 1000000.0, "minimum": 1e-6, "title": "Z Exaggeration", "type": "number" }
            }
        }),
        "aspect" => serde_json::json!({
            "title": "Aspect",
            "description": "Calculate compass aspect in degrees (0-360) from DEM dataset.",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 1, "dtype": "float32", "min": [0.0], "max": [360.0] },
            "parameters": {
                "buffer": { "default": 0, "description": "Buffer size for edge effects", "maximum": 99, "minimum": 0, "title": "Buffer", "type": "integer" },
                "z_exaggeration": { "default": 1.0, "maximum": 1000000.0, "minimum": 1e-6, "title": "Z Exaggeration", "type": "number" }
            }
        }),
        "terrarium" => serde_json::json!({
            "title": "Terrarium",
            "description": "Encode DEM into RGB (Mapzen Terrarium).",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 3, "dtype": "uint8", "min": null, "max": null },
            "parameters": {
                "nodata_height": { "anyOf": [{ "maximum": 99999.0, "minimum": -99999.0, "type": "number" }, { "type": "null" }], "default": null, "title": "Nodata Height" }
            }
        }),
        "terrainrgb" => serde_json::json!({
            "title": "TerrainRGB",
            "description": "Encode DEM into RGB (Mapbox Terrain RGB).",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 3, "dtype": "uint8", "min": null, "max": null },
            "parameters": {
                "interval": { "default": 0.1, "maximum": 1.0, "minimum": 0.0, "title": "Interval", "type": "number" },
                "baseval": { "default": -10000.0, "maximum": 99999.0, "minimum": -99999.0, "title": "Baseval", "type": "number" },
                "nodata_height": { "anyOf": [{ "maximum": 99999.0, "minimum": -99999.0, "type": "number" }, { "type": "null" }], "default": null, "title": "Nodata Height" }
            }
        }),
        "normalizedIndex" | "normalizedindex" => serde_json::json!({
            "title": "Normalized Difference Index",
            "description": "Compute normalized difference index from two bands.",
            "inputs": { "nbands": 2 },
            "outputs": { "nbands": 1, "dtype": "float32", "min": [-1.0], "max": [1.0] },
            "parameters": {}
        }),
        "cast" => serde_json::json!({
            "title": "Cast data to Integer",
            "description": "Cast data to Integer.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": null, "dtype": "uint8", "min": [0], "max": [255] },
            "parameters": {}
        }),
        "ceil" => serde_json::json!({
            "title": "Round data to the smallest integer",
            "description": "Round data to the smallest integer.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": null, "dtype": "uint8", "min": [0], "max": [255] },
            "parameters": {}
        }),
        "floor" => serde_json::json!({
            "title": "Round data to the largest integer",
            "description": "Round data to the largest integer.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": null, "dtype": "uint8", "min": [0], "max": [255] },
            "parameters": {}
        }),
        "min" => serde_json::json!({
            "title": "Minimum",
            "description": "Return the Minimum along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": null, "min": null, "max": null },
            "parameters": {}
        }),
        "max" => serde_json::json!({
            "title": "Maximum",
            "description": "Return the Maximum along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": null, "min": null, "max": null },
            "parameters": {}
        }),
        "median" => serde_json::json!({
            "title": "Median",
            "description": "Return the Median along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": "float64", "min": null, "max": null },
            "parameters": {}
        }),
        "mean" => serde_json::json!({
            "title": "Mean",
            "description": "Return the Mean along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": "float64", "min": null, "max": null },
            "parameters": {}
        }),
        "std" => serde_json::json!({
            "title": "Standard deviation",
            "description": "Return the Standard Deviation along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": "float64", "min": null, "max": null },
            "parameters": {}
        }),
        "var" => serde_json::json!({
            "title": "Variance",
            "description": "Return Variance along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": "float64", "min": null, "max": null },
            "parameters": {}
        }),
        "sum" => serde_json::json!({
            "title": "Sum",
            "description": "Return the Sum along the `bands` axis.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": null, "min": null, "max": null },
            "parameters": {}
        }),
        "grayscale" => serde_json::json!({
            "title": "Transform a RGB Image to Grayscale",
            "description": "Transform a RGB Image to Grayscale using the ITU-R 601-2 luma.",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": null, "min": null, "max": null },
            "parameters": {}
        }),
        "bitonal" => serde_json::json!({
            "title": "Transform an Image to Bitonal",
            "description": "All values larger than 127 are set to 255 (white), all other values to 0 (black).",
            "inputs": { "nbands": null },
            "outputs": { "nbands": 1, "dtype": "uint8", "min": null, "max": null },
            "parameters": {}
        }),
        "contours" => serde_json::json!({
            "title": "Contours",
            "description": "Create contours from DEM dataset.",
            "inputs": { "nbands": 1 },
            "outputs": { "nbands": 3, "dtype": "uint8", "min": null, "max": null },
            "parameters": {
                "increment": { "default": 35, "maximum": 999, "minimum": 0, "title": "Increment", "type": "integer" },
                "thickness": { "default": 1, "maximum": 10, "minimum": 0, "title": "Thickness", "type": "integer" },
                "minz": { "default": -12000, "maximum": 99999, "minimum": -99999, "title": "Minz", "type": "integer" },
                "maxz": { "default": 8000, "maximum": 99999, "minimum": -99999, "title": "Maxz", "type": "integer" }
            }
        }),
        _ => {
            return Err(EngineError::NotFound(format!(
                "unknown algorithm '{id}' (supported: {})",
                SUPPORTED_ALGORITHMS.join(", ")
            )));
        }
    };
    Ok(meta)
}

/// Escape XML text and attribute special characters.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Build an OGC WMTS 1.0.0 `WMTSCapabilities.xml` document for `/cog`, `/stac`, or `/mosaicjson`
/// advertising a single TileMatrixSet.
#[allow(clippy::too_many_arguments)]
pub fn build_wmts_capabilities(
    host: &str,
    family: &str,
    tms: &str,
    title: &str,
    bounds: [f64; 4],
    minzoom: u8,
    maxzoom: u8,
    tile_format: &str,
    tile_scale: u32,
    tile_query: &str,
    capabilities_query: &str,
    use_epsg: bool,
) -> String {
    build_wmts_capabilities_multi(
        host,
        family,
        &[(tms, minzoom, maxzoom)],
        Some(tms),
        title,
        bounds,
        tile_format,
        tile_scale,
        tile_query,
        capabilities_query,
        use_epsg,
    )
}

/// Build an OGC WMTS 1.0.0 `WMTSCapabilities.xml` document advertising one or more TileMatrixSets.
/// When `explicit_tms` is `None` (root `/{family}/WMTSCapabilities.xml`), advertises a `<Layer>`
/// and `<TileMatrixSet>` for every entry in `tms_entries` (all 13 supported TMSs).
#[allow(clippy::too_many_arguments)]
pub fn build_wmts_capabilities_multi(
    host: &str,
    family: &str,
    tms_entries: &[(&str, u8, u8)],
    explicit_tms: Option<&str>,
    title: &str,
    bounds: [f64; 4],
    tile_format: &str,
    tile_scale: u32,
    tile_query: &str,
    capabilities_query: &str,
    use_epsg: bool,
) -> String {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let scale_str = if tile_scale > 1 {
        format!("@{tile_scale}x")
    } else {
        String::new()
    };
    let media_type = match tile_format {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "tif" | "tiff" => "image/tiff; application=geotiff",
        "npy" => "application/x-binary",
        "f32" | "bin" | "raw" => "application/octet-stream",
        _ => "image/png",
    };
    let q_tile = if tile_query.is_empty() {
        String::new()
    } else {
        format!("?{tile_query}")
    };
    let q_cap = if capabilities_query.is_empty() {
        String::new()
    } else {
        format!("?{capabilities_query}")
    };
    let capabilities_url = if let Some(tms) = explicit_tms {
        xml_escape(&format!("{origin}/{family}/{tms}/WMTSCapabilities.xml{q_cap}"))
    } else {
        xml_escape(&format!("{origin}/{family}/WMTSCapabilities.xml{q_cap}"))
    };
    let tile_base = if let Some(tms) = explicit_tms {
        xml_escape(&format!("{origin}/{family}/tiles/{tms}"))
    } else {
        xml_escape(&format!("{origin}/{family}/tiles"))
    };
    let title_esc = xml_escape(title);
    let multi = explicit_tms.is_none() && tms_entries.len() > 1;

    let mut layers_xml = String::new();
    let mut tms_sets_xml = String::new();
    let scale = tile_scale.clamp(1, 4);

    for &(tms, minzoom, maxzoom) in tms_entries {
        let tms_esc = xml_escape(tms);
        let tms_def = crate::tms::lookup_tms(tms)
            .unwrap_or_else(|| crate::tms::lookup_tms("WebMercatorQuad").unwrap());
        let crs_urn = if use_epsg {
            format!("EPSG:{}", tms_def.epsg)
        } else {
            tms_def.crs_uri.to_string()
        };
        let max_tms_z = tms_def
            .matrices
            .iter()
            .filter_map(|m| m.id.parse::<u8>().ok())
            .max()
            .unwrap_or(0);
        let min_tms_z = tms_def
            .matrices
            .iter()
            .filter_map(|m| m.id.parse::<u8>().ok())
            .min()
            .unwrap_or(0);
        let eff_minz = minzoom.clamp(min_tms_z, max_tms_z);
        let eff_maxz = maxzoom.clamp(eff_minz, max_tms_z);

        let mut matrices_xml = String::new();
        let mut limits_xml = String::new();
        for m in tms_def.matrices {
            if let Ok(z) = m.id.parse::<u8>()
                && (eff_minz..=eff_maxz).contains(&z)
            {
                let (tl0, tl1) = if tms_def.invert_axis {
                    (m.origin_y, m.origin_x)
                } else {
                    (m.origin_x, m.origin_y)
                };
                let tw = m.tile_width * scale;
                let th = m.tile_height * scale;
                let scale_denom = m.scale_denominator / (scale as f64);
                matrices_xml.push_str(&format!(
                    "            <TileMatrix>\n\
                     \x20               <ows:Identifier>{}</ows:Identifier>\n\
                     \x20               <ScaleDenominator>{scale_denom:.6}</ScaleDenominator>\n\
                     \x20               <TopLeftCorner>{tl0} {tl1}</TopLeftCorner>\n\
                     \x20               <TileWidth>{tw}</TileWidth>\n\
                     \x20               <TileHeight>{th}</TileHeight>\n\
                     \x20               <MatrixWidth>{}</MatrixWidth>\n\
                     \x20               <MatrixHeight>{}</MatrixHeight>\n\
                     \x20           </TileMatrix>\n",
                    m.id, m.matrix_width, m.matrix_height
                ));
                let (min_col, max_col, min_row, max_row) =
                    tms_def.tile_limits_for_wgs84_bounds(m, bounds);
                limits_xml.push_str(&format!(
                    "                    <TileMatrixLimits>\n\
                     \x20                       <TileMatrix>{}</TileMatrix>\n\
                     \x20                       <MinTileRow>{min_row}</MinTileRow>\n\
                     \x20                       <MaxTileRow>{max_row}</MaxTileRow>\n\
                     \x20                       <MinTileCol>{min_col}</MinTileCol>\n\
                     \x20                       <MaxTileCol>{max_col}</MaxTileCol>\n\
                     \x20                   </TileMatrixLimits>\n",
                    m.id
                ));
            }
        }

        let tile_template = xml_escape(&format!(
            "{origin}/{family}/tiles/{tms}/{{TileMatrix}}/{{TileCol}}/{{TileRow}}{scale_str}.{tile_format}{q_tile}"
        ));
        let layer_title = if multi {
            format!("{title_esc}_{tms_esc}")
        } else {
            title_esc.clone()
        };
        let layer_id = if multi {
            tms_esc.as_str()
        } else {
            "cogeo"
        };
        layers_xml.push_str(&format!(
            "        <Layer>\n\
             \x20           <ows:Title>{layer_title}</ows:Title>\n\
             \x20           <ows:Identifier>{layer_id}</ows:Identifier>\n\
             \x20           <ows:WGS84BoundingBox crs=\"urn:ogc:def:crs:OGC:2:84\">\n\
             \x20               <ows:LowerCorner>{} {}</ows:LowerCorner>\n\
             \x20               <ows:UpperCorner>{} {}</ows:UpperCorner>\n\
             \x20           </ows:WGS84BoundingBox>\n\
             \x20           <Style isDefault=\"true\">\n\
             \x20               <ows:Identifier>default</ows:Identifier>\n\
             \x20           </Style>\n\
             \x20           <Format>{media_type}</Format>\n\
             \x20           <TileMatrixSetLink>\n\
             \x20               <TileMatrixSet>{tms_esc}</TileMatrixSet>\n\
             \x20               <TileMatrixSetLimits>\n\
             {limits_xml}\x20               </TileMatrixSetLimits>\n\
             \x20           </TileMatrixSetLink>\n\
             \x20           <ResourceURL format=\"{media_type}\" resourceType=\"tile\" template=\"{tile_template}\"/>\n\
             \x20       </Layer>\n",
            bounds[0], bounds[1], bounds[2], bounds[3]
        ));
        tms_sets_xml.push_str(&format!(
            "        <TileMatrixSet>\n\
             \x20           <ows:Title>{tms_esc}</ows:Title>\n\
             \x20           <ows:Identifier>{tms_esc}</ows:Identifier>\n\
             \x20           <ows:SupportedCRS>{crs_urn}</ows:SupportedCRS>\n\
             {matrices_xml}        </TileMatrixSet>\n"
        ));
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Capabilities xmlns="http://www.opengis.net/wmts/1.0"
    xmlns:ows="http://www.opengis.net/ows/1.1"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:gml="http://www.opengis.net/gml"
    xsi:schemaLocation="http://www.opengis.net/wmts/1.0 http://schemas.opengis.net/wmts/1.0/wmtsGetCapabilities_response.xsd"
    version="1.0.0">
    <ows:ServiceIdentification>
        <ows:Title>{title_esc}</ows:Title>
        <ows:ServiceType>OGC WMTS</ows:ServiceType>
        <ows:ServiceTypeVersion>1.0.0</ows:ServiceTypeVersion>
    </ows:ServiceIdentification>
    <ows:OperationsMetadata>
        <ows:Operation name="GetCapabilities">
            <ows:DCP>
                <ows:HTTP>
                    <ows:Get xlink:href="{capabilities_url}">
                        <ows:Constraint name="GetEncoding">
                            <ows:AllowedValues>
                                <ows:Value>RESTful</ows:Value>
                            </ows:AllowedValues>
                        </ows:Constraint>
                    </ows:Get>
                </ows:HTTP>
            </ows:DCP>
        </ows:Operation>
        <ows:Operation name="GetTile">
            <ows:DCP>
                <ows:HTTP>
                    <ows:Get xlink:href="{tile_base}">
                        <ows:Constraint name="GetEncoding">
                            <ows:AllowedValues>
                                <ows:Value>RESTful</ows:Value>
                            </ows:AllowedValues>
                        </ows:Constraint>
                    </ows:Get>
                </ows:HTTP>
            </ows:DCP>
        </ows:Operation>
    </ows:OperationsMetadata>
    <Contents>
{layers_xml}{tms_sets_xml}    </Contents>
    <ServiceMetadataURL xlink:href="{capabilities_url}"/>
</Capabilities>
"#
    )
}

/// TiTiler-compatible TileJSON 3.0.0 (`/cog/{tms}/tilejson.json`).
pub fn build_tilejson(
    engine: &TileEngine,
    cog: &OpenedCog,
    host: &str,
    tms: &str,
    raw_query: &str,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);
    let (minzoom, maxzoom) = if let Some(def) = crate::tms::lookup_tms(tms) {
        def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
    } else if tms == "WorldCRS84Quad" {
        crs.world_crs84_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
    } else {
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
    };
    let center = serde_json::json!([
        0.5 * (bounds[0] + bounds[2]),
        0.5 * (bounds[1] + bounds[3]),
        minzoom,
    ]);
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let tile_url = if raw_query.is_empty() {
        format!("{origin}/cog/tiles/{tms}/{{z}}/{{x}}/{{y}}")
    } else {
        format!("{origin}/cog/tiles/{tms}/{{z}}/{{x}}/{{y}}?{raw_query}")
    };
    let count = full.samples_per_pixel as usize;
    let band_descriptions: Vec<_> = (0..count)
        .map(|i| {
            let bname = format!("b{}", i + 1);
            let desc = cog
                .index
                .band_descriptions
                .get(i)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| bname.clone());
            serde_json::json!([bname, desc])
        })
        .collect();
    Ok(serde_json::json!({
        "tilejson": "3.0.0",
        "name": cog.uri,
        "version": "1.0.0",
        "scheme": "xyz",
        "tiles": [tile_url],
        "minzoom": minzoom,
        "maxzoom": maxzoom,
        "bounds": bounds,
        "center": center,
        "band_descriptions": band_descriptions,
        "data_type": full.sample_type.gdal_name(),
    }))
}

/// TiTiler-compatible `/cog/point/{lon},{lat}` exact point query (fetches only 1 internal tile!).
pub async fn query_point(
    engine: &TileEngine,
    cog: &OpenedCog,
    lon: f64,
    lat: f64,
    unscale: bool,
    nodata_override: Option<f32>,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let ifd = cog.index.full_res();
    let q_lon = if let Some(lo) = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width)
        && lon < lo
    {
        lon + 360.0
    } else {
        lon
    };
    let (sx, sy) = crs.from_lonlat_deg(q_lon, lat);
    let (col_f, row_f) = ifd.crs_to_pixel_pt(sx, sy);
    let col = col_f.floor() as i64;
    let row = row_f.floor() as i64;

    let count = ifd.samples_per_pixel as usize;
    let alpha_idx = cogzilla_codec::alpha_channel_index(count, &ifd.extra_samples);
    let data_bands: Vec<usize> = (0..count).filter(|&b| Some(b) != alpha_idx).collect();
    let band_names: Vec<String> = data_bands.iter().map(|&b| format!("b{}", b + 1)).collect();

    if !col_f.is_finite()
        || !row_f.is_finite()
        || col < 0
        || row < 0
        || col >= ifd.width as i64
        || row >= ifd.height as i64
    {
        return Err(EngineError::NotFound(format!(
            "point ({lon}, {lat}) is outside dataset bounds"
        )));
    }
    let tx = (col as u32) / ifd.tile_width;
    let ty = (row as u32) / ifd.tile_height;
    let lx = (col as u32) % ifd.tile_width;
    let ly = (row as u32) % ifd.tile_height;
    let pidx = (ly as usize) * (ifd.tile_width as usize) + (lx as usize);

    let maybe_unscale = |b: usize, v: f32| -> f32 {
        if let Some(nd) = nodata_override {
            if !nd.is_nan() && v == nd {
                return f32::NAN;
            }
        }
        if unscale && !v.is_nan() {
            let sc = cog.index.scales.get(b).copied().unwrap_or(1.0);
            let off = cog.index.offsets.get(b).copied().unwrap_or(0.0);
            v * sc + off
        } else {
            v
        }
    };

    let mut values: Vec<serde_json::Value> = Vec::with_capacity(data_bands.len());
    match ifd.planar_config {
        PlanarConfig::Chunky => {
            let idx = ifd.tile_index(tx, ty, 0) as u32;
            let tiles = engine.store.get_blocks(cog, 0, &[Some(idx)]).await?;
            for &b in &data_bands {
                let v = maybe_unscale(
                    b,
                    tiles[0]
                        .as_ref()
                        .and_then(|t| t.bands.get(b))
                        .map(|band| band.data.get_f32(pidx))
                        .unwrap_or(f32::NAN),
                );
                values.push(if v.is_nan() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(v)
                });
            }
        }
        PlanarConfig::Planar => {
            let indices: Vec<Option<u32>> = data_bands
                .iter()
                .map(|&b| Some(ifd.tile_index(tx, ty, b as u16) as u32))
                .collect();
            let tiles = engine.store.get_blocks(cog, 0, &indices).await?;
            for (slot, &b) in data_bands.iter().enumerate() {
                let v = maybe_unscale(
                    b,
                    tiles[slot]
                        .as_ref()
                        .and_then(|t| t.bands.first())
                        .map(|band| band.data.get_f32(pidx))
                        .unwrap_or(f32::NAN),
                );
                values.push(if v.is_nan() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(v)
                });
            }
        }
    }

    Ok(serde_json::json!({
        "coordinates": [lon, lat],
        "values": values,
        "band_names": band_names,
        "band_descriptions": band_names,
    }))
}

/// Per-band statistics accumulator: uses exact O(1)-memory counting histograms for `U8`/`U16`
/// bands (zero sort, zero per-pixel enum dispatch) and O(N) Quickselect for `F32` bands.
enum BandStatsAcc {
    HistU8([u64; 256]),
    HistU16(Vec<u64>),
    Floats(Vec<f32>),
}

impl BandStatsAcc {
    fn for_sample_type(st: cogzilla_tiff::SampleType, capacity_hint: usize) -> Self {
        match st {
            cogzilla_tiff::SampleType::Uint8 => Self::HistU8([0u64; 256]),
            cogzilla_tiff::SampleType::Uint16 => Self::HistU16(vec![0u64; 65_536]),
            _ => Self::Floats(Vec::with_capacity(capacity_hint)),
        }
    }

    /// Accumulate an unmasked tile region `[0..x_max) x [0..y_max)` directly from `BandData`.
    #[allow(clippy::too_many_arguments)]
    fn push_tile_unmasked(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        use cogzilla_codec::BandData;
        match (self, data) {
            (Self::HistU8(hist), BandData::U8 { data: slice, nodata }) => {
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            hist[v as usize] += 1;
                        }
                    }
                }
            }
            (
                Self::HistU8(hist),
                BandData::RgbU8Channel {
                    rgb,
                    channel,
                    nodata,
                },
            ) => {
                let ch = *channel as usize;
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let base = ly * tw * 3 + ch;
                        for lx in 0..x_max {
                            let v = rgb[base + lx * 3];
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let base = ly * tw * 3 + ch;
                        for lx in 0..x_max {
                            hist[rgb[base + lx * 3] as usize] += 1;
                        }
                    }
                }
            }
            (Self::HistU16(hist), BandData::U16 { data: slice, nodata }) => {
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            hist[v as usize] += 1;
                        }
                    }
                }
            }
            (Self::Floats(vals), BandData::F32(slab)) => {
                let slice = slab.as_slice();
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                if unscale {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() && nd_ov != Some(v) {
                                vals.push(v * scale + offset);
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() && nd_ov != Some(v) {
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            // Fallback if tile storage variant differs from IFD sample type
            (Self::Floats(vals), other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let mut v = other.get_f32(base + lx);
                        if v.is_finite() && nd_ov != Some(v) {
                            if unscale {
                                v = v * scale + offset;
                            }
                            vals.push(v);
                        }
                    }
                }
            }
            (Self::HistU8(hist), other) => {
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let v = other.get_f32(base + lx);
                        if v.is_finite() {
                            hist[(v.clamp(0.0, 255.0)) as usize] += 1;
                        }
                    }
                }
            }
            (Self::HistU16(hist), other) => {
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let v = other.get_f32(base + lx);
                        if v.is_finite() {
                            hist[(v.clamp(0.0, 65535.0)) as usize] += 1;
                        }
                    }
                }
            }
        }
    }

    /// Accumulate masked pixels from a tile where `mask_bits[ly * x_max + lx] != 0`.
    #[allow(clippy::too_many_arguments)]
    fn push_tile_masked(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        mask_bits: &[u8],
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        use cogzilla_codec::BandData;
        match (self, data) {
            (Self::HistU8(hist), BandData::U8 { data: slice, nodata }) => {
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = row[lx];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (
                Self::HistU8(hist),
                BandData::RgbU8Channel {
                    rgb,
                    channel,
                    nodata,
                },
            ) => {
                let ch = *channel as usize;
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw * 3 + ch;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = rgb[base + lx * 3];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (Self::HistU16(hist), BandData::U16 { data: slice, nodata }) => {
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = row[lx];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (Self::Floats(vals), BandData::F32(slab)) => {
                let slice = slab.as_slice();
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = row[lx];
                            if v.is_finite() && nd_ov != Some(v) {
                                if unscale {
                                    v = v * scale + offset;
                                }
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            (acc, other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = other.get_f32(base + lx);
                            if v.is_finite() && nd_ov != Some(v) {
                                match acc {
                                    Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                                    Self::HistU16(hist) => {
                                        hist[(v.clamp(0.0, 65535.0)) as usize] += 1
                                    }
                                    Self::Floats(vals) => {
                                        if unscale {
                                            v = v * scale + offset;
                                        }
                                        vals.push(v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Accumulate strided pixels (`lx0, lx0+step, ...` × `ly0, ly0+step, ...`) when an overview-less
    /// IFD exceeds `max_size`. `mask_bits`, when present, is compact (`nx * ny` bytes matching the
    /// strided grid).
    #[allow(clippy::too_many_arguments)]
    fn push_tile_strided(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        lx0: usize,
        ly0: usize,
        step: usize,
        mask_bits: Option<&[u8]>,
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        let nd_ov = nodata_override.filter(|v| !v.is_nan());
        let mut mi = 0usize;
        let mut ly = ly0;
        while ly < y_max {
            let base = ly * tw;
            let mut lx = lx0;
            while lx < x_max {
                let keep = match mask_bits {
                    Some(bits) => {
                        let k = bits.get(mi).copied().unwrap_or(0) != 0;
                        mi += 1;
                        k
                    }
                    None => true,
                };
                if keep {
                    let mut v = data.get_f32(base + lx);
                    if v.is_finite() && nd_ov != Some(v) {
                        match self {
                            Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                            Self::HistU16(hist) => hist[(v.clamp(0.0, 65535.0)) as usize] += 1,
                            Self::Floats(vals) => {
                                if unscale {
                                    v = v * scale + offset;
                                }
                                vals.push(v);
                            }
                        }
                    }
                }
                lx += step;
            }
            ly += step;
        }
    }

    fn push_tile_coords(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        xs: &[usize],
        ys: &[usize],
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        let nd_ov = nodata_override.filter(|v| !v.is_nan());
        for &ly in ys {
            let base = ly * tw;
            for &lx in xs {
                let mut v = data.get_f32(base + lx);
                if v.is_finite() && nd_ov != Some(v) {
                    match self {
                        Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                        Self::HistU16(hist) => hist[(v.clamp(0.0, 65535.0)) as usize] += 1,
                        Self::Floats(vals) => {
                            if unscale {
                                v = v * scale + offset;
                            }
                            vals.push(v);
                        }
                    }
                }
            }
        }
    }
}

/// Optional statistics customization matching TiTiler's `/statistics` query parameters.
#[derive(Debug, Clone)]
pub struct StatsOptions {
    pub percentiles: Vec<f64>,
    pub histogram_bins: usize,
    pub histogram_edges: Option<Vec<f64>>,
    pub histogram_range: Option<(f64, f64)>,
    pub categorical: bool,
    pub categories: Option<Vec<f64>>,
    pub bidx: Option<Vec<u16>>,
    pub expressions: Option<Vec<(String, cogzilla_kernel::ExprProgram)>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub max_size_explicit: bool,
    pub area_weighted: bool,
}

impl Default for StatsOptions {
    fn default() -> Self {
        Self {
            percentiles: vec![2.0, 98.0],
            histogram_bins: 10,
            histogram_edges: None,
            histogram_range: None,
            categorical: false,
            categories: None,
            bidx: None,
            expressions: None,
            width: None,
            height: None,
            max_size_explicit: false,
            area_weighted: true,
        }
    }
}

/// Bin a value `v` into sorted `edges` (`edges.len() >= 2`) using NumPy `np.histogram` semantics:
/// `[edges[i], edges[i+1])` for `i < n-1`, and `[edges[n-2], edges[n-1]]` inclusive on the last bin.
#[inline]
fn bin_into_edges(edges: &[f64], v: f64) -> Option<usize> {
    let n_bins = edges.len().checked_sub(1)?;
    if n_bins == 0 || v < edges[0] || v > edges[n_bins] {
        return None;
    }
    if v == edges[n_bins] {
        return Some(n_bins - 1);
    }
    let idx = edges.partition_point(|&e| e <= v);
    if idx == 0 {
        None
    } else {
        Some((idx - 1).min(n_bins - 1))
    }
}

fn build_categorical_histogram(
    observed: &[(f64, u64)],
    req_cats: &Option<Vec<f64>>,
) -> serde_json::Value {
    let val_json = |v: f64| -> serde_json::Value {
        if v.fract() == 0.0 && v >= (i64::MIN as f64) && v <= (i64::MAX as f64) {
            serde_json::json!(v as i64)
        } else {
            serde_json::json!(v)
        }
    };
    if let Some(req) = req_cats {
        let mut counts = Vec::with_capacity(req.len());
        let mut vals = Vec::with_capacity(req.len());
        for &c in req {
            let cnt = observed
                .iter()
                .find(|(v, _)| (v - c).abs() <= 1e-6)
                .map(|(_, n)| *n)
                .unwrap_or(0);
            counts.push(cnt);
            vals.push(val_json(c));
        }
        serde_json::json!([counts, vals])
    } else {
        let counts: Vec<u64> = observed.iter().map(|(_, n)| *n).collect();
        let vals: Vec<serde_json::Value> = observed.iter().map(|(v, _)| val_json(*v)).collect();
        serde_json::json!([counts, vals])
    }
}

/// Compute TiTiler-compatible band statistics from a dense `f32` slice where `NaN` marks masked pixels.
pub fn compute_f32_band_stats(
    pixels: &[f32],
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    let denom = pixels.len();
    let valid: Vec<f32> = pixels.iter().copied().filter(|v| v.is_finite()).collect();
    BandStatsAcc::Floats(valid).finalize(denom, false, 1.0, 0.0, None, description, opts)
}

/// GPU-accelerated variant of `compute_f32_band_stats`: when built with `--features gpu` and
/// above the adapter-calibrated pixel crossover (non-categorical, standard histogram bins),
/// runs the two-pass WGSL `GpuStatsReducer` (20×–51× faster than CPU `sort_unstable_by` on
/// 4096² rasters) and formats the identical TiTiler JSON object.
pub fn compute_f32_band_stats_with_engine(
    engine: &TileEngine,
    pixels: &[f32],
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    #[cfg(feature = "gpu")]
    {
        if !opts.categorical
            && opts.histogram_edges.is_none()
            && let Some(gpu) = engine.gpu.as_ref()
            && let Some(gs) = gpu.try_reduce_f32_stats(
                pixels,
                None,
                opts.histogram_bins.clamp(1, 1000),
                opts.histogram_range,
                &opts.percentiles,
            )
        {
            return gpu_band_stats_to_json(&gs, pixels.len(), description, opts);
        }
    }
    let _ = engine;
    compute_f32_band_stats(pixels, description, opts)
}

#[cfg(feature = "gpu")]
fn gpu_band_stats_to_json(
    gs: &cogzilla_gpu::GpuBandStats,
    denom_pixels: usize,
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    let masked_pixels = denom_pixels.saturating_sub(gs.valid_pixels);
    if gs.valid_pixels == 0 {
        return empty_band_stats_json(masked_pixels, description, opts);
    }
    let valid_percent = if denom_pixels > 0 {
        ((gs.valid_pixels as f64 / denom_pixels as f64) * 100.0 * 100.0).round() / 100.0
    } else {
        0.0
    };
    let mut obj = serde_json::Map::new();
    obj.insert("min".into(), serde_json::json!(gs.min));
    obj.insert("max".into(), serde_json::json!(gs.max));
    obj.insert("mean".into(), serde_json::json!(gs.mean));
    obj.insert("count".into(), serde_json::json!(gs.valid_pixels));
    obj.insert("sum".into(), serde_json::json!(gs.sum));
    obj.insert("std".into(), serde_json::json!(gs.std));
    obj.insert("median".into(), serde_json::json!(gs.median));
    obj.insert("majority".into(), serde_json::json!(gs.majority));
    obj.insert("minority".into(), serde_json::json!(gs.minority));
    obj.insert("unique".into(), serde_json::json!(gs.unique_bins));
    obj.insert("valid_percent".into(), serde_json::json!(valid_percent));
    obj.insert("masked_pixels".into(), serde_json::json!(masked_pixels));
    obj.insert("valid_pixels".into(), serde_json::json!(gs.valid_pixels));
    for &(p, val) in &gs.percentiles {
        obj.insert(percentile_key(p), serde_json::json!(val));
    }
    obj.insert(
        "histogram".into(),
        serde_json::json!([gs.hist_counts, gs.hist_edges]),
    );
    obj.insert("description".into(), serde_json::json!(description));
    serde_json::Value::Object(obj)
}

fn percentile_key(p: f64) -> String {
    if p.fract() == 0.0 {
        format!("percentile_{}", p as i64)
    } else {
        format!("percentile_{p}")
    }
}

fn format_category_key(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn matches_category_filter(v: f64, filter: &Option<Vec<f64>>) -> bool {
    match filter {
        None => true,
        Some(list) => list.iter().any(|&c| (c - v).abs() <= 1e-6),
    }
}

fn empty_band_stats_json(
    masked_px: usize,
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("min".into(), serde_json::Value::Null);
    obj.insert("max".into(), serde_json::Value::Null);
    obj.insert("mean".into(), serde_json::Value::Null);
    obj.insert("count".into(), serde_json::json!(0));
    obj.insert("sum".into(), serde_json::json!(0.0));
    obj.insert("std".into(), serde_json::Value::Null);
    obj.insert("median".into(), serde_json::Value::Null);
    obj.insert("majority".into(), serde_json::Value::Null);
    obj.insert("minority".into(), serde_json::Value::Null);
    obj.insert("unique".into(), serde_json::json!(0));
    obj.insert("valid_percent".into(), serde_json::json!(0.0));
    obj.insert("masked_pixels".into(), serde_json::json!(masked_px));
    obj.insert("valid_pixels".into(), serde_json::json!(0));
    for &p in &opts.percentiles {
        obj.insert(percentile_key(p), serde_json::Value::Null);
    }
    obj.insert("histogram".into(), serde_json::json!([[], []]));
    if opts.categorical {
        obj.insert("categories".into(), serde_json::json!({}));
    }
    obj.insert("description".into(), serde_json::json!(description));
    serde_json::Value::Object(obj)
}

impl BandStatsAcc {
    fn finalize_hist_with_opts(
        hist: &[u64],
        denom_pixels: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
        description: &str,
        opts: &StatsOptions,
    ) -> serde_json::Value {
        let valid_px: usize = hist.iter().map(|&c| c as usize).sum();
        let masked_px = denom_pixels.saturating_sub(valid_px);
        let valid_pct = if denom_pixels > 0 {
            (valid_px as f64 / denom_pixels as f64) * 100.0
        } else {
            0.0
        };
        if valid_px == 0 {
            return empty_band_stats_json(masked_px, description, opts);
        }
        let reverse = unscale && scale < 0.0;
        let to_val = |k: usize| -> f32 {
            if unscale {
                (k as f32) * scale + offset
            } else {
                k as f32
            }
        };
        let rank_med = valid_px / 2;
        let p_ranks: Vec<usize> = opts
            .percentiles
            .iter()
            .map(|&p| (((valid_px as f64) * (p.clamp(0.0, 100.0) / 100.0)) as usize).min(valid_px - 1))
            .collect();
        let mut p_vals: Vec<f32> = vec![f32::NAN; p_ranks.len()];

        let mut min = f32::NAN;
        let mut max = f32::NAN;
        let mut median = f32::NAN;
        let mut sum = 0.0f64;
        let mut cum = 0usize;
        let mut unique = 0usize;
        let mut maj_val = f32::NAN;
        let mut maj_cnt = 0usize;
        let mut min_val = f32::NAN;
        let mut min_cnt = usize::MAX;
        let mut categories = if opts.categorical {
            Some(serde_json::Map::new())
        } else {
            None
        };
        let mut cat_observed: Vec<(f64, u64)> = Vec::new();

        let n = hist.len();
        for step in 0..n {
            let k = if reverse { n - 1 - step } else { step };
            let c = hist[k] as usize;
            if c == 0 {
                continue;
            }
            unique += 1;
            let v = to_val(k);
            if let Some(ref mut cats) = categories {
                let vf = v as f64;
                cat_observed.push((vf, c as u64));
                if matches_category_filter(vf, &opts.categories) {
                    cats.insert(format_category_key(vf), serde_json::json!(c));
                }
            }
            if c > maj_cnt {
                maj_cnt = c;
                maj_val = v;
            }
            if c < min_cnt {
                min_cnt = c;
                min_val = v;
            }
            if min.is_nan() {
                min = v;
            }
            max = v;
            let prev = cum;
            cum += c;
            if prev <= rank_med && rank_med < cum {
                median = v;
            }
            for (idx, &rk) in p_ranks.iter().enumerate() {
                if prev <= rk && rk < cum {
                    p_vals[idx] = v;
                }
            }
            sum += (v as f64) * (c as f64);
        }

        let mean = sum / (valid_px as f64);
        let mut var_sum = 0.0f64;
        let (counts, edges) = if let Some(ref custom_edges) = opts.histogram_edges
            && custom_edges.len() >= 2
        {
            let mut counts = vec![0u64; custom_edges.len() - 1];
            for (k, &c) in hist.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let v = to_val(k) as f64;
                let d = v - mean;
                var_sum += d * d * (c as f64);
                if let Some(bin) = bin_into_edges(custom_edges, v) {
                    counts[bin] += c;
                }
            }
            (counts, custom_edges.clone())
        } else {
            let (h_min, h_max) = opts.histogram_range.unwrap_or((min as f64, max as f64));
            let span = (h_max - h_min).max(1e-12);
            let n_bins = opts.histogram_bins.clamp(1, 1000);
            let mut counts = vec![0u64; n_bins];
            for (k, &c) in hist.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let v = to_val(k) as f64;
                let d = v - mean;
                var_sum += d * d * (c as f64);
                if opts.histogram_range.is_none() || (v >= h_min && v <= h_max) {
                    let bin = (((v - h_min) / span) * (n_bins as f64))
                        .floor()
                        .clamp(0.0, (n_bins - 1) as f64) as usize;
                    counts[bin] += c;
                }
            }
            let edges: Vec<f64> = (0..=n_bins)
                .map(|i| h_min + span * (i as f64 / n_bins as f64))
                .collect();
            (counts, edges)
        };
        let std = (var_sum / (valid_px as f64)).sqrt();
        let mut obj = serde_json::Map::new();
        obj.insert("min".into(), serde_json::json!(min));
        obj.insert("max".into(), serde_json::json!(max));
        obj.insert("mean".into(), serde_json::json!(mean));
        obj.insert("count".into(), serde_json::json!(valid_px));
        obj.insert("sum".into(), serde_json::json!(sum));
        obj.insert("std".into(), serde_json::json!(std));
        obj.insert("median".into(), serde_json::json!(median));
        obj.insert("majority".into(), serde_json::json!(maj_val));
        obj.insert("minority".into(), serde_json::json!(min_val));
        obj.insert("unique".into(), serde_json::json!(unique));
        obj.insert("valid_percent".into(), serde_json::json!((valid_pct * 100.0).round() / 100.0));
        obj.insert("masked_pixels".into(), serde_json::json!(masked_px));
        obj.insert("valid_pixels".into(), serde_json::json!(valid_px));
        for (&p, &pv) in opts.percentiles.iter().zip(p_vals.iter()) {
            obj.insert(percentile_key(p), serde_json::json!(pv));
        }
        if let Some(cats) = categories {
            obj.insert(
                "histogram".into(),
                build_categorical_histogram(&cat_observed, &opts.categories),
            );
            obj.insert("categories".into(), serde_json::Value::Object(cats));
        } else {
            obj.insert("histogram".into(), serde_json::json!([counts, edges]));
        }
        obj.insert("description".into(), serde_json::json!(description));
        serde_json::Value::Object(obj)
    }

    fn finalize(
        self,
        denom_pixels: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
        description: &str,
        opts: &StatsOptions,
    ) -> serde_json::Value {
        match self {
            Self::HistU8(mut hist) => {
                if let Some(nd) = nodata_override {
                    if nd.is_finite() && nd >= 0.0 && nd <= 255.0 && nd.fract() == 0.0 {
                        hist[nd as usize] = 0;
                    }
                }
                Self::finalize_hist_with_opts(
                    &hist,
                    denom_pixels,
                    unscale,
                    scale,
                    offset,
                    description,
                    opts,
                )
            }
            Self::HistU16(mut hist) => {
                if let Some(nd) = nodata_override {
                    if nd.is_finite() && nd >= 0.0 && nd <= 65535.0 && nd.fract() == 0.0 {
                        hist[nd as usize] = 0;
                    }
                }
                Self::finalize_hist_with_opts(
                    &hist,
                    denom_pixels,
                    unscale,
                    scale,
                    offset,
                    description,
                    opts,
                )
            }
            Self::Floats(mut vals) => {
                let valid_px = vals.len();
                let masked_px = denom_pixels.saturating_sub(valid_px);
                let valid_pct = if denom_pixels > 0 {
                    (valid_px as f64 / denom_pixels as f64) * 100.0
                } else {
                    0.0
                };
                if valid_px == 0 {
                    return empty_band_stats_json(masked_px, description, opts);
                }
                vals.sort_unstable_by(|a, b| a.total_cmp(b));
                let min = vals[0];
                let max = vals[valid_px - 1];
                let mut sum = 0.0f64;
                let mut unique = 0usize;
                let mut maj_val = min;
                let mut maj_cnt = 0usize;
                let mut min_val = min;
                let mut min_cnt = usize::MAX;
                let mut run_val = min;
                let mut run_cnt = 0usize;
                let mut categories = if opts.categorical {
                    Some(serde_json::Map::new())
                } else {
                    None
                };
                let mut cat_observed: Vec<(f64, u64)> = Vec::new();
                for &v in &vals {
                    sum += v as f64;
                    if v == run_val {
                        run_cnt += 1;
                    } else {
                        unique += 1;
                        if let Some(ref mut cats) = categories {
                            let vf = run_val as f64;
                            cat_observed.push((vf, run_cnt as u64));
                            if matches_category_filter(vf, &opts.categories) {
                                cats.insert(format_category_key(vf), serde_json::json!(run_cnt));
                            }
                        }
                        if run_cnt > maj_cnt {
                            maj_cnt = run_cnt;
                            maj_val = run_val;
                        }
                        if run_cnt < min_cnt {
                            min_cnt = run_cnt;
                            min_val = run_val;
                        }
                        run_val = v;
                        run_cnt = 1;
                    }
                }
                unique += 1;
                if let Some(ref mut cats) = categories {
                    let vf = run_val as f64;
                    cat_observed.push((vf, run_cnt as u64));
                    if matches_category_filter(vf, &opts.categories) {
                        cats.insert(format_category_key(vf), serde_json::json!(run_cnt));
                    }
                }
                if run_cnt > maj_cnt {
                    maj_val = run_val;
                }
                if run_cnt < min_cnt {
                    min_val = run_val;
                }
                let mean = sum / (valid_px as f64);
                let mut var_sum = 0.0f64;
                let (counts, edges) = if let Some(ref custom_edges) = opts.histogram_edges
                    && custom_edges.len() >= 2
                {
                    let mut counts = vec![0u64; custom_edges.len() - 1];
                    for &v in &vals {
                        let vf = v as f64;
                        let d = vf - mean;
                        var_sum += d * d;
                        if let Some(bin) = bin_into_edges(custom_edges, vf) {
                            counts[bin] += 1;
                        }
                    }
                    (counts, custom_edges.clone())
                } else {
                    let (h_min, h_max) = opts.histogram_range.unwrap_or((min as f64, max as f64));
                    let span = (h_max - h_min).max(1e-12);
                    let n_bins = opts.histogram_bins.clamp(1, 1000);
                    let mut counts = vec![0u64; n_bins];
                    for &v in &vals {
                        let vf = v as f64;
                        let d = vf - mean;
                        var_sum += d * d;
                        if opts.histogram_range.is_none() || (vf >= h_min && vf <= h_max) {
                            let bin = (((vf - h_min) / span) * (n_bins as f64))
                                .floor()
                                .clamp(0.0, (n_bins - 1) as f64) as usize;
                            counts[bin] += 1;
                        }
                    }
                    let edges: Vec<f64> = (0..=n_bins)
                        .map(|i| h_min + span * (i as f64 / n_bins as f64))
                        .collect();
                    (counts, edges)
                };
                let std = (var_sum / (valid_px as f64)).sqrt();

                let med_idx = valid_px / 2;
                let median = vals[med_idx];
                let mut obj = serde_json::Map::new();
                obj.insert("min".into(), serde_json::json!(min));
                obj.insert("max".into(), serde_json::json!(max));
                obj.insert("mean".into(), serde_json::json!(mean));
                obj.insert("count".into(), serde_json::json!(valid_px));
                obj.insert("sum".into(), serde_json::json!(sum));
                obj.insert("std".into(), serde_json::json!(std));
                obj.insert("median".into(), serde_json::json!(median));
                obj.insert("majority".into(), serde_json::json!(maj_val));
                obj.insert("minority".into(), serde_json::json!(min_val));
                obj.insert("unique".into(), serde_json::json!(unique));
                obj.insert("valid_percent".into(), serde_json::json!((valid_pct * 100.0).round() / 100.0));
                obj.insert("masked_pixels".into(), serde_json::json!(masked_px));
                obj.insert("valid_pixels".into(), serde_json::json!(valid_px));
                for &p in &opts.percentiles {
                    let idx = (((valid_px as f64) * (p.clamp(0.0, 100.0) / 100.0)) as usize)
                        .min(valid_px - 1);
                    obj.insert(percentile_key(p), serde_json::json!(vals[idx]));
                }
                if let Some(cats) = categories {
                    obj.insert(
                        "histogram".into(),
                        build_categorical_histogram(&cat_observed, &opts.categories),
                    );
                    obj.insert("categories".into(), serde_json::Value::Object(cats));
                } else {
                    obj.insert("histogram".into(), serde_json::json!([counts, edges]));
                }
                obj.insert("description".into(), serde_json::json!(description));
                serde_json::Value::Object(obj)
            }
        }
    }
}

/// Zero-allocation 4×3 probe fast-reject before allocating per-tile polygon grids/masks.
/// Cuts OOB/small-polygon tile rejection from ~1,089 projections + heap Vecs down to 12 projections,
/// while sampling thirds in X so odd-width global 0..360 tiles never miss the Western Hemisphere.
fn tile_probe_outside_polygon(
    ifd: &cogzilla_tiff::IfdMeta,
    crs: &cogzilla_proj::CrsModel,
    pm: &PolygonMask,
    tx: u32,
    ty: u32,
    x_max: usize,
    y_max: usize,
) -> bool {
    let base_col = (tx * ifd.tile_width) as usize;
    let base_row = (ty * ifd.tile_height) as usize;
    let cols = [0usize, x_max / 3, (2 * x_max) / 3, x_max];
    let rows = [0usize, y_max / 2, y_max];
    let mut p_min_lon = f64::INFINITY;
    let mut p_max_lon = f64::NEG_INFINITY;
    let mut p_min_lat = f64::INFINITY;
    let mut p_max_lat = f64::NEG_INFINITY;
    let mut non_finite = false;
    let mut probe_antimeridian = false;
    for &ry in &rows {
        let row_f = (base_row + ry) as f64;
        let mut prev_lon: Option<f64> = None;
        for &cx in &cols {
            let col_f = (base_col + cx) as f64;
            let (sx, sy) = ifd.pixel_to_crs.forward(col_f, row_f);
            let (lon, lat) = crs.to_lonlat_deg(sx, sy);
            if !lon.is_finite() || !lat.is_finite() {
                non_finite = true;
                break;
            }
            if let Some(pl) = prev_lon
                && (lon - pl).abs() > 180.0
            {
                probe_antimeridian = true;
            }
            prev_lon = Some(lon);
            p_min_lon = p_min_lon.min(lon);
            p_max_lon = p_max_lon.max(lon);
            p_min_lat = p_min_lat.min(lat);
            p_max_lat = p_max_lat.max(lat);
        }
        if non_finite {
            break;
        }
    }
    if !non_finite && p_min_lat > -85.0 && p_max_lat < 85.0 {
        let lon_span = p_max_lon - p_min_lon;
        let lat_span = p_max_lat - p_min_lat;
        let lat_pad = (lat_span * 0.15).max(1e-4);
        if p_max_lat + lat_pad < pm.bbox[1] || p_min_lat - lat_pad > pm.bbox[3] {
            return true;
        }
        if !probe_antimeridian && lon_span < 120.0 {
            let lon_pad = (lon_span * 0.15).max(1e-4);
            if p_max_lon + lon_pad < pm.bbox[0] || p_min_lon - lon_pad > pm.bbox[2] {
                return true;
            }
        }
    }
    false
}

/// Build a 1-byte/px polygon mask for a single source tile using a coarse 16-px inverse CRS grid.
/// Returns `(mask_bits, count_inside)` or `None` if the tile's lon/lat bounds do not intersect `pm.bbox`.
fn build_tile_polygon_mask(
    ifd: &cogzilla_tiff::IfdMeta,
    crs: &cogzilla_proj::CrsModel,
    pm: &PolygonMask,
    tx: u32,
    ty: u32,
    x_max: usize,
    y_max: usize,
) -> Option<(Vec<u8>, usize)> {
    const STEP: usize = 16;
    let base_col = (tx * ifd.tile_width) as usize;
    let base_row = (ty * ifd.tile_height) as usize;

    if tile_probe_outside_polygon(ifd, crs, pm, tx, ty, x_max, y_max) {
        return None;
    }

    let gw = x_max.div_ceil(STEP) + 1;
    let gh = y_max.div_ceil(STEP) + 1;

    let mut grid_lon = vec![0.0f64; gw * gh];
    let mut grid_lat = vec![0.0f64; gw * gh];
    let mut t_min_lon = f64::INFINITY;
    let mut t_max_lon = f64::NEG_INFINITY;
    let mut t_min_lat = f64::INFINITY;
    let mut t_max_lat = f64::NEG_INFINITY;
    let mut has_non_finite = false;
    let mut crosses_antimeridian = false;

    for gy in 0..gh {
        let row_f = (base_row + (gy * STEP).min(y_max)) as f64;
        for gx in 0..gw {
            let col_f = (base_col + (gx * STEP).min(x_max)) as f64;
            let (sx, sy) = ifd.pixel_to_crs.forward(col_f, row_f);
            let (lon, lat) = crs.to_lonlat_deg(sx, sy);
            grid_lon[gy * gw + gx] = lon;
            grid_lat[gy * gw + gx] = lat;
            if !lon.is_finite() || !lat.is_finite() {
                has_non_finite = true;
                continue;
            }
            if gx > 0 {
                let prev = grid_lon[gy * gw + gx - 1];
                if prev.is_finite() && (lon - prev).abs() > 180.0 {
                    crosses_antimeridian = true;
                }
            }
            if gy > 0 {
                let above = grid_lon[(gy - 1) * gw + gx];
                if above.is_finite() && (lon - above).abs() > 180.0 {
                    crosses_antimeridian = true;
                }
            }
            t_min_lon = t_min_lon.min(lon);
            t_max_lon = t_max_lon.max(lon);
            t_min_lat = t_min_lat.min(lat);
            t_max_lat = t_max_lat.max(lat);
        }
    }
    if t_max_lon - t_min_lon > 180.0 {
        crosses_antimeridian = true;
    }

    // Fast tile-level reject against polygon bbox (skip lon reject across antimeridian jumps).
    if !has_non_finite {
        if t_max_lat < pm.bbox[1] || t_min_lat > pm.bbox[3] {
            return None;
        }
        if !crosses_antimeridian && (t_max_lon < pm.bbox[0] || t_min_lon > pm.bbox[2]) {
            return None;
        }
    }

    let mut bits = vec![0u8; x_max * y_max];
    let mut inside_count = 0usize;

    for gy in 0..(gh - 1) {
        let y0 = gy * STEP;
        let y1 = ((gy + 1) * STEP).min(y_max);
        if y0 >= y1 {
            continue;
        }
        let inv_dy = 1.0 / ((y1 - y0) as f64);
        for gx in 0..(gw - 1) {
            let x0 = gx * STEP;
            let x1 = ((gx + 1) * STEP).min(x_max);
            if x0 >= x1 {
                continue;
            }
            let i00 = gy * gw + gx;
            let lo00 = grid_lon[i00];
            let lo10 = grid_lon[i00 + 1];
            let lo01 = grid_lon[i00 + gw];
            let lo11 = grid_lon[i00 + gw + 1];
            let la00 = grid_lat[i00];
            let la10 = grid_lat[i00 + 1];
            let la01 = grid_lat[i00 + gw];
            let la11 = grid_lat[i00 + gw + 1];

            let all_finite = lo00.is_finite()
                && lo10.is_finite()
                && lo01.is_finite()
                && lo11.is_finite()
                && la00.is_finite()
                && la10.is_finite()
                && la01.is_finite()
                && la11.is_finite();
            let c_min_lo = lo00.min(lo10).min(lo01).min(lo11);
            let c_max_lo = lo00.max(lo10).max(lo01).max(lo11);
            let c_min_la = la00.min(la10).min(la01).min(la11);
            let c_max_la = la00.max(la10).max(la01).max(la11);
            // Fall back to exact per-pixel projection on Sinusoidal/LAEA boundary cells (non-finite
            // corners) or antimeridian/polar cells where linear longitude interpolation wraps across 0°.
            if !all_finite || (c_max_lo - c_min_lo) > 180.0 {
                let base_x = (tx * ifd.tile_width) as f64;
                let base_y = (ty * ifd.tile_height) as f64;
                for ly in y0..y1 {
                    let row_off = ly * x_max;
                    let py = base_y + ly as f64 + 0.5;
                    for lx in x0..x1 {
                        let px = base_x + lx as f64 + 0.5;
                        let (sx, sy) = ifd.pixel_to_crs.forward(px, py);
                        let (lon, lat) = crs.to_lonlat_deg(sx, sy);
                        if lon.is_finite() && lat.is_finite() && pm.contains(lon, lat) {
                            bits[row_off + lx] = 1;
                            inside_count += 1;
                        }
                    }
                }
                continue;
            }
            // Cell-level reject against polygon bbox
            if c_max_lo < pm.bbox[0]
                || c_min_lo > pm.bbox[2]
                || c_max_la < pm.bbox[1]
                || c_min_la > pm.bbox[3]
            {
                continue;
            }

            let inv_dx = 1.0 / ((x1 - x0) as f64);
            for ly in y0..y1 {
                let ty_f = (ly as f64 + 0.5 - y0 as f64) * inv_dy;
                let lo_l = lo00 + (lo01 - lo00) * ty_f;
                let lo_r = lo10 + (lo11 - lo10) * ty_f;
                let la_l = la00 + (la01 - la00) * ty_f;
                let la_r = la10 + (la11 - la10) * ty_f;
                let dlo = (lo_r - lo_l) * inv_dx;
                let dla = (la_r - la_l) * inv_dx;
                let mut lon = lo_l + 0.5 * dlo;
                let mut lat = la_l + 0.5 * dla;
                let row_off = ly * x_max;
                for lx in x0..x1 {
                    if pm.contains(lon, lat) {
                        bits[row_off + lx] = 1;
                        inside_count += 1;
                    }
                    lon += dlo;
                    lat += dla;
                }
            }
        }
    }

    if inside_count == 0 {
        None
    } else {
        Some((bits, inside_count))
    }
}

/// Build a compact polygon mask on the strided sample grid `(lx0..x_max).step_by(step)` ×
/// `(ly0..y_max).step_by(step)` for overview-less IFDs exceeding `max_size`.
#[allow(clippy::too_many_arguments)]
fn build_tile_polygon_mask_strided(
    ifd: &cogzilla_tiff::IfdMeta,
    crs: &cogzilla_proj::CrsModel,
    pm: &PolygonMask,
    tx: u32,
    ty: u32,
    x_max: usize,
    y_max: usize,
    lx0: usize,
    ly0: usize,
    step: usize,
) -> Option<(Vec<u8>, usize)> {
    if tile_probe_outside_polygon(ifd, crs, pm, tx, ty, x_max, y_max) {
        return None;
    }
    let nx = (x_max - lx0).div_ceil(step);
    let ny = (y_max - ly0).div_ceil(step);
    let base_x = (tx * ifd.tile_width) as f64;
    let base_y = (ty * ifd.tile_height) as f64;
    let mut bits = vec![0u8; nx * ny];
    let mut inside_count = 0usize;
    let mut idx = 0usize;
    let mut ly = ly0;
    while ly < y_max {
        let py = base_y + ly as f64 + 0.5;
        let mut lx = lx0;
        while lx < x_max {
            let px = base_x + lx as f64 + 0.5;
            let (sx, sy) = ifd.pixel_to_crs.forward(px, py);
            let (lon, lat) = crs.to_lonlat_deg(sx, sy);
            if lon.is_finite() && lat.is_finite() && pm.contains(lon, lat) {
                bits[idx] = 1;
                inside_count += 1;
            }
            idx += 1;
            lx += step;
        }
        ly += step;
    }
    if inside_count == 0 {
        None
    } else {
        Some((bits, inside_count))
    }
}

enum StatsTiles {
    Chunky {
        tiles: Vec<Option<Arc<cogzilla_codec::DecodedTile>>>,
        tile_masks: Option<Vec<Option<Vec<u8>>>>,
        mask_total_pixels: usize,
    },
    Planar {
        band_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>,
        tile_masks: Option<Vec<Option<Vec<u8>>>>,
        mask_total_pixels: usize,
    },
}

#[allow(clippy::too_many_arguments)]
fn compute_statistics_cpu(
    cog: &OpenedCog,
    ifd_idx: usize,
    _crs: &cogzilla_proj::CrsModel,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
    nodata_override: Option<f32>,
    step: usize,
    sample_dims: Option<(usize, usize)>,
    fetched: StatsTiles,
    opts: &StatsOptions,
    #[cfg(feature = "gpu")] gpu: Option<&crate::gpu_backend::GpuBackend>,
) -> serde_json::Value {
    let ifd = &cog.index.ifds[ifd_idx];
    let count = ifd.samples_per_pixel as usize;
    let total_pixels = if let Some((ow, oh)) = sample_dims {
        ow * oh
    } else if step > 1 {
        (ifd.width as usize).div_ceil(step) * (ifd.height as usize).div_ceil(step)
    } else {
        (ifd.width as usize) * (ifd.height as usize)
    };
    let mask_total_pixels = match &fetched {
        StatsTiles::Chunky {
            mask_total_pixels, ..
        }
        | StatsTiles::Planar {
            mask_total_pixels, ..
        } => *mask_total_pixels,
    };
    let capacity_hint = if poly_mask.is_some() {
        mask_total_pixels
    } else {
        total_pixels
    };

    // Precompute exact target grid local coordinates per tile column/row when sampling
    // unmasked full-image statistics at exact `(out_w, out_h)` (matching rio-tiler's max_size).
    let (tile_xs, tile_ys): (Option<Vec<Vec<usize>>>, Option<Vec<Vec<usize>>>) =
        if let Some((ow, oh)) = sample_dims {
            let mut xs_by_tx = vec![Vec::new(); ifd.tiles_across as usize];
            for px in 0..ow {
                let col = (((px as f64 + 0.5) * (ifd.width as f64) / (ow as f64)).floor() as u32)
                    .min(ifd.width - 1);
                let tx = (col / ifd.tile_width) as usize;
                let lx = (col % ifd.tile_width) as usize;
                xs_by_tx[tx].push(lx);
            }
            let mut ys_by_ty = vec![Vec::new(); ifd.tiles_down as usize];
            for py in 0..oh {
                let row = (((py as f64 + 0.5) * (ifd.height as f64) / (oh as f64)).floor() as u32)
                    .min(ifd.height - 1);
                let ty = (row / ifd.tile_height) as usize;
                let ly = (row % ifd.tile_height) as usize;
                ys_by_ty[ty].push(ly);
            }
            (Some(xs_by_tx), Some(ys_by_ty))
        } else {
            (None, None)
        };

    if let Some(ref exprs) = opts.expressions
        && !exprs.is_empty()
    {
        return compute_expression_statistics_cpu(
            cog,
            ifd_idx,
            ifd,
            poly_mask,
            unscale,
            nodata_override,
            step,
            &tile_xs,
            &tile_ys,
            total_pixels,
            mask_total_pixels,
            fetched,
            opts,
            exprs,
            #[cfg(feature = "gpu")]
            gpu,
        );
    }

    let mut per_band: Vec<BandStatsAcc> = (0..count)
        .map(|_| BandStatsAcc::for_sample_type(ifd.sample_type, capacity_hint))
        .collect();

    match fetched {
        StatsTiles::Chunky {
            tiles, tile_masks, ..
        } => {
            for (ti, tile_opt) in tiles.iter().enumerate() {
                let Some(tile) = tile_opt else { continue };
                let tx = (ti as u32) % ifd.tiles_across;
                let ty = (ti as u32) / ifd.tiles_across;
                let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                let tw = ifd.tile_width as usize;

                if let (Some(xs_all), Some(ys_all)) = (&tile_xs, &tile_ys) {
                    let xs = &xs_all[tx as usize];
                    let ys = &ys_all[ty as usize];
                    if xs.is_empty() || ys.is_empty() {
                        continue;
                    }
                    for b in 0..count.min(tile.bands.len()) {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
                        let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
                        per_band[b].push_tile_coords(
                            &tile.bands[b].data,
                            tw,
                            xs,
                            ys,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    }
                } else if step > 1 {
                    let x0_tile = (tx * ifd.tile_width) as usize;
                    let y0_tile = (ty * ifd.tile_height) as usize;
                    let lx0 = (step - (x0_tile % step)) % step;
                    let ly0 = (step - (y0_tile % step)) % step;
                    if lx0 >= x_max || ly0 >= y_max {
                        continue;
                    }
                    let mask_bits = match tile_masks {
                        Some(ref masks) => match masks[ti] {
                            Some(ref bits) => Some(bits.as_slice()),
                            None => continue,
                        },
                        None => None,
                    };
                    for b in 0..count.min(tile.bands.len()) {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
                        let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
                        per_band[b].push_tile_strided(
                            &tile.bands[b].data,
                            tw,
                            x_max,
                            y_max,
                            lx0,
                            ly0,
                            step,
                            mask_bits,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    }
                } else if let Some(ref masks) = tile_masks {
                    let Some(ref mask_bits) = masks[ti] else {
                        continue;
                    };
                    for b in 0..count.min(tile.bands.len()) {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
                        let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
                        per_band[b].push_tile_masked(
                            &tile.bands[b].data,
                            tw,
                            x_max,
                            y_max,
                            mask_bits,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    }
                } else {
                    for b in 0..count.min(tile.bands.len()) {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
                        let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
                        per_band[b].push_tile_unmasked(
                            &tile.bands[b].data,
                            tw,
                            x_max,
                            y_max,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    }
                }
            }
        }
        StatsTiles::Planar {
            band_tiles,
            tile_masks,
            ..
        } => {
            for (b, tiles) in band_tiles.iter().enumerate() {
                let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
                let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
                for (ti, tile_opt) in tiles.iter().enumerate() {
                    let Some(tile) = tile_opt else { continue };
                    let band = &tile.bands[0];
                    if band.stats.all_nodata {
                        continue;
                    }
                    let tx = (ti as u32) % ifd.tiles_across;
                    let ty = (ti as u32) / ifd.tiles_across;
                    let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                    let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                    let tw = ifd.tile_width as usize;
                    if let (Some(xs_all), Some(ys_all)) = (&tile_xs, &tile_ys) {
                        let xs = &xs_all[tx as usize];
                        let ys = &ys_all[ty as usize];
                        if xs.is_empty() || ys.is_empty() {
                            continue;
                        }
                        per_band[b].push_tile_coords(
                            &band.data,
                            tw,
                            xs,
                            ys,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    } else if step > 1 {
                        let x0_tile = (tx * ifd.tile_width) as usize;
                        let y0_tile = (ty * ifd.tile_height) as usize;
                        let lx0 = (step - (x0_tile % step)) % step;
                        let ly0 = (step - (y0_tile % step)) % step;
                        if lx0 >= x_max || ly0 >= y_max {
                            continue;
                        }
                        let mask_bits = match tile_masks {
                            Some(ref masks) => match masks[ti] {
                                Some(ref bits) => Some(bits.as_slice()),
                                None => continue,
                            },
                            None => None,
                        };
                        per_band[b].push_tile_strided(
                            &band.data,
                            tw,
                            x_max,
                            y_max,
                            lx0,
                            ly0,
                            step,
                            mask_bits,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    } else if let Some(ref masks) = tile_masks {
                        if let Some(ref mask_bits) = masks[ti] {
                            per_band[b].push_tile_masked(
                                &band.data,
                                tw,
                                x_max,
                                y_max,
                                mask_bits,
                                unscale,
                                scale,
                                offset,
                                nodata_override,
                            );
                        }
                    } else {
                        per_band[b].push_tile_unmasked(
                            &band.data,
                            tw,
                            x_max,
                            y_max,
                            unscale,
                            scale,
                            offset,
                            nodata_override,
                        );
                    }
                }
            }
        }
    }

    let denom_pixels = if poly_mask.is_some() {
        mask_total_pixels
    } else {
        total_pixels
    };

    let alpha_idx = cogzilla_codec::alpha_channel_index(count, &ifd.extra_samples);
    let mut out = serde_json::Map::new();
    let selected_bands: Option<&[u16]> = opts.bidx.as_deref().filter(|b| !b.is_empty());
    for (b, acc) in per_band.into_iter().enumerate() {
        let band_1based = (b + 1) as u16;
        if let Some(sel) = selected_bands {
            if !sel.contains(&band_1based) {
                continue;
            }
        } else if Some(b) == alpha_idx {
            continue;
        }
        let scale = cog.index.scales.get(b).copied().unwrap_or(1.0);
        let offset = cog.index.offsets.get(b).copied().unwrap_or(0.0);
        let bname = format!("b{band_1based}");
        #[cfg(feature = "gpu")]
        if let BandStatsAcc::Floats(ref vals) = acc
            && !opts.categorical
            && opts.histogram_edges.is_none()
            && let Some(gpu_ref) = gpu
            && let Some(gs) = gpu_ref.try_reduce_f32_stats(
                vals,
                None,
                opts.histogram_bins.clamp(1, 1000),
                opts.histogram_range,
                &opts.percentiles,
            )
        {
            out.insert(
                bname.clone(),
                gpu_band_stats_to_json(&gs, denom_pixels, &bname, opts),
            );
            continue;
        }
        let val = acc.finalize(
            denom_pixels,
            unscale,
            scale,
            offset,
            nodata_override,
            &bname,
            opts,
        );
        out.insert(bname, val);
    }

    serde_json::Value::Object(out)
}

#[allow(clippy::too_many_arguments)]
fn compute_expression_statistics_cpu(
    cog: &OpenedCog,
    ifd_idx: usize,
    ifd: &cogzilla_tiff::IfdMeta,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
    nodata_override: Option<f32>,
    step: usize,
    tile_xs: &Option<Vec<Vec<usize>>>,
    tile_ys: &Option<Vec<Vec<usize>>>,
    total_pixels: usize,
    mask_total_pixels: usize,
    fetched: StatsTiles,
    opts: &StatsOptions,
    exprs: &[(String, cogzilla_kernel::ExprProgram)],
    #[cfg(feature = "gpu")] gpu: Option<&crate::gpu_backend::GpuBackend>,
) -> serde_json::Value {
    #[cfg(not(feature = "gpu"))]
    let _ = ifd_idx;
    let denom_pixels = if poly_mask.is_some() {
        mask_total_pixels
    } else {
        total_pixels
    };
    let tile_masks = match &fetched {
        StatsTiles::Chunky { tile_masks, .. } | StatsTiles::Planar { tile_masks, .. } => tile_masks,
    };

    // 1-pass fused GPU fast path for unmasked `/cog/statistics?expression=...`:
    // evaluates each expression + 3-stage histogram/percentile reduction directly on the GPU
    // (12 KB readback, zero CPU pixel loop, zero 67 MB Vec<f32> alloc/upload).
    #[cfg(feature = "gpu")]
    if poly_mask.is_none()
        && tile_masks.is_none()
        && !opts.categorical
        && opts.histogram_edges.is_none()
        && let Some(gpu_ref) = gpu
        && exprs
            .iter()
            .all(|(_, prog)| !prog.used_bands.is_empty() && prog.used_bands.len() <= 8)
    {
        let (u_cols, v_rows): (Vec<f32>, Vec<f32>) =
            if let (Some(xs_all), Some(ys_all)) = (tile_xs, tile_ys) {
                let mut u = Vec::new();
                for (tx, xs) in xs_all.iter().enumerate() {
                    let base_x = tx as u32 * ifd.tile_width;
                    for &lx in xs {
                        u.push((base_x + lx as u32) as f32);
                    }
                }
                let mut v = Vec::new();
                for (ty, ys) in ys_all.iter().enumerate() {
                    let base_y = ty as u32 * ifd.tile_height;
                    for &ly in ys {
                        v.push((base_y + ly as u32) as f32);
                    }
                }
                (u, v)
            } else {
                let st = step.max(1);
                let u = (0..ifd.width as usize)
                    .step_by(st)
                    .map(|x| x as f32)
                    .collect();
                let v = (0..ifd.height as usize)
                    .step_by(st)
                    .map(|y| y as f32)
                    .collect();
                (u, v)
            };
        if u_cols.len().saturating_mul(v_rows.len()) == denom_pixels && !u_cols.is_empty() && !v_rows.is_empty() {
            let coords = crate::gpu_backend::BboxCoordSource::Separable {
                u: &u_cols,
                v: &v_rows,
            };
            let hist_bins = opts.histogram_bins.clamp(1, 1000);
            let n_tiles = (ifd.tiles_across * ifd.tiles_down) as usize;
            let mut gpu_out = serde_json::Map::new();
            let mut all_ok = true;
            for (i, (expr_str, prog)) in exprs.iter().enumerate() {
                let ref_vecs: Vec<(Vec<Option<&cogzilla_codec::DecodedBand>>, (f32, f32))> = prog
                    .used_bands
                    .iter()
                    .map(|&b1| {
                        let bi = b1.saturating_sub(1) as usize;
                        let so = if unscale {
                            (
                                cog.index.scales.get(bi).copied().unwrap_or(1.0),
                                cog.index.offsets.get(bi).copied().unwrap_or(0.0),
                            )
                        } else {
                            (1.0, 0.0)
                        };
                        let blocks: Vec<Option<&cogzilla_codec::DecodedBand>> = match &fetched {
                            StatsTiles::Chunky { tiles, .. } => tiles
                                .iter()
                                .take(n_tiles)
                                .map(|o| {
                                    o.as_ref()
                                        .and_then(|t| t.bands.get(bi))
                                        .filter(|tb| !tb.stats.all_nodata)
                                })
                                .collect(),
                            StatsTiles::Planar { band_tiles, .. } => {
                                if let Some(bt) = band_tiles.get(bi) {
                                    bt.iter()
                                        .take(n_tiles)
                                        .map(|o| {
                                            o.as_ref()
                                                .and_then(|t| t.bands.first())
                                                .filter(|tb| !tb.stats.all_nodata)
                                        })
                                        .collect()
                                } else {
                                    vec![None; n_tiles]
                                }
                            }
                        };
                        (blocks, so)
                    })
                    .collect();
                let srcs: Vec<BandSource<'_>> = ref_vecs
                    .iter()
                    .map(|(blocks, (sc, off))| {
                        BandSource::new(
                            ifd.width,
                            ifd.height,
                            ifd.tile_width,
                            ifd.tile_height,
                            0,
                            0,
                            ifd.tiles_across,
                            ifd.tiles_down,
                            blocks,
                        )
                        .with_nodata_override(nodata_override)
                        .with_scale_offset(*sc, *off)
                    })
                    .collect();
                let cache_key = crate::gpu_backend::GpuBackend::multiband_key(
                    cog.id,
                    ifd_idx,
                    &prog.used_bands,
                    &srcs,
                );
                if let Some(gs) = gpu_ref.try_render_bbox_expr_stats(
                    cache_key,
                    coords,
                    &srcs,
                    prog,
                    ResampleMode::Nearest,
                    hist_bins,
                    opts.histogram_range,
                    &opts.percentiles,
                ) {
                    gpu_out.insert(
                        format!("b{}", i + 1),
                        gpu_band_stats_to_json(&gs, denom_pixels, expr_str, opts),
                    );
                } else {
                    all_ok = false;
                    break;
                }
            }
            if all_ok {
                return serde_json::Value::Object(gpu_out);
            }
        }
    }

    let mut accs: Vec<Vec<f32>> = (0..exprs.len())
        .map(|_| Vec::with_capacity(denom_pixels.min(1 << 20)))
        .collect();
    let n_tiles = (ifd.tiles_across * ifd.tiles_down) as usize;
    let count = ifd.samples_per_pixel as usize;
    let tw = ifd.tile_width as usize;

    let has_where: Vec<bool> = exprs
        .iter()
        .map(|(_, prog)| prog.ops.contains(&cogzilla_kernel::expr_vm::Op::Where))
        .collect();
    let has_const_expr = exprs.iter().any(|(_, prog)| prog.used_bands.is_empty());
    let needs_empty_tiles = has_const_expr || has_where.iter().any(|&w| w);

    let mut eval_px = |bands: &[Option<&cogzilla_codec::BandData>], idx: usize| {
        let mut slot_vals = [0.0f32; 16];
        for (ei, (_, prog)) in exprs.iter().enumerate() {
            let allow_nan = has_where[ei];
            let mut valid = true;
            for (si, &b1) in prog.used_bands.iter().enumerate() {
                let bi = (b1.saturating_sub(1)) as usize;
                let Some(Some(bd)) = bands.get(bi) else {
                    if allow_nan {
                        slot_vals[si] = f32::NAN;
                        continue;
                    }
                    valid = false;
                    break;
                };
                let raw = bd.get_f32(idx);
                if !raw.is_finite() {
                    if allow_nan {
                        slot_vals[si] = f32::NAN;
                        continue;
                    }
                    valid = false;
                    break;
                }
                if let Some(nd) = nodata_override
                    && raw == nd
                {
                    if allow_nan {
                        slot_vals[si] = f32::NAN;
                        continue;
                    }
                    valid = false;
                    break;
                }
                let v = if unscale {
                    let sc = cog.index.scales.get(bi).copied().unwrap_or(1.0);
                    let off = cog.index.offsets.get(bi).copied().unwrap_or(0.0);
                    raw * sc + off
                } else {
                    raw
                };
                slot_vals[si] = v;
            }
            if valid {
                let r = prog.eval(&slot_vals[..prog.used_bands.len()]);
                if r.is_finite() {
                    accs[ei].push(r);
                }
            }
        }
    };

    let mut band_ptrs: Vec<Option<&cogzilla_codec::BandData>> = vec![None; count];
    for ti in 0..n_tiles {
        let mut any_band = false;
        match &fetched {
            StatsTiles::Chunky { tiles, .. } => {
                if let Some(tile) = &tiles[ti] {
                    for (b, slot) in band_ptrs.iter_mut().enumerate().take(count) {
                        *slot = tile
                            .bands
                            .get(b)
                            .filter(|tb| !tb.stats.all_nodata)
                            .map(|tb| &tb.data);
                        any_band |= slot.is_some();
                    }
                } else if needs_empty_tiles {
                    band_ptrs.fill(None);
                } else {
                    continue;
                }
            }
            StatsTiles::Planar { band_tiles, .. } => {
                for (b, slot) in band_ptrs.iter_mut().enumerate().take(count) {
                    *slot = band_tiles
                        .get(b)
                        .and_then(|bt| bt[ti].as_ref())
                        .and_then(|t| t.bands.first())
                        .filter(|tb| !tb.stats.all_nodata)
                        .map(|tb| &tb.data);
                    any_band |= slot.is_some();
                }
            }
        }
        if !any_band && !needs_empty_tiles {
            continue;
        }
        let tx = (ti as u32) % ifd.tiles_across;
        let ty = (ti as u32) / ifd.tiles_across;
        let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
        let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;

        if let (Some(xs_all), Some(ys_all)) = (tile_xs, tile_ys) {
            let xs = &xs_all[tx as usize];
            let ys = &ys_all[ty as usize];
            for &ly in ys {
                let row = ly * tw;
                for &lx in xs {
                    eval_px(&band_ptrs, row + lx);
                }
            }
        } else if step > 1 {
            let x0_tile = (tx * ifd.tile_width) as usize;
            let y0_tile = (ty * ifd.tile_height) as usize;
            let lx0 = (step - (x0_tile % step)) % step;
            let ly0 = (step - (y0_tile % step)) % step;
            let mask_bits = match tile_masks {
                Some(masks) => match &masks[ti] {
                    Some(bits) => Some(bits.as_slice()),
                    None => continue,
                },
                None => None,
            };
            let mut ly = ly0;
            let mut m_idx = 0usize;
            while ly < y_max {
                let row = ly * tw;
                let mut lx = lx0;
                while lx < x_max {
                    if mask_bits.is_none_or(|m| m.get(m_idx).copied().unwrap_or(0) != 0) {
                        eval_px(&band_ptrs, row + lx);
                    }
                    m_idx += 1;
                    lx += step;
                }
                ly += step;
            }
        } else if let Some(masks) = tile_masks {
            let Some(ref mask_bits) = masks[ti] else {
                continue;
            };
            for ly in 0..y_max {
                let row = ly * tw;
                let mrow = ly * x_max;
                for lx in 0..x_max {
                    if mask_bits[mrow + lx] != 0 {
                        eval_px(&band_ptrs, row + lx);
                    }
                }
            }
        } else {
            for ly in 0..y_max {
                let row = ly * tw;
                for lx in 0..x_max {
                    eval_px(&band_ptrs, row + lx);
                }
            }
        }
    }

    let mut out = serde_json::Map::new();
    for (i, ((expr_str, _), vals)) in exprs.iter().zip(accs.into_iter()).enumerate() {
        #[cfg(feature = "gpu")]
        if !opts.categorical
            && opts.histogram_edges.is_none()
            && let Some(gpu_ref) = gpu
            && let Some(gs) = gpu_ref.try_reduce_f32_stats(
                &vals,
                None,
                opts.histogram_bins.clamp(1, 1000),
                opts.histogram_range,
                &opts.percentiles,
            )
        {
            out.insert(
                format!("b{}", i + 1),
                gpu_band_stats_to_json(&gs, denom_pixels, expr_str, opts),
            );
            continue;
        }
        let acc = BandStatsAcc::Floats(vals);
        let val = acc.finalize(denom_pixels, false, 1.0, 0.0, None, expr_str, opts);
        out.insert(format!("b{}", i + 1), val);
    }
    serde_json::Value::Object(out)
}

/// Encode an RGBA buffer as a spec-compliant uncompressed Classic GeoTIFF (`EPSG:3857` or `EPSG:4326`).
pub fn encode_geotiff_rgba(
    width: u32,
    height: u32,
    rgba: &[u8],
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    let width = width.clamp(1, 4096);
    let height = height.clamp(1, 4096);
    let n_px = (width as usize) * (height as usize);
    let mut has_opaque = false;
    let mut all_gray = true;
    for px in rgba.chunks_exact(4).take(n_px) {
        if px[3] > 0 {
            has_opaque = true;
            if px[0] != px[1] || px[1] != px[2] {
                all_gray = false;
                break;
            }
        }
    }
    let is_gray = has_opaque && all_gray;
    let spp: u16 = if is_gray {
        if return_mask { 2 } else { 1 }
    } else if return_mask {
        4
    } else {
        3
    };
    let pixel_bytes = n_px * (spp as usize);
    let n_entries: u16 = if return_mask { 14 } else { 13 };

    // Layout:
    // 0..8                     : TIFF header ("II", 42, ifd_offset=8)
    // 8..8+2+12*n_entries+4    : IFD
    // out_off..                : BitsPerSample (spp*2 if spp>2), ModelPixelScale (24), ModelTiepoint (48), GeoKeyDir (32)
    // pixel_off..              : Raw strip pixel data
    let ifd_size = 2 + 12 * (n_entries as usize) + 4;
    let bps_bytes: u32 = if spp > 2 { (spp as u32) * 2 } else { 0 };
    let bps_off = (8 + ifd_size) as u32;
    let scale_off = bps_off + bps_bytes;
    let tie_off = scale_off + 24;
    let gkd_off = tie_off + 48;
    let pixel_off = gkd_off + 32;

    let mut buf = BytesMut::with_capacity((pixel_off as usize) + pixel_bytes);
    // TIFF little-endian header
    buf.put_slice(b"II");
    buf.put_u16_le(42);
    buf.put_u32_le(8);

    // IFD entry count
    buf.put_u16_le(n_entries);
    let mut put_entry = |tag: u16, ty: u16, count: u32, val: u32| {
        buf.put_u16_le(tag);
        buf.put_u16_le(ty);
        buf.put_u32_le(count);
        buf.put_u32_le(val);
    };
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const DOUBLE: u16 = 12;

    put_entry(256, LONG, 1, width);
    put_entry(257, LONG, 1, height);
    match spp {
        1 => put_entry(258, SHORT, 1, 8),
        2 => put_entry(258, SHORT, 2, 8 | (8 << 16)),
        _ => put_entry(258, SHORT, spp as u32, bps_off),
    }
    put_entry(259, SHORT, 1, 1); // Compression = None
    put_entry(262, SHORT, 1, if is_gray { 1 } else { 2 }); // Photometric = MinIsBlack (1) or RGB (2)
    put_entry(273, LONG, 1, pixel_off); // StripOffsets
    put_entry(277, SHORT, 1, spp as u32); // SamplesPerPixel
    put_entry(278, LONG, 1, height); // RowsPerStrip
    put_entry(279, LONG, 1, pixel_bytes as u32); // StripByteCounts
    put_entry(284, SHORT, 1, 1); // PlanarConfiguration = Chunky
    if return_mask {
        put_entry(338, SHORT, 1, 2); // ExtraSamples = UnassociatedAlpha
    }
    put_entry(33550, DOUBLE, 3, scale_off); // ModelPixelScaleTag
    put_entry(33922, DOUBLE, 6, tie_off); // ModelTiepointTag
    put_entry(34735, SHORT, 16, gkd_off); // GeoKeyDirectoryTag
    buf.put_u32_le(0); // Next IFD = 0

    // BitsPerSample (only out-of-line when spp > 2)
    if spp > 2 {
        for _ in 0..spp {
            buf.put_u16_le(8);
        }
    }
    let is_geographic = matches!(
        cogzilla_proj::CrsModel::from_epsg(epsg),
        Ok(cogzilla_proj::CrsModel::Geographic)
    );
    // ModelPixelScaleTag: [dx, dy, 0.0] (unwrap antimeridian-crossing geographic bboxes w > e)
    let span_x = if is_geographic && bbox[2] < bbox[0] {
        (bbox[2] + 360.0) - bbox[0]
    } else {
        (bbox[2] - bbox[0]).abs()
    };
    let dx = (span_x / (width.max(1) as f64)).max(1e-12);
    let dy = ((bbox[3] - bbox[1]).abs() / (height.max(1) as f64)).max(1e-12);
    buf.put_f64_le(dx);
    buf.put_f64_le(dy);
    buf.put_f64_le(0.0);
    // ModelTiepointTag: [0, 0, 0, minx, maxy, 0]
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(bbox[0]);
    buf.put_f64_le(bbox[3]);
    buf.put_f64_le(0.0);
    // GeoKeyDirectoryTag (16 u16s = 32 bytes)
    let (model_type, crs_key): (u16, u16) = if is_geographic {
        (2, 2048) // ModelTypeGeographic, GeographicTypeGeoKey
    } else {
        (1, 3072) // ModelTypeProjected, ProjectedCSTypeGeoKey
    };
    let gkd: [u16; 16] = [
        1, 1, 0, 3, // version 1.1.0, 3 keys
        1024, 0, 1, model_type, // GTModelTypeGeoKey
        1025, 0, 1, 1, // GTRasterTypeGeoKey = PixelIsArea
        crs_key, 0, 1, epsg, // EPSG code
    ];
    for &k in &gkd {
        buf.put_u16_le(k);
    }

    // Pixel data
    if is_gray {
        if return_mask {
            for px in rgba.chunks_exact(4).take(n_px) {
                buf.put_u8(px[0]);
                buf.put_u8(px[3]);
            }
        } else {
            for px in rgba.chunks_exact(4).take(n_px) {
                buf.put_u8(px[0]);
            }
        }
        let target_len = (pixel_off as usize) + pixel_bytes;
        if buf.len() < target_len {
            buf.resize(target_len, 0);
        }
    } else if return_mask {
        let take = pixel_bytes.min(rgba.len());
        buf.put_slice(&rgba[..take]);
        if take < pixel_bytes {
            buf.resize((pixel_off as usize) + pixel_bytes, 0);
        }
    } else {
        for px in rgba.chunks_exact(4).take(n_px) {
            buf.put_u8(px[0]);
            buf.put_u8(px[1]);
            buf.put_u8(px[2]);
        }
        let target_len = (pixel_off as usize) + pixel_bytes;
        if buf.len() < target_len {
            buf.resize(target_len, 0);
        }
    }
    buf.freeze()
}

/// Encode a CHW `f32` little-endian buffer as a spec-compliant uncompressed Classic IEEE Float32
/// GeoTIFF (`SampleFormat=3`, `BitsPerSample=32`, `GDAL_NODATA="nan"`). When `return_mask` is
/// `true`, appends an extra float32 validity mask band (`255.0` valid / `0.0` nodata) matching
/// TiTiler / `rio-tiler` GeoTIFF output.
pub fn encode_geotiff_f32(
    width: u32,
    height: u32,
    raw_f32_chw: &[u8],
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    let width = width.clamp(1, 4096);
    let height = height.clamp(1, 4096);
    let n_px = (width as usize) * (height as usize);
    let data_spp: u16 = if n_px > 0 && raw_f32_chw.len() >= n_px * 4 {
        (raw_f32_chw.len() / (n_px * 4)).clamp(1, 15) as u16
    } else {
        1
    };
    let spp: u16 = if return_mask { data_spp + 1 } else { data_spp };
    let pixel_bytes = n_px * (spp as usize) * 4;
    let photometric: u32 = if data_spp == 3 { 2 } else { 1 }; // RGB when data has 3 bands, MinIsBlack otherwise
    let extra_samples_count: u16 = if photometric == 2 {
        spp.saturating_sub(3)
    } else {
        spp.saturating_sub(1)
    };
    let mut extra_samples = Vec::with_capacity(extra_samples_count as usize);
    for idx in 0..extra_samples_count {
        let is_mask = return_mask && idx + 1 == extra_samples_count;
        // 2 = UnassociatedAlpha for the appended validity mask, 0 = Unspecified for extra data bands.
        extra_samples.push(if is_mask { 2u16 } else { 0u16 });
    }
    let n_entries: u16 = if extra_samples_count > 0 { 16 } else { 15 };

    let ifd_size = 2 + 12 * (n_entries as usize) + 4;
    let mut extra_off = (8 + ifd_size) as u32;
    let bps_val = if spp == 1 {
        32u32
    } else if spp == 2 {
        32u32 | (32u32 << 16)
    } else {
        let off = extra_off;
        extra_off += (spp as u32) * 2;
        off
    };
    let extra_samples_val = match extra_samples_count {
        0 => 0u32,
        1 => extra_samples[0] as u32,
        2 => (extra_samples[0] as u32) | ((extra_samples[1] as u32) << 16),
        _ => {
            let off = extra_off;
            extra_off += (extra_samples_count as u32) * 2;
            off
        }
    };
    let sf_val = if spp == 1 {
        3u32
    } else if spp == 2 {
        3u32 | (3u32 << 16)
    } else {
        let off = extra_off;
        extra_off += (spp as u32) * 2;
        off
    };
    let scale_off = extra_off;
    let tie_off = scale_off + 24;
    let gkd_off = tie_off + 48;
    let pixel_off = gkd_off + 32;

    let mut buf = BytesMut::with_capacity((pixel_off as usize) + pixel_bytes);
    // TIFF little-endian header
    buf.put_slice(b"II");
    buf.put_u16_le(42);
    buf.put_u32_le(8);

    // IFD entries (must be in strictly ascending tag order)
    buf.put_u16_le(n_entries);
    let mut put_entry = |tag: u16, ty: u16, count: u32, val: u32| {
        buf.put_u16_le(tag);
        buf.put_u16_le(ty);
        buf.put_u32_le(count);
        buf.put_u32_le(val);
    };
    const ASCII: u16 = 2;
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const DOUBLE: u16 = 12;

    put_entry(256, LONG, 1, width);
    put_entry(257, LONG, 1, height);
    put_entry(258, SHORT, spp as u32, bps_val); // BitsPerSample = 32
    put_entry(259, SHORT, 1, 1); // Compression = None
    put_entry(262, SHORT, 1, photometric); // Photometric = RGB (2) or MinIsBlack (1)
    put_entry(273, LONG, 1, pixel_off); // StripOffsets
    put_entry(277, SHORT, 1, spp as u32); // SamplesPerPixel
    put_entry(278, LONG, 1, height); // RowsPerStrip
    put_entry(279, LONG, 1, pixel_bytes as u32); // StripByteCounts
    put_entry(284, SHORT, 1, 1); // PlanarConfiguration = Chunky
    if extra_samples_count > 0 {
        put_entry(338, SHORT, extra_samples_count as u32, extra_samples_val); // ExtraSamples = Unspecified (0)
    }
    put_entry(339, SHORT, spp as u32, sf_val); // SampleFormat = IEEEFloat (3)
    put_entry(33550, DOUBLE, 3, scale_off); // ModelPixelScaleTag
    put_entry(33922, DOUBLE, 6, tie_off); // ModelTiepointTag
    put_entry(34735, SHORT, 16, gkd_off); // GeoKeyDirectoryTag
    put_entry(42113, ASCII, 4, u32::from_le_bytes(*b"nan\0")); // GDAL_NODATA = "nan"
    buf.put_u32_le(0); // Next IFD = 0

    if spp >= 3 {
        for _ in 0..spp {
            buf.put_u16_le(32);
        }
    }
    if extra_samples_count >= 3 {
        for &es in &extra_samples {
            buf.put_u16_le(es);
        }
    }
    if spp >= 3 {
        for _ in 0..spp {
            buf.put_u16_le(3);
        }
    }
    let is_geographic = matches!(
        cogzilla_proj::CrsModel::from_epsg(epsg),
        Ok(cogzilla_proj::CrsModel::Geographic)
    );
    // ModelPixelScaleTag: [dx, dy, 0.0] (unwrap antimeridian-crossing geographic bboxes w > e)
    let span_x = if is_geographic && bbox[2] < bbox[0] {
        (bbox[2] + 360.0) - bbox[0]
    } else {
        (bbox[2] - bbox[0]).abs()
    };
    let dx = (span_x / (width.max(1) as f64)).max(1e-12);
    let dy = ((bbox[3] - bbox[1]).abs() / (height.max(1) as f64)).max(1e-12);
    buf.put_f64_le(dx);
    buf.put_f64_le(dy);
    buf.put_f64_le(0.0);
    // ModelTiepointTag: [0, 0, 0, minx, maxy, 0]
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(bbox[0]);
    buf.put_f64_le(bbox[3]);
    buf.put_f64_le(0.0);
    // GeoKeyDirectoryTag
    let (model_type, crs_key): (u16, u16) = if is_geographic {
        (2, 2048)
    } else {
        (1, 3072)
    };
    let gkd: [u16; 16] = [
        1, 1, 0, 3,
        1024, 0, 1, model_type,
        1025, 0, 1, 1,
        crs_key, 0, 1, epsg,
    ];
    for &k in &gkd {
        buf.put_u16_le(k);
    }

    // Pixel data: interleave CHW f32 planes into Chunky f32 (plus optional validity mask band)
    if spp == 1 && !return_mask {
        let take = (pixel_bytes.min(raw_f32_chw.len()) / 4) * 4;
        buf.put_slice(&raw_f32_chw[..take]);
        if take < pixel_bytes {
            let rem_f32 = (pixel_bytes - take) / 4;
            for _ in 0..rem_f32 {
                buf.put_f32_le(f32::NAN);
            }
        }
    } else {
        let data_spp_us = data_spp as usize;
        let plane_bytes = n_px * 4;
        let valid_le = 255.0f32.to_le_bytes();
        let zero_le = 0.0f32.to_le_bytes();
        for px in 0..n_px {
            let px_off = px * 4;
            let mut any_valid = false;
            for b in 0..data_spp_us {
                let off = b * plane_bytes + px_off;
                if off + 4 <= raw_f32_chw.len() {
                    let sl = &raw_f32_chw[off..off + 4];
                    if return_mask
                        && !any_valid
                        && f32::from_le_bytes([sl[0], sl[1], sl[2], sl[3]]).is_finite()
                    {
                        any_valid = true;
                    }
                    buf.put_slice(sl);
                } else {
                    buf.put_f32_le(f32::NAN);
                }
            }
            if return_mask {
                buf.put_slice(if any_valid { &valid_le } else { &zero_le });
            }
        }
    }
    buf.freeze()
}

/// Project a WGS84 polygon bbox into an IFD's pixel space and return `(win_w, win_h, intersects)`.
/// Compute the native pixel dimensions `(w, h)` for a WGS84 bounding box against `ifd`,
/// matching GDAL `WarpedVRT` / `rio-tiler` suggested output sizing (preserving the target bbox
/// aspect ratio without upsampling past native resolution).
pub fn native_bbox_dims_in_crs(
    ifd: &cogzilla_tiff::IfdMeta,
    src_crs: &cogzilla_proj::CrsModel,
    wgs_bbox: [f64; 4],
    dst_bbox: [f64; 4],
) -> (u32, u32) {
    let [w, s, mut e, n] = wgs_bbox;
    if e < w {
        e += 360.0;
    }
    let mx = 0.5 * (w + e);
    let my = 0.5 * (s + n);
    let pts = [
        (w, s),
        (e, s),
        (w, n),
        (e, n),
        (mx, s),
        (mx, n),
        (w, my),
        (e, my),
    ];
    let mut cmin = f64::INFINITY;
    let mut cmax = f64::NEG_INFINITY;
    let mut rmin = f64::INFINITY;
    let mut rmax = f64::NEG_INFINITY;
    let Some(crs_to_px) = ifd.pixel_to_crs.invert() else {
        return (1, 1);
    };
    let wrap_lo = src_crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
    for (mut lon, lat) in pts {
        if let Some(lo) = wrap_lo
            && lon < lo
        {
            lon += 360.0;
        }
        let (sx, sy) = src_crs.from_lonlat_deg(lon, lat);
        if sx.is_finite() && sy.is_finite() {
            let (col, row) = crs_to_px.forward(sx, sy);
            if col.is_finite() && row.is_finite() {
                cmin = cmin.min(col);
                cmax = cmax.max(col);
                rmin = rmin.min(row);
                rmax = rmax.max(row);
            }
        }
    }
    if !cmin.is_finite() || !rmin.is_finite() {
        return (1, 1);
    }
    let nat_w = (cmax - cmin).abs();
    let nat_h = (rmax - rmin).abs();
    let dx = if dst_bbox[2] < dst_bbox[0] && dst_bbox[0].abs() <= 180.0 && dst_bbox[2].abs() <= 180.0 {
        (dst_bbox[2] + 360.0) - dst_bbox[0]
    } else {
        (dst_bbox[2] - dst_bbox[0]).abs()
    };
    let dy = (dst_bbox[3] - dst_bbox[1]).abs();
    if dx <= 1e-12 || dy <= 1e-12 || nat_w <= 1e-6 || nat_h <= 1e-6 {
        return (
            nat_w.round().clamp(1.0, 4096.0) as u32,
            nat_h.round().clamp(1.0, 4096.0) as u32,
        );
    }
    // Equal-pixel-density isotropic resolution in dst_crs matching GDAL WarpedVRT:
    let area_px = (nat_w * nat_h).max(1.0);
    let res = ((dx * dy) / area_px).sqrt();
    if !res.is_finite() || res <= 0.0 {
        return (
            nat_w.round().clamp(1.0, 4096.0) as u32,
            nat_h.round().clamp(1.0, 4096.0) as u32,
        );
    }
    let ow = (dx / res).round().clamp(1.0, 4096.0) as u32;
    let oh = (dy / res).round().clamp(1.0, 4096.0) as u32;
    (ow, oh)
}

fn polygon_window_dims(
    ifd: &cogzilla_tiff::IfdMeta,
    crs: &cogzilla_proj::CrsModel,
    bbox: [f64; 4],
) -> (u32, u32, bool) {
    let [w, s, e, n] = bbox;
    let mx = 0.5 * (w + e);
    let my = 0.5 * (s + n);
    let pts = [
        (w, s),
        (e, s),
        (w, n),
        (e, n),
        (mx, s),
        (mx, n),
        (w, my),
        (e, my),
    ];
    let mut cmin = f64::INFINITY;
    let mut cmax = f64::NEG_INFINITY;
    let mut rmin = f64::INFINITY;
    let mut rmax = f64::NEG_INFINITY;
    let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
    for (mut lon, lat) in pts {
        if let Some(lo) = wrap_lo
            && lon < lo
        {
            lon += 360.0;
        }
        let (sx, sy) = crs.from_lonlat_deg(lon, lat);
        if sx.is_finite() && sy.is_finite() {
            let (col, row) = ifd.crs_to_pixel.forward(sx, sy);
            if col.is_finite() && row.is_finite() {
                cmin = cmin.min(col);
                cmax = cmax.max(col);
                rmin = rmin.min(row);
                rmax = rmax.max(row);
            }
        }
    }
    if !cmin.is_finite() {
        return (ifd.width, ifd.height, true);
    }
    let c0 = cmin.floor().clamp(0.0, ifd.width as f64);
    let c1 = cmax.ceil().clamp(0.0, ifd.width as f64);
    let r0 = rmin.floor().clamp(0.0, ifd.height as f64);
    let r1 = rmax.ceil().clamp(0.0, ifd.height as f64);
    let intersects = c1 > c0 && r1 > r0;
    (((c1 - c0).max(1.0)) as u32, ((r1 - r0).max(1.0)) as u32, intersects)
}

/// TiTiler-compatible `/cog/statistics` computed from the finest overview whose target window
/// fits within `max_size` (for `POST /statistics` with a polygon, sized to the polygon's pixel
/// window rather than the whole raster so small polygons sample at full native resolution).
/// When the coarsest IFD still exceeds `max_size` (e.g. an overview-less GeoTIFF), stride-decimates
/// by `step = ceil(win_max_dim / max_size)` so memory and I/O stay bounded to `<= max_size^2` samples.
pub async fn compute_statistics(
    engine: &TileEngine,
    cog: &Arc<OpenedCog>,
    max_size: u32,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
    nodata_override: Option<f32>,
) -> Result<serde_json::Value, EngineError> {
    compute_statistics_with_options(
        engine,
        cog,
        max_size,
        poly_mask,
        unscale,
        nodata_override,
        &StatsOptions::default(),
    )
    .await
}

pub async fn compute_statistics_with_options(
    engine: &TileEngine,
    cog: &Arc<OpenedCog>,
    max_size: u32,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
    nodata_override: Option<f32>,
    opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    if let Some(ref bidx) = opts.bidx {
        let count = cog.index.full_res().samples_per_pixel;
        for &b in bidx {
            if b == 0 || b > count {
                return Err(EngineError::BadRequest(format!(
                    "band {b} out of range (dataset has {count} bands)"
                )));
            }
        }
    }
    if let Some(ref exprs) = opts.expressions {
        let count = cog.index.full_res().samples_per_pixel;
        for (_, prog) in exprs {
            for &b in &prog.used_bands {
                if b == 0 || b > count {
                    return Err(EngineError::BadRequest(format!(
                        "expression references band {b} out of range (dataset has {count} bands)"
                    )));
                }
            }
        }
    }
    const MAX_STATS_BLOCKS: usize = 512;
    const MAX_STATS_MASK_TILES: usize = 65_536;
    let crs = if poly_mask.is_none() && !cog.index.is_georeferenced {
        cogzilla_proj::CrsModel::Geographic
    } else {
        engine.get_or_build_crs(cog)?
    };
    if let Some(pm) = poly_mask {
        if pm.is_empty() || (opts.width.is_none() && opts.height.is_none()) {
            return crate::zonal::compute_native_exact_polygon_stats(
                engine,
                cog,
                &crs,
                pm,
                max_size,
                unscale,
                nodata_override,
                opts,
            )
            .await;
        }
        let full = cog.index.full_res();
        let count = full.samples_per_pixel;
        // Pad degenerate zero-span axes (e.g. Point or axis-aligned LineString) by half a source pixel.
        let mut eff_bbox = pm.bbox;
        if (eff_bbox[2] - eff_bbox[0]).abs() < 1e-12 || (eff_bbox[3] - eff_bbox[1]).abs() < 1e-12 {
            let cx = 0.5 * (eff_bbox[0] + eff_bbox[2]);
            let cy = 0.5 * (eff_bbox[1] + eff_bbox[3]);
            let (dlon, dlat) = if let Some(crs_to_px) = full.pixel_to_crs.invert() {
                let (sx, sy) = crs.from_lonlat_deg(cx, cy);
                let (col, row) = crs_to_px.forward(sx, sy);
                let (sx1, sy1) = full.pixel_to_crs.forward(col + 1.0, row + 1.0);
                let (lo1, la1) = crs.to_lonlat_deg(sx1, sy1);
                ((lo1 - cx).abs().max(1e-6), (la1 - cy).abs().max(1e-6))
            } else {
                (1e-5, 1e-5)
            };
            if (eff_bbox[2] - eff_bbox[0]).abs() < 1e-12 {
                eff_bbox[0] = cx - 0.5 * dlon;
                eff_bbox[2] = cx + 0.5 * dlon;
            }
            if (eff_bbox[3] - eff_bbox[1]).abs() < 1e-12 {
                eff_bbox[1] = cy - 0.5 * dlat;
                eff_bbox[3] = cy + 0.5 * dlat;
            }
        }
        let (nat_w, nat_h) = native_bbox_dims_in_crs(full, &crs, eff_bbox, eff_bbox);
        let (out_w, out_h) = match (opts.width, opts.height) {
            (Some(w), Some(h)) => (w.clamp(1, 4096), h.clamp(1, 4096)),
            (Some(w), None) => {
                let w = w.clamp(1, 4096);
                let h = ((nat_h as f64 / nat_w.max(1) as f64) * w as f64)
                    .round()
                    .clamp(1.0, 4096.0) as u32;
                (w, h)
            }
            (None, Some(h)) => {
                let h = h.clamp(1, 4096);
                let w = ((nat_w as f64 / nat_h.max(1) as f64) * h as f64)
                    .round()
                    .clamp(1.0, 4096.0) as u32;
                (w, h)
            }
            (None, None) => {
                let cap = if opts.max_size_explicit && max_size > 0 {
                    max_size.clamp(1, 4096)
                } else {
                    4096
                };
                if nat_w.max(nat_h) <= cap {
                    (nat_w.max(1), nat_h.max(1))
                } else if nat_w >= nat_h {
                    let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    (cap, h)
                } else {
                    let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
                        .round()
                        .clamp(1.0, 4096.0) as u32;
                    (w, cap)
                }
            }
        };
        let cogs = [cog.clone()];
        let n_px = (out_w as usize) * (out_h as usize);
        let feature_covered_px = pm
            .rasterize_all_touched(
                out_w as usize,
                out_h as usize,
                eff_bbox,
                &cogzilla_proj::CrsModel::Geographic,
            )
            .iter()
            .filter(|&&b| b)
            .count();
        let patch_valid_percent = |mut stat: serde_json::Value| -> serde_json::Value {
            if feature_covered_px > 0
                && let Some(obj) = stat.as_object_mut()
                && let Some(vp) = obj.get("valid_pixels").and_then(|v| v.as_f64())
            {
                let pct = ((vp / feature_covered_px as f64) * 100.0).min(100.0);
                obj.insert(
                    "valid_percent".into(),
                    serde_json::json!((pct * 100.0).round() / 100.0),
                );
            }
            stat
        };
        let mut out = serde_json::Map::new();
        if let Some(ref exprs) = opts.expressions {
            for (i, (desc, prog)) in exprs.iter().enumerate() {
                let bands = BandRequest::CustomExpr {
                    prog: prog.clone(),
                    rescale: (0.0, 1.0),
                    colormap: ColormapName::Grayscale,
                };
                let stat = match render_wgs84_bbox_stats(
                    engine,
                    &cogs,
                    eff_bbox,
                    out_w,
                    out_h,
                    Some(pm),
                    &bands,
                    ResampleMode::Nearest,
                    nodata_override,
                    unscale,
                    desc,
                    opts,
                )
                .await
                {
                    Ok(v) => v,
                    Err(EngineError::NotFound(_)) => {
                        compute_f32_band_stats_with_engine(engine, &[], desc, opts)
                    }
                    Err(e) => return Err(e),
                };
                out.insert(format!("b{}", i + 1), patch_valid_percent(stat));
            }
            return Ok(serde_json::Value::Object(out));
        }
        let alpha_idx = cogzilla_codec::alpha_channel_index(count as usize, &full.extra_samples);
        let sel_bands: Vec<u16> = match &opts.bidx {
            Some(b) if !b.is_empty() => b.clone(),
            _ => (1..=count)
                .filter(|&b| Some((b - 1) as usize) != alpha_idx)
                .collect(),
        };
        for chunk in sel_bands.chunks(15) {
            if chunk.len() == 1 {
                let b1 = chunk[0];
                let bands_req = BandRequest::Single {
                    band_1based: b1,
                    rescale: (0.0, 1.0),
                    colormap: ColormapName::Grayscale,
                };
                let desc = cog
                    .index
                    .band_descriptions
                    .get((b1 - 1) as usize)
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("b{b1}"));
                let stat = match render_wgs84_bbox_stats(
                    engine,
                    &cogs,
                    eff_bbox,
                    out_w,
                    out_h,
                    Some(pm),
                    &bands_req,
                    ResampleMode::Nearest,
                    nodata_override,
                    unscale,
                    &desc,
                    opts,
                )
                .await
                {
                    Ok(v) => v,
                    Err(EngineError::NotFound(_)) => {
                        compute_f32_band_stats_with_engine(engine, &[], &desc, opts)
                    }
                    Err(e) => return Err(e),
                };
                out.insert(format!("b{b1}"), patch_valid_percent(stat));
                continue;
            }
            let bands_req = BandRequest::MultiBand {
                bands_1based: chunk.to_vec(),
            };
            let raw = match render_wgs84_bbox(
                engine,
                &cogs,
                eff_bbox,
                out_w,
                out_h,
                Some(pm),
                &bands_req,
                ResampleMode::Nearest,
                OutputFormat::RawF32,
                nodata_override,
                None,
                None,
                true,
                unscale,
            )
            .await
            {
                Ok(b) => b,
                Err(EngineError::NotFound(_)) => Bytes::new(),
                Err(e) => return Err(e),
            };
            for (ci, &b1) in chunk.iter().enumerate() {
                let start = ci * n_px * 4;
                let end = (start + n_px * 4).min(raw.len());
                let slice = if start < end { &raw[start..end] } else { &[] };
                let pixels: Vec<f32> = slice
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let desc = cog
                    .index
                    .band_descriptions
                    .get((b1 - 1) as usize)
                    .map(String::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("b{b1}"));
                out.insert(
                    format!("b{b1}"),
                    patch_valid_percent(compute_f32_band_stats_with_engine(
                        engine, &pixels, &desc, opts,
                    )),
                );
            }
        }
        return Ok(serde_json::Value::Object(out));
    }
    let max_size = max_size.max(1);
    let (ifd_idx, step, sample_dims, total_pixels, poly_win_pixels) = if let Some(pm) = poly_mask {
        let mut idx = cog.index.ifds.len() - 1;
        let mut win_w = cog.index.ifds[idx].width;
        let mut win_h = cog.index.ifds[idx].height;
        let mut win_max_dim = win_w.max(win_h);
        for (i, ifd) in cog.index.ifds.iter().enumerate() {
            let (ww, wh, intersects) = polygon_window_dims(ifd, &crs, pm.bbox);
            let dim = ww.max(wh);
            let ifd_tiles = (ifd.tiles_across as usize).saturating_mul(ifd.tiles_down as usize);
            if intersects && dim <= max_size && ifd_tiles <= MAX_STATS_MASK_TILES {
                idx = i;
                win_w = ww;
                win_h = wh;
                win_max_dim = dim;
                break;
            }
            win_w = ww;
            win_h = wh;
            win_max_dim = dim;
        }
        let ifd = &cog.index.ifds[idx];
        let step = win_max_dim.div_ceil(max_size).max(1) as usize;
        let total = if step > 1 {
            (ifd.width as usize).div_ceil(step) * (ifd.height as usize).div_ceil(step)
        } else {
            (ifd.width as usize) * (ifd.height as usize)
        };
        let est_w = (win_w.saturating_add(ifd.tile_width).min(ifd.width) as usize).div_ceil(step);
        let est_h = (win_h.saturating_add(ifd.tile_height).min(ifd.height) as usize).div_ceil(step);
        (idx, step, None, total, est_w.saturating_mul(est_h))
    } else {
        // Unmasked full-image statistics: match rio-tiler's exact `max_size` target dimensions
        // and select the coarsest IFD that is >= target dimensions (falling back to IFD 0).
        let full = cog.index.full_res();
        let (out_w, out_h) = match (opts.width, opts.height) {
            (Some(w), Some(h)) => (w.clamp(1, 4096) as usize, h.clamp(1, 4096) as usize),
            (Some(w), None) => {
                let w = w.clamp(1, 4096) as usize;
                let h = ((full.height as f64 / full.width.max(1) as f64) * w as f64)
                    .round()
                    .clamp(1.0, 4096.0) as usize;
                (w, h)
            }
            (None, Some(h)) => {
                let h = h.clamp(1, 4096) as usize;
                let w = ((full.width as f64 / full.height.max(1) as f64) * h as f64)
                    .round()
                    .clamp(1.0, 4096.0) as usize;
                (w, h)
            }
            (None, None) => {
                if full.width.max(full.height) <= max_size {
                    (full.width as usize, full.height as usize)
                } else if full.width >= full.height {
                    let h = ((full.height as f64 / full.width as f64) * max_size as f64)
                        .round()
                        .max(1.0) as usize;
                    (max_size as usize, h)
                } else {
                    let w = ((full.width as f64 / full.height as f64) * max_size as f64)
                        .round()
                        .max(1.0) as usize;
                    (w, max_size as usize)
                }
            }
        };
        let mut idx = 0usize;
        for (i, ifd) in cog.index.ifds.iter().enumerate() {
            if (ifd.width as usize) >= out_w && (ifd.height as usize) >= out_h {
                idx = i;
            }
        }
        let ifd = &cog.index.ifds[idx];
        let dims = if (ifd.width as usize) == out_w && (ifd.height as usize) == out_h {
            None
        } else {
            Some((out_w, out_h))
        };
        (idx, 1usize, dims, out_w * out_h, 0usize)
    };
    let ifd = &cog.index.ifds[ifd_idx];
    let count = ifd.samples_per_pixel as usize;

    let n_tiles = (ifd.tiles_across as usize).saturating_mul(ifd.tiles_down as usize);
    let (sample_has_tx, sample_has_ty): (Option<Vec<bool>>, Option<Vec<bool>>) =
        if let Some((ow, oh)) = sample_dims {
            let mut has_tx = vec![false; ifd.tiles_across as usize];
            let mut has_ty = vec![false; ifd.tiles_down as usize];
            for px in 0..ow {
                let col = (((px as f64 + 0.5) * (ifd.width as f64) / (ow as f64)).floor() as u32)
                    .min(ifd.width - 1);
                let tx = (col / ifd.tile_width) as usize;
                if let Some(slot) = has_tx.get_mut(tx) {
                    *slot = true;
                }
            }
            for py in 0..oh {
                let row = (((py as f64 + 0.5) * (ifd.height as f64) / (oh as f64)).floor() as u32)
                    .min(ifd.height - 1);
                let ty = (row / ifd.tile_height) as usize;
                if let Some(slot) = has_ty.get_mut(ty) {
                    *slot = true;
                }
            }
            (Some(has_tx), Some(has_ty))
        } else {
            (None, None)
        };
    let tile_has_strided_sample = |tx: u32, ty: u32, x_max: usize, y_max: usize| -> bool {
        if let (Some(has_tx), Some(has_ty)) = (&sample_has_tx, &sample_has_ty) {
            return has_tx.get(tx as usize).copied().unwrap_or(false)
                && has_ty.get(ty as usize).copied().unwrap_or(false);
        }
        if step == 1 {
            return true;
        }
        let x0_tile = (tx * ifd.tile_width) as usize;
        let y0_tile = (ty * ifd.tile_height) as usize;
        let lx0 = (step - (x0_tile % step)) % step;
        let ly0 = (step - (y0_tile % step)) % step;
        lx0 < x_max && ly0 < y_max
    };

    let (tile_masks, mask_total_pixels): (Option<Vec<Option<Vec<u8>>>>, usize) =
        if let Some(pm) = poly_mask {
            pm.check_work_budget(poly_win_pixels)?;
            if n_tiles > MAX_STATS_MASK_TILES {
                return Err(EngineError::BadRequest(format!(
                    "statistics IFD tile count ({n_tiles}) exceeds polygon mask limit ({MAX_STATS_MASK_TILES})"
                )));
            }
            let build_masks = {
                let ifd = ifd.clone();
                let crs = crs.clone();
                let pm = pm.clone();
                move || -> Result<(Option<Vec<Option<Vec<u8>>>>, usize), EngineError> {
                    let mut masks = Vec::with_capacity(n_tiles);
                    let mut inside_total = 0usize;
                    let mut active_blocks = 0usize;
                    let mut tested_pixels = 0usize;
                    for ty in 0..ifd.tiles_down {
                        for tx in 0..ifd.tiles_across {
                            let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                            let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                            let has_sample = if step == 1 {
                                true
                            } else {
                                let x0_tile = (tx * ifd.tile_width) as usize;
                                let y0_tile = (ty * ifd.tile_height) as usize;
                                let lx0 = (step - (x0_tile % step)) % step;
                                let ly0 = (step - (y0_tile % step)) % step;
                                lx0 < x_max && ly0 < y_max
                            };
                            if !has_sample
                                || tile_probe_outside_polygon(&ifd, &crs, &pm, tx, ty, x_max, y_max)
                            {
                                masks.push(None);
                                continue;
                            }
                            let tile_px = if step > 1 {
                                x_max.div_ceil(step).saturating_mul(y_max.div_ceil(step))
                            } else {
                                x_max.saturating_mul(y_max)
                            };
                            tested_pixels = tested_pixels.saturating_add(tile_px);
                            pm.check_work_budget(tested_pixels)?;
                            let built = if step > 1 {
                                let x0_tile = (tx * ifd.tile_width) as usize;
                                let y0_tile = (ty * ifd.tile_height) as usize;
                                let lx0 = (step - (x0_tile % step)) % step;
                                let ly0 = (step - (y0_tile % step)) % step;
                                build_tile_polygon_mask_strided(
                                    &ifd, &crs, &pm, tx, ty, x_max, y_max, lx0, ly0, step,
                                )
                            } else {
                                build_tile_polygon_mask(&ifd, &crs, &pm, tx, ty, x_max, y_max)
                            };
                            if let Some((bits, inside)) = built {
                                active_blocks += 1;
                                if active_blocks > MAX_STATS_BLOCKS {
                                    return Err(EngineError::BadRequest(format!(
                                        "statistics request exceeds maximum block limit ({MAX_STATS_BLOCKS})"
                                    )));
                                }
                                inside_total += inside;
                                masks.push(Some(bits));
                            } else {
                                masks.push(None);
                            }
                        }
                    }
                    Ok((Some(masks), inside_total))
                }
            };
            if n_tiles >= 4 || total_pixels >= 65_536 {
                engine
                    .store
                    .run_heavy_cpu(build_masks)
                    .await??
            } else {
                build_masks()?
            }
        } else {
            (None, 0)
        };

    let fetched = match ifd.planar_config {
        PlanarConfig::Chunky => {
            let mut indices = Vec::with_capacity(n_tiles.min(MAX_STATS_MASK_TILES));
            let mut active_blocks = 0usize;
            for ty in 0..ifd.tiles_down {
                for tx in 0..ifd.tiles_across {
                    let ti = (ty * ifd.tiles_across + tx) as usize;
                    let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                    let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                    if !tile_has_strided_sample(tx, ty, x_max, y_max) {
                        indices.push(None);
                        continue;
                    }
                    if let Some(ref masks) = tile_masks {
                        if masks[ti].is_none() {
                            indices.push(None);
                            continue;
                        }
                    }
                    active_blocks += 1;
                    if active_blocks > MAX_STATS_BLOCKS {
                        return Err(EngineError::BadRequest(format!(
                            "statistics request exceeds maximum block limit ({MAX_STATS_BLOCKS})"
                        )));
                    }
                    indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                }
            }
            let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
            StatsTiles::Chunky {
                tiles,
                tile_masks,
                mask_total_pixels,
            }
        }
        PlanarConfig::Planar => {
            // Only fetch and decode bands actually referenced by `?expression=` or `?bidx=`.
            let needed_bands: Vec<bool> = if let Some(ref exprs) = opts.expressions
                && !exprs.is_empty()
            {
                let mut mask = vec![false; count];
                for (_, prog) in exprs {
                    for &b1 in &prog.used_bands {
                        if let Some(slot) = mask.get_mut((b1.saturating_sub(1)) as usize) {
                            *slot = true;
                        }
                    }
                }
                mask
            } else if let Some(ref bidx) = opts.bidx
                && !bidx.is_empty()
            {
                let mut mask = vec![false; count];
                for &b1 in bidx {
                    if let Some(slot) = mask.get_mut((b1.saturating_sub(1)) as usize) {
                        *slot = true;
                    }
                }
                mask
            } else {
                vec![true; count]
            };

            // Coalesce all `count * n_tiles` band-plane block indices into one concurrent
            // `get_blocks` batch instead of `count` sequential remote round trips.
            let mut all_indices =
                Vec::with_capacity(count.saturating_mul(n_tiles).min(MAX_STATS_MASK_TILES));
            let mut active_blocks = 0usize;
            for (b, &need_b) in needed_bands.iter().enumerate().take(count) {
                if !need_b {
                    all_indices.resize(all_indices.len() + n_tiles, None);
                    continue;
                }
                for ty in 0..ifd.tiles_down {
                    for tx in 0..ifd.tiles_across {
                        let ti = (ty * ifd.tiles_across + tx) as usize;
                        let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                        let y_max =
                            (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                        if !tile_has_strided_sample(tx, ty, x_max, y_max) {
                            all_indices.push(None);
                            continue;
                        }
                        if let Some(ref masks) = tile_masks {
                            if masks[ti].is_none() {
                                all_indices.push(None);
                                continue;
                            }
                        }
                        active_blocks += 1;
                        if active_blocks > MAX_STATS_BLOCKS {
                            return Err(EngineError::BadRequest(format!(
                                "statistics request exceeds maximum block limit ({MAX_STATS_BLOCKS})"
                            )));
                        }
                        all_indices.push(Some(ifd.tile_index(tx, ty, b as u16) as u32));
                    }
                }
            }
            let mut flat = engine.store.get_blocks(cog, ifd_idx, &all_indices).await?;
            let mut band_tiles = Vec::with_capacity(count);
            for _ in 0..count {
                let tail = flat.split_off(n_tiles);
                band_tiles.push(flat);
                flat = tail;
            }
            StatsTiles::Planar {
                band_tiles,
                tile_masks,
                mask_total_pixels,
            }
        }
    };

    // Offload large (>256x256) statistics accumulation/Quickselect to bounded spawn_blocking so
    // heavy analytics requests never stall Tokio's async I/O reactor on mixed workloads.
    if total_pixels > 256 * 256 {
        let cog_owned = cog.clone();
        let pm_owned = poly_mask.cloned();
        let opts_owned = opts.clone();
        #[cfg(feature = "gpu")]
        let gpu_owned = engine.gpu.clone();
        let val = engine
            .store
            .run_heavy_cpu(move || {
                compute_statistics_cpu(
                    &cog_owned,
                    ifd_idx,
                    &crs,
                    pm_owned.as_ref(),
                    unscale,
                    nodata_override,
                    step,
                    sample_dims,
                    fetched,
                    &opts_owned,
                    #[cfg(feature = "gpu")]
                    gpu_owned.as_deref(),
                )
            })
            .await?;
        Ok(val)
    } else {
        Ok(compute_statistics_cpu(
            cog,
            ifd_idx,
            &crs,
            poly_mask,
            unscale,
            nodata_override,
            step,
            sample_dims,
            fetched,
            opts,
            #[cfg(feature = "gpu")]
            engine.gpu.as_deref(),
        ))
    }
}

/// Simple ray-casting / ALL_TOUCHED rasterizer for GeoJSON Feature masking
/// (`POST /*/feature`, `POST /*/statistics`), supporting Polygon, MultiPolygon,
/// LineString, MultiLineString, Point, MultiPoint, and GeometryCollection.
#[derive(Clone)]
pub struct PolygonMask {
    /// List of polygons; each polygon is `[outer_ring, hole1, ...]`, each ring is `Vec<(lon, lat)>`.
    pub polygons: Vec<Vec<Vec<(f64, f64)>>>,
    /// List of polylines (`LineString` / `MultiLineString`) in `(lon, lat)`.
    pub lines: Vec<Vec<(f64, f64)>>,
    /// List of points (`Point` / `MultiPoint`) in `(lon, lat)`.
    pub points: Vec<(f64, f64)>,
    /// Precomputed `[min_lon, min_lat, max_lon, max_lat]` for fast bounding-box rejection.
    pub bbox: [f64; 4],
    /// Raw `[min_x, min_y, max_x, max_y]` in the incoming GeoJSON `coord_crs` (exact before WGS84 round-trip).
    pub raw_bbox: [f64; 4],
    /// True when the incoming GeoJSON coordinates were provided in a projected `coord_crs`.
    pub coord_crs_projected: bool,
}

/// Extract a legacy GeoJSON 2008 `"crs"` member (e.g.
/// `{"type": "name", "properties": {"name": "EPSG:32610"}}` or
/// `{"type": "EPSG", "properties": {"code": 32610}}`) from a GeoJSON object.
///
/// Checks `val["crs"]`, and if `val` is a `Feature` without its own `"crs"`, also checks
/// `val["geometry"]["crs"]`. Returns `Ok(None)` when absent or `null`.
pub fn extract_geojson_crs(
    val: &serde_json::Value,
) -> Result<Option<cogzilla_proj::CrsModel>, EngineError> {
    let crs_val = val
        .get("crs")
        .filter(|v| !v.is_null())
        .or_else(|| {
            if val.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                val.get("geometry")
                    .and_then(|g| g.get("crs"))
                    .filter(|v| !v.is_null())
            } else {
                None
            }
        });
    let Some(crs_obj) = crs_val else {
        return Ok(None);
    };
    if let Some(s) = crs_obj.as_str() {
        let (model, _) = crate::http::parse_coord_crs_with_epsg(s)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    let crs_type = crs_obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let props = crs_obj.get("properties");
    if crs_type == "name" {
        let name = props
            .and_then(|p| p.get("name"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                EngineError::BadRequest(
                    "invalid GeoJSON crs: type 'name' requires properties.name string".into(),
                )
            })?;
        let (model, _) = crate::http::parse_coord_crs_with_epsg(name)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    if crs_type == "epsg" {
        let code_val = props.and_then(|p| p.get("code"));
        let raw = if let Some(n) = code_val.and_then(|v| v.as_u64()) {
            format!("EPSG:{n}")
        } else if let Some(s) = code_val.and_then(|v| v.as_str()) {
            if s.to_ascii_lowercase().starts_with("epsg:") {
                s.to_string()
            } else {
                format!("EPSG:{s}")
            }
        } else {
            return Err(EngineError::BadRequest(
                "invalid GeoJSON crs: type 'EPSG' requires properties.code".into(),
            ));
        };
        let (model, _) = crate::http::parse_coord_crs_with_epsg(&raw)
            .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON crs: {e}")))?;
        return Ok(Some(model));
    }
    Err(EngineError::BadRequest(format!(
        "unsupported GeoJSON crs member (expected type 'name' or 'EPSG'; pass ?coord_crs=EPSG:<code>)"
    )))
}

impl PolygonMask {
    /// Empty mask (e.g. RFC 7946 §3.2 Feature with `"geometry": null`), producing zero-count
    /// null statistics identical to a feature outside the dataset.
    pub fn empty() -> Self {
        Self {
            polygons: Vec::new(),
            lines: Vec::new(),
            points: Vec::new(),
            bbox: [0.0; 4],
            raw_bbox: [0.0; 4],
            coord_crs_projected: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }

    pub fn from_geojson(val: &serde_json::Value) -> Result<(Self, [f64; 4]), EngineError> {
        Self::from_geojson_in_crs(val, None)
    }

    pub fn from_geojson_in_crs(
        val: &serde_json::Value,
        coord_crs: Option<&cogzilla_proj::CrsModel>,
    ) -> Result<(Self, [f64; 4]), EngineError> {
        let owned_geojson_crs = if coord_crs.is_none() {
            extract_geojson_crs(val)?
        } else {
            None
        };
        let effective_crs = coord_crs.or(owned_geojson_crs.as_ref());
        let mut geometries: Vec<&serde_json::Value> = Vec::new();
        match val.get("type").and_then(|v| v.as_str()) {
            Some("FeatureCollection") => {
                let feats = val
                    .get("features")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        EngineError::BadRequest("GeoJSON FeatureCollection missing features".into())
                    })?;
                if feats.is_empty() {
                    return Err(EngineError::BadRequest(
                        "GeoJSON FeatureCollection has no features".into(),
                    ));
                }
                if feats.len() > 64 {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON FeatureCollection has {} features (max 64)",
                        feats.len()
                    )));
                }
                for f in feats {
                    let g = f.get("geometry").ok_or_else(|| {
                        EngineError::BadRequest("GeoJSON Feature missing geometry".into())
                    })?;
                    geometries.push(g);
                }
            }
            Some("Feature") => {
                let g = val.get("geometry").ok_or_else(|| {
                    EngineError::BadRequest("GeoJSON Feature missing geometry".into())
                })?;
                geometries.push(g);
            }
            _ => {
                geometries.push(val);
            }
        }

        const MAX_GEOJSON_VERTICES: usize = 65_536;
        const MAX_GEOJSON_RINGS: usize = 4_096;
        const MAX_GEOM_COLLECTION_DEPTH: usize = 8;
        const MAX_LEAF_GEOMETRIES: usize = 64;

        // Iteratively flatten any GeometryCollection nodes with bounded depth and leaf count.
        let mut flat_geoms: Vec<&serde_json::Value> = Vec::with_capacity(geometries.len());
        let mut stack: Vec<(&serde_json::Value, usize)> =
            geometries.into_iter().rev().map(|g| (g, 0)).collect();
        while let Some((g, depth)) = stack.pop() {
            if g.get("type").and_then(|v| v.as_str()) == Some("GeometryCollection") {
                if depth >= MAX_GEOM_COLLECTION_DEPTH {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON GeometryCollection exceeds max nesting depth {MAX_GEOM_COLLECTION_DEPTH}"
                    )));
                }
                let children = g
                    .get("geometries")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        EngineError::BadRequest(
                            "GeoJSON GeometryCollection missing geometries array".into(),
                        )
                    })?;
                if children.len().saturating_add(stack.len()).saturating_add(flat_geoms.len())
                    > MAX_LEAF_GEOMETRIES
                {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON exceeds max {MAX_LEAF_GEOMETRIES} geometries"
                    )));
                }
                for child in children.iter().rev() {
                    stack.push((child, depth + 1));
                }
            } else {
                flat_geoms.push(g);
                if flat_geoms.len() > MAX_LEAF_GEOMETRIES {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON exceeds max {MAX_LEAF_GEOMETRIES} geometries"
                    )));
                }
            }
        }

        let mut total_vertices = 0usize;
        let mut total_rings = 0usize;
        let raw_bounds = std::cell::Cell::new([
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ]);

        let parse_coord = |pt: &serde_json::Value| -> Result<(f64, f64), EngineError> {
            let p = pt
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coord".into()))?;
            if p.len() < 2 {
                return Err(EngineError::BadRequest("coord needs [lon, lat]".into()));
            }
            let (Some(x), Some(y)) = (p[0].as_f64(), p[1].as_f64()) else {
                return Err(EngineError::BadRequest(
                    "GeoJSON coordinates must be numeric".into(),
                ));
            };
            if !x.is_finite() || !y.is_finite() {
                return Err(EngineError::BadRequest(
                    "GeoJSON coordinates must be finite".into(),
                ));
            }
            let mut rb = raw_bounds.get();
            rb[0] = rb[0].min(x);
            rb[1] = rb[1].min(y);
            rb[2] = rb[2].max(x);
            rb[3] = rb[3].max(y);
            raw_bounds.set(rb);
            if let Some(crs) = effective_crs {
                let (lo, la) = crs.to_lonlat_deg(x, y);
                if !lo.is_finite() || !la.is_finite() {
                    return Err(EngineError::BadRequest(
                        "reprojected GeoJSON coordinates are non-finite".into(),
                    ));
                }
                Ok((lo, la))
            } else {
                if x.abs() > 360.0 || y.abs() > 90.0 {
                    return Err(EngineError::BadRequest(format!(
                        "GeoJSON coordinate [{x}, {y}] is outside WGS84 lon/lat bounds ([-360, 360] x [-90, 90]); \
                         if coordinates are in a projected CRS, pass ?coord_crs=EPSG:<code> or include a GeoJSON 'crs' member"
                    )));
                }
                Ok((x, y))
            }
        };

        let parse_ring = |arr: &serde_json::Value,
                          min_len: usize,
                          total_rings: &mut usize,
                          total_vertices: &mut usize|
         -> Result<Vec<(f64, f64)>, EngineError> {
            let list = arr
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coordinate array".into()))?;
            if list.len() < min_len {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry ring/line must have at least {min_len} vertices"
                )));
            }
            *total_rings += 1;
            if *total_rings > MAX_GEOJSON_RINGS {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry exceeds max {MAX_GEOJSON_RINGS} rings/lines"
                )));
            }
            *total_vertices = total_vertices.saturating_add(list.len());
            if *total_vertices > MAX_GEOJSON_VERTICES {
                return Err(EngineError::BadRequest(format!(
                    "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                )));
            }
            let mut ring = Vec::with_capacity(list.len());
            for pt in list {
                ring.push(parse_coord(pt)?);
            }
            Ok(ring)
        };

        let mut polygons = Vec::new();
        let mut lines = Vec::new();
        let mut points = Vec::new();
        for geom in flat_geoms {
            let gtype = geom
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| EngineError::BadRequest("GeoJSON missing geometry.type".into()))?;
            let coords = geom
                .get("coordinates")
                .ok_or_else(|| EngineError::BadRequest("GeoJSON missing coordinates".into()))?;
            match gtype {
                "Polygon" => {
                    let rings = coords
                        .as_array()
                        .ok_or_else(|| EngineError::BadRequest("invalid Polygon coords".into()))?;
                    let mut poly = Vec::new();
                    for r in rings {
                        poly.push(parse_ring(r, 3, &mut total_rings, &mut total_vertices)?);
                    }
                    polygons.push(poly);
                }
                "MultiPolygon" => {
                    let polys = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiPolygon coords".into())
                    })?;
                    for p in polys {
                        let rings = p
                            .as_array()
                            .ok_or_else(|| EngineError::BadRequest("invalid polygon".into()))?;
                        let mut poly = Vec::new();
                        for r in rings {
                            poly.push(parse_ring(r, 3, &mut total_rings, &mut total_vertices)?);
                        }
                        polygons.push(poly);
                    }
                }
                "LineString" => {
                    lines.push(parse_ring(coords, 2, &mut total_rings, &mut total_vertices)?);
                }
                "MultiLineString" => {
                    let ls = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiLineString coords".into())
                    })?;
                    for l in ls {
                        lines.push(parse_ring(l, 2, &mut total_rings, &mut total_vertices)?);
                    }
                }
                "Point" => {
                    total_vertices = total_vertices.saturating_add(1);
                    if total_vertices > MAX_GEOJSON_VERTICES {
                        return Err(EngineError::BadRequest(format!(
                            "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                        )));
                    }
                    points.push(parse_coord(coords)?);
                }
                "MultiPoint" => {
                    let pts = coords.as_array().ok_or_else(|| {
                        EngineError::BadRequest("invalid MultiPoint coords".into())
                    })?;
                    total_vertices = total_vertices.saturating_add(pts.len());
                    if total_vertices > MAX_GEOJSON_VERTICES {
                        return Err(EngineError::BadRequest(format!(
                            "GeoJSON geometry exceeds max {MAX_GEOJSON_VERTICES} vertices"
                        )));
                    }
                    for pt in pts {
                        points.push(parse_coord(pt)?);
                    }
                }
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported GeoJSON geometry type '{other}'"
                    )));
                }
            }
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for poly in &polygons {
            if let Some(outer) = poly.first() {
                for &(x, y) in outer {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        for line in &lines {
            for &(x, y) in line {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
        for &(x, y) in &points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let has_only_polygons = lines.is_empty() && points.is_empty();
        if !min_x.is_finite()
            || !min_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
            || (has_only_polygons && (min_x >= max_x || min_y >= max_y))
        {
            return Err(EngineError::BadRequest(
                "empty or degenerate GeoJSON geometry".into(),
            ));
        }
        let bbox = [min_x, min_y, max_x, max_y];
        let rb = raw_bounds.get();
        let raw_bbox = if rb.iter().all(|v| v.is_finite()) {
            rb
        } else {
            bbox
        };
        let coord_crs_projected =
            effective_crs.is_some_and(|c| !matches!(c, cogzilla_proj::CrsModel::Geographic));
        Ok((Self { polygons, lines, points, bbox, raw_bbox, coord_crs_projected }, bbox))
    }

    /// Compute the exact tight bounding box of this feature's vertices directly in `dst_crs`
    /// (avoiding the area expansion caused by reprojecting an axis-aligned WGS84 envelope).
    pub fn bbox_in_crs(&self, dst_crs: &cogzilla_proj::CrsModel) -> Option<[f64; 4]> {
        let mut x0 = f64::INFINITY;
        let mut y0 = f64::INFINITY;
        let mut x1 = f64::NEG_INFINITY;
        let mut y1 = f64::NEG_INFINITY;
        let mut visit = |lon: f64, lat: f64| {
            let (x, y) = dst_crs.from_lonlat_deg(lon, lat);
            if x.is_finite() && y.is_finite() {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        };
        for poly in &self.polygons {
            if let Some(outer) = poly.first() {
                for &(lo, la) in outer {
                    visit(lo, la);
                }
            }
        }
        for line in &self.lines {
            for &(lo, la) in line {
                visit(lo, la);
            }
        }
        for &(lo, la) in &self.points {
            visit(lo, la);
        }
        if x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite() && x0 < x1 && y0 < y1 {
            Some([x0, y0, x1, y1])
        } else {
            None
        }
    }

    fn ring_contains(ring: &[(f64, f64)], x: f64, y: f64) -> bool {
        let mut inside = false;
        let n = ring.len();
        if n < 3 {
            return false;
        }
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = ring[i];
            let (xj, yj) = ring[j];
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Total number of geometry vertices across all rings, lines, and points.
    pub fn vertex_count(&self) -> usize {
        let poly_v: usize = self
            .polygons
            .iter()
            .flat_map(|rings| rings.iter())
            .map(|ring| ring.len())
            .sum();
        let line_v: usize = self.lines.iter().map(|l| l.len()).sum();
        poly_v + line_v + self.points.len()
    }

    /// Guard against `O(pixels * vertices)` point-in-polygon CPU exhaustion.
    pub fn check_work_budget(&self, pixels: usize) -> Result<(), EngineError> {
        const MAX_POLYGON_WORK: usize = 256_000_000;
        let verts = self.vertex_count();
        if pixels.saturating_mul(verts) > MAX_POLYGON_WORK {
            return Err(EngineError::BadRequest(format!(
                "polygon complexity ({verts} vertices x {pixels} pixels) exceeds work budget ({MAX_POLYGON_WORK})"
            )));
        }
        Ok(())
    }

    #[inline(always)]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        if lon < self.bbox[0] || lon > self.bbox[2] || lat < self.bbox[1] || lat > self.bbox[3] {
            return false;
        }
        for poly in &self.polygons {
            if let Some(outer) = poly.first() {
                if Self::ring_contains(outer, lon, lat) {
                    let in_hole = poly[1..].iter().any(|h| Self::ring_contains(h, lon, lat));
                    if !in_hole {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Rasterize the polygon onto a `(w, h)` grid covering `bbox` in `dst_crs` using GDAL's
    /// `ALL_TOUCHED=TRUE` semantics: a pixel is `true` if its center is inside the polygon OR
    /// any polygon ring edge intersects the pixel cell `[px, px+1] x [py, py+1]`.
    pub fn rasterize_all_touched(
        &self,
        w: usize,
        h: usize,
        bbox: [f64; 4],
        dst_crs: &cogzilla_proj::CrsModel,
    ) -> Vec<bool> {
        let mut mask = vec![false; w * h];
        if w == 0 || h == 0 {
            return mask;
        }
        let dlx = (bbox[2] - bbox[0]) / (w as f64);
        let dly = (bbox[3] - bbox[1]) / (h as f64);
        if !dlx.is_finite() || !dly.is_finite() || dlx <= 0.0 || dly <= 0.0 {
            return mask;
        }
        let wrap_hi = self.bbox[2] > 180.0;
        let wrap_lo = self.bbox[0] < -180.0;
        for py in 0..h {
            let y = bbox[3] - (py as f64 + 0.5) * dly;
            let row_off = py * w;
            for px in 0..w {
                let x = bbox[0] + (px as f64 + 0.5) * dlx;
                let (lon, lat) = dst_crs.to_lonlat_deg(x, y);
                if lon.is_finite()
                    && lat.is_finite()
                    && (self.contains(lon, lat)
                        || (wrap_hi && self.contains(lon + 360.0, lat))
                        || (wrap_lo && self.contains(lon - 360.0, lat)))
                {
                    mask[row_off + px] = true;
                }
            }
        }
        let mut mark_segment = |ax: f64, ay: f64, bx: f64, by: f64| {
            let x0 = (ax - bbox[0]) / dlx;
            let y0 = (bbox[3] - ay) / dly;
            let x1 = (bx - bbox[0]) / dlx;
            let y1 = (bbox[3] - by) / dly;
            if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
                return;
            }
            let dx = x1 - x0;
            let dy = y1 - y0;
            let mut t0 = 0.0f64;
            let mut t1 = 1.0f64;
            for (p, q) in [
                (-dx, x0),
                (dx, w as f64 - x0),
                (-dy, y0),
                (dy, h as f64 - y0),
            ] {
                if p.abs() < 1e-15 {
                    if q < 0.0 {
                        return;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 {
                        if r > t1 {
                            return;
                        }
                        if r > t0 {
                            t0 = r;
                        }
                    } else {
                        if r < t0 {
                            return;
                        }
                        if r < t1 {
                            t1 = r;
                        }
                    }
                }
            }
            let sx = (x0 + t0 * dx).clamp(0.0, w as f64);
            let sy = (y0 + t0 * dy).clamp(0.0, h as f64);
            let ex = (x0 + t1 * dx).clamp(0.0, w as f64);
            let ey = (y0 + t1 * dy).clamp(0.0, h as f64);
            let mut ix = (sx.floor() as isize).clamp(0, w as isize - 1);
            let mut iy = (sy.floor() as isize).clamp(0, h as isize - 1);
            let end_x = (ex.floor() as isize).clamp(0, w as isize - 1);
            let end_y = (ey.floor() as isize).clamp(0, h as isize - 1);
            mask[iy as usize * w + ix as usize] = true;
            let sdx = ex - sx;
            let sdy = ey - sy;
            let step_x: isize = match end_x.cmp(&ix) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            };
            let step_y: isize = match end_y.cmp(&iy) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            };
            let mut t_max_x = if step_x > 0 {
                ((ix + 1) as f64 - sx) / sdx
            } else if step_x < 0 {
                (ix as f64 - sx) / sdx
            } else {
                f64::INFINITY
            };
            let mut t_max_y = if step_y > 0 {
                ((iy + 1) as f64 - sy) / sdy
            } else if step_y < 0 {
                (iy as f64 - sy) / sdy
            } else {
                f64::INFINITY
            };
            let t_delta_x = if step_x != 0 {
                (1.0 / sdx).abs()
            } else {
                f64::INFINITY
            };
            let t_delta_y = if step_y != 0 {
                (1.0 / sdy).abs()
            } else {
                f64::INFINITY
            };
            let max_steps = (w + h) * 2 + 4;
            for _ in 0..max_steps {
                if ix == end_x && iy == end_y {
                    break;
                }
                if t_max_x < t_max_y {
                    ix += step_x;
                    t_max_x += t_delta_x;
                } else if t_max_y < t_max_x {
                    iy += step_y;
                    t_max_y += t_delta_y;
                } else {
                    ix += step_x;
                    iy += step_y;
                    t_max_x += t_delta_x;
                    t_max_y += t_delta_y;
                }
                if ix < 0 || ix >= w as isize || iy < 0 || iy >= h as isize {
                    break;
                }
                mask[iy as usize * w + ix as usize] = true;
            }
        };
        for poly in &self.polygons {
            for ring in poly {
                let n = ring.len();
                if n < 2 {
                    continue;
                }
                for i in 0..n {
                    let (lon0, lat0) = ring[i];
                    let (lon1, lat1) = ring[(i + 1) % n];
                    let (ax, ay) = dst_crs.from_lonlat_deg(lon0, lat0);
                    let (bx, by) = dst_crs.from_lonlat_deg(lon1, lat1);
                    mark_segment(ax, ay, bx, by);
                }
            }
        }
        for line in &self.lines {
            for seg in line.windows(2) {
                let (ax, ay) = dst_crs.from_lonlat_deg(seg[0].0, seg[0].1);
                let (bx, by) = dst_crs.from_lonlat_deg(seg[1].0, seg[1].1);
                mark_segment(ax, ay, bx, by);
            }
        }
        for &(lon, lat) in &self.points {
            let (px_crs, py_crs) = dst_crs.from_lonlat_deg(lon, lat);
            let fx = (px_crs - bbox[0]) / dlx;
            let fy = (bbox[3] - py_crs) / dly;
            if fx.is_finite() && fy.is_finite() {
                let ix = fx.floor() as isize;
                let iy = fy.floor() as isize;
                if ix >= 0 && ix < w as isize && iy >= 0 && iy < h as isize {
                    mask[iy as usize * w + ix as usize] = true;
                }
            }
        }
        mask
    }
}

/// TiTiler auto-format encoder for extension-less routes: emits `image/jpeg` when all pixels
/// are opaque (alpha == 255) and `image/png` when any pixel is transparent/masked.
pub(crate) fn encode_auto_image(
    w: usize,
    h: usize,
    rgba: &[u8],
    jpeg_quality: u8,
    return_mask: bool,
) -> (Bytes, &'static str) {
    let pixels = w.saturating_mul(h);
    let all_opaque = !return_mask
        || (pixels > 0 && rgba.chunks_exact(4).take(pixels).all(|p| p[3] == 255));
    if all_opaque {
        (
            encode_rgba_output(w, h, rgba, OutputFormat::Jpeg { quality: jpeg_quality }, false),
            "image/jpeg",
        )
    } else {
        (
            encode_rgba_output(w, h, rgba, OutputFormat::Png, return_mask),
            "image/png",
        )
    }
}

/// Encode an RGBA8 buffer to the requested `OutputFormat`, optionally stripping the alpha mask.
pub(crate) fn encode_rgba_output(
    w: usize,
    h: usize,
    rgba: &[u8],
    format: OutputFormat,
    return_mask: bool,
) -> Bytes {
    thread_local! {
        static PNG_SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    const MAX_TLS_SCRATCH: usize = 4 * 1024 * 1024;
    match format {
        OutputFormat::Png => {
            let pixels = w.saturating_mul(h);
            let is_gray = rgba
                .chunks_exact(4)
                .take(pixels)
                .all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
            let all_opaque = return_mask
                && rgba.chunks_exact(4).take(pixels).all(|p| p[3] == 255);
            PNG_SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let (ref mut filtered, ref mut row_buf) = *guard;
                let out = if is_gray && (!return_mask || all_opaque) {
                    let stride = 1 + w;
                    filtered.resize(h * stride, 0);
                    row_buf.resize(w, 0);
                    for py in 0..h {
                        let src_row = &rgba[py * w * 4..(py + 1) * w * 4];
                        for (src, dst) in src_row.chunks_exact(4).zip(row_buf.iter_mut()) {
                            *dst = if src[3] != 0 { src[0] } else { 0 };
                        }
                        cogzilla_kernel::png_fast::filter_sub_u8_row(
                            row_buf,
                            &mut filtered[py * stride..(py + 1) * stride],
                        );
                    }
                    encode_filtered_png(w as u32, h as u32, 0, filtered)
                } else if is_gray {
                    let stride = 1 + w * 2;
                    filtered.resize(h * stride, 0);
                    row_buf.resize(w * 2, 0);
                    for py in 0..h {
                        let src_row = &rgba[py * w * 4..(py + 1) * w * 4];
                        for (src, dst) in
                            src_row.chunks_exact(4).zip(row_buf.chunks_exact_mut(2))
                        {
                            dst[0] = if src[3] != 0 { src[0] } else { 0 };
                            dst[1] = src[3];
                        }
                        cogzilla_kernel::png_fast::filter_sub_ga_row(
                            row_buf,
                            &mut filtered[py * stride..(py + 1) * stride],
                        );
                    }
                    encode_filtered_png(w as u32, h as u32, 4, filtered)
                } else if !return_mask {
                    let stride = 1 + w * 3;
                    filtered.resize(h * stride, 0);
                    row_buf.resize(w * 3, 0);
                    for py in 0..h {
                        let src_row = &rgba[py * w * 4..(py + 1) * w * 4];
                        for (src, dst) in
                            src_row.chunks_exact(4).zip(row_buf.chunks_exact_mut(3))
                        {
                            if src[3] != 0 {
                                dst[0] = src[0];
                                dst[1] = src[1];
                                dst[2] = src[2];
                            } else {
                                dst[0] = 0;
                                dst[1] = 0;
                                dst[2] = 0;
                            }
                        }
                        filter_sub_rgb_row(
                            row_buf,
                            &mut filtered[py * stride..(py + 1) * stride],
                        );
                    }
                    encode_filtered_png(w as u32, h as u32, 2, filtered)
                } else {
                    let stride = 1 + w * 4;
                    filtered.resize(h * stride, 0);
                    for py in 0..h {
                        filter_sub_rgba_row(
                            &rgba[py * w * 4..(py + 1) * w * 4],
                            &mut filtered[py * stride..(py + 1) * stride],
                        );
                    }
                    encode_filtered_png(w as u32, h as u32, 6, filtered)
                };
                if filtered.capacity() > MAX_TLS_SCRATCH {
                    *filtered = Vec::new();
                }
                if row_buf.capacity() > MAX_TLS_SCRATCH {
                    *row_buf = Vec::new();
                }
                out
            })
        }
        OutputFormat::Webp { quality } => {
            if !return_mask {
                let mut rgb = vec![0u8; w * h * 3];
                for i in 0..(w * h) {
                    if rgba[i * 4 + 3] != 0 {
                        rgb[i * 3] = rgba[i * 4];
                        rgb[i * 3 + 1] = rgba[i * 4 + 1];
                        rgb[i * 3 + 2] = rgba[i * 4 + 2];
                    }
                }
                encode_webp_rgb(w as u32, h as u32, &rgb, quality)
            } else {
                encode_webp_rgba(w as u32, h as u32, rgba, quality)
            }
        }
        OutputFormat::Jpeg { quality } => encode_jpeg_rgba(w as u32, h as u32, rgba, quality),
        OutputFormat::RawF32 | OutputFormat::Npy | OutputFormat::RawRgba => {
            Bytes::copy_from_slice(rgba)
        }
    }
}

/// Native pixel-space preview renderer backing `/cog/preview*` and `/stac/preview*`,
/// matching TiTiler / `rio-tiler`'s `Reader.preview()` (reads directly in source pixel space
/// without reprojecting through an axis-aligned WGS84 bounding box).
#[allow(clippy::too_many_arguments)]
pub async fn render_native_preview(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox_wgs84: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox_wgs84,
        &cogzilla_proj::CrsModel::Geographic,
        true,
        width,
        height,
        None,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-dimensions WGS84 bounding-box renderer backing `/cog/bbox`,
/// `POST /cog/feature`, and `WorldCRS84Quad` tiles.
#[allow(clippy::too_many_arguments)]
pub async fn render_wgs84_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4], // [min_lon, min_lat, max_lon, max_lat]
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-CRS bounding-box renderer backing `?dst_crs=` and all 13 OGC TileMatrixSets.
#[allow(clippy::too_many_arguments)]
pub async fn render_crs_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox,
        dst_crs,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-dimensions EPSG:3857 Web Mercator bounding-box renderer backing `WebMercatorQuad`
/// tiles with `?nodata=`, `?buffer=`, or custom JSON `?colormap=` in true Web Mercator meters.
#[allow(clippy::too_many_arguments)]
pub async fn render_mercator_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox_3857: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox_3857,
        &cogzilla_proj::CrsModel::Epsg3857,
        false,
        width,
        height,
        None,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

enum BboxRenderOutput {
    Bytes(Bytes),
    #[cfg(feature = "gpu")]
    Stats(serde_json::Value),
}

/// 1-pass fused GPU (with seamless CPU/2-pass fallback) native-preview band/expression statistics.
#[allow(clippy::too_many_arguments)]
pub async fn render_native_preview_stats(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    nodata_override: Option<f32>,
    unscale: bool,
    description: &str,
    stats_opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let n_px = (width.clamp(1, 4096) as usize) * (height.clamp(1, 4096) as usize);
    match render_bbox_core(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        true,
        width,
        height,
        None,
        bands,
        resample,
        OutputFormat::RawF32,
        nodata_override,
        None,
        None,
        true,
        unscale,
        Some((stats_opts, description)),
    )
    .await?
    {
        #[cfg(feature = "gpu")]
        BboxRenderOutput::Stats(val) => Ok(val),
        BboxRenderOutput::Bytes(raw) => {
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .take(n_px)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(compute_f32_band_stats_with_engine(
                engine,
                &pixels,
                description,
                stats_opts,
            ))
        }
    }
}

/// 1-pass fused GPU (with seamless CPU/2-pass fallback) WGS84 bbox / polygon-masked statistics.
#[allow(clippy::too_many_arguments)]
pub async fn render_wgs84_bbox_stats(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    nodata_override: Option<f32>,
    unscale: bool,
    description: &str,
    stats_opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let n_px = (width.clamp(1, 4096) as usize) * (height.clamp(1, 4096) as usize);
    match render_bbox_core(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        OutputFormat::RawF32,
        nodata_override,
        None,
        None,
        true,
        unscale,
        Some((stats_opts, description)),
    )
    .await?
    {
        #[cfg(feature = "gpu")]
        BboxRenderOutput::Stats(val) => Ok(val),
        BboxRenderOutput::Bytes(raw) => {
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .take(n_px)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(compute_f32_band_stats_with_engine(
                engine,
                &pixels,
                description,
                stats_opts,
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn render_bbox_impl(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    match render_bbox_core(
        engine,
        cogs,
        bbox,
        dst_crs,
        native_preview,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
        None,
    )
    .await?
    {
        BboxRenderOutput::Bytes(b) => Ok(b),
        #[cfg(feature = "gpu")]
        BboxRenderOutput::Stats(_) => unreachable!("fused_stats is None"),
    }
}

#[allow(clippy::too_many_arguments)]
async fn render_bbox_core(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    mut bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
    fused_stats: Option<(&StatsOptions, &str)>,
) -> Result<BboxRenderOutput, EngineError> {
    if cogs.is_empty() || cogs.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "render_bbox supports 1..=16 assets, got {}",
            cogs.len()
        )));
    }
    if !native_preview {
        if !bbox.iter().all(|v| v.is_finite()) || bbox[1] >= bbox[3] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
        // RFC 7946 antimeridian-crossing geographic bboxes use w > e (e.g. [177, -20, -178, -16]);
        // unwrap e into (180, 540] so dlx > 0 and clamp() never sees min > max.
        if matches!(dst_crs, cogzilla_proj::CrsModel::Geographic) && bbox[0] > bbox[2] {
            bbox[2] += 360.0;
        }
        if bbox[0] >= bbox[2] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
    }
    if let BandRequest::MultiBand { bands_1based } = bands
        && !matches!(format, OutputFormat::RawF32 | OutputFormat::Npy)
    {
        return Err(EngineError::BadRequest(format!(
            "unsupported bidx count {} for visual image output (expected 1 or 3 bands; use .tif, .npy, or .f32)",
            bands_1based.len()
        )));
    }
    let w = width.clamp(1, 4096) as usize;
    let h = height.clamp(1, 4096) as usize;
    let primary = &cogs[0];
    let crs = if native_preview && !primary.index.is_georeferenced {
        cogzilla_proj::CrsModel::Geographic
    } else {
        engine.get_or_build_crs(primary)?
    };

    let cx = 0.5 * (bbox[0] + bbox[2]);
    let cy = 0.5 * (bbox[1] + bbox[3]);
    let dlx = (bbox[2] - bbox[0]) / (w as f64);
    let dly = (bbox[3] - bbox[1]) / (h as f64);

    let (ifd_idx, target_rx, target_ry) = if native_preview {
        let full = primary.index.full_res();
        let rx0 = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d);
        let ry0 = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e);
        let target_rx = rx0 * (full.width as f64) / (w as f64);
        let target_ry = ry0 * (full.height as f64) / (h as f64);
        (primary.index.select_ifd(target_rx, target_ry), target_rx, target_ry)
    } else {
        let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
            match dst_crs {
                cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                cogzilla_proj::CrsModel::Geographic => {
                    let lon = if x > 180.0 { x - 360.0 } else { x };
                    crs.from_lonlat_deg(lon, y)
                }
                other => {
                    let (lon, lat) = other.to_lonlat_deg(x, y);
                    if lon.is_finite() && lat.is_finite() {
                        crs.from_lonlat_deg(lon, lat)
                    } else {
                        (f64::NAN, f64::NAN)
                    }
                }
            }
        };

        // Estimate target CRS resolution to pick optimal overview IFD.
        // Probe bbox center first; if out of CRS domain (e.g. WorldCRS84Quad z=0/1 on a UTM COG >75°
        // from center), fall back to the raster's own center clamped into bbox, then 4 quadrant probes.
        let probe_res = |px: f64, py: f64| -> (f64, f64) {
            let (s0x, s0y) = to_src_crs(px, py);
            let (s1x, s1y) = to_src_crs(px + dlx, py);
            let (s2x, s2y) = to_src_crs(px, py + dly);
            let rx = ((s1x - s0x).powi(2) + (s1y - s0y).powi(2)).sqrt();
            let ry = ((s2x - s0x).powi(2) + (s2y - s0y).powi(2)).sqrt();
            (rx, ry)
        };
        let (mut target_rx, mut target_ry) = probe_res(cx, cy);
        if !target_rx.is_finite() || !target_ry.is_finite() {
            let full = primary.index.full_res();
            let rb = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
            let (mut rcx_raw, rcy_raw) =
                dst_crs.from_lonlat_deg(0.5 * (rb[0] + rb[2]), 0.5 * (rb[1] + rb[3]));
            if matches!(dst_crs, cogzilla_proj::CrsModel::Geographic)
                && rcx_raw < bbox[0]
                && rcx_raw + 360.0 <= bbox[2]
            {
                rcx_raw += 360.0;
            }
            let rcx = rcx_raw.clamp(bbox[0], bbox[2]);
            let rcy = rcy_raw.clamp(bbox[1], bbox[3]);
            let probes = [
                (rcx, rcy),
                (bbox[0] + 0.25 * (bbox[2] - bbox[0]), bbox[1] + 0.25 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.75 * (bbox[2] - bbox[0]), bbox[1] + 0.25 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.25 * (bbox[2] - bbox[0]), bbox[1] + 0.75 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.75 * (bbox[2] - bbox[0]), bbox[1] + 0.75 * (bbox[3] - bbox[1])),
            ];
            for (px, py) in probes {
                let (rx, ry) = probe_res(px, py);
                if rx.is_finite() && ry.is_finite() && rx > 0.0 && ry > 0.0 {
                    target_rx = rx;
                    target_ry = ry;
                    break;
                }
            }
        }
        (primary.index.select_ifd(target_rx, target_ry), target_rx, target_ry)
    };
    if let Some(pm) = poly_mask {
        pm.check_work_budget(w.saturating_mul(h))?;
    }
    if cogs.len() > 1 && crate::engine::validate_multi_cog_grid(cogs, ifd_idx).is_err() {
        return render_mixed_grid_bbox(
            engine,
            cogs,
            bbox,
            dst_crs,
            native_preview,
            w,
            h,
            poly_mask,
            bands,
            resample,
            format,
            nodata_override,
            custom_lut,
            color_formula,
            return_mask,
            unscale,
            None,
        )
        .await
        .map(BboxRenderOutput::Bytes);
    }
    let ifd = &primary.index.ifds[ifd_idx];

    // Precompute source pixel coordinates `(u, v)`.
    // For native preview (`native_preview == true`), map directly in source pixel space without
    // reprojecting through an axis-aligned WGS84 bbox (matching `rio-tiler`'s `Reader.preview()`).
    // Otherwise use a coarse 16-px projection grid + bilinear interpolation.
    let (u_map, v_map, min_u, max_u, min_v, max_v) = if native_preview {
        let mut u_map = vec![0.0f32; w * h];
        let mut v_map = vec![0.0f32; w * h];
        let sx = (ifd.width as f64) / (w as f64);
        let sy = (ifd.height as f64) / (h as f64);
        let u_cols: Vec<f32> = (0..w).map(|px| ((px as f64 + 0.5) * sx - 0.5) as f32).collect();
        for py in 0..h {
            let vv = ((py as f64 + 0.5) * sy - 0.5) as f32;
            let row_off = py * w;
            u_map[row_off..row_off + w].copy_from_slice(&u_cols);
            v_map[row_off..row_off + w].fill(vv);
        }
        let min_u = u_cols.first().copied().unwrap_or(0.0);
        let max_u = u_cols.last().copied().unwrap_or(0.0);
        let min_v = ((0.5 * sy) - 0.5) as f32;
        let max_v = (((h as f64 - 0.5) * sy) - 0.5) as f32;
        (u_map, v_map, min_u, max_u, min_v, max_v)
    } else {
        let build_uv_maps = {
        let crs = crs.clone();
        let dst_crs = dst_crs.clone();
        let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
        let crs_to_pixel = ifd.crs_to_pixel;
        let gcp_poly = ifd.gcp_poly.clone();
        let ifd_width = ifd.width;
        let ifd_height = ifd.height;
        let poly_mask = poly_mask.cloned();
        move || {
            let map_crs_to_px = |sx: f64, sy: f64| -> (f64, f64) {
                if let Some(ref poly) = gcp_poly {
                    poly.crs_to_pixel(sx, sy)
                } else {
                    crs_to_pixel.forward(sx, sy)
                }
            };
            let touched_mask: Option<Vec<bool>> = poly_mask
                .as_ref()
                .map(|pm| pm.rasterize_all_touched(w, h, bbox, &dst_crs));
            let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
                let (mut sx, sy) = match &dst_crs {
                    cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                    cogzilla_proj::CrsModel::Geographic => {
                        let lon = if x > 180.0 { x - 360.0 } else { x };
                        crs.from_lonlat_deg(lon, y)
                    }
                    other => {
                        let (lon, lat) = other.to_lonlat_deg(x, y);
                        if lon.is_finite() && lat.is_finite() {
                            crs.from_lonlat_deg(lon, lat)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    }
                };
                if let Some(lo) = wrap_lo
                    && sx.is_finite()
                    && sx < lo
                {
                    sx += 360.0;
                }
                (sx, sy)
            };
            let mut u_map = vec![f32::NAN; w * h];
            let mut v_map = vec![f32::NAN; w * h];
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            let ifd_w_f = ifd_width as f32;
            let ifd_h_f = ifd_height as f32;

            // Separable 1D fast path: when both dst_crs and src_crs are separable cylindrical/geographic
            // projections and the raster is axis-aligned, u depends only on px and v depends only on py.
            // Uses w + h exact evaluations (faster than a 16-px bilinear grid and 100% exact at every row).
            let is_separable = gcp_poly.is_none()
                && crs_to_pixel.is_axis_aligned()
                && matches!(
                    dst_crs,
                    cogzilla_proj::CrsModel::Geographic
                        | cogzilla_proj::CrsModel::Epsg3857
                        | cogzilla_proj::CrsModel::WorldMercator
                        | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
                )
                && matches!(
                    crs,
                    cogzilla_proj::CrsModel::Geographic
                        | cogzilla_proj::CrsModel::Epsg3857
                        | cogzilla_proj::CrsModel::WorldMercator
                        | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
                );
            if is_separable {
                // Probe separable axes at the destination CRS's origin (0, 0) / (false_easting, false_northing)
                // so a partially overlapping CylindricalEqualArea bbox whose midpoint falls beyond ±y_pole
                // or ±x_max never produces NaN for every column/row.
                let (origin_x, origin_y) = match &dst_crs {
                    cogzilla_proj::CrsModel::CylindricalEqualArea(cea) => {
                        (cea.false_easting, cea.false_northing)
                    }
                    _ => (0.0, 0.0),
                };
                let mut u_cols = vec![f32::NAN; w];
                for px in 0..w {
                    let x_coord = bbox[0] + (px as f64 + 0.5) * dlx;
                    let (sx, _) = to_src_crs(x_coord, origin_y);
                    if sx.is_finite() {
                        let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                        if uu >= -1.0 && uu <= ifd_w_f {
                            u_cols[px] = uu;
                        }
                    }
                }
                let mut v_rows = vec![f32::NAN; h];
                for py in 0..h {
                    let y_coord = bbox[3] - (py as f64 + 0.5) * dly;
                    let (_, sy) = to_src_crs(origin_x, y_coord);
                    if sy.is_finite() {
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        if vv >= -1.0 && vv <= ifd_h_f {
                            v_rows[py] = vv;
                        }
                    }
                }
                for py in 0..h {
                    let vv = v_rows[py];
                    if !vv.is_finite() {
                        continue;
                    }
                    let row_off = py * w;
                    if let Some(ref tm) = touched_mask {
                        for px in 0..w {
                            let uu = u_cols[px];
                            if tm[row_off + px] && uu.is_finite() {
                                u_map[row_off + px] = uu;
                                v_map[row_off + px] = vv;
                                min_u = min_u.min(uu);
                                max_u = max_u.max(uu);
                                min_v = min_v.min(vv);
                                max_v = max_v.max(vv);
                            }
                        }
                    } else {
                        for px in 0..w {
                            let uu = u_cols[px];
                            if uu.is_finite() {
                                u_map[row_off + px] = uu;
                                v_map[row_off + px] = vv;
                                min_u = min_u.min(uu);
                                max_u = max_u.max(uu);
                                min_v = min_v.min(vv);
                                max_v = max_v.max(vv);
                            }
                        }
                    }
                }
                return (u_map, v_map, min_u, max_u, min_v, max_v);
            }

            const STEP: usize = 16;
            let gw = w.div_ceil(STEP) + 1;
            let gh = h.div_ceil(STEP) + 1;
            let mut grid_u = vec![f64::NAN; gw * gh];
            let mut grid_v = vec![f64::NAN; gw * gh];
            for gy in 0..gh {
                let py_f = (gy * STEP).min(h) as f64;
                let y_coord = bbox[3] - py_f * dly;
                for gx in 0..gw {
                    let px_f = (gx * STEP).min(w) as f64;
                    let x_coord = bbox[0] + px_f * dlx;
                    let (sx, sy) = to_src_crs(x_coord, y_coord);
                    if sx.is_finite() && sy.is_finite() {
                        let (col, row) = map_crs_to_px(sx, sy);
                        grid_u[gy * gw + gx] = col - 0.5;
                        grid_v[gy * gw + gx] = row - 0.5;
                    }
                }
            }

            for gy in 0..(gh - 1) {
                let y0 = gy * STEP;
                let y1 = ((gy + 1) * STEP).min(h);
                if y0 >= y1 {
                    continue;
                }
                let inv_dy = 1.0 / ((y1 - y0) as f64);
                for gx in 0..(gw - 1) {
                    let x0 = gx * STEP;
                    let x1 = ((gx + 1) * STEP).min(w);
                    if x0 >= x1 {
                        continue;
                    }
                    let inv_dx = 1.0 / ((x1 - x0) as f64);
                    let i00 = gy * gw + gx;
                    let u00 = grid_u[i00];
                    let u10 = grid_u[i00 + 1];
                    let u01 = grid_u[i00 + gw];
                    let u11 = grid_u[i00 + gw + 1];
                    let v00 = grid_v[i00];
                    let v10 = grid_v[i00 + 1];
                    let v01 = grid_v[i00 + gw];
                    let v11 = grid_v[i00 + gw + 1];
                    let all_finite = u00.is_finite()
                        && u10.is_finite()
                        && u01.is_finite()
                        && u11.is_finite()
                        && v00.is_finite()
                        && v10.is_finite()
                        && v01.is_finite()
                        && v11.is_finite();

                    // Skip entire cell if all 4 corners + midpoint are strictly outside the source image
                    // bounds, and fall back to exact per-pixel projection if the cell has high curvature
                    // (> 0.05 px midpoint error) or wraps across the antimeridian / pole in X or Y.
                    let mut use_bilinear = all_finite;
                    if all_finite {
                        let x_mid = bbox[0] + (0.5 * (x0 + x1) as f64) * dlx;
                        let y_mid = bbox[3] - (0.5 * (y0 + y1) as f64) * dly;
                        let (msx, msy) = to_src_crs(x_mid, y_mid);
                        let mid_ok = if msx.is_finite() && msy.is_finite() {
                            let (mcol, mrow) = map_crs_to_px(msx, msy);
                            let mu = mcol - 0.5;
                            let mv = mrow - 0.5;
                            let u_avg = 0.25 * (u00 + u10 + u01 + u11);
                            let v_avg = 0.25 * (v00 + v10 + v01 + v11);
                            mu.is_finite()
                                && mv.is_finite()
                                && (mu - u_avg).abs() <= 1e-3
                                && (mv - v_avg).abs() <= 1e-3
                        } else {
                            false
                        };
                        if !mid_ok {
                            use_bilinear = false;
                        } else {
                            let cu_min = u00.min(u10).min(u01).min(u11) as f32;
                            let cu_max = u00.max(u10).max(u01).max(u11) as f32;
                            let cv_min = v00.min(v10).min(v01).min(v11) as f32;
                            let cv_max = v00.max(v10).max(v01).max(v11) as f32;
                            if cu_max < -1.0 || cu_min > ifd_w_f || cv_max < -1.0 || cv_min > ifd_h_f
                            {
                                continue;
                            }
                            if (ifd_width >= 64 && w > STEP * 2 && (cu_max - cu_min) > 0.5 * ifd_w_f)
                                || (ifd_height >= 64
                                    && h > STEP * 2
                                    && (cv_max - cv_min) > 0.5 * ifd_h_f)
                            {
                                use_bilinear = false;
                            }
                        }
                    }

                    for py in y0..y1 {
                        let row_off = py * w;
                        let ty = (py as f64 + 0.5 - y0 as f64) * inv_dy;
                        let lat = bbox[3] - (py as f64 + 0.5) * dly;
                        if use_bilinear {
                            let ul = u00 + (u01 - u00) * ty;
                            let ur = u10 + (u11 - u10) * ty;
                            let vl = v00 + (v01 - v00) * ty;
                            let vr = v10 + (v11 - v10) * ty;
                            let du = (ur - ul) * inv_dx;
                            let dv = (vr - vl) * inv_dx;
                            let mut u_cur = ul + 0.5 * du;
                            let mut v_cur = vl + 0.5 * dv;
                            if let Some(ref tm) = touched_mask {
                                for px in x0..x1 {
                                    if tm[row_off + px] {
                                        let uu = u_cur as f32;
                                        let vv = v_cur as f32;
                                        if uu >= -1.0
                                            && uu <= ifd_w_f
                                            && vv >= -1.0
                                            && vv <= ifd_h_f
                                        {
                                            u_map[row_off + px] = uu;
                                            v_map[row_off + px] = vv;
                                            min_u = min_u.min(uu);
                                            max_u = max_u.max(uu);
                                            min_v = min_v.min(vv);
                                            max_v = max_v.max(vv);
                                        }
                                    }
                                    u_cur += du;
                                    v_cur += dv;
                                }
                            } else {
                                for px in x0..x1 {
                                    let uu = u_cur as f32;
                                    let vv = v_cur as f32;
                                    if uu >= -1.0
                                        && uu <= ifd_w_f
                                        && vv >= -1.0
                                        && vv <= ifd_h_f
                                    {
                                        u_map[row_off + px] = uu;
                                        v_map[row_off + px] = vv;
                                        min_u = min_u.min(uu);
                                        max_u = max_u.max(uu);
                                        min_v = min_v.min(vv);
                                        max_v = max_v.max(vv);
                                    }
                                    u_cur += du;
                                    v_cur += dv;
                                }
                            }
                        } else {
                            for px in x0..x1 {
                                let lon = bbox[0] + (px as f64 + 0.5) * dlx;
                                if let Some(ref tm) = touched_mask {
                                    if !tm[row_off + px] {
                                        continue;
                                    }
                                }
                                let (sx, sy) = to_src_crs(lon, lat);
                                if !sx.is_finite() || !sy.is_finite() {
                                    continue;
                                }
                                let (col, row) = map_crs_to_px(sx, sy);
                                let uu = (col - 0.5) as f32;
                                let vv = (row - 0.5) as f32;
                                if uu >= -1.0 && uu <= ifd_w_f && vv >= -1.0 && vv <= ifd_h_f {
                                    u_map[row_off + px] = uu;
                                    v_map[row_off + px] = vv;
                                    min_u = min_u.min(uu);
                                    max_u = max_u.max(uu);
                                    min_v = min_v.min(vv);
                                    max_v = max_v.max(vv);
                                }
                            }
                        }
                    }
                }
            }
            (u_map, v_map, min_u, max_u, min_v, max_v)
        }
    };
    let (u_map, v_map, min_u, max_u, min_v, max_v) = if w * h >= 65_536 || poly_mask.is_some() {
        engine
            .store
            .run_heavy_cpu(build_uv_maps)
            .await?
    } else {
        build_uv_maps()
    };
    (u_map, v_map, min_u, max_u, min_v, max_v)
    };

    // Empty check: zero source pixels mapped into the requested tile/bbox/feature.
    if !min_u.is_finite() {
        let full = primary.index.full_res();
        let rb = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
        return Err(EngineError::NotFound(format!(
            "request is outside dataset bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
            rb[0], rb[1], rb[2], rb[3]
        )));
    }

    let (footprint_sx, footprint_sy) = if native_preview {
        (
            (ifd.width as f64 / w as f64) as f32,
            (ifd.height as f64 / h as f64) as f32,
        )
    } else {
        let ifd_rx = ifd.pixel_to_crs.a.hypot(ifd.pixel_to_crs.d);
        let ifd_ry = ifd.pixel_to_crs.b.hypot(ifd.pixel_to_crs.e);
        let sx = if target_rx.is_finite() && ifd_rx > 0.0 {
            (target_rx / ifd_rx).max(1e-3) as f32
        } else {
            1.0
        };
        let sy = if target_ry.is_finite() && ifd_ry > 0.0 {
            (target_ry / ifd_ry).max(1e-3) as f32
        } else {
            1.0
        };
        (sx, sy)
    };
    let is_footprint_mode = matches!(
        resample,
        ResampleMode::Average
            | ResampleMode::Mode
            | ResampleMode::Rms
            | ResampleMode::Min
            | ResampleMode::Max
            | ResampleMode::Med
            | ResampleMode::Q1
            | ResampleMode::Q3
            | ResampleMode::Sum
    );
    let max_c = (ifd.width as i32 - 1).max(0);
    let max_r = (ifd.height as i32 - 1).max(0);
    let is_dem = matches!(bands, BandRequest::Dem { .. });
    let (c0, c1, r0, r1) = if is_dem {
        let (lo, hi) = crate::engine::resample_fetch_margin(resample, true);
        (
            (min_u.floor() as i32).saturating_sub(lo).clamp(0, max_c) as u32,
            (max_u.floor() as i32).saturating_add(hi).clamp(0, max_c) as u32,
            (min_v.floor() as i32).saturating_sub(lo).clamp(0, max_r) as u32,
            (max_v.floor() as i32).saturating_add(hi).clamp(0, max_r) as u32,
        )
    } else if resample == ResampleMode::Nearest {
        (
            ((min_u + 0.5).floor() as i32).clamp(0, max_c) as u32,
            ((max_u + 0.5).floor() as i32).clamp(0, max_c) as u32,
            ((min_v + 0.5).floor() as i32).clamp(0, max_r) as u32,
            ((max_v + 0.5).floor() as i32).clamp(0, max_r) as u32,
        )
    } else if is_footprint_mode {
        let hx = 0.5 * footprint_sx;
        let hy = 0.5 * footprint_sy;
        (
            ((min_u + 0.5 - hx).floor() as i32).clamp(0, max_c) as u32,
            (((max_u + 0.5 + hx).ceil() as i32) - 1).clamp(0, max_c) as u32,
            ((min_v + 0.5 - hy).floor() as i32).clamp(0, max_r) as u32,
            (((max_v + 0.5 + hy).ceil() as i32) - 1).clamp(0, max_r) as u32,
        )
    } else {
        let (lo, hi) = crate::engine::resample_fetch_margin(resample, false);
        (
            (min_u.floor() as i32).saturating_sub(lo).clamp(0, max_c) as u32,
            (max_u.floor() as i32).saturating_add(hi).clamp(0, max_c) as u32,
            (min_v.floor() as i32).saturating_sub(lo).clamp(0, max_r) as u32,
            (max_v.floor() as i32).saturating_add(hi).clamp(0, max_r) as u32,
        )
    };
    let tx1 = (c1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
    let tx0 = (c0 / ifd.tile_width).min(tx1);
    let ty1 = (r1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
    let ty0 = (r0 / ifd.tile_height).min(ty1);
    let grid_w = tx1 - tx0 + 1;
    let grid_h = ty1 - ty0 + 1;

    let grid_len = (grid_w * grid_h) as usize;
    let is_single_planar =
        cogs.len() == 1 && ifd.planar_config == cogzilla_tiff::PlanarConfig::Planar;

    // Validate 1-based band indices on both single- and multi-COG requests and collect unique
    // 0-based planes needed when a single COG is `PlanarConfig::Planar`.
    let mut planar_bands: Vec<u16> = Vec::new();
    let max_bands = if cogs.len() == 1 {
        ifd.samples_per_pixel
    } else {
        cogs.len() as u16
    };
    {
        let mut check_band = |b_1based: u16| -> Result<(), EngineError> {
            let b0 = b_1based.saturating_sub(1);
            if b_1based == 0 || b0 >= max_bands {
                return Err(EngineError::BadRequest(format!(
                    "band {} out of range (dataset has {} bands)",
                    b_1based, max_bands
                )));
            }
            if is_single_planar && !planar_bands.contains(&b0) {
                planar_bands.push(b0);
            }
            Ok(())
        };
        match bands {
            BandRequest::Single { band_1based, .. }
            | BandRequest::Terrain { band_1based, .. }
            | BandRequest::Dem { band_1based, .. }
            | BandRequest::Spectral1 { band_1based, .. } => check_band(*band_1based)?,
            BandRequest::SarRgb { bands_1based, .. }
            | BandRequest::Spectral2 { bands_1based, .. } => {
                check_band(bands_1based[0])?;
                check_band(bands_1based[1])?;
            }
            BandRequest::Rgb { bands_1based, .. }
            | BandRequest::Spectral3 { bands_1based, .. } => {
                check_band(bands_1based[0])?;
                check_band(bands_1based[1])?;
                check_band(bands_1based[2])?;
            }
            BandRequest::CustomExpr { prog, .. } => {
                for &b in &prog.used_bands {
                    check_band(b)?;
                }
            }
            BandRequest::CustomExprRgb { progs, .. } => {
                for prog in progs {
                    for &b in &prog.used_bands {
                        check_band(b)?;
                    }
                }
            }
            BandRequest::MultiBand { bands_1based } => {
                for &b in bands_1based {
                    check_band(b)?;
                }
            }
        }
    }

    // Skip block I/O completely for 0-band constant expressions (u_map/v_map already enforce
    // raster footprint + polygon bounds).
    let zero_band_expr = match bands {
        BandRequest::CustomExpr { prog, .. } => prog.used_bands.is_empty(),
        BandRequest::CustomExprRgb { progs, .. } => {
            progs.iter().all(|p| p.used_bands.is_empty())
        }
        _ => false,
    };

    // Fetch needed bands (single COG chunky/planar or multi-COG)
    let cog_tiles = if zero_band_expr {
        vec![Vec::new(); cogs.len()]
    } else if cogs.len() == 1 {
        if is_single_planar {
            let mut indices = Vec::with_capacity(grid_len * planar_bands.len().max(1));
            for &band in &planar_bands {
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                    }
                }
            }
            vec![engine.store.get_blocks(&cogs[0], ifd_idx, &indices).await?]
        } else {
            let mut indices = Vec::with_capacity(grid_len);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                }
            }
            vec![engine.store.get_blocks(&cogs[0], ifd_idx, &indices).await?]
        }
    } else {
        let mut indices = Vec::with_capacity(grid_len);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
            }
        }
        let indices_arc = Arc::new(indices);
        let mut js = tokio::task::JoinSet::new();
        for (slot, c) in cogs.iter().enumerate() {
            let store = engine.store.clone();
            let cog = c.clone();
            let idx_ref = indices_arc.clone();
            js.spawn(async move {
                let blocks = store.get_blocks(&cog, ifd_idx, &idx_ref).await?;
                Ok::<_, cogzilla_io::IoError>((slot, blocks))
            });
        }
        let mut out: Vec<Option<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>> =
            vec![None; cogs.len()];
        while let Some(res) = js.join_next().await {
            let (slot, blocks) = res.map_err(|e| {
                EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                    "multi-cog bbox fetch task failed: {e}"
                ))))
            })??;
            out[slot] = Some(blocks);
        }
        out.into_iter().map(|o| o.unwrap()).collect()
    };

    let ifd_width = ifd.width;
    let ifd_height = ifd.height;
    let ifd_tile_width = ifd.tile_width;
    let ifd_tile_height = ifd.tile_height;
    let num_cogs = cogs.len();
    let band_scale_offsets: Vec<(f32, f32)> = if num_cogs > 1 {
        cogs.iter()
            .map(|c| {
                (
                    c.index.scales.first().copied().unwrap_or(1.0),
                    c.index.offsets.first().copied().unwrap_or(0.0),
                )
            })
            .collect()
    } else {
        let n = primary.index.scales.len().max(primary.index.offsets.len());
        (0..n)
            .map(|i| {
                (
                    primary.index.scales.get(i).copied().unwrap_or(1.0),
                    primary.index.offsets.get(i).copied().unwrap_or(0.0),
                )
            })
            .collect()
    };

    let dst_is_merc = matches!(
        dst_crs,
        cogzilla_proj::CrsModel::Epsg3857 | cogzilla_proj::CrsModel::WorldMercator
    );
    let dst_is_geo = matches!(dst_crs, cogzilla_proj::CrsModel::Geographic);
    let dst_unit_m = dst_crs.linear_unit_metres();
    let dst_center_lat = if dst_is_merc {
        dst_crs.to_lonlat_deg(cx, cy).1
    } else {
        0.5 * (bbox[1] + bbox[3])
    };
    let native_dem_dxy: Option<(f32, f32)> = if native_preview {
        let full = primary.index.full_res();
        let rx = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d) * (full.width as f64) / (w as f64);
        let ry = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e) * (full.height as f64) / (h as f64);
        let (dx_m, dy_m) = match &crs {
            cogzilla_proj::CrsModel::Geographic => {
                let lat_rad = (0.5 * (bbox[1] + bbox[3])).to_radians();
                (
                    (rx * 111_320.0 * lat_rad.cos().max(0.05)).max(1e-3) as f32,
                    (ry * 111_320.0).max(1e-3) as f32,
                )
            }
            cogzilla_proj::CrsModel::Epsg3857 | cogzilla_proj::CrsModel::WorldMercator => {
                let lat_rad = (0.5 * (bbox[1] + bbox[3])).to_radians();
                let cos_lat = lat_rad.cos().max(0.05);
                (
                    (rx * cos_lat).max(1e-3) as f32,
                    (ry * cos_lat).max(1e-3) as f32,
                )
            }
            other => {
                let unit_m = other.linear_unit_metres();
                (
                    (rx * unit_m).max(1e-3) as f32,
                    (ry * unit_m).max(1e-3) as f32,
                )
            }
        };
        Some((dx_m, dy_m))
    } else {
        None
    };
    #[cfg(feature = "gpu")]
    let gpu_ctx = engine
        .gpu
        .as_ref()
        .map(|g| (Arc::clone(g), Arc::clone(&engine.stats), primary.id));
    let render_cpu = move |u_map: Vec<f32>,
                           v_map: Vec<f32>,
                           cog_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>,
                           bands: &BandRequest,
                           custom_lut: Option<&CustomColormap>,
                           color_formula: Option<&ColorFormula>,
                           fused_stats: Option<(&StatsOptions, &str)>|
          -> BboxRenderOutput {
        #[cfg(not(feature = "gpu"))]
        let _ = fused_stats;
        let ifd_w_i = ifd_width as i32;
        let ifd_h_i = ifd_height as i32;
        let sample = |src: &BandSource<'_>, u: f32, v: f32| -> f32 {
            if !u.is_finite() || !v.is_finite() {
                return f32::NAN;
            }
            match resample {
                ResampleMode::Nearest => {
                    let xi = (u + 0.5).floor() as i32;
                    let yi = (v + 0.5).floor() as i32;
                    if xi >= 0
                        && yi >= 0
                        && xi < ifd_w_i
                        && yi < ifd_h_i
                        && src.nodata_override.is_none()
                    {
                        src.sample_inbounds(xi as u32, yi as u32)
                    } else {
                        src.sample_checked(xi, yi)
                    }
                }
                ResampleMode::Cubic => cogzilla_kernel::sample_cubic(src, u, v),
                ResampleMode::Bilinear => {
                    let x0_f = u.floor();
                    let y0_f = v.floor();
                    let fx = u - x0_f;
                    let fy = v - y0_f;
                    let x0 = x0_f as i32;
                    let y0 = y0_f as i32;
                    let (v00, v10, v01, v11) = if x0 >= 0
                        && y0 >= 0
                        && (x0 + 1) < ifd_w_i
                        && (y0 + 1) < ifd_h_i
                        && src.nodata_override.is_none()
                    {
                        src.sample_quad_inbounds(x0 as u32, y0 as u32)
                    } else {
                        (
                            src.sample_checked(x0, y0),
                            src.sample_checked(x0 + 1, y0),
                            src.sample_checked(x0, y0 + 1),
                            src.sample_checked(x0 + 1, y0 + 1),
                        )
                    };
                    let omx = 1.0 - fx;
                    let omy = 1.0 - fy;
                    // Fast branchless path when all 4 corners are valid (99.9% of interior pixels)
                    if !(v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan()) {
                        let top = v00 * omx + v10 * fx;
                        let bot = v01 * omx + v11 * fx;
                        return top * omy + bot * fy;
                    }
                    let (mut sum, mut wsum) = (0.0f32, 0.0f32);
                    for (val, wt) in [
                        (v00, omx * omy),
                        (v10, fx * omy),
                        (v01, omx * fy),
                        (v11, fx * fy),
                    ] {
                        if !val.is_nan() {
                            sum += val * wt;
                            wsum += wt;
                        }
                    }
                    if wsum > 1e-6 { sum / wsum } else { f32::NAN }
                }
                other => cogzilla_kernel::sample_general_scaled(
                    src,
                    u,
                    v,
                    footprint_sx,
                    footprint_sy,
                    other,
                ),
            }
        };

        let apply_source_unscale = unscale
            && (crate::engine::needs_source_unscale(bands, format)
                || custom_lut.is_some_and(CustomColormap::uses_direct_lookup));

        // Helper to build BandSource for 1-based band index across single (chunky/planar) or multi-COG
        let get_band_refs = |b_1based: u16,
                             _asset_slot: usize|
         -> (Vec<Option<&cogzilla_codec::DecodedBand>>, (f32, f32)) {
            let slot = (b_1based.saturating_sub(1) as usize).min(num_cogs.saturating_sub(1));
            let so = if !apply_source_unscale {
                (1.0, 0.0)
            } else if num_cogs > 1 {
                band_scale_offsets
                    .get(slot)
                    .copied()
                    .unwrap_or((1.0, 0.0))
            } else {
                band_scale_offsets
                    .get(b_1based.saturating_sub(1) as usize)
                    .copied()
                    .unwrap_or((1.0, 0.0))
            };
            let refs = if num_cogs > 1 {
                cog_tiles[slot]
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect()
            } else if is_single_planar {
                let b0 = b_1based.saturating_sub(1);
                if let Some(pos) = planar_bands.iter().position(|&b| b == b0) {
                    let start = pos * grid_len;
                    let end = (start + grid_len).min(cog_tiles[0].len());
                    cog_tiles[0][start..end]
                        .iter()
                        .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                        .collect()
                } else {
                    vec![None; grid_len]
                }
            } else {
                let b0 = b_1based.saturating_sub(1) as usize;
                cog_tiles[0]
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.get(b0)))
                    .collect()
            };
            (refs, so)
        };

        macro_rules! make_src {
            ($refs:expr) => {{
                let (ref blocks, (sc, off)) = *$refs;
                BandSource::new(
                    ifd_width,
                    ifd_height,
                    ifd_tile_width,
                    ifd_tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    blocks,
                )
                .with_nodata_override(nodata_override)
                .with_scale_offset(sc, off)
            }};
        }

        // Sample a (w + 2) x (h + 2) elevation ring by linearly extrapolating (u, v) 1 px past
        // each edge so Horn 3x3 gradients stay seamless across adjacent tiles (WorldCRS84Quad /
        // render_mercator_bbox) and non-square bboxes.
        let sample_dem_ring = |src: &BandSource<'_>| -> Vec<f32> {
            let rw = w + 2;
            let rh = h + 2;
            let mut ring = vec![f32::NAN; rw * rh];
            for py in 0..h {
                let row = py * w;
                let rrow = (py + 1) * rw;
                for px in 0..w {
                    ring[rrow + px + 1] = sample(src, u_map[row + px], v_map[row + px]);
                }
                let (u_l, v_l) = if w >= 2 {
                    (2.0 * u_map[row] - u_map[row + 1], 2.0 * v_map[row] - v_map[row + 1])
                } else {
                    (u_map[row], v_map[row])
                };
                let (u_r, v_r) = if w >= 2 {
                    (
                        2.0 * u_map[row + w - 1] - u_map[row + w - 2],
                        2.0 * v_map[row + w - 1] - v_map[row + w - 2],
                    )
                } else {
                    (u_map[row + w - 1], v_map[row + w - 1])
                };
                ring[rrow] = sample(src, u_l, v_l);
                ring[rrow + w + 1] = sample(src, u_r, v_r);
            }
            let extrap_row = |src_y0: usize, src_y1: usize, dst_ry: usize, ring: &mut [f32]| {
                let r0 = src_y0 * w;
                let r1 = src_y1 * w;
                let dr = dst_ry * rw;
                for rx in 0..rw {
                    let px_signed = rx as isize - 1;
                    let (u0, v0, u1, v1) = if px_signed < 0 {
                        if w >= 2 {
                            (
                                2.0 * u_map[r0] - u_map[r0 + 1],
                                2.0 * v_map[r0] - v_map[r0 + 1],
                                2.0 * u_map[r1] - u_map[r1 + 1],
                                2.0 * v_map[r1] - v_map[r1 + 1],
                            )
                        } else {
                            (u_map[r0], v_map[r0], u_map[r1], v_map[r1])
                        }
                    } else if (px_signed as usize) >= w {
                        if w >= 2 {
                            (
                                2.0 * u_map[r0 + w - 1] - u_map[r0 + w - 2],
                                2.0 * v_map[r0 + w - 1] - v_map[r0 + w - 2],
                                2.0 * u_map[r1 + w - 1] - u_map[r1 + w - 2],
                                2.0 * v_map[r1 + w - 1] - v_map[r1 + w - 2],
                            )
                        } else {
                            (
                                u_map[r0 + w - 1],
                                v_map[r0 + w - 1],
                                u_map[r1 + w - 1],
                                v_map[r1 + w - 1],
                            )
                        }
                    } else {
                        let p = px_signed as usize;
                        (u_map[r0 + p], v_map[r0 + p], u_map[r1 + p], v_map[r1 + p])
                    };
                    let (u, v) = if h >= 2 {
                        (2.0 * u0 - u1, 2.0 * v0 - v1)
                    } else {
                        (u0, v0)
                    };
                    ring[dr + rx] = sample(src, u, v);
                }
            };
            extrap_row(0, 1.min(h - 1), 0, &mut ring);
            extrap_row(h - 1, h.saturating_sub(2), h + 1, &mut ring);
            ring
        };

        let dem_inv_8dxy = || -> (f32, f32) {
            let (dx_m, dy_m) = if let Some(dxy) = native_dem_dxy {
                dxy
            } else if dst_is_merc {
                let cos_lat = dst_center_lat.to_radians().cos().max(0.05);
                (
                    (dlx.abs() * cos_lat).max(1e-3) as f32,
                    (dly.abs() * cos_lat).max(1e-3) as f32,
                )
            } else if dst_is_geo {
                let lat_rad = dst_center_lat.to_radians();
                (
                    (dlx.abs() * 111_320.0 * lat_rad.cos().max(0.05)).max(1e-3) as f32,
                    (dly.abs() * 111_320.0).max(1e-3) as f32,
                )
            } else {
                (
                    (dlx.abs() * dst_unit_m).max(1e-3) as f32,
                    (dly.abs() * dst_unit_m).max(1e-3) as f32,
                )
            };
            (1.0 / (8.0 * dx_m), 1.0 / (8.0 * dy_m))
        };

        // Real RawF32 / Npy output for single-band / expression / DEM / Terrain / RGB requests
        if matches!(format, OutputFormat::RawF32 | OutputFormat::Npy) {
            #[cfg(feature = "gpu")]
            if let Some((ref gpu, ref stats, primary_id)) = gpu_ctx {
                let v_rows: Vec<f32> = if native_preview {
                    (0..h).map(|py| v_map[py * w]).collect()
                } else {
                    Vec::new()
                };
                let coords = if native_preview {
                    crate::gpu_backend::BboxCoordSource::Separable {
                        u: &u_map[..w],
                        v: &v_rows,
                    }
                } else {
                    crate::gpu_backend::BboxCoordSource::Exact2D {
                        dst_w: w,
                        dst_h: h,
                        u_map: &u_map,
                        v_map: &v_map,
                    }
                };
                if format == OutputFormat::RawF32
                    && let Some((sopts, desc)) = fused_stats
                    && !sopts.categorical
                    && sopts.histogram_edges.is_none()
                {
                    let hist_bins = sopts.histogram_bins.clamp(1, 1000);
                    let gpu_stats: Option<cogzilla_gpu::GpuBandStats> = match bands {
                        BandRequest::Single { band_1based, .. } => {
                            let refs = get_band_refs(*band_1based, 0);
                            let src = make_src!(&refs);
                            gpu.try_render_bbox_single_band_stats(
                                primary_id,
                                ifd_idx,
                                band_1based.saturating_sub(1),
                                coords,
                                &src,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        BandRequest::CustomExpr { prog, .. } => {
                            let ref_vecs: Vec<_> = prog
                                .used_bands
                                .iter()
                                .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    &prog.used_bands,
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_expr_stats(
                                cache_key,
                                coords,
                                &srcs,
                                prog,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        BandRequest::Dem {
                            band_1based,
                            algorithm,
                            ..
                        } => {
                            let refs = get_band_refs(*band_1based, 0);
                            let src = make_src!(&refs);
                            let (inv_8dx, inv_8dy) = dem_inv_8dxy();
                            let dx_m = (1.0 / (8.0 * inv_8dx)).max(1e-3);
                            let dy_m = (1.0 / (8.0 * inv_8dy)).max(1e-3);
                            gpu.try_render_bbox_dem_stats(
                                primary_id,
                                ifd_idx,
                                band_1based.saturating_sub(1),
                                coords,
                                &src,
                                *algorithm,
                                dx_m,
                                dy_m,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        BandRequest::Spectral1 {
                            expr, band_1based, ..
                        } => {
                            let refs = get_band_refs(*band_1based, 0);
                            let srcs = [make_src!(&refs)];
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    &[*band_1based],
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_spectral_stats(
                                cache_key,
                                coords,
                                &srcs,
                                *expr,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        BandRequest::Spectral2 {
                            expr, bands_1based, ..
                        } => {
                            let r0 = get_band_refs(bands_1based[0], 0);
                            let r1 = get_band_refs(bands_1based[1], 1);
                            let srcs = [make_src!(&r0), make_src!(&r1)];
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    bands_1based,
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_spectral_stats(
                                cache_key,
                                coords,
                                &srcs,
                                *expr,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        BandRequest::Spectral3 {
                            expr, bands_1based, ..
                        } => {
                            let r0 = get_band_refs(bands_1based[0], 0);
                            let r1 = get_band_refs(bands_1based[1], 1);
                            let r2 = get_band_refs(bands_1based[2], 2);
                            let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    bands_1based,
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_spectral_stats(
                                cache_key,
                                coords,
                                &srcs,
                                *expr,
                                resample,
                                hist_bins,
                                sopts.histogram_range,
                                &sopts.percentiles,
                            )
                        }
                        _ => None,
                    };
                    if let Some(gs) = gpu_stats {
                        stats
                            .gpu_dispatches
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return BboxRenderOutput::Stats(gpu_band_stats_to_json(
                            &gs,
                            w * h,
                            desc,
                            sopts,
                        ));
                    }
                }
                let gpu_raw: Option<Bytes> = match bands {
                    BandRequest::Single { band_1based, .. } => {
                        let refs = get_band_refs(*band_1based, 0);
                        let src = make_src!(&refs);
                        gpu.try_render_bbox_single_band_f32(
                            primary_id,
                            ifd_idx,
                            band_1based.saturating_sub(1),
                            coords,
                            &src,
                            resample,
                        )
                    }
                    BandRequest::Dem {
                        band_1based,
                        algorithm,
                        ..
                    } => {
                        let refs = get_band_refs(*band_1based, 0);
                        let src = make_src!(&refs);
                        let (inv_8dx, inv_8dy) = dem_inv_8dxy();
                        let dx_m = (1.0 / (8.0 * inv_8dx)).max(1e-3);
                        let dy_m = (1.0 / (8.0 * inv_8dy)).max(1e-3);
                        gpu.try_render_bbox_dem_f32(
                            primary_id,
                            ifd_idx,
                            band_1based.saturating_sub(1),
                            coords,
                            &src,
                            *algorithm,
                            dx_m,
                            dy_m,
                            resample,
                        )
                    }
                    BandRequest::CustomExpr { prog, .. } => {
                        let ref_vecs: Vec<_> = prog
                            .used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                &prog.used_bands,
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_expr_f32(cache_key, coords, &srcs, prog, resample)
                    }
                    BandRequest::Spectral1 {
                        expr, band_1based, ..
                    } => {
                        let refs = get_band_refs(*band_1based, 0);
                        let srcs = [make_src!(&refs)];
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                &[*band_1based],
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_spectral_f32(cache_key, coords, &srcs, *expr, resample)
                    }
                    BandRequest::Spectral2 {
                        expr, bands_1based, ..
                    } => {
                        let r0 = get_band_refs(bands_1based[0], 0);
                        let r1 = get_band_refs(bands_1based[1], 1);
                        let srcs = [make_src!(&r0), make_src!(&r1)];
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                bands_1based,
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_spectral_f32(cache_key, coords, &srcs, *expr, resample)
                    }
                    BandRequest::Spectral3 {
                        expr, bands_1based, ..
                    } => {
                        let r0 = get_band_refs(bands_1based[0], 0);
                        let r1 = get_band_refs(bands_1based[1], 1);
                        let r2 = get_band_refs(bands_1based[2], 2);
                        let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                bands_1based,
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_spectral_f32(cache_key, coords, &srcs, *expr, resample)
                    }
                    BandRequest::Rgb { bands_1based, .. } => {
                        let r0 = get_band_refs(bands_1based[0], 0);
                        let r1 = get_band_refs(bands_1based[1], 1);
                        let r2 = get_band_refs(bands_1based[2], 2);
                        let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                bands_1based,
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_multiband_f32(cache_key, coords, &srcs, resample)
                    }
                    BandRequest::MultiBand { bands_1based } => {
                        if bands_1based.is_empty() || bands_1based.len() > 8 {
                            None
                        } else {
                            let ref_vecs: Vec<_> = bands_1based
                                .iter()
                                .enumerate()
                                .map(|(slot, &b)| get_band_refs(b, slot))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    bands_1based,
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_multiband_f32(cache_key, coords, &srcs, resample)
                        }
                    }
                    BandRequest::CustomExprRgb { progs, .. } => {
                        let mut unique_bands: Vec<u16> = Vec::new();
                        for prog in progs {
                            for &b in &prog.used_bands {
                                if !unique_bands.contains(&b) {
                                    unique_bands.push(b);
                                }
                            }
                        }
                        if unique_bands.is_empty() || unique_bands.len() > 8 {
                            None
                        } else {
                            let ref_vecs: Vec<_> = unique_bands
                                .iter()
                                .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            let cache_key = if num_cogs == 1 {
                                crate::gpu_backend::GpuBackend::multiband_key(
                                    primary_id,
                                    ifd_idx,
                                    &unique_bands,
                                    &srcs,
                                )
                            } else {
                                None
                            };
                            gpu.try_render_bbox_expr_rgb_f32(
                                cache_key,
                                coords,
                                &unique_bands,
                                &srcs,
                                progs,
                                resample,
                            )
                        }
                    }
                    _ => None,
                };
                if let Some(raw) = gpu_raw {
                    stats
                        .gpu_dispatches
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if format == OutputFormat::Npy {
                        return BboxRenderOutput::Bytes(
                            cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                                raw,
                                1,
                                h as u32,
                                w as u32,
                                return_mask,
                            ),
                        );
                    }
                    return BboxRenderOutput::Bytes(raw);
                }
            }

            let mut out = BytesMut::with_capacity(w * h * 4);
            match bands {
                BandRequest::Dem {
                    band_1based,
                    algorithm,
                    ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let ring = sample_dem_ring(&src);
                    let rw = w + 2;
                    let (inv_8dx, inv_8dy) = dem_inv_8dxy();
                    for py in 0..h {
                        let r0 = py * rw;
                        let r1 = (py + 1) * rw;
                        let r2 = (py + 2) * rw;
                        for px in 0..w {
                            let val = cogzilla_kernel::eval_horn_3x3_aniso(
                                ring[r0 + px],
                                ring[r0 + px + 1],
                                ring[r0 + px + 2],
                                ring[r1 + px],
                                ring[r1 + px + 1],
                                ring[r1 + px + 2],
                                ring[r2 + px],
                                ring[r2 + px + 1],
                                ring[r2 + px + 2],
                                inv_8dx,
                                inv_8dy,
                                *algorithm,
                            );
                            out.put_slice(&val.to_le_bytes());
                        }
                    }
                }
                BandRequest::Single { band_1based, .. } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    for i in 0..(w * h) {
                        out.put_slice(&sample(&src, u_map[i], v_map[i]).to_le_bytes());
                    }
                }
                BandRequest::Terrain {
                    band_1based,
                    encoding,
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let mut g_plane = Vec::with_capacity(w * h * 4);
                    let mut b_plane = Vec::with_capacity(w * h * 4);
                    for i in 0..(w * h) {
                        let elev = sample(&src, u_map[i], v_map[i]);
                        if elev.is_nan() {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            g_plane.extend_from_slice(&f32::NAN.to_le_bytes());
                            b_plane.extend_from_slice(&f32::NAN.to_le_bytes());
                        } else {
                            let [r, g, b] = encoding.encode_rgb(elev);
                            out.put_slice(&(r as f32).to_le_bytes());
                            g_plane.extend_from_slice(&(g as f32).to_le_bytes());
                            b_plane.extend_from_slice(&(b as f32).to_le_bytes());
                        }
                    }
                    out.put_slice(&g_plane);
                    out.put_slice(&b_plane);
                }
                BandRequest::CustomExpr { prog, .. } => {
                    let ref_vecs: Vec<_> = prog
                        .used_bands
                        .iter()
                        .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    let mut vals = [0.0f32; 16];
                    let nb = srcs.len().min(16);
                    for i in 0..(w * h) {
                        if u_map[i].is_nan() {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            continue;
                        }
                        for k in 0..nb {
                            vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                        }
                        out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
                    }
                }
                BandRequest::CustomExprRgb { progs, .. } => {
                    for prog in progs {
                        let ref_vecs: Vec<_> = prog
                            .used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let mut vals = [0.0f32; 16];
                        let nb = srcs.len().min(16);
                        for i in 0..(w * h) {
                            if u_map[i].is_nan() {
                                out.put_slice(&f32::NAN.to_le_bytes());
                                continue;
                            }
                            for k in 0..nb {
                                vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                            }
                            out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
                        }
                    }
                }
                BandRequest::Spectral1 {
                    expr, band_1based, ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    for i in 0..(w * h) {
                        out.put_slice(&expr.eval1(sample(&src, u_map[i], v_map[i])).to_le_bytes());
                    }
                }
                BandRequest::Spectral2 {
                    expr, bands_1based, ..
                } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    for i in 0..(w * h) {
                        let a = sample(&s0, u_map[i], v_map[i]);
                        let b = sample(&s1, u_map[i], v_map[i]);
                        let v = if a.is_nan() || b.is_nan() { f32::NAN } else { expr.eval2(a, b) };
                        out.put_slice(&v.to_le_bytes());
                    }
                }
                BandRequest::Spectral3 {
                    expr, bands_1based, ..
                } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    let s2 = make_src!(&r2);
                    for i in 0..(w * h) {
                        let a = sample(&s0, u_map[i], v_map[i]);
                        let b = sample(&s1, u_map[i], v_map[i]);
                        let c = sample(&s2, u_map[i], v_map[i]);
                        let v = if a.is_nan() || b.is_nan() || c.is_nan() {
                            f32::NAN
                        } else {
                            expr.eval3(a, b, c)
                        };
                        out.put_slice(&v.to_le_bytes());
                    }
                }
                BandRequest::Rgb { bands_1based, .. } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                    out.reserve(w * h * 8);
                    for s in &srcs {
                        for i in 0..(w * h) {
                            out.put_slice(&sample(s, u_map[i], v_map[i]).to_le_bytes());
                        }
                    }
                }
                BandRequest::SarRgb { bands_1based, .. } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    let mut vh_plane = Vec::with_capacity(w * h * 4);
                    let mut diff_plane = Vec::with_capacity(w * h * 4);
                    out.reserve(w * h * 8);
                    for i in 0..(w * h) {
                        let vv = fast_db_from_power(sample(&s0, u_map[i], v_map[i]));
                        let vh = fast_db_from_power(sample(&s1, u_map[i], v_map[i]));
                        out.put_slice(&vv.to_le_bytes());
                        vh_plane.extend_from_slice(&vh.to_le_bytes());
                        diff_plane.extend_from_slice(&(vv - vh).to_le_bytes());
                    }
                    out.put_slice(&vh_plane);
                    out.put_slice(&diff_plane);
                }
                BandRequest::MultiBand { bands_1based } => {
                    out.reserve(w * h * 4 * bands_1based.len().saturating_sub(1));
                    for (slot, &b1) in bands_1based.iter().enumerate() {
                        let r = get_band_refs(b1, slot);
                        let s = make_src!(&r);
                        for i in 0..(w * h) {
                            out.put_slice(&sample(&s, u_map[i], v_map[i]).to_le_bytes());
                        }
                    }
                }
            }
            let raw = out.freeze();
            if format == OutputFormat::Npy {
                let nb = match bands {
                    BandRequest::Rgb { .. }
                    | BandRequest::SarRgb { .. }
                    | BandRequest::CustomExprRgb { .. }
                    | BandRequest::Terrain { .. } => 3,
                    BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
                    _ => 1,
                };
                return BboxRenderOutput::Bytes(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    nb,
                    h as u32,
                    w as u32,
                    return_mask,
                ));
            }
            return BboxRenderOutput::Bytes(raw);
        }

        let cf_single_lut: Option<[u8; 256]> = if matches!(
            bands,
            BandRequest::Single { .. }
                | BandRequest::Dem { .. }
                | BandRequest::CustomExpr { .. }
                | BandRequest::Spectral1 { .. }
                | BandRequest::Spectral2 { .. }
                | BandRequest::Spectral3 { .. }
        ) {
            color_formula.map(|cf| {
                let mut t = [0u8; 256];
                for i in 0..256 {
                    let mut px = [i as u8, i as u8, i as u8, 255];
                    cf.apply_buffer_in_place(&mut px, 4);
                    t[i] = px[0];
                }
                t
            })
        } else {
            None
        };
        let map_color = |v: f32, rmin: f32, inv: f32, default_cmap: ColormapName| -> [u8; 4] {
            if let Some(cmap) = custom_lut {
                cmap.lookup(v, rmin, inv)
            } else {
                let mut idx = ((v - rmin) * inv).clamp(0.0, 255.0) as usize;
                if let Some(ref cf_lut) = cf_single_lut {
                    idx = cf_lut[idx] as usize;
                }
                default_cmap.lut().rgba_u32[idx].to_le_bytes()
            }
        };

        #[cfg(feature = "gpu")]
        if matches!(
            format,
            OutputFormat::Png
                | OutputFormat::Webp { .. }
                | OutputFormat::Jpeg { .. }
                | OutputFormat::RawRgba
        ) && let Some((ref gpu, ref stats, primary_id)) = gpu_ctx
        {
            let v_rows: Vec<f32> = if native_preview {
                (0..h).map(|py| v_map[py * w]).collect()
            } else {
                Vec::new()
            };
            let coords = if native_preview {
                crate::gpu_backend::BboxCoordSource::Separable {
                    u: &u_map[..w],
                    v: &v_rows,
                }
            } else {
                crate::gpu_backend::BboxCoordSource::Exact2D {
                    dst_w: w,
                    dst_h: h,
                    u_map: &u_map,
                    v_map: &v_map,
                }
            };
            let gpu_out: Option<Bytes> = match bands {
                BandRequest::Single {
                    band_1based,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox_single_band_with_lut(
                        primary_id,
                        ifd_idx,
                        band_1based.saturating_sub(1),
                        coords,
                        &src,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                BandRequest::Dem {
                    band_1based,
                    algorithm,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let (inv_8dx, inv_8dy) = dem_inv_8dxy();
                    let dx_m = (1.0 / (8.0 * inv_8dx)).max(1e-3);
                    let dy_m = (1.0 / (8.0 * inv_8dy)).max(1e-3);
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox_dem_with_lut(
                        primary_id,
                        ifd_idx,
                        band_1based.saturating_sub(1),
                        coords,
                        &src,
                        *algorithm,
                        dx_m,
                        dy_m,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                BandRequest::Terrain {
                    band_1based,
                    encoding,
                } if custom_lut.is_none() && color_formula.is_none() => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    gpu.try_render_bbox_terrain(
                        primary_id,
                        ifd_idx,
                        band_1based.saturating_sub(1),
                        coords,
                        &src,
                        *encoding,
                        resample,
                        format,
                        return_mask,
                    )
                }
                BandRequest::CustomExpr {
                    prog,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let ref_vecs: Vec<_> = prog
                        .used_bands
                        .iter()
                        .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            &prog.used_bands,
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_expr_with_lut(
                        cache_key,
                        coords,
                        &srcs,
                        prog,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                BandRequest::CustomExprRgb {
                    progs,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let mut unique_bands: Vec<u16> = Vec::new();
                    for prog in progs {
                        for &b in &prog.used_bands {
                            if !unique_bands.contains(&b) {
                                unique_bands.push(b);
                            }
                        }
                    }
                    if unique_bands.is_empty() || unique_bands.len() > 8 {
                        None
                    } else {
                        let ref_vecs: Vec<_> = unique_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let style = cogzilla_kernel::RgbStyle {
                            rescale_min: *rescale_min,
                            rescale_max: *rescale_max,
                            resample,
                            format,
                        };
                        let cache_key = if num_cogs == 1 {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id,
                                ifd_idx,
                                &unique_bands,
                                &srcs,
                            )
                        } else {
                            None
                        };
                        gpu.try_render_bbox_expr_rgb(
                            cache_key,
                            coords,
                            &unique_bands,
                            &srcs,
                            progs,
                            &style,
                            color_formula,
                            return_mask,
                        )
                    }
                }
                BandRequest::Rgb {
                    bands_1based,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                    let style = cogzilla_kernel::RgbStyle {
                        rescale_min: *rescale_min,
                        rescale_max: *rescale_max,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            bands_1based,
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_rgb(
                        cache_key,
                        coords,
                        &srcs,
                        &style,
                        color_formula,
                        return_mask,
                    )
                }
                BandRequest::SarRgb {
                    bands_1based,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let srcs = [make_src!(&r0), make_src!(&r1)];
                    let style = cogzilla_kernel::RgbStyle {
                        rescale_min: *rescale_min,
                        rescale_max: *rescale_max,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            bands_1based,
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_sar_rgb(
                        cache_key,
                        coords,
                        &srcs,
                        &style,
                        color_formula,
                        return_mask,
                    )
                }
                BandRequest::Spectral1 {
                    expr,
                    band_1based,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let refs = get_band_refs(*band_1based, 0);
                    let srcs = [make_src!(&refs)];
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            &[*band_1based],
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_spectral_with_lut(
                        cache_key,
                        coords,
                        &srcs,
                        *expr,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                BandRequest::Spectral2 {
                    expr,
                    bands_1based,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let srcs = [make_src!(&r0), make_src!(&r1)];
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            bands_1based,
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_spectral_with_lut(
                        cache_key,
                        coords,
                        &srcs,
                        *expr,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                BandRequest::Spectral3 {
                    expr,
                    bands_1based,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let cache_key = if num_cogs == 1 {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id,
                            ifd_idx,
                            bands_1based,
                            &srcs,
                        )
                    } else {
                        None
                    };
                    gpu.try_render_bbox_spectral_with_lut(
                        cache_key,
                        coords,
                        &srcs,
                        *expr,
                        &style,
                        &rgb_lut,
                        return_mask,
                    )
                }),
                _ => None,
            };
            if let Some(bytes) = gpu_out {
                stats
                    .gpu_dispatches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return BboxRenderOutput::Bytes(bytes);
            }
        }

        let mut rgba = vec![0u8; w * h * 4];
        match bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let v = sample(&src, u_map[i], v_map[i]);
                    if !v.is_nan() {
                        let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                }
            }
            BandRequest::Terrain {
                band_1based,
                encoding,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                for i in 0..(w * h) {
                    let v = sample(&src, u_map[i], v_map[i]);
                    if !v.is_nan() {
                        let [r, g, b] = encoding.encode_rgb(v);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = 255;
                    }
                }
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let ring = sample_dem_ring(&src);
                let rw = w + 2;
                let (inv_8dx, inv_8dy) = dem_inv_8dxy();
                for py in 0..h {
                    let r0 = py * rw;
                    let r1 = (py + 1) * rw;
                    let r2 = (py + 2) * rw;
                    for px in 0..w {
                        let v = cogzilla_kernel::eval_horn_3x3_aniso(
                            ring[r0 + px],
                            ring[r0 + px + 1],
                            ring[r0 + px + 2],
                            ring[r1 + px],
                            ring[r1 + px + 1],
                            ring[r1 + px + 2],
                            ring[r2 + px],
                            ring[r2 + px + 1],
                            ring[r2 + px + 2],
                            inv_8dx,
                            inv_8dy,
                            *algorithm,
                        );
                        if !v.is_nan() {
                            let i = py * w + px;
                            let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                            rgba[i * 4] = r;
                            rgba[i * 4 + 1] = g;
                            rgba[i * 4 + 2] = b;
                            rgba[i * 4 + 3] = a;
                        }
                    }
                }
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let r2 = get_band_refs(bands_1based[2], 2);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let s2 = make_src!(&r2);
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                for i in 0..(w * h) {
                    let r = sample(&s0, u_map[i], v_map[i]);
                    let g = sample(&s1, u_map[i], v_map[i]);
                    let b = sample(&s2, u_map[i], v_map[i]);
                    if !r.is_nan() && !g.is_nan() && !b.is_nan() {
                        rgba[i * 4] = ((r - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 1] = ((g - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 2] = ((b - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 3] = 255;
                    }
                }
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                for i in 0..(w * h) {
                    let vv = fast_db_from_power(sample(&s0, u_map[i], v_map[i]));
                    let vh = fast_db_from_power(sample(&s1, u_map[i], v_map[i]));
                    if !vv.is_nan() && !vh.is_nan() {
                        rgba[i * 4] = ((vv - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 1] = ((vh - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 2] =
                            (((vv - vh) - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 3] = 255;
                    }
                }
            }
            BandRequest::CustomExpr {
                prog,
                rescale,
                colormap,
            } => {
                let ref_vecs: Vec<_> = prog
                    .used_bands
                    .iter()
                    .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                    .collect();
                let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let mut vals = [0.0f32; 16];
                let nb = srcs.len().min(16);
                for i in 0..(w * h) {
                    if u_map[i].is_nan() {
                        continue;
                    }
                    for k in 0..nb {
                        vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                    }
                    let v = prog.eval(&vals[..nb]);
                    if !v.is_nan() {
                        let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                }
            }
            BandRequest::CustomExprRgb {
                progs,
                rescale_min,
                rescale_max,
            } => {
                let all_refs: Vec<Vec<_>> = progs
                    .iter()
                    .map(|prog| {
                        prog.used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect()
                    })
                    .collect();
                let all_srcs: Vec<Vec<_>> = all_refs
                    .iter()
                    .map(|rvec| rvec.iter().map(|r| make_src!(r)).collect())
                    .collect();
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                let mut vals = [0.0f32; 16];
                for i in 0..(w * h) {
                    if u_map[i].is_nan() {
                        continue;
                    }
                    let mut rgb = [0.0f32; 3];
                    let mut ok = true;
                    for c in 0..3 {
                        let srcs = &all_srcs[c];
                        let nb = srcs.len().min(16);
                        for k in 0..nb {
                            vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                        }
                        let v = progs[c].eval(&vals[..nb]);
                        if v.is_nan() {
                            ok = false;
                            break;
                        }
                        rgb[c] = v;
                    }
                    if ok {
                        rgba[i * 4] = ((rgb[0] - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 1] =
                            ((rgb[1] - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 2] =
                            ((rgb[2] - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 3] = 255;
                    }
                }
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let v = expr.eval1(sample(&src, u_map[i], v_map[i]));
                    if !v.is_nan() {
                        let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                }
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let a = sample(&s0, u_map[i], v_map[i]);
                    let b = sample(&s1, u_map[i], v_map[i]);
                    let v = if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        expr.eval2(a, b)
                    };
                    if !v.is_nan() {
                        let [r, g, b, al] = map_color(v, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = al;
                    }
                }
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let r2 = get_band_refs(bands_1based[2], 2);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let s2 = make_src!(&r2);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let a = sample(&s0, u_map[i], v_map[i]);
                    let b = sample(&s1, u_map[i], v_map[i]);
                    let c = sample(&s2, u_map[i], v_map[i]);
                    let v = if a.is_nan() || b.is_nan() || c.is_nan() {
                        f32::NAN
                    } else {
                        expr.eval3(a, b, c)
                    };
                    if !v.is_nan() {
                        let [r, g, b, al] = map_color(v, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = al;
                    }
                }
            }
            BandRequest::MultiBand { .. } => {}
        }

        if cf_single_lut.is_none()
            && let Some(cf) = color_formula
        {
            cf.apply_buffer_in_place(&mut rgba, 4);
        }

        BboxRenderOutput::Bytes(encode_rgba_output(w, h, &rgba, format, return_mask))
    };

    // Offload large (>512x512) bbox renders to bounded spawn_blocking so giant previews/crops
    // never stall Tokio's async I/O workers.
    if w * h > 512 * 512 {
        let bands_owned = bands.clone();
        let lut_owned = custom_lut.cloned();
        let cf_owned = color_formula.cloned();
        let stats_owned = fused_stats.map(|(o, d)| (o.clone(), d.to_string()));
        let out = engine
            .store
            .run_heavy_cpu(move || {
                render_cpu(
                    u_map,
                    v_map,
                    cog_tiles,
                    &bands_owned,
                    lut_owned.as_ref(),
                    cf_owned.as_ref(),
                    stats_owned.as_ref().map(|(o, d)| (o, d.as_str())),
                )
            })
            .await?;
        Ok(out)
    } else {
        Ok(render_cpu(
            u_map,
            v_map,
            cog_tiles,
            bands,
            custom_lut,
            color_formula,
            fused_stats,
        ))
    }
}

/// Interactive MapLibre HTML viewer (`/cog/{tms}/map.html`), matching TiTiler's built-in viewer.
pub fn build_map_html(tilejson_url: &str) -> String {
    // Escape `</` so an untrusted query string can never break out of the inline `<script>` tag.
    let safe_url_js = serde_json::to_string(tilejson_url)
        .unwrap_or_else(|_| "\"\"".to_string())
        .replace("</", "<\\/");
    format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>cogzilla Map Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>body {{ margin: 0; padding: 0; }} #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}</style>
</head>
<body>
  <div id="map"></div>
  <script>
    fetch({safe_url_js}).then(r => r.json()).then(tj => {{
      const map = new maplibregl.Map({{
        container: 'map',
        style: {{
          version: 8,
          sources: {{
            osm: {{ type: 'raster', tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'], tileSize: 256 }},
            cog: {{ type: 'raster', tiles: tj.tiles, tileSize: 256, bounds: tj.bounds }}
          }},
          layers: [
            {{ id: 'osm', type: 'raster', source: 'osm' }},
            {{ id: 'cog', type: 'raster', source: 'cog' }}
          ]
        }},
        center: [tj.center[0], tj.center[1]],
        zoom: tj.center[2]
      }});
      map.fitBounds([[tj.bounds[0], tj.bounds[1]], [tj.bounds[2], tj.bounds[3]]], {{ padding: 40 }});
    }});
  </script>
</body>
</html>"#
    )
}

/// Fallback multi-COG / STAC renderer when assets have mixed resolutions, tile grids, overview
/// counts, or CRSes (e.g. Sentinel-2 10m + 20m bands). Samples each needed asset independently
/// into a `(w * h)` f32 plane in the target CRS, then evaluates the band math / styling / encoding.
/// Render a multi-asset request where each slot in `cogs` selects a specific 1-based band
/// from `asset_bands` (for multi-asset STAC `?asset_bidx=a|2&asset_bidx=b|1`).
#[allow(clippy::too_many_arguments)]
pub async fn render_multi_asset_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    asset_bands: &[u16],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    let w = (width as usize).clamp(1, 4096);
    let h = (height as usize).clamp(1, 4096);
    if let Some(pm) = poly_mask {
        pm.check_work_budget(w.saturating_mul(h))?;
    }
    render_mixed_grid_bbox(
        engine,
        cogs,
        bbox,
        dst_crs,
        native_preview,
        w,
        h,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
        Some(asset_bands),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn render_mixed_grid_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    w: usize,
    h: usize,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
    asset_bands: Option<&[u16]>,
) -> Result<Bytes, EngineError> {
    let max_b = cogs.len() as u16;
    // DEM algorithms (hillshade, slope, aspect, contours) operate on a single band using a 3x3
    // neighborhood; delegate directly to `render_bbox_impl` on that asset so Horn derivatives and
    // native preview georeferencing work identically to single-COG renders.
    if let BandRequest::Dem {
        band_1based,
        algorithm,
        rescale,
        colormap,
    } = bands
    {
        if *band_1based == 0 || *band_1based > max_b {
            return Err(EngineError::BadRequest(format!(
                "band {band_1based} out of range (dataset has {max_b} assets)"
            )));
        }
        let single_cog = [cogs[(*band_1based - 1) as usize].clone()];
        let src_band = asset_bands
            .and_then(|ab| ab.get((*band_1based - 1) as usize).copied())
            .unwrap_or(1);
        let single_dem = BandRequest::Dem {
            band_1based: src_band,
            algorithm: *algorithm,
            rescale: *rescale,
            colormap: *colormap,
        };
        return Box::pin(render_bbox_impl(
            engine,
            &single_cog,
            bbox,
            dst_crs,
            native_preview,
            w as u32,
            h as u32,
            poly_mask,
            &single_dem,
            resample,
            format,
            nodata_override,
            custom_lut,
            color_formula,
            return_mask,
            unscale,
        ))
        .await;
    }

    // When `native_preview` is requested on a mixed-grid multi-COG / STAC composite, sample all
    // assets into `cogs[0]`'s native CRS and native full-resolution bounds so output pixels match
    // the native GeoTIFF georeferencing header stamped by `/{cog,stac}/preview.tif`.
    let native_crs_owned;
    let (eff_bbox, eff_crs) = if native_preview {
        native_crs_owned = cogzilla_proj::CrsModel::from_geo_keys(&cogs[0].index.geo_keys)?;
        let full = cogs[0].index.full_res();
        let a = &full.pixel_to_crs;
        let corners = [
            a.forward(0.0, 0.0),
            a.forward(full.width as f64, 0.0),
            a.forward(0.0, full.height as f64),
            a.forward(full.width as f64, full.height as f64),
        ];
        let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let max_x = corners.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let max_y = corners.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
        ([min_x, min_y, max_x, max_y], &native_crs_owned)
    } else {
        (bbox, dst_crs)
    };
    let mut needed: Vec<u16> = Vec::new();
    let mut push_b = |b: u16| -> Result<(), EngineError> {
        if b == 0 || b > max_b {
            return Err(EngineError::BadRequest(format!(
                "band {b} out of range (dataset has {max_b} assets)"
            )));
        }
        if !needed.contains(&b) {
            needed.push(b);
        }
        Ok(())
    };
    match bands {
        BandRequest::Single { band_1based, .. }
        | BandRequest::Terrain { band_1based, .. }
        | BandRequest::Dem { band_1based, .. }
        | BandRequest::Spectral1 { band_1based, .. } => push_b(*band_1based)?,
        BandRequest::SarRgb { bands_1based, .. } | BandRequest::Spectral2 { bands_1based, .. } => {
            push_b(bands_1based[0])?;
            push_b(bands_1based[1])?;
        }
        BandRequest::Rgb { bands_1based, .. } | BandRequest::Spectral3 { bands_1based, .. } => {
            push_b(bands_1based[0])?;
            push_b(bands_1based[1])?;
            push_b(bands_1based[2])?;
        }
        BandRequest::CustomExpr { prog, .. } => {
            for &b in &prog.used_bands {
                push_b(b)?;
            }
        }
        BandRequest::CustomExprRgb { progs, .. } => {
            for p in progs {
                for &b in &p.used_bands {
                    push_b(b)?;
                }
            }
        }
        BandRequest::MultiBand { bands_1based } => {
            for &b in bands_1based {
                push_b(b)?;
            }
        }
    }

    // Sample each needed COG independently into a w*h f32 plane in target CRS.
    // Avoid double-unscaling when Single/Rgb already folded GDAL scale/offset into rescale bounds.
    let sub_unscale = unscale
        && (crate::engine::needs_source_unscale(bands, format)
            || custom_lut.is_some_and(CustomColormap::uses_direct_lookup));
    let mut planes: Vec<(u16, Vec<f32>)> = Vec::with_capacity(needed.len());
    for &b1 in &needed {
        let single_cog = [cogs[(b1 - 1) as usize].clone()];
        let src_band = asset_bands
            .and_then(|ab| ab.get((b1 - 1) as usize).copied())
            .unwrap_or(1);
        let single_band = BandRequest::Single {
            band_1based: src_band,
            rescale: (0.0, 1.0),
            colormap: ColormapName::Grayscale,
        };
        let raw = Box::pin(render_bbox_impl(
            engine,
            &single_cog,
            eff_bbox,
            eff_crs,
            false,
            w as u32,
            h as u32,
            poly_mask,
            &single_band,
            resample,
            OutputFormat::RawF32,
            nodata_override,
            None,
            None,
            return_mask,
            sub_unscale,
        ))
        .await?;
        let mut vals = vec![f32::NAN; w * h];
        for (i, chunk) in raw.chunks_exact(4).take(w * h).enumerate() {
            vals[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        planes.push((b1, vals));
    }

    // For constant or where() expressions in mixed-grid mode, build per-channel polygon +
    // referenced-asset footprint validity masks (logical OR across the channel's referenced COGs,
    // falling back to cogs[0] only when the channel references no bands) so non-primary assets
    // are never clipped to cogs[0]'s footprint.
    let prog_needs_mask = |p: &cogzilla_kernel::ExprProgram| {
        p.used_bands.is_empty() || p.ops.contains(&cogzilla_kernel::expr_vm::Op::Where)
    };
    let mut single_cog_masks: Vec<(usize, Vec<bool>)> = Vec::new();
    let one_prog = cogzilla_kernel::ExprProgram::compile("1").map_err(EngineError::BadRequest)?;
    let mask_band = BandRequest::CustomExpr {
        prog: one_prog,
        rescale: (0.0, 1.0),
        colormap: ColormapName::Grayscale,
    };
    let mut expr_valid_mask: Option<Vec<bool>> = None;
    let mut expr_rgb_masks: [Option<Vec<bool>>; 3] = [None, None, None];
    match bands {
        BandRequest::CustomExpr { prog, .. } if prog_needs_mask(prog) => {
            let slots: Vec<usize> = if prog.used_bands.is_empty() {
                vec![0]
            } else {
                let mut s = Vec::new();
                for &b in &prog.used_bands {
                    let idx = (b - 1) as usize;
                    if !s.contains(&idx) {
                        s.push(idx);
                    }
                }
                s
            };
            let mut combined = vec![false; w * h];
            for slot in slots {
                let single_cog = [cogs[slot].clone()];
                let raw = Box::pin(render_bbox_impl(
                    engine,
                    &single_cog,
                    eff_bbox,
                    eff_crs,
                    false,
                    w as u32,
                    h as u32,
                    poly_mask,
                    &mask_band,
                    ResampleMode::Nearest,
                    OutputFormat::RawF32,
                    None,
                    None,
                    None,
                    false,
                    false,
                ))
                .await?;
                for (i, c) in raw.chunks_exact(4).take(w * h).enumerate() {
                    if !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan() {
                        combined[i] = true;
                    }
                }
            }
            expr_valid_mask = Some(combined);
        }
        BandRequest::CustomExprRgb { progs, .. } => {
            for c_idx in 0..3 {
                if !prog_needs_mask(&progs[c_idx]) {
                    continue;
                }
                let slots: Vec<usize> = if progs[c_idx].used_bands.is_empty() {
                    vec![0]
                } else {
                    let mut s = Vec::new();
                    for &b in &progs[c_idx].used_bands {
                        let idx = (b - 1) as usize;
                        if !s.contains(&idx) {
                            s.push(idx);
                        }
                    }
                    s
                };
                let mut combined = vec![false; w * h];
                for slot in slots {
                    if let Some((_, cached)) = single_cog_masks.iter().find(|(k, _)| *k == slot) {
                        for (dst, &v) in combined.iter_mut().zip(cached.iter()) {
                            *dst |= v;
                        }
                        continue;
                    }
                    let single_cog = [cogs[slot].clone()];
                    let raw = Box::pin(render_bbox_impl(
                        engine,
                        &single_cog,
                        eff_bbox,
                        eff_crs,
                        false,
                        w as u32,
                        h as u32,
                        poly_mask,
                        &mask_band,
                        ResampleMode::Nearest,
                        OutputFormat::RawF32,
                        None,
                        None,
                        None,
                        false,
                        false,
                    ))
                    .await?;
                    let mask_vec: Vec<bool> = raw
                        .chunks_exact(4)
                        .take(w * h)
                        .map(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan())
                        .collect();
                    for (dst, &v) in combined.iter_mut().zip(mask_vec.iter()) {
                        *dst |= v;
                    }
                    single_cog_masks.push((slot, mask_vec));
                }
                expr_rgb_masks[c_idx] = Some(combined);
            }
        }
        _ => {}
    }

    let get_plane = |b1: u16| -> &[f32] {
        planes
            .iter()
            .find(|(k, _)| *k == b1)
            .map(|(_, v)| v.as_slice())
            .unwrap()
    };

    let npix = w * h;
    if matches!(format, OutputFormat::RawF32 | OutputFormat::Npy) {
        let mut out = BytesMut::with_capacity(npix * 4);
        let nb: usize = match bands {
            BandRequest::Single { band_1based, .. } | BandRequest::Dem { band_1based, .. } => {
                for &v in get_plane(*band_1based) {
                    out.put_slice(&v.to_le_bytes());
                }
                1
            }
            BandRequest::Terrain {
                band_1based,
                encoding,
            } => {
                let p = get_plane(*band_1based);
                let mut g_p = Vec::with_capacity(npix * 4);
                let mut b_p = Vec::with_capacity(npix * 4);
                for &elev in p {
                    if elev.is_nan() {
                        out.put_slice(&f32::NAN.to_le_bytes());
                        g_p.extend_from_slice(&f32::NAN.to_le_bytes());
                        b_p.extend_from_slice(&f32::NAN.to_le_bytes());
                    } else {
                        let [r, g, b] = encoding.encode_rgb(elev);
                        out.put_slice(&(r as f32).to_le_bytes());
                        g_p.extend_from_slice(&(g as f32).to_le_bytes());
                        b_p.extend_from_slice(&(b as f32).to_le_bytes());
                    }
                }
                out.put_slice(&g_p);
                out.put_slice(&b_p);
                3
            }
            BandRequest::Rgb { bands_1based, .. } => {
                for &b in bands_1based {
                    for &v in get_plane(b) {
                        out.put_slice(&v.to_le_bytes());
                    }
                }
                3
            }
            BandRequest::Spectral1 {
                band_1based, expr, ..
            } => {
                for &v in get_plane(*band_1based) {
                    out.put_slice(&expr.eval1(v).to_le_bytes());
                }
                1
            }
            BandRequest::Spectral2 {
                bands_1based, expr, ..
            } => {
                let p0 = get_plane(bands_1based[0]);
                let p1 = get_plane(bands_1based[1]);
                for i in 0..npix {
                    out.put_slice(&expr.eval2(p0[i], p1[i]).to_le_bytes());
                }
                1
            }
            BandRequest::Spectral3 {
                bands_1based, expr, ..
            } => {
                let p0 = get_plane(bands_1based[0]);
                let p1 = get_plane(bands_1based[1]);
                let p2 = get_plane(bands_1based[2]);
                for i in 0..npix {
                    out.put_slice(&expr.eval3(p0[i], p1[i], p2[i]).to_le_bytes());
                }
                1
            }
            BandRequest::SarRgb { bands_1based, .. } => {
                let p0 = get_plane(bands_1based[0]);
                let p1 = get_plane(bands_1based[1]);
                let mut g_p = Vec::with_capacity(npix * 4);
                let mut b_p = Vec::with_capacity(npix * 4);
                for i in 0..npix {
                    let vv = cogzilla_kernel::fast_db_from_power(p0[i]);
                    let vh = cogzilla_kernel::fast_db_from_power(p1[i]);
                    out.put_slice(&vv.to_le_bytes());
                    g_p.extend_from_slice(&vh.to_le_bytes());
                    b_p.extend_from_slice(&(vv - vh).to_le_bytes());
                }
                out.put_slice(&g_p);
                out.put_slice(&b_p);
                3
            }
            BandRequest::CustomExpr { prog, .. } => {
                let p_refs: Vec<&[f32]> = prog.used_bands.iter().map(|&b| get_plane(b)).collect();
                let mut vals = [0.0f32; 16];
                let k = p_refs.len().min(16);
                for i in 0..npix {
                    if expr_valid_mask.as_ref().is_some_and(|m| !m[i]) {
                        out.put_slice(&f32::NAN.to_le_bytes());
                        continue;
                    }
                    for j in 0..k {
                        vals[j] = p_refs[j][i];
                    }
                    out.put_slice(&prog.eval(&vals[..k]).to_le_bytes());
                }
                1
            }
            BandRequest::CustomExprRgb { progs, .. } => {
                for (c_idx, prog) in progs.iter().enumerate() {
                    let p_refs: Vec<&[f32]> =
                        prog.used_bands.iter().map(|&b| get_plane(b)).collect();
                    let mut vals = [0.0f32; 16];
                    let k = p_refs.len().min(16);
                    for i in 0..npix {
                        if expr_rgb_masks[c_idx].as_ref().is_some_and(|m| !m[i]) {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            continue;
                        }
                        for j in 0..k {
                            vals[j] = p_refs[j][i];
                        }
                        out.put_slice(&prog.eval(&vals[..k]).to_le_bytes());
                    }
                }
                3
            }
            BandRequest::MultiBand { bands_1based } => {
                for &b1 in bands_1based {
                    for &v in get_plane(b1) {
                        out.put_slice(&v.to_le_bytes());
                    }
                }
                bands_1based.len().max(1)
            }
        };
        let raw = out.freeze();
        if format == OutputFormat::Npy {
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                raw,
                nb,
                h as u32,
                w as u32,
                return_mask,
            ));
        }
        return Ok(raw);
    }

    let cf_single_lut: Option<[u8; 256]> = if matches!(
        bands,
        BandRequest::Single { .. }
            | BandRequest::Dem { .. }
            | BandRequest::CustomExpr { .. }
            | BandRequest::Spectral1 { .. }
            | BandRequest::Spectral2 { .. }
            | BandRequest::Spectral3 { .. }
    ) {
        color_formula.map(|cf| {
            let mut t = [0u8; 256];
            for i in 0..256 {
                let mut px = [i as u8, i as u8, i as u8, 255];
                cf.apply_buffer_in_place(&mut px, 4);
                t[i] = px[0];
            }
            t
        })
    } else {
        None
    };
    let map_color = |v: f32, rmin: f32, inv: f32, default_cmap: ColormapName| -> [u8; 4] {
        if let Some(cmap) = custom_lut {
            cmap.lookup(v, rmin, inv)
        } else {
            let mut idx = ((v - rmin) * inv).clamp(0.0, 255.0) as usize;
            if let Some(ref cf_lut) = cf_single_lut {
                idx = cf_lut[idx] as usize;
            }
            let [r, g, b] = default_cmap.lut().rgb[idx];
            [r, g, b, 255]
        }
    };

    let mut rgba = vec![0u8; npix * 4];
    match bands {
        BandRequest::Single {
            band_1based,
            rescale: (rescale_min, rescale_max),
            colormap,
        }
        | BandRequest::Dem {
            band_1based,
            rescale: (rescale_min, rescale_max),
            colormap,
            ..
        } => {
            let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(*rescale_min, *rescale_max);
            for (i, &v) in get_plane(*band_1based).iter().enumerate() {
                if !v.is_nan() {
                    let c = map_color(v, *rescale_min, inv, *colormap);
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&c);
                }
            }
        }
        BandRequest::Rgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            let p0 = get_plane(bands_1based[0]);
            let p1 = get_plane(bands_1based[1]);
            let p2 = get_plane(bands_1based[2]);
            let inv = [
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
            ];
            for i in 0..npix {
                let (v0, v1, v2) = (p0[i], p1[i], p2[i]);
                if !(v0.is_nan() || v1.is_nan() || v2.is_nan()) {
                    rgba[i * 4] = ((v0 - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((v1 - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = ((v2 - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::Terrain {
            band_1based,
            encoding,
        } => {
            for (i, &v) in get_plane(*band_1based).iter().enumerate() {
                if !v.is_nan() {
                    let [r, g, b] = encoding.encode_rgb(v);
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
                }
            }
        }
        BandRequest::Spectral1 {
            band_1based,
            expr,
            rescale,
            colormap,
        } => {
            let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
            for (i, &v) in get_plane(*band_1based).iter().enumerate() {
                let out_v = expr.eval1(v);
                if !out_v.is_nan() {
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&map_color(out_v, rescale.0, inv, *colormap));
                }
            }
        }
        BandRequest::Spectral2 {
            bands_1based,
            expr,
            rescale,
            colormap,
        } => {
            let p0 = get_plane(bands_1based[0]);
            let p1 = get_plane(bands_1based[1]);
            let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
            for i in 0..npix {
                let out_v = expr.eval2(p0[i], p1[i]);
                if !out_v.is_nan() {
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&map_color(out_v, rescale.0, inv, *colormap));
                }
            }
        }
        BandRequest::Spectral3 {
            bands_1based,
            expr,
            rescale,
            colormap,
        } => {
            let p0 = get_plane(bands_1based[0]);
            let p1 = get_plane(bands_1based[1]);
            let p2 = get_plane(bands_1based[2]);
            let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
            for i in 0..npix {
                let out_v = expr.eval3(p0[i], p1[i], p2[i]);
                if !out_v.is_nan() {
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&map_color(out_v, rescale.0, inv, *colormap));
                }
            }
        }
        BandRequest::SarRgb {
            bands_1based,
            rescale_min,
            rescale_max,
        } => {
            let p0 = get_plane(bands_1based[0]);
            let p1 = get_plane(bands_1based[1]);
            let inv = [
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
            ];
            for i in 0..npix {
                let vv = cogzilla_kernel::fast_db_from_power(p0[i]);
                let vh = cogzilla_kernel::fast_db_from_power(p1[i]);
                if !vv.is_nan() && !vh.is_nan() {
                    rgba[i * 4] = ((vv - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((vh - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = (((vv - vh) - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::CustomExpr {
            prog,
            rescale,
            colormap,
        } => {
            let p_refs: Vec<&[f32]> = prog.used_bands.iter().map(|&b| get_plane(b)).collect();
            let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
            let mut vals = [0.0f32; 16];
            let k = p_refs.len().min(16);
            for i in 0..npix {
                if expr_valid_mask.as_ref().is_some_and(|m| !m[i]) {
                    continue;
                }
                for j in 0..k {
                    vals[j] = p_refs[j][i];
                }
                let out_v = prog.eval(&vals[..k]);
                if !out_v.is_nan() {
                    rgba[i * 4..i * 4 + 4].copy_from_slice(&map_color(out_v, rescale.0, inv, *colormap));
                }
            }
        }
        BandRequest::CustomExprRgb {
            progs,
            rescale_min,
            rescale_max,
        } => {
            let p_refs: [Vec<&[f32]>; 3] = [
                progs[0].used_bands.iter().map(|&b| get_plane(b)).collect(),
                progs[1].used_bands.iter().map(|&b| get_plane(b)).collect(),
                progs[2].used_bands.iter().map(|&b| get_plane(b)).collect(),
            ];
            let inv = [
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
            ];
            let mut vals = [0.0f32; 16];
            for i in 0..npix {
                let mut ch = [f32::NAN; 3];
                for c in 0..3 {
                    if expr_rgb_masks[c].as_ref().is_some_and(|m| !m[i]) {
                        ch[c] = f32::NAN;
                        break;
                    }
                    let k = p_refs[c].len().min(16);
                    for j in 0..k {
                        vals[j] = p_refs[c][j][i];
                    }
                    ch[c] = progs[c].eval(&vals[..k]);
                }
                if !(ch[0].is_nan() || ch[1].is_nan() || ch[2].is_nan()) {
                    rgba[i * 4] = ((ch[0] - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 1] = ((ch[1] - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 2] = ((ch[2] - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                    rgba[i * 4 + 3] = 255;
                }
            }
        }
        BandRequest::MultiBand { bands_1based } => {
            return Err(EngineError::BadRequest(format!(
                "unsupported bidx count {} for visual image output (expected 1 or 3 bands; use .tif, .npy, or .f32)",
                bands_1based.len()
            )));
        }
    }
    if cf_single_lut.is_none()
        && let Some(cf) = color_formula
    {
        cf.apply_buffer_in_place(&mut rgba, 4);
    }
    Ok(encode_rgba_output(w, h, &rgba, format, return_mask))
}
