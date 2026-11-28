//! TiTiler metadata, validation, WMTS capabilities, TileJSON, and map viewer builders.

use cogzilla_io::OpenedCog;
use cogzilla_tiff::PlanarConfig;

use crate::engine::{EngineError, TileEngine};

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
    let nodata_exact = cog.index.nodata_f64.or_else(|| cog.index.nodata.map(|v| v as f64));
    let (nodata_type, nodata_value) = match nodata_exact {
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
        let bounds = crs.wgs84_bounds(
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
///
/// Structural COG validation (tiling, overviews, IFD/block layout) works for any readable TIFF,
/// including ungeoreferenced and RPC-only TIFFs; missing/unsupported georeferencing is reported
/// as a warning in `COG_warnings` with `GEO` fields set to `null` rather than failing with 400.
pub fn build_cog_validate(
    engine: &TileEngine,
    cog: &OpenedCog,
    url: &str,
    strict: bool,
) -> Result<serde_json::Value, EngineError> {
    let full = cog.index.full_res();
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let crs_opt = if cog.index.is_georeferenced {
        match engine.get_or_build_crs(cog) {
            Ok(c) => Some(c),
            Err(e) => {
                warnings.push(format!("Unsupported CRS metadata: {e}"));
                None
            }
        }
    } else {
        let msg = if cog.index.has_rpc {
            "Dataset has no supported georeferencing (missing GeoTransform/GCPs; RPC georeferencing is not supported)"
        } else {
            "Dataset has no georeferencing (missing GeoTransform/GCPs; RPCs are not supported)"
        };
        warnings.push(msg.into());
        None
    };

    let zoom_range = crs_opt
        .as_ref()
        .map(|crs| crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor));

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
    // rio-cogeo COG header & block-offset layout checks:
    // 1. Main IFD must sit near the front of the file (< 300 bytes) so a single initial range read
    //    can parse the header without extra round-trips.
    if cog.index.main_ifd_offset >= 300 {
        errors.push(format!(
            "The offset of the main IFD should be < 300. It is {} instead",
            cog.index.main_ifd_offset
        ));
    }
    let first_block_offset = |ifd: &cogzilla_tiff::IfdMeta| -> Option<u64> {
        ifd.tile_offsets.iter().copied().find(|&o| o > 0)
    };
    // 2. Image block data must be stored after its IFD (plain GTiff writes data first, IFD at EOF).
    if let (Some(first_blk), Some(&main_ifd_off)) = (
        first_block_offset(&cog.index.ifds[0]),
        cog.index.ifd_offsets.first(),
    ) && first_blk < main_ifd_off
    {
        errors.push(format!(
            "The offset of the first block of the image should be after its IFD (block offset {first_blk} < IFD offset {main_ifd_off})"
        ));
    }
    for (idx, ifd) in cog.index.ifds.iter().enumerate().skip(1) {
        if let (Some(first_blk), Some(&ifd_off)) =
            (first_block_offset(ifd), cog.index.ifd_offsets.get(idx))
            && first_blk < ifd_off
        {
            errors.push(format!(
                "The offset of the first block of overview of index {idx} should be after its IFD (block offset {first_blk} < IFD offset {ifd_off})"
            ));
        }
    }
    // 3. Coarser overviews must be stored before finer levels.
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
    let nodata_json = match cog.index.nodata_f64.or_else(|| cog.index.nodata.map(|v| v as f64)) {
        Some(v) if v.is_nan() => serde_json::json!("nan"),
        Some(v) => serde_json::json!(v),
        None => serde_json::Value::Null,
    };

    let geo_json = if crs_opt.is_some() {
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
                .map(|c| c.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height))
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
        let (minzoom, maxzoom) = zoom_range.unwrap_or((0, 0));
        serde_json::json!({
            "CRS": crs_str,
            "BoundingBox": native_bounds,
            "Origin": [full.pixel_to_crs.c, full.pixel_to_crs.f],
            "Resolution": [res_x, res_y],
            "MinZoom": minzoom,
            "MaxZoom": maxzoom,
        })
    } else {
        serde_json::json!({
            "CRS": null,
            "BoundingBox": null,
            "Origin": null,
            "Resolution": null,
            "MinZoom": null,
            "MaxZoom": null,
        })
    };

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
        "GEO": geo_json,
        "Tags": {},
        "IFD": ifds,
    }))
}

/// Escape XML text and attribute special characters.
pub(crate) fn xml_escape(s: &str) -> String {
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
    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
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
        "raster_layers": {},
        "band_descriptions": band_descriptions,
        "data_type": full.sample_type.gdal_name(),
    }))
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
