//! OGC WMS 1.1.1 / 1.3.0 (`GET /cog/wms`) `GetCapabilities` and `GetMap` handler.

use bytes::Bytes;
use cogzilla_kernel::OutputFormat;

use crate::engine::{EngineError, TileEngine};
use crate::titiler::render_crs_bbox;

use super::{
    ParsedQuery, apply_unscale_cogs, build_band_request_for_output, data_band_count,
    geotiff_render_format, open_query_urls, parse_bbox4, parse_bool_param,
    parse_coord_crs_with_epsg, parse_ext_format, percent_decode, query_key, resolve_custom_lut,
    resolve_input_bbox, resolve_resample, widest_sample_type, wrap_geotiff_fmt,
};

/// Strip WMS protocol parameters (case-insensitively) from a query string so `OnlineResource`
/// in `GetCapabilities` retains only dataset/rendering parameters (e.g. `url`, `bidx`, `rescale`)
/// and ends with `?` or `&` per OGC WMS 1.3.0 §6.3.3.
fn wms_online_resource(host: &str, raw_query: &str) -> String {
    const WMS_STRIP: &[&str] = &[
        "service", "request", "version", "wmtver", "layers", "styles", "crs", "srs", "bbox",
        "width", "height", "format", "transparent", "bgcolor", "exceptions", "time", "elevation",
        "updatesequence", "minzoom", "maxzoom", "tile_format", "tile_scale", "use_epsg", "zooms",
    ];
    let base_url = format!("{}/cog/wms", host.trim_end_matches('/'));
    let kept: Vec<&str> = raw_query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .filter(|pair| {
            let k = pair.split_once('=').map_or(*pair, |(k, _)| k);
            let kl = query_key(k).to_ascii_lowercase();
            !WMS_STRIP.contains(&kl.as_str())
        })
        .collect();
    if kept.is_empty() {
        format!("{base_url}?")
    } else {
        format!("{base_url}?{}&", kept.join("&"))
    }
}

/// Build an OGC WMS 1.3.0 `WMS_Capabilities` XML document for a COG dataset.
pub(super) fn build_wms_capabilities(
    host: &str,
    raw_query: &str,
    bounds: [f64; 4],
    native_epsg: Option<u16>,
) -> String {
    let bounds = if bounds.iter().all(|v| v.is_finite()) {
        bounds
    } else {
        [-180.0, -90.0, 180.0, 90.0]
    };
    let online_href = wms_online_resource(host, raw_query);
    let href_esc = crate::titiler::xml_escape(&online_href);
    let [w, s, e, n] = bounds;
    let [mw, ms, me, mn] = crate::catalog::wgs84_bbox_to_3857(bounds);
    let extra_crs = match native_epsg {
        Some(code) if code != 4326 && code != 3857 => {
            format!("      <CRS>EPSG:{code}</CRS>\n")
        }
        _ => String::new(),
    };
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<WMS_Capabilities version="1.3.0" xmlns="http://www.opengis.net/wms" xmlns:xlink="http://www.w3.org/1999/xlink">
  <Service>
    <Name>WMS</Name>
    <Title>cogzilla COG WMS</Title>
    <OnlineResource xlink:type="simple" xlink:href="{href_esc}"/>
    <MaxWidth>4096</MaxWidth>
    <MaxHeight>4096</MaxHeight>
  </Service>
  <Capability>
    <Request>
      <GetCapabilities>
        <Format>text/xml</Format>
        <Format>application/xml</Format>
        <DCPType><HTTP><Get><OnlineResource xlink:type="simple" xlink:href="{href_esc}"/></Get></HTTP></DCPType>
      </GetCapabilities>
      <GetMap>
        <Format>image/png</Format>
        <Format>image/jpeg</Format>
        <Format>image/webp</Format>
        <Format>image/tiff</Format>
        <DCPType><HTTP><Get><OnlineResource xlink:type="simple" xlink:href="{href_esc}"/></Get></HTTP></DCPType>
      </GetMap>
    </Request>
    <Exception>
      <Format>XML</Format>
    </Exception>
    <Layer queryable="0">
      <Name>cog</Name>
      <Title>Cloud Optimized GeoTIFF</Title>
      <CRS>CRS:84</CRS>
      <CRS>EPSG:4326</CRS>
      <CRS>EPSG:3857</CRS>
{extra_crs}      <EX_GeographicBoundingBox>
        <westBoundLongitude>{w}</westBoundLongitude>
        <eastBoundLongitude>{e}</eastBoundLongitude>
        <southBoundLatitude>{s}</southBoundLatitude>
        <northBoundLatitude>{n}</northBoundLatitude>
      </EX_GeographicBoundingBox>
      <BoundingBox CRS="CRS:84" minx="{w}" miny="{s}" maxx="{e}" maxy="{n}"/>
      <BoundingBox CRS="EPSG:4326" minx="{s}" miny="{w}" maxx="{n}" maxy="{e}"/>
      <BoundingBox CRS="EPSG:3857" minx="{mw}" miny="{ms}" maxx="{me}" maxy="{mn}"/>
    </Layer>
  </Capability>
</WMS_Capabilities>"#
    )
}

/// Parse an OGC WMS `FORMAT` parameter (`image/png`, `image/jpeg`, `image/webp`, `image/tiff`,
/// or short extensions) into `(OutputFormat, content_type)`.
pub(super) fn parse_wms_format(
    raw: Option<&str>,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    let norm = raw.map(|s| s.split(';').next().unwrap_or(s).trim().to_ascii_lowercase());
    let ext = match norm.as_deref() {
        None | Some("") | Some("image/png") | Some("png") => "png",
        Some("image/jpeg") | Some("image/jpg") | Some("jpeg") | Some("jpg") => "jpg",
        Some("image/webp") | Some("webp") => "webp",
        Some("image/tiff") | Some("image/geotiff") | Some("tif") | Some("tiff") => "tif",
        Some(other) => {
            return Err(EngineError::BadRequest(format!(
                "unsupported WMS FORMAT '{other}' (use image/png, image/jpeg, image/webp, or image/tiff)"
            )));
        }
    };
    parse_ext_format(ext, webp_quality, jpeg_quality)
}

/// Handle OGC WMS 1.1.1 / 1.3.0 (`GET /cog/wms`) `GetCapabilities` and `GetMap` requests.
pub(super) async fn handle_cog_wms(
    engine: &TileEngine,
    host: &str,
    raw_query: &str,
    pq: &ParsedQuery,
    geo_meta: &mut Option<([f64; 4], Option<u16>, bool)>,
) -> Result<(Bytes, &'static str), EngineError> {
    let mut wms_request: Option<String> = None;
    let mut wms_version: Option<String> = None;
    let mut wms_srs: Option<String> = None;
    let mut wms_crs_raw: Option<String> = None;
    let mut wms_bbox_raw: Option<String> = None;
    let mut wms_width_raw: Option<String> = None;
    let mut wms_height_raw: Option<String> = None;
    let mut wms_format_raw: Option<String> = None;
    let mut wms_transparent: Option<bool> = None;
    for pair in raw_query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        match query_key(k).to_ascii_lowercase().as_str() {
            "request" => wms_request = Some(percent_decode(v).trim().to_ascii_lowercase()),
            "version" => wms_version = Some(percent_decode(v).trim().to_string()),
            "srs" => wms_srs = Some(percent_decode(v)),
            "crs" => wms_crs_raw = Some(percent_decode(v)),
            "bbox" => wms_bbox_raw = Some(percent_decode(v)),
            "width" => wms_width_raw = Some(percent_decode(v)),
            "height" => wms_height_raw = Some(percent_decode(v)),
            "format" | "f" => wms_format_raw = Some(percent_decode(v)),
            "transparent" => {
                let dec = percent_decode(v);
                match parse_bool_param(dec.trim()) {
                    Some(b) => wms_transparent = Some(b),
                    None => {
                        return Err(EngineError::BadRequest(format!(
                            "invalid WMS TRANSPARENT '{dec}' (expected TRUE or FALSE)"
                        )));
                    }
                }
            }
            _ => {}
        }
    }

    match wms_request.as_deref() {
        Some("getcapabilities") => {
            let cogs = open_query_urls(engine, &pq.urls, None).await?;
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            let bounds = crs.wgs84_bounds_with_gcp(
                &full.pixel_to_crs,
                full.gcp_poly.as_deref(),
                full.width,
                full.height,
            );
            let native_epsg = cogs[0]
                .index
                .geo_keys
                .projected_cs_type
                .or(cogs[0].index.geo_keys.geographic_type)
                .filter(|&c| c != 32767);
            let xml = build_wms_capabilities(host, raw_query, bounds, native_epsg);
            Ok((Bytes::from(xml), "application/xml"))
        }
        Some("getmap") => {
            let bbox_str = wms_bbox_raw
                .as_deref()
                .ok_or_else(|| EngineError::BadRequest("WMS GetMap requires BBOX=minx,miny,maxx,maxy".into()))?;
            let (mut bbox, None) = parse_bbox4(bbox_str)
                .ok_or_else(|| EngineError::BadRequest(format!("invalid WMS BBOX '{bbox_str}'")))?
            else {
                return Err(EngineError::BadRequest(format!("invalid WMS BBOX '{bbox_str}'")));
            };
            let parse_dim = |raw: Option<&str>, name: &str| -> Result<u32, EngineError> {
                let s = raw.ok_or_else(|| EngineError::BadRequest(format!("WMS GetMap requires {name}")))?;
                let n = s
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| EngineError::BadRequest(format!("invalid WMS {name} '{s}'")))?;
                if !(1..=4096).contains(&n) {
                    return Err(EngineError::BadRequest(format!(
                        "WMS {name} must be between 1 and 4096 (got {n})"
                    )));
                }
                Ok(n)
            };
            let out_w = parse_dim(wms_width_raw.as_deref(), "WIDTH")?;
            let out_h = parse_dim(wms_height_raw.as_deref(), "HEIGHT")?;
            let crs_str = wms_crs_raw
                .as_deref()
                .or(wms_srs.as_deref())
                .unwrap_or("EPSG:3857");
            let (dst_crs, dst_epsg) =
                parse_coord_crs_with_epsg(crs_str).map_err(EngineError::BadRequest)?;
            // Default to WMS 1.3.0 unless VERSION explicitly says 1.1.x or client sent WMS 1.1.1 SRS= without CRS=.
            let is_wms_130 = match wms_version.as_deref() {
                Some(v) => v.starts_with("1.3"),
                None => !(wms_crs_raw.is_none() && wms_srs.is_some()),
            };
            let crs_lower = crs_str.trim().to_ascii_lowercase();
            let is_crs84 = crs_lower == "crs:84"
                || crs_lower == "crs84"
                || crs_lower.ends_with(":crs84")
                || crs_lower.ends_with("/crs84");
            // OGC WMS 1.3.0 specifies (lat, lon) axis order for geographic EPSG CRSes (unlike CRS:84 and WMS 1.1.1).
            if is_wms_130 && dst_crs.is_geographic() && !is_crs84 {
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]];
            }
            let bbox = resolve_input_bbox(bbox, Some(&dst_crs))?;
            let (fmt, ct) = parse_wms_format(
                wms_format_raw.as_deref(),
                pq.webp_quality,
                pq.jpeg_quality,
            )?;
            // Default transparent nodata mask for COG overlays unless TRANSPARENT=FALSE is passed.
            let return_mask = wms_transparent.unwrap_or(pq.return_mask);
            let cogs = open_query_urls(engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(pq, db, st, fmt, ct)?;
            let fmt = geotiff_render_format(fmt, ct, pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), dst_epsg);
            let data = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                &dst_crs,
                out_w,
                out_h,
                None,
                &bands,
                resample,
                fmt,
                pq.nodata,
                custom_lut.as_ref(),
                cf.as_ref(),
                return_mask,
                pq.unscale,
            )
            .await?;
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, bbox, dst_epsg, return_mask);
            *geo_meta = Some((bbox, Some(dst_epsg), is_crs84));
            Ok((data, ct))
        }
        Some(other) => Err(EngineError::BadRequest(format!(
            "unsupported WMS REQUEST '{other}' (use GetCapabilities or GetMap)"
        ))),
        None => Err(EngineError::BadRequest(
            "WMS requires REQUEST=GetCapabilities or REQUEST=GetMap".into(),
        )),
    }
}
