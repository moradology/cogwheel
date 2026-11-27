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
pub mod histogram;
pub use self::histogram::HistDtype;

mod band_stats;
mod bbox;
pub use self::bbox::{
    render_crs_bbox, render_exact_f64_array, render_mercator_bbox, render_native_preview,
    render_native_preview_stats, render_wgs84_bbox, render_wgs84_bbox_stats,
};
pub(crate) use self::bbox::render_bbox_impl;
mod compute;
mod metadata;
mod polygon;

use self::band_stats::gpu_band_stats_to_json;
pub use self::compute::{compute_statistics, compute_statistics_with_options};
pub(crate) use self::band_stats::format_category_key;
pub use self::band_stats::{
    BandHistogram, BandStats, StatCount, StatScalar, StatsOptions, compute_f32_band_stats,
    compute_f32_band_stats_with_engine,
};
pub use self::metadata::{
    build_cog_info, build_cog_info_geojson, build_cog_validate, build_map_html, build_tilejson,
    build_wmts_capabilities, build_wmts_capabilities_multi,
};
pub(crate) use self::metadata::xml_escape;
pub use self::polygon::{PolygonMask, extract_geojson_crs};
pub use crate::algorithms::{
    SUPPORTED_ALGORITHMS, get_algorithm_json, is_supported_algorithm, list_algorithms_json,
};

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
    let Some(norm_lon) = cogzilla_proj::normalize_geographic_lon(lon) else {
        return Err(EngineError::NotFound(format!(
            "point ({lon}, {lat}) is outside dataset bounds"
        )));
    };
    let q_lon = if let Some(lo) = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width)
        && norm_lon < lo
    {
        norm_lon + 360.0
    } else if let Some(hi) = crs.geographic_neg_180_max_lon(&ifd.pixel_to_crs, ifd.width)
        && norm_lon > hi
    {
        norm_lon - 360.0
    } else {
        norm_lon
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

    let maybe_unscale = |b: usize, v: f64| -> f64 {
        if let Some(nd) = nodata_override
            && !nd.is_nan() && v == nd as f64 {
                return f64::NAN;
            }
        if unscale && !v.is_nan() {
            let sc = cog.index.scales_f64.get(b).copied().unwrap_or(1.0);
            let off = cog.index.offsets_f64.get(b).copied().unwrap_or(0.0);
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
                        .map(|band| band.data.get_f64(pidx))
                        .unwrap_or(f64::NAN),
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
                        .map(|band| band.data.get_f64(pidx))
                        .unwrap_or(f64::NAN),
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
    encode_rgba_output_as(w, h, rgba, format, return_mask, true)
}

/// [`encode_rgba_output`] for a picture known to be of three bands: it is written in
/// colour whatever its pixels hold, where the other writes a grey picture when every
/// pixel that is there happens to be grey (which an empty picture always is).
pub(crate) fn encode_rgba_output_colour(
    w: usize,
    h: usize,
    rgba: &[u8],
    format: OutputFormat,
    return_mask: bool,
) -> Bytes {
    encode_rgba_output_as(w, h, rgba, format, return_mask, false)
}

fn encode_rgba_output_as(
    w: usize,
    h: usize,
    rgba: &[u8],
    format: OutputFormat,
    return_mask: bool,
    grey_if_grey: bool,
) -> Bytes {
    thread_local! {
        static PNG_SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    const MAX_TLS_SCRATCH: usize = 4 * 1024 * 1024;
    match format {
        OutputFormat::Png => {
            let pixels = w.saturating_mul(h);
            let is_gray = grey_if_grey
                && rgba
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
        OutputFormat::Jpeg { quality } if grey_if_grey => encode_jpeg_rgba(w as u32, h as u32, rgba, quality),
        OutputFormat::Jpeg { quality } => {
            cogzilla_kernel::jpeg_fast::encode_jpeg_rgba_colour(w as u32, h as u32, rgba, quality)
        }
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 | OutputFormat::RawRgba => {
            Bytes::copy_from_slice(rgba)
        }
    }
}

