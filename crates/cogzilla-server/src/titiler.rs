use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_io::OpenedCog;
use cogzilla_kernel::jpeg_fast::encode_jpeg_rgba;
use cogzilla_kernel::png_fast::{encode_filtered_png, filter_sub_rgb_row, filter_sub_rgba_row};
use cogzilla_kernel::webp_fast::{encode_webp_rgb, encode_webp_rgba};
use cogzilla_kernel::{
    BandSource, ColorFormula, ColormapLut, OutputFormat, ResampleMode, fast_db_from_power,
};
use cogzilla_tiff::PlanarConfig;

use crate::engine::{BandRequest, EngineError, TileEngine};

/// TiTiler-compatible `/cog/info` JSON payload built directly from our `CogIndex`.
pub fn build_cog_info(engine: &TileEngine, cog: &OpenedCog) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let full = cog.index.full_res();
    let bounds = crs.wgs84_bounds(&full.pixel_to_crs, full.width, full.height);
    let coarsest = cog.index.ifds.last().unwrap();
    let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
        .max(full.height as f64 / coarsest.height.max(1) as f64);
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);

    let count = full.samples_per_pixel as usize;
    let band_metadata: Vec<_> = (1..=count)
        .map(|i| serde_json::json!([format!("b{i}"), {}]))
        .collect();
    let band_descriptions: Vec<_> = (1..=count)
        .map(|i| serde_json::json!([format!("b{i}"), ""]))
        .collect();
    let colorinterp: Vec<&str> = match count {
        3 => vec!["red", "green", "blue"],
        4 => vec!["red", "green", "blue", "alpha"],
        _ => vec!["gray"; count],
    };
    let overviews: Vec<u32> = cog
        .index
        .ifds
        .iter()
        .skip(1)
        .map(|ifd| (full.width as f64 / ifd.width.max(1) as f64).round() as u32)
        .collect();

    let (nodata_type, nodata_value) = match cog.index.nodata {
        Some(v) if v.is_nan() => ("Nodata", serde_json::json!("nan")),
        Some(v) => ("Nodata", serde_json::json!(v)),
        None => ("None", serde_json::Value::Null),
    };

    let (x00, y00) = full.pixel_to_crs.forward(0.0, 0.0);
    let (x10, y10) = full.pixel_to_crs.forward(full.width as f64, 0.0);
    let (x01, y01) = full.pixel_to_crs.forward(0.0, full.height as f64);
    let (x11, y11) = full
        .pixel_to_crs
        .forward(full.width as f64, full.height as f64);
    let native_bounds = [
        x00.min(x10).min(x01).min(x11),
        y00.min(y10).min(y01).min(y11),
        x00.max(x10).max(x01).max(x11),
        y00.max(y10).max(y01).max(y11),
    ];

    Ok(serde_json::json!({
        "bounds": native_bounds,
        "wgs84_bounds": bounds,
        "minzoom": minzoom,
        "maxzoom": maxzoom,
        "band_metadata": band_metadata,
        "band_descriptions": band_descriptions,
        "dtype": full.sample_type.gdal_name(),
        "nodata_type": nodata_type,
        "nodata_value": nodata_value,
        "colorinterp": colorinterp,
        "driver": "GTiff",
        "count": count,
        "width": full.width,
        "height": full.height,
        "overviews": overviews,
    }))
}

/// TiTiler-compatible `/cog/info.geojson` Feature wrapping the raster footprint polygon.
pub fn build_cog_info_geojson(
    engine: &TileEngine,
    cog: &OpenedCog,
) -> Result<serde_json::Value, EngineError> {
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
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
        },
        "properties": info,
    }))
}

/// TiTiler-compatible TileJSON 2.2.0 (`/cog/{tms}/tilejson.json`).
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
    let (minzoom, maxzoom) =
        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor);
    let center = [
        0.5 * (bounds[0] + bounds[2]),
        0.5 * (bounds[1] + bounds[3]),
        minzoom as f64,
    ];
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
    }))
}

/// TiTiler-compatible `/cog/point/{lon},{lat}` exact point query (fetches only 1 internal tile!).
pub async fn query_point(
    engine: &TileEngine,
    cog: &OpenedCog,
    lon: f64,
    lat: f64,
    unscale: bool,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let ifd = cog.index.full_res();
    let (sx, sy) = crs.from_lonlat_deg(lon, lat);
    let (col_f, row_f) = ifd.crs_to_pixel.forward(sx, sy);
    let col = col_f.floor() as i64;
    let row = row_f.floor() as i64;

    let count = ifd.samples_per_pixel as usize;
    let band_names: Vec<String> = (1..=count).map(|i| format!("b{i}")).collect();

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
        if unscale && !v.is_nan() {
            let sc = cog.index.scales.get(b).copied().unwrap_or(1.0);
            let off = cog.index.offsets.get(b).copied().unwrap_or(0.0);
            v * sc + off
        } else {
            v
        }
    };

    let mut values: Vec<serde_json::Value> = Vec::with_capacity(count);
    match ifd.planar_config {
        PlanarConfig::Chunky => {
            let idx = ifd.tile_index(tx, ty, 0) as u32;
            let tiles = engine.store.get_blocks(cog, 0, &[Some(idx)]).await?;
            for b in 0..count {
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
            let indices: Vec<Option<u32>> = (0..count)
                .map(|b| Some(ifd.tile_index(tx, ty, b as u16) as u32))
                .collect();
            let tiles = engine.store.get_blocks(cog, 0, &indices).await?;
            for b in 0..count {
                let v = maybe_unscale(
                    b,
                    tiles[b]
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
    fn push_tile_unmasked(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
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
                if unscale {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() {
                                vals.push(v * scale + offset);
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() {
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            // Fallback if tile storage variant differs from IFD sample type
            (Self::Floats(vals), other) => {
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let mut v = other.get_f32(base + lx);
                        if v.is_finite() {
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
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = row[lx];
                            if v.is_finite() {
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
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = other.get_f32(base + lx);
                            if v.is_finite() {
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

    fn finalize_hist(
        hist: &[u64],
        denom_pixels: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
    ) -> serde_json::Value {
        let valid_px: usize = hist.iter().map(|&c| c as usize).sum();
        let masked_px = denom_pixels.saturating_sub(valid_px);
        let valid_pct = if denom_pixels > 0 {
            (valid_px as f64 / denom_pixels as f64) * 100.0
        } else {
            0.0
        };
        if valid_px == 0 {
            return serde_json::json!({
                "min": null, "max": null, "mean": null, "count": 0, "sum": 0.0,
                "std": null, "median": null, "valid_percent": 0.0,
                "masked_pixels": masked_px, "valid_pixels": 0,
                "percentile_2": null, "percentile_98": null,
                "histogram": [[], []],
            });
        }
        let reverse = unscale && scale < 0.0;
        let to_val = |k: usize| -> f32 {
            if unscale {
                (k as f32) * scale + offset
            } else {
                k as f32
            }
        };
        let rank_p2 = ((valid_px as f64) * 0.02) as usize;
        let rank_med = valid_px / 2;
        let rank_p98 = (((valid_px as f64) * 0.98) as usize).min(valid_px - 1);

        let mut min = f32::NAN;
        let mut max = f32::NAN;
        let mut p2 = f32::NAN;
        let mut median = f32::NAN;
        let mut p98 = f32::NAN;
        let mut sum = 0.0f64;
        let mut cum = 0usize;

        let n = hist.len();
        for step in 0..n {
            let k = if reverse { n - 1 - step } else { step };
            let c = hist[k] as usize;
            if c == 0 {
                continue;
            }
            let v = to_val(k);
            if min.is_nan() {
                min = v;
            }
            max = v;
            let prev = cum;
            cum += c;
            if prev <= rank_p2 && rank_p2 < cum {
                p2 = v;
            }
            if prev <= rank_med && rank_med < cum {
                median = v;
            }
            if prev <= rank_p98 && rank_p98 < cum {
                p98 = v;
            }
            sum += (v as f64) * (c as f64);
        }

        let mean = sum / (valid_px as f64);
        let min_f64 = min as f64;
        let max_f64 = max as f64;
        let span = (max_f64 - min_f64).max(1e-12);
        let mut var_sum = 0.0f64;
        let mut counts = [0u64; 10];
        for (k, &c) in hist.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let v = to_val(k) as f64;
            let d = v - mean;
            var_sum += d * d * (c as f64);
            let bin = (((v - min_f64) / span) * 10.0).floor().clamp(0.0, 9.0) as usize;
            counts[bin] += c;
        }
        let std = (var_sum / (valid_px as f64)).sqrt();
        let edges: Vec<f64> = (0..=10)
            .map(|i| min_f64 + span * (i as f64 / 10.0))
            .collect();
        serde_json::json!({
            "min": min,
            "max": max,
            "mean": mean,
            "count": valid_px,
            "sum": sum,
            "std": std,
            "median": median,
            "valid_percent": valid_pct,
            "masked_pixels": masked_px,
            "valid_pixels": valid_px,
            "percentile_2": p2,
            "percentile_98": p98,
            "histogram": [counts, edges],
        })
    }

    fn finalize(
        self,
        denom_pixels: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
    ) -> serde_json::Value {
        match self {
            Self::HistU8(hist) => Self::finalize_hist(&hist, denom_pixels, unscale, scale, offset),
            Self::HistU16(hist) => Self::finalize_hist(&hist, denom_pixels, unscale, scale, offset),
            Self::Floats(mut vals) => {
                let valid_px = vals.len();
                let masked_px = denom_pixels.saturating_sub(valid_px);
                let valid_pct = if denom_pixels > 0 {
                    (valid_px as f64 / denom_pixels as f64) * 100.0
                } else {
                    0.0
                };
                if valid_px == 0 {
                    return serde_json::json!({
                        "min": null, "max": null, "mean": null, "count": 0, "sum": 0.0,
                        "std": null, "median": null, "valid_percent": 0.0,
                        "masked_pixels": masked_px, "valid_pixels": 0,
                        "percentile_2": null, "percentile_98": null,
                        "histogram": [[], []],
                    });
                }
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                let mut sum = 0.0f64;
                for &v in &vals {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                    sum += v as f64;
                }
                let mean = sum / (valid_px as f64);
                let min_f64 = min as f64;
                let max_f64 = max as f64;
                let span = (max_f64 - min_f64).max(1e-12);
                let mut var_sum = 0.0f64;
                let mut counts = [0u64; 10];
                for &v in &vals {
                    let vf = v as f64;
                    let d = vf - mean;
                    var_sum += d * d;
                    let bin = (((vf - min_f64) / span) * 10.0).floor().clamp(0.0, 9.0) as usize;
                    counts[bin] += 1;
                }
                let std = (var_sum / (valid_px as f64)).sqrt();

                // O(N) Quickselect for median, p2, p98 (bit-exact match with full sort)
                let p2_idx = ((valid_px as f64) * 0.02) as usize;
                let med_idx = valid_px / 2;
                let p98_idx = (((valid_px as f64) * 0.98) as usize).min(valid_px - 1);
                let (left, med_ref, right) = vals.select_nth_unstable_by(med_idx, |a, b| a.total_cmp(b));
                let median = *med_ref;
                let p2 = if p2_idx < med_idx {
                    *left.select_nth_unstable_by(p2_idx, |a, b| a.total_cmp(b)).1
                } else {
                    median
                };
                let p98 = if p98_idx > med_idx {
                    *right
                        .select_nth_unstable_by(p98_idx - med_idx - 1, |a, b| a.total_cmp(b))
                        .1
                } else {
                    median
                };
                let edges: Vec<f64> = (0..=10)
                    .map(|i| min_f64 + span * (i as f64 / 10.0))
                    .collect();
                serde_json::json!({
                    "min": min,
                    "max": max,
                    "mean": mean,
                    "count": valid_px,
                    "sum": sum,
                    "std": std,
                    "median": median,
                    "valid_percent": valid_pct,
                    "masked_pixels": masked_px,
                    "valid_pixels": valid_px,
                    "percentile_2": p2,
                    "percentile_98": p98,
                    "histogram": [counts, edges],
                })
            }
        }
    }
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
    let gw = x_max.div_ceil(STEP) + 1;
    let gh = y_max.div_ceil(STEP) + 1;
    let base_col = (tx * ifd.tile_width) as usize;
    let base_row = (ty * ifd.tile_height) as usize;

    let mut grid_lon = vec![0.0f64; gw * gh];
    let mut grid_lat = vec![0.0f64; gw * gh];
    let mut t_min_lon = f64::INFINITY;
    let mut t_max_lon = f64::NEG_INFINITY;
    let mut t_min_lat = f64::INFINITY;
    let mut t_max_lat = f64::NEG_INFINITY;

    for gy in 0..gh {
        let row_f = (base_row + (gy * STEP).min(y_max)) as f64;
        for gx in 0..gw {
            let col_f = (base_col + (gx * STEP).min(x_max)) as f64;
            let (sx, sy) = ifd.pixel_to_crs.forward(col_f, row_f);
            let (lon, lat) = crs.to_lonlat_deg(sx, sy);
            grid_lon[gy * gw + gx] = lon;
            grid_lat[gy * gw + gx] = lat;
            t_min_lon = t_min_lon.min(lon);
            t_max_lon = t_max_lon.max(lon);
            t_min_lat = t_min_lat.min(lat);
            t_max_lat = t_max_lat.max(lat);
        }
    }

    // Fast tile-level reject against polygon bbox
    if t_max_lon < pm.bbox[0]
        || t_min_lon > pm.bbox[2]
        || t_max_lat < pm.bbox[1]
        || t_min_lat > pm.bbox[3]
    {
        return None;
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

fn compute_statistics_cpu(
    cog: &OpenedCog,
    ifd_idx: usize,
    _crs: &cogzilla_proj::CrsModel,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
    fetched: StatsTiles,
) -> serde_json::Value {
    let ifd = &cog.index.ifds[ifd_idx];
    let count = ifd.samples_per_pixel as usize;
    let total_pixels = (ifd.width as usize) * (ifd.height as usize);
    let mask_total_pixels;

    let mut per_band: Vec<BandStatsAcc> = (0..count)
        .map(|_| BandStatsAcc::for_sample_type(ifd.sample_type, total_pixels))
        .collect();

    match fetched {
        StatsTiles::Chunky {
            tiles,
            tile_masks,
            mask_total_pixels: mtp,
        } => {
            mask_total_pixels = mtp;
            for (ti, tile_opt) in tiles.iter().enumerate() {
                let Some(tile) = tile_opt else { continue };
                let tx = (ti as u32) % ifd.tiles_across;
                let ty = (ti as u32) / ifd.tiles_across;
                let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                let tw = ifd.tile_width as usize;

                if let Some(ref masks) = tile_masks {
                    let Some(ref mask_bits) = masks[ti] else {
                        continue;
                    };
                    for b in 0..count {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        per_band[b].push_tile_masked(
                            &tile.bands[b].data,
                            tw,
                            x_max,
                            y_max,
                            mask_bits,
                            unscale,
                            cog.index.scales[b],
                            cog.index.offsets[b],
                        );
                    }
                } else {
                    for b in 0..count {
                        if tile.bands[b].stats.all_nodata {
                            continue;
                        }
                        per_band[b].push_tile_unmasked(
                            &tile.bands[b].data,
                            tw,
                            x_max,
                            y_max,
                            unscale,
                            cog.index.scales[b],
                            cog.index.offsets[b],
                        );
                    }
                }
            }
        }
        StatsTiles::Planar {
            band_tiles,
            tile_masks,
            mask_total_pixels: planar_mask_px,
        } => {
            mask_total_pixels = planar_mask_px;
            for (b, tiles) in band_tiles.iter().enumerate() {
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
                    if let Some(ref masks) = tile_masks {
                        if let Some(ref mask_bits) = masks[ti] {
                            per_band[b].push_tile_masked(
                                &band.data,
                                tw,
                                x_max,
                                y_max,
                                mask_bits,
                                unscale,
                                cog.index.scales[b],
                                cog.index.offsets[b],
                            );
                        }
                    } else {
                        per_band[b].push_tile_unmasked(
                            &band.data,
                            tw,
                            x_max,
                            y_max,
                            unscale,
                            cog.index.scales[b],
                            cog.index.offsets[b],
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

    let mut out = serde_json::Map::new();
    for (b, acc) in per_band.into_iter().enumerate() {
        out.insert(
            format!("b{}", b + 1),
            acc.finalize(denom_pixels, unscale, cog.index.scales[b], cog.index.offsets[b]),
        );
    }

    serde_json::Value::Object(out)
}

/// TiTiler-compatible `/cog/statistics` computed from the coarsest overview <= `max_size`.
/// When `poly_mask` is `Some`, clips statistics strictly to pixels inside the posted GeoJSON polygon.
pub async fn compute_statistics(
    engine: &TileEngine,
    cog: &Arc<OpenedCog>,
    max_size: u32,
    poly_mask: Option<&PolygonMask>,
    unscale: bool,
) -> Result<serde_json::Value, EngineError> {
    let crs = engine.get_or_build_crs(cog)?;
    let mut ifd_idx = cog.index.ifds.len() - 1;
    for (i, ifd) in cog.index.ifds.iter().enumerate() {
        if ifd.width.max(ifd.height) <= max_size {
            ifd_idx = i;
            break;
        }
    }
    let ifd = &cog.index.ifds[ifd_idx];
    let count = ifd.samples_per_pixel as usize;
    let total_pixels = (ifd.width as usize) * (ifd.height as usize);

    let n_tiles = (ifd.tiles_across * ifd.tiles_down) as usize;
    let mut mask_total_pixels = 0usize;
    let tile_masks: Option<Vec<Option<Vec<u8>>>> = poly_mask.map(|pm| {
        let mut masks = Vec::with_capacity(n_tiles);
        for ty in 0..ifd.tiles_down {
            for tx in 0..ifd.tiles_across {
                let x_max = (ifd.width - tx * ifd.tile_width).min(ifd.tile_width) as usize;
                let y_max = (ifd.height - ty * ifd.tile_height).min(ifd.tile_height) as usize;
                if let Some((bits, inside)) =
                    build_tile_polygon_mask(ifd, &crs, pm, tx, ty, x_max, y_max)
                {
                    mask_total_pixels += inside;
                    masks.push(Some(bits));
                } else {
                    masks.push(None);
                }
            }
        }
        masks
    });

    let fetched = match ifd.planar_config {
        PlanarConfig::Chunky => {
            let mut indices = Vec::with_capacity(n_tiles);
            for ty in 0..ifd.tiles_down {
                for tx in 0..ifd.tiles_across {
                    let ti = (ty * ifd.tiles_across + tx) as usize;
                    if let Some(ref masks) = tile_masks {
                        if masks[ti].is_none() {
                            indices.push(None);
                            continue;
                        }
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
            let mut band_tiles = Vec::with_capacity(count);
            for b in 0..count {
                let mut indices = Vec::with_capacity(n_tiles);
                for ty in 0..ifd.tiles_down {
                    for tx in 0..ifd.tiles_across {
                        let ti = (ty * ifd.tiles_across + tx) as usize;
                        if let Some(ref masks) = tile_masks {
                            if masks[ti].is_none() {
                                indices.push(None);
                                continue;
                            }
                        }
                        indices.push(Some(ifd.tile_index(tx, ty, b as u16) as u32));
                    }
                }
                band_tiles.push(engine.store.get_blocks(cog, ifd_idx, &indices).await?);
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
        let val = engine
            .store
            .run_heavy_cpu(move || {
                compute_statistics_cpu(
                    &cog_owned,
                    ifd_idx,
                    &crs,
                    pm_owned.as_ref(),
                    unscale,
                    fetched,
                )
            })
            .await?;
        Ok(val)
    } else {
        Ok(compute_statistics_cpu(
            cog, ifd_idx, &crs, poly_mask, unscale, fetched,
        ))
    }
}

/// Simple ray-casting point-in-polygon check for GeoJSON Feature crop masking (`POST /cog/feature`).
#[derive(Clone)]
pub struct PolygonMask {
    /// List of polygons; each polygon is `[outer_ring, hole1, ...]`, each ring is `Vec<(lon, lat)>`.
    pub polygons: Vec<Vec<Vec<(f64, f64)>>>,
    /// Precomputed `[min_lon, min_lat, max_lon, max_lat]` for fast bounding-box rejection.
    pub bbox: [f64; 4],
}

impl PolygonMask {
    pub fn from_geojson(val: &serde_json::Value) -> Result<(Self, [f64; 4]), EngineError> {
        let geom = if val.get("type").and_then(|v| v.as_str()) == Some("Feature") {
            val.get("geometry")
                .ok_or_else(|| EngineError::BadRequest("GeoJSON Feature missing geometry".into()))?
        } else {
            val
        };
        let gtype = geom
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::BadRequest("GeoJSON missing geometry.type".into()))?;
        let coords = geom
            .get("coordinates")
            .ok_or_else(|| EngineError::BadRequest("GeoJSON missing coordinates".into()))?;

        let parse_ring = |arr: &serde_json::Value| -> Result<Vec<(f64, f64)>, EngineError> {
            let list = arr
                .as_array()
                .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON ring".into()))?;
            let mut ring = Vec::with_capacity(list.len());
            for pt in list {
                let p = pt
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid GeoJSON coord".into()))?;
                if p.len() < 2 {
                    return Err(EngineError::BadRequest("coord needs [lon, lat]".into()));
                }
                ring.push((
                    p[0].as_f64().unwrap_or(0.0),
                    p[1].as_f64().unwrap_or(0.0),
                ));
            }
            Ok(ring)
        };

        let mut polygons = Vec::new();
        match gtype {
            "Polygon" => {
                let rings = coords
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid Polygon coords".into()))?;
                let mut poly = Vec::new();
                for r in rings {
                    poly.push(parse_ring(r)?);
                }
                polygons.push(poly);
            }
            "MultiPolygon" => {
                let polys = coords
                    .as_array()
                    .ok_or_else(|| EngineError::BadRequest("invalid MultiPolygon coords".into()))?;
                for p in polys {
                    let rings = p
                        .as_array()
                        .ok_or_else(|| EngineError::BadRequest("invalid polygon".into()))?;
                    let mut poly = Vec::new();
                    for r in rings {
                        poly.push(parse_ring(r)?);
                    }
                    polygons.push(poly);
                }
            }
            other => {
                return Err(EngineError::BadRequest(format!(
                    "unsupported GeoJSON geometry type '{other}' (expected Polygon or MultiPolygon)"
                )));
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
        if !min_x.is_finite() {
            return Err(EngineError::BadRequest("empty GeoJSON polygon".into()));
        }
        let bbox = [min_x, min_y, max_x, max_y];
        Ok((Self { polygons, bbox }, bbox))
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
}

/// Encode an RGBA8 buffer to the requested `OutputFormat`, optionally stripping the alpha mask.
pub(crate) fn encode_rgba_output(
    w: usize,
    h: usize,
    rgba: &[u8],
    format: OutputFormat,
    return_mask: bool,
) -> Bytes {
    match format {
        OutputFormat::Png => {
            if !return_mask {
                let mut rgb = vec![0u8; w * h * 3];
                for i in 0..(w * h) {
                    if rgba[i * 4 + 3] != 0 {
                        rgb[i * 3] = rgba[i * 4];
                        rgb[i * 3 + 1] = rgba[i * 4 + 1];
                        rgb[i * 3 + 2] = rgba[i * 4 + 2];
                    }
                }
                let stride = 1 + w * 3;
                let mut filtered = vec![0u8; h * stride];
                for py in 0..h {
                    filter_sub_rgb_row(
                        &rgb[py * w * 3..(py + 1) * w * 3],
                        &mut filtered[py * stride..(py + 1) * stride],
                    );
                }
                encode_filtered_png(w as u32, h as u32, 2, &filtered)
            } else {
                let stride = 1 + w * 4;
                let mut filtered = vec![0u8; h * stride];
                for py in 0..h {
                    filter_sub_rgba_row(
                        &rgba[py * w * 4..(py + 1) * w * 4],
                        &mut filtered[py * stride..(py + 1) * stride],
                    );
                }
                encode_filtered_png(w as u32, h as u32, 6, &filtered)
            }
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
        OutputFormat::RawF32 | OutputFormat::RawRgba => Bytes::copy_from_slice(rgba),
    }
}

/// Arbitrary-dimensions WGS84 bounding-box renderer backing `/cog/preview`, `/cog/bbox`,
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
    custom_lut: Option<&ColormapLut>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox,
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
    custom_lut: Option<&ColormapLut>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox_3857,
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

#[allow(clippy::too_many_arguments)]
async fn render_bbox_impl(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    bbox_is_3857: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&ColormapLut>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    let w = width.clamp(1, 4096) as usize;
    let h = height.clamp(1, 4096) as usize;
    let primary = &cogs[0];
    let crs = engine.get_or_build_crs(primary)?;

    let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
        if bbox_is_3857 {
            crs.from_3857(x, y)
        } else {
            crs.from_lonlat_deg(x, y)
        }
    };

    // Estimate target CRS resolution to pick optimal overview IFD.
    // Probe bbox center first; if out of CRS domain (e.g. WorldCRS84Quad z=0/1 on a UTM COG >75°
    // from center), fall back to the raster's own center clamped into bbox, then 4 quadrant probes.
    let cx = 0.5 * (bbox[0] + bbox[2]);
    let cy = 0.5 * (bbox[1] + bbox[3]);
    let dlx = (bbox[2] - bbox[0]) / (w as f64);
    let dly = (bbox[3] - bbox[1]) / (h as f64);
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
        let (rcx_raw, rcy_raw) = if bbox_is_3857 {
            let lat_rad = (0.5 * (rb[1] + rb[3])).clamp(-85.05112878, 85.05112878).to_radians();
            let lon_rad = (0.5 * (rb[0] + rb[2])).to_radians();
            const R: f64 = 6_378_137.0;
            (R * lon_rad, R * (std::f64::consts::FRAC_PI_4 + 0.5 * lat_rad).tan().ln())
        } else {
            (0.5 * (rb[0] + rb[2]), 0.5 * (rb[1] + rb[3]))
        };
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
    let ifd_idx = primary.index.select_ifd(target_rx, target_ry);
    crate::engine::validate_multi_cog_grid(cogs, ifd_idx)?;
    let ifd = &primary.index.ifds[ifd_idx];

    // Precompute source pixel coordinates `(u, v)` using a coarse 16-px projection grid
    // + bilinear interpolation (reduces UTM/LCC trig calls by ~225× while keeping <0.005 px error).
    // Any grid cell with a non-finite corner falls back to exact per-pixel projection.
    let mut u_map = vec![f32::NAN; w * h];
    let mut v_map = vec![f32::NAN; w * h];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;

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
                let (col, row) = ifd.crs_to_pixel.forward(sx, sy);
                grid_u[gy * gw + gx] = col - 0.5;
                grid_v[gy * gw + gx] = row - 0.5;
            }
        }
    }

    let ifd_w_f = ifd.width as f32;
    let ifd_h_f = ifd.height as f32;

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

            // Skip entire cell if all 4 corners are strictly outside the source image bounds,
            // and fall back to exact per-pixel projection if the cell wraps across the antimeridian.
            let mut use_bilinear = all_finite;
            if all_finite {
                let cu_min = u00.min(u10).min(u01).min(u11) as f32;
                let cu_max = u00.max(u10).max(u01).max(u11) as f32;
                let cv_min = v00.min(v10).min(v01).min(v11) as f32;
                let cv_max = v00.max(v10).max(v01).max(v11) as f32;
                if cu_max < -1.0 || cu_min > ifd_w_f || cv_max < -1.0 || cv_min > ifd_h_f {
                    continue;
                }
                if ifd.width >= 64 && w > STEP * 2 && (cu_max - cu_min) > 0.5 * ifd_w_f {
                    use_bilinear = false;
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
                    if let Some(pm) = poly_mask {
                        for px in x0..x1 {
                            let lon = bbox[0] + (px as f64 + 0.5) * dlx;
                            if pm.contains(lon, lat) {
                                let uu = u_cur as f32;
                                let vv = v_cur as f32;
                                if uu >= -1.0 && uu <= ifd_w_f && vv >= -1.0 && vv <= ifd_h_f {
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
                            if uu >= -1.0 && uu <= ifd_w_f && vv >= -1.0 && vv <= ifd_h_f {
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
                        if let Some(pm) = poly_mask {
                            if !pm.contains(lon, lat) {
                                continue;
                            }
                        }
                        let (sx, sy) = to_src_crs(lon, lat);
                        if !sx.is_finite() || !sy.is_finite() {
                            continue;
                        }
                        let (col, row) = ifd.crs_to_pixel.forward(sx, sy);
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

    // Empty check
    if !min_u.is_finite() {
        return Ok(match format {
            OutputFormat::RawF32 => {
                let nb: usize = match bands {
                    BandRequest::Rgb { .. } | BandRequest::SarRgb { .. } => 3,
                    _ => 1,
                };
                let mut b = BytesMut::with_capacity(nb * w * h * 4);
                for _ in 0..(nb * w * h) {
                    b.put_slice(&f32::NAN.to_le_bytes());
                }
                b.freeze()
            }
            OutputFormat::RawRgba => Bytes::from(vec![0u8; w * h * 4]),
            _ => encode_rgba_output(w, h, &vec![0u8; w * h * 4], format, return_mask),
        });
    }

    let c0 = (min_u.floor() as i32 - 1).max(0) as u32;
    let c1 = (max_u.ceil() as i32 + 1).min(ifd.width as i32 - 1).max(0) as u32;
    let r0 = (min_v.floor() as i32 - 1).max(0) as u32;
    let r1 = (max_v.ceil() as i32 + 1).min(ifd.height as i32 - 1).max(0) as u32;
    let tx0 = c0 / ifd.tile_width;
    let tx1 = (c1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
    let ty0 = r0 / ifd.tile_height;
    let ty1 = (r1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
    let grid_w = tx1 - tx0 + 1;
    let grid_h = ty1 - ty0 + 1;

    let grid_len = (grid_w * grid_h) as usize;
    let is_single_planar =
        cogs.len() == 1 && ifd.planar_config == cogzilla_tiff::PlanarConfig::Planar;

    // Validate 1-based band indices on single-COG requests and collect unique 0-based planes
    // needed when the COG is `PlanarConfig::Planar`.
    let mut planar_bands: Vec<u16> = Vec::new();
    if cogs.len() == 1 {
        let mut check_band = |b_1based: u16| -> Result<(), EngineError> {
            let b0 = b_1based.saturating_sub(1);
            if b0 >= ifd.samples_per_pixel {
                return Err(EngineError::BadRequest(format!(
                    "band {} out of range (dataset has {} bands)",
                    b_1based, ifd.samples_per_pixel
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
        }
    }

    // Fetch needed bands (single COG chunky/planar or multi-COG)
    let cog_tiles = if cogs.len() == 1 {
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

    let render_cpu = move |u_map: Vec<f32>,
                           v_map: Vec<f32>,
                           cog_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>,
                           bands: &BandRequest,
                           custom_lut: Option<&ColormapLut>,
                           color_formula: Option<&ColorFormula>|
          -> Bytes {
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
            }
        };

        let apply_source_unscale =
            unscale && crate::engine::needs_source_unscale(bands, format);

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

        // Real RawF32 output for single-band / expression / DEM / Terrain requests
        if format == OutputFormat::RawF32 {
            let mut out = BytesMut::with_capacity(w * h * 4);
            match bands {
                BandRequest::Dem {
                    band_1based,
                    algorithm,
                    ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let mut elev = vec![f32::NAN; w * h];
                    for i in 0..(w * h) {
                        elev[i] = sample(&src, u_map[i], v_map[i]);
                    }
                    let px_m = if bbox_is_3857 {
                        let (_, center_lat) = cogzilla_proj::mercator_to_lonlat_deg(cx, cy);
                        (dlx.abs().max(dly.abs()) * center_lat.to_radians().cos()).max(1e-3) as f32
                    } else {
                        let lat_rad = (0.5 * (bbox[1] + bbox[3])).to_radians();
                        ((dlx * 111_320.0 * lat_rad.cos()).abs()
                            .max((dly * 111_320.0).abs()))
                            .max(1e-3) as f32
                    };
                    let inv_8dx = 1.0 / (8.0 * px_m);
                    for py in 0..h {
                        let ym = py.saturating_sub(1);
                        let yp = (py + 1).min(h - 1);
                        for px in 0..w {
                            let xm = px.saturating_sub(1);
                            let xp = (px + 1).min(w - 1);
                            let val = cogzilla_kernel::eval_horn_3x3(
                                elev[ym * w + xm],
                                elev[ym * w + px],
                                elev[ym * w + xp],
                                elev[py * w + xm],
                                elev[py * w + px],
                                elev[py * w + xp],
                                elev[yp * w + xm],
                                elev[yp * w + px],
                                elev[yp * w + xp],
                                inv_8dx,
                                *algorithm,
                            );
                            out.put_slice(&val.to_le_bytes());
                        }
                    }
                }
                BandRequest::Single { band_1based, .. }
                | BandRequest::Terrain { band_1based, .. } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    for i in 0..(w * h) {
                        out.put_slice(&sample(&src, u_map[i], v_map[i]).to_le_bytes());
                    }
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
                        for k in 0..nb {
                            vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                        }
                        out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
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
            }
            return out.freeze();
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let v = sample(&src, u_map[i], v_map[i]);
                    if !v.is_nan() {
                        let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                        let [r, g, b, a] = lut.rgba_u32[idx].to_le_bytes();
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
                        let [r, g, b] = match encoding {
                            cogzilla_kernel::TerrainEncoding::Terrarium => {
                                cogzilla_kernel::encode_terrarium_rgb(v)
                            }
                            cogzilla_kernel::TerrainEncoding::TerrainRgb => {
                                cogzilla_kernel::encode_terrain_rgb(v)
                            }
                        };
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let mut elev = vec![f32::NAN; w * h];
                for i in 0..(w * h) {
                    elev[i] = sample(&src, u_map[i], v_map[i]);
                }
                let px_m = if bbox_is_3857 {
                    let (_, center_lat) = cogzilla_proj::mercator_to_lonlat_deg(cx, cy);
                    (dlx.abs().max(dly.abs()) * center_lat.to_radians().cos()).max(1e-3) as f32
                } else {
                    let lat_rad = (0.5 * (bbox[1] + bbox[3])).to_radians();
                    ((dlx * 111_320.0 * lat_rad.cos()).abs()
                        .max((dly * 111_320.0).abs()))
                        .max(1e-3) as f32
                };
                let inv_8dx = 1.0 / (8.0 * px_m);
                for py in 0..h {
                    let ym = py.saturating_sub(1);
                    let yp = (py + 1).min(h - 1);
                    for px in 0..w {
                        let xm = px.saturating_sub(1);
                        let xp = (px + 1).min(w - 1);
                        let v = cogzilla_kernel::eval_horn_3x3(
                            elev[ym * w + xm],
                            elev[ym * w + px],
                            elev[ym * w + xp],
                            elev[py * w + xm],
                            elev[py * w + px],
                            elev[py * w + xp],
                            elev[yp * w + xm],
                            elev[yp * w + px],
                            elev[yp * w + xp],
                            inv_8dx,
                            *algorithm,
                        );
                        if !v.is_nan() {
                            let i = py * w + px;
                            let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                            let [r, g, b, a] = lut.rgba_u32[idx].to_le_bytes();
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let mut vals = [0.0f32; 16];
                let nb = srcs.len().min(16);
                for i in 0..(w * h) {
                    for k in 0..nb {
                        vals[k] = sample(&srcs[k], u_map[i], v_map[i]);
                    }
                    let v = prog.eval(&vals[..nb]);
                    if !v.is_nan() {
                        let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                        let [r, g, b, a] = lut.rgba_u32[idx].to_le_bytes();
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                for i in 0..(w * h) {
                    let v = expr.eval1(sample(&src, u_map[i], v_map[i]));
                    if !v.is_nan() {
                        let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                        let [r, g, b, a] = lut.rgba_u32[idx].to_le_bytes();
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
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
                        let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                        let [r, g, b, al] = lut.rgba_u32[idx].to_le_bytes();
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
                let lut = custom_lut.unwrap_or_else(|| colormap.lut());
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
                        let idx = ((v - rescale.0) * inv).clamp(0.0, 255.0) as usize;
                        let [r, g, b, al] = lut.rgba_u32[idx].to_le_bytes();
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = al;
                    }
                }
            }
        }

        if let Some(cf) = color_formula {
            cf.apply_buffer_in_place(&mut rgba, 4);
        }

        encode_rgba_output(w, h, &rgba, format, return_mask)
    };

    // Offload large (>512x512) bbox renders to bounded spawn_blocking so giant previews/crops
    // never stall Tokio's async I/O workers.
    if w * h > 512 * 512 {
        let bands_owned = bands.clone();
        let lut_owned = custom_lut.cloned();
        let cf_owned = color_formula.cloned();
        let bytes = engine
            .store
            .run_heavy_cpu(move || {
                render_cpu(
                    u_map,
                    v_map,
                    cog_tiles,
                    &bands_owned,
                    lut_owned.as_ref(),
                    cf_owned.as_ref(),
                )
            })
            .await?;
        Ok(bytes)
    } else {
        Ok(render_cpu(
            u_map,
            v_map,
            cog_tiles,
            bands,
            custom_lut,
            color_formula,
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
