//! COG and polygon-masked band statistics computation (CPU + GPU fast paths).

use std::sync::Arc;

use bytes::Bytes;
use cogzilla_io::OpenedCog;
use cogzilla_kernel::{BandSource, ColormapName, OutputFormat, ResampleMode};
use cogzilla_tiff::PlanarConfig;

use crate::engine::{BandRequest, EngineError, TileEngine};
use crate::http::geo::native_bbox_dims_in_crs;

use super::band_stats::{BandStatsAcc, BandStatsFinalize, gpu_band_stats_to_json};
use super::{
    HistDtype, PolygonMask, StatsOptions, compute_f32_band_stats_with_engine,
    render_native_preview, render_wgs84_bbox, render_wgs84_bbox_stats,
};

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
    gpu: Option<&crate::gpu_backend::GpuBackend>,
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
            gpu,
        );
    }

    // 1-pass fused GPU fast path for unmasked float/int `/cog/statistics`:
    // skips CPU pixel extraction and per-band Vec<f32> allocations completely, and
    // submits multi-band requests concurrently so GpuCoalescer packs all bands into 1 submit.
    {
        let tile_masks_none = match &fetched {
            StatsTiles::Chunky { tile_masks, .. } | StatsTiles::Planar { tile_masks, .. } => {
                tile_masks.is_none()
            }
        };
        if poly_mask.is_none()
            && tile_masks_none
            && !opts.categorical
            && opts.histogram_edges.is_none()
            && !matches!(
                ifd.sample_type,
                cogzilla_raster::SampleType::Uint8 | cogzilla_raster::SampleType::Uint16
            )
            && !ifd.sample_type.needs_f64()
            && let Some(gpu_ref) = gpu
        {
            let alpha_idx = cogzilla_codec::alpha_channel_index(count, &ifd.extra_samples);
            let selected_bands: Option<&[u16]> = opts.bidx.as_deref().filter(|b| !b.is_empty());
            let sel_bands: Vec<usize> = (0..count)
                .filter(|&b| {
                    let b1 = (b + 1) as u16;
                    if let Some(sel) = selected_bands {
                        sel.contains(&b1)
                    } else {
                        Some(b) != alpha_idx
                    }
                })
                .collect();
            let (u_cols, v_rows) = build_stats_separable_coords(ifd, step, &tile_xs, &tile_ys);
            if !sel_bands.is_empty()
                && u_cols.len().saturating_mul(v_rows.len()) == total_pixels
                && !u_cols.is_empty()
                && !v_rows.is_empty()
            {
                let coords = crate::gpu_backend::BboxCoordSource::Separable {
                    u: &u_cols,
                    v: &v_rows,
                };
                let hist_bins = opts.histogram_bins.clamp(1, 1000);
                let n_tiles = (ifd.tiles_across * ifd.tiles_down) as usize;
                let eval_band = |b: usize| -> Option<(String, serde_json::Value)> {
                    let blocks = stats_band_blocks(&fetched, n_tiles, b);
                    let (sc, off) = if unscale {
                        (
                            cog.index.scales.get(b).copied().unwrap_or(1.0),
                            cog.index.offsets.get(b).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    };
                    let src = BandSource::new(
                        ifd.width,
                        ifd.height,
                        ifd.tile_width,
                        ifd.tile_height,
                        0,
                        0,
                        ifd.tiles_across,
                        ifd.tiles_down,
                        &blocks,
                    )
                    .with_nodata_override(nodata_override)
                    .with_scale_offset(sc, off);
                    let gs = gpu_ref
                        .try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::Single {
                                cog_id: cog.id,
                                ifd_idx,
                                band0: b as u16,
                                src: &src,
                            },
                            &crate::gpu_backend::GpuBboxOutput::Stats {
                                resample: ResampleMode::Nearest,
                                num_bins: hist_bins,
                                hist_range: opts.histogram_range,
                                percentiles: &opts.percentiles,
                            },
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_stats)?;
                    let bname = format!("b{}", b + 1);
                    let dtype = HistDtype::for_sample_type(ifd.sample_type, unscale);
                    let json = gpu_band_stats_to_json(&gs, total_pixels, &bname, opts, dtype, None);
                    Some((bname, json))
                };
                let results: Vec<Option<(String, serde_json::Value)>> = if sel_bands.len() == 1 {
                    vec![eval_band(sel_bands[0])]
                } else {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = sel_bands
                            .iter()
                            .map(|&b| s.spawn(move || eval_band(b)))
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| h.join().ok().flatten())
                            .collect()
                    })
                };
                if results.iter().all(Option::is_some) {
                    let mut gpu_out = serde_json::Map::new();
                    for item in results.into_iter().flatten() {
                        gpu_out.insert(item.0, item.1);
                    }
                    return serde_json::Value::Object(gpu_out);
                }
            }
        }
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
                gpu_band_stats_to_json(
                    &gs,
                    denom_pixels,
                    &bname,
                    opts,
                    HistDtype::for_sample_type(ifd.sample_type, unscale),
                    Some(vals),
                ),
            );
            continue;
        }
        let val = acc.finalize(&BandStatsFinalize {
            denom_pixels,
            unscale,
            scale,
            offset,
            nodata_override,
            description: &bname,
            opts,
            dtype: Some(HistDtype::for_sample_type(ifd.sample_type, unscale)),
        });
        out.insert(bname, val);
    }

    serde_json::Value::Object(out)
}

fn build_stats_separable_coords(
    ifd: &cogzilla_tiff::IfdMeta,
    step: usize,
    tile_xs: &Option<Vec<Vec<usize>>>,
    tile_ys: &Option<Vec<Vec<usize>>>,
) -> (Vec<f32>, Vec<f32>) {
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
    }
}

fn stats_band_blocks(
    fetched: &StatsTiles,
    n_tiles: usize,
    bi: usize,
) -> Vec<Option<&cogzilla_codec::DecodedBand>> {
    match fetched {
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
    }
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
    gpu: Option<&crate::gpu_backend::GpuBackend>,
) -> serde_json::Value {
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
    if poly_mask.is_none()
        && tile_masks.is_none()
        && !opts.categorical
        && opts.histogram_edges.is_none()
        && let Some(gpu_ref) = gpu
        && exprs
            .iter()
            .all(|(_, prog)| !prog.used_bands.is_empty() && prog.used_bands.len() <= 8)
    {
        let (u_cols, v_rows) = build_stats_separable_coords(ifd, step, tile_xs, tile_ys);
        if u_cols.len().saturating_mul(v_rows.len()) == denom_pixels
            && !u_cols.is_empty()
            && !v_rows.is_empty()
        {
            let coords = crate::gpu_backend::BboxCoordSource::Separable {
                u: &u_cols,
                v: &v_rows,
            };
            let hist_bins = opts.histogram_bins.clamp(1, 1000);
            let n_tiles = (ifd.tiles_across * ifd.tiles_down) as usize;
            let eval_expr = |i: usize, expr_str: &str, prog: &cogzilla_kernel::ExprProgram| -> Option<(String, serde_json::Value)> {
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
                        (stats_band_blocks(&fetched, n_tiles, bi), so)
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
                let gs = gpu_ref
                    .try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Expr {
                            cache_key,
                            srcs: &srcs,
                            prog,
                        },
                        &crate::gpu_backend::GpuBboxOutput::Stats {
                            resample: ResampleMode::Nearest,
                            num_bins: hist_bins,
                            hist_range: opts.histogram_range,
                            percentiles: &opts.percentiles,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_stats)?;
                Some((
                    format!("b{}", i + 1),
                    gpu_band_stats_to_json(
                        &gs,
                        denom_pixels,
                        expr_str,
                        opts,
                        HistDtype::for_expression(
                            expr_str,
                            HistDtype::for_sample_type(ifd.sample_type, unscale),
                        ),
                        None,
                    ),
                ))
            };
            let results: Vec<Option<(String, serde_json::Value)>> = if exprs.len() == 1 {
                vec![eval_expr(0, &exprs[0].0, &exprs[0].1)]
            } else {
                std::thread::scope(|s| {
                    let handles: Vec<_> = exprs
                        .iter()
                        .enumerate()
                        .map(|(i, (expr_str, prog))| s.spawn(move || eval_expr(i, expr_str, prog)))
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().ok().flatten())
                        .collect()
                })
            };
            if results.iter().all(Option::is_some) {
                let mut gpu_out = serde_json::Map::new();
                for item in results.into_iter().flatten() {
                    gpu_out.insert(item.0, item.1);
                }
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
    for (i, ((expr_str, _), vals)) in exprs.iter().zip(accs).enumerate() {
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
                gpu_band_stats_to_json(
                    &gs,
                    denom_pixels,
                    expr_str,
                    opts,
                    opts.hist_dtype.unwrap_or_else(|| {
                        HistDtype::for_expression(
                            expr_str,
                            HistDtype::for_sample_type(ifd.sample_type, unscale),
                        )
                    }),
                    Some(&vals),
                ),
            );
            continue;
        }
        let acc = BandStatsAcc::Floats(vals);
        let dtype = opts.hist_dtype.unwrap_or_else(|| {
            HistDtype::for_expression(expr_str, HistDtype::for_sample_type(ifd.sample_type, unscale))
        });
        let val = acc.finalize(&BandStatsFinalize {
            denom_pixels,
            unscale: false,
            scale: 1.0,
            offset: 0.0,
            nodata_override: None,
            description: expr_str,
            opts,
            dtype: Some(dtype),
        });
        out.insert(format!("b{}", i + 1), val);
    }
    serde_json::Value::Object(out)
}

/// A whole raster's statistics read at `out_w`×`out_h` with a resampling other than nearest,
/// as rio-tiler takes them: from its preview, which GDAL reads with that resampling. The read
/// is the one `/preview` draws (GDAL's kernels, widened when shrinking); a resampled value
/// is put into an integer band's type as GDAL puts it, rounded to the nearest and held to
/// the type's range, and one that lands on the nodata value is masked, as rio-tiler masks
/// it (a cubic kernel's undershoot beside nodata 0); then scales and offsets (`unscale`) and
/// band math are applied to those values, as rio-tiler applies them after the read.
#[allow(clippy::too_many_arguments)]
async fn compute_resampled_statistics(
    engine: &TileEngine,
    cog: &Arc<OpenedCog>,
    crs: &cogzilla_proj::CrsModel,
    out_w: usize,
    out_h: usize,
    unscale: bool,
    nodata_override: Option<f32>,
    opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let full = cog.index.full_res();
    let count = full.samples_per_pixel;
    let alpha_idx = cogzilla_codec::alpha_channel_index(count as usize, &full.extra_samples);
    let read_bands: Vec<u16> = if let Some(exprs) = &opts.expressions {
        let mut used: Vec<u16> = exprs.iter().flat_map(|(_, prog)| prog.used_bands.iter().copied()).collect();
        used.sort_unstable();
        used.dedup();
        used
    } else if let Some(bidx) = opts.bidx.as_ref().filter(|b| !b.is_empty()) {
        bidx.clone()
    } else {
        (1..=count).filter(|&b| Some((b - 1) as usize) != alpha_idx).collect()
    };
    if read_bands.is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }
    let bbox = crs.wgs84_bounds_with_gcp(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
    let request = BandRequest::MultiBand { bands_1based: read_bands.clone() };
    let raw = render_native_preview(
        engine,
        std::slice::from_ref(cog),
        bbox,
        out_w as u32,
        out_h as u32,
        &request,
        opts.resample,
        OutputFormat::RawF32,
        nodata_override,
        None,
        None,
        false,
        false,
    )
    .await?;
    let n_px = out_w * out_h;
    if raw.len() < n_px * 4 * read_bands.len() {
        return Err(EngineError::BadRequest("statistics read returned fewer bands than asked".into()));
    }
    // An integer band's range, as GDAL holds a resampled value to it.
    let whole = crate::http::array_type_range(full.sample_type);
    let nodata = nodata_override.map(f64::from).or(cog.index.nodata_f64).or(cog.index.nodata.map(f64::from));
    let opts = opts.clone();
    let sample_type = full.sample_type;
    let scales: Vec<(f64, f64)> = read_bands
        .iter()
        .map(|&b| {
            let i = (b - 1) as usize;
            (
                f64::from(cog.index.scales.get(i).copied().unwrap_or(1.0)),
                f64::from(cog.index.offsets.get(i).copied().unwrap_or(0.0)),
            )
        })
        .collect();
    let compute = move || -> serde_json::Value {
        // Each band as read: GDAL rounds a resampled value into an integer type; NaN is masked.
        let planes: Vec<Vec<f32>> = raw
            .chunks_exact(n_px * 4)
            .take(read_bands.len())
            .zip(&scales)
            .map(|(plane, &(scale, offset))| {
                plane
                    .chunks_exact(4)
                    .map(|c| {
                        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        let v = match whole {
                            Some((lo, hi)) if v.is_finite() => {
                                let r = f64::from(v).round().clamp(lo, hi);
                                if nodata == Some(r) { f32::NAN } else { r as f32 }
                            }
                            _ => v,
                        };
                        if unscale && v.is_finite() { (f64::from(v) * scale + offset) as f32 } else { v }
                    })
                    .collect()
            })
            .collect();
        let band_dtype = HistDtype::for_sample_type(sample_type, unscale);
        let finish = |values: Vec<f32>, desc: &str, dtype: HistDtype| {
            BandStatsAcc::Floats(values.into_iter().filter(|v| v.is_finite()).collect())
                .finalize(&BandStatsFinalize {
                    denom_pixels: n_px,
                    unscale: false,
                    scale: 1.0,
                    offset: 0.0,
                    nodata_override: None,
                    description: desc,
                    opts: &opts,
                    dtype: Some(opts.hist_dtype.unwrap_or(dtype)),
                })
        };
        let mut out = serde_json::Map::new();
        if let Some(exprs) = &opts.expressions {
            let slot = |b: u16| read_bands.iter().position(|&r| r == b);
            for (i, (expr_str, prog)) in exprs.iter().enumerate() {
                let width = prog.used_bands.iter().copied().max().unwrap_or(1) as usize;
                let mut px = vec![f32::NAN; width];
                let values: Vec<f32> = (0..n_px)
                    .map(|p| {
                        for &b in &prog.used_bands {
                            px[(b - 1) as usize] = slot(b).map_or(f32::NAN, |s| planes[s][p]);
                        }
                        if prog.used_bands.iter().any(|&b| px[(b - 1) as usize].is_nan()) {
                            f32::NAN
                        } else {
                            prog.eval(&px)
                        }
                    })
                    .collect();
                let dtype = HistDtype::for_expression(expr_str, band_dtype);
                out.insert(format!("b{}", i + 1), finish(values, expr_str, dtype));
            }
        } else {
            for (plane, &b) in planes.into_iter().zip(&read_bands) {
                let name = format!("b{b}");
                let val = finish(plane, &name, band_dtype);
                out.insert(name, val);
            }
        }
        serde_json::Value::Object(out)
    };
    if n_px > 256 * 256 {
        Ok(engine.store.run_heavy_cpu(compute).await?)
    } else {
        Ok(compute())
    }
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
        let (nat_w, nat_h) = native_bbox_dims_in_crs(
            full,
            &crs,
            &cogzilla_proj::CrsModel::Geographic,
            eff_bbox,
            eff_bbox,
        );
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
        let (_, feature_covered_px) = pm.rasterize_all_touched_bits(
            out_w as usize,
            out_h as usize,
            eff_bbox,
            &cogzilla_proj::CrsModel::Geographic,
        );
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
                let expr_opts = StatsOptions {
                    hist_dtype: opts.hist_dtype.or(Some(HistDtype::for_expression(
                        desc,
                        HistDtype::for_sample_type(full.sample_type, unscale),
                    ))),
                    ..opts.clone()
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
                    &expr_opts,
                )
                .await
                {
                    Ok(v) => v,
                    Err(EngineError::NotFound(_)) => {
                        compute_f32_band_stats_with_engine(engine, &[], desc, &expr_opts)
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
        // Plain bands: numpy builds the histogram's edges in the band's own type.
        let band_opts = StatsOptions {
            hist_dtype: opts
                .hist_dtype
                .or(Some(HistDtype::for_sample_type(full.sample_type, unscale))),
            ..opts.clone()
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
                    &band_opts,
                )
                .await
                {
                    Ok(v) => v,
                    Err(EngineError::NotFound(_)) => {
                        compute_f32_band_stats_with_engine(engine, &[], &desc, &band_opts)
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
                        engine, &pixels, &desc, &band_opts,
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
        if dims.is_some() && opts.resample != ResampleMode::Nearest {
            return compute_resampled_statistics(engine, cog, &crs, out_w, out_h, unscale, nodata_override, opts).await;
        }
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
                    if let Some(ref masks) = tile_masks
                        && masks[ti].is_none() {
                            indices.push(None);
                            continue;
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
                        if let Some(ref masks) = tile_masks
                            && masks[ti].is_none() {
                                all_indices.push(None);
                                continue;
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
            engine.gpu.as_deref(),
        ))
    }
}
