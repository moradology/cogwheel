use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_io::OpenedCog;
use cogzilla_kernel::{
    ColorFormula, ColormapName, CustomColormap, OutputFormat, ResampleMode,
};

use super::{PolygonMask, encode_rgba_output, render_bbox_impl};
use crate::engine::{BandRequest, EngineError, TileEngine};

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
pub(super) async fn render_mixed_grid_bbox(
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
        BandRequest::CustomExprMulti { progs } => {
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
    let mut expr_multi_masks: Vec<Option<Vec<bool>>> = Vec::new();
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
        BandRequest::CustomExprMulti { progs } => {
            expr_multi_masks.resize_with(progs.len(), || None);
            for (p_idx, prog) in progs.iter().enumerate() {
                if !prog_needs_mask(prog) {
                    continue;
                }
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
                expr_multi_masks[p_idx] = Some(combined);
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
            BandRequest::CustomExprMulti { progs } => {
                for (p_idx, prog) in progs.iter().enumerate() {
                    let p_refs: Vec<&[f32]> =
                        prog.used_bands.iter().map(|&b| get_plane(b)).collect();
                    let mut vals = [0.0f32; 16];
                    let k = p_refs.len().min(16);
                    for i in 0..npix {
                        if expr_multi_masks
                            .get(p_idx)
                            .and_then(|m| m.as_ref())
                            .is_some_and(|m| !m[i])
                        {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            continue;
                        }
                        for j in 0..k {
                            vals[j] = p_refs[j][i];
                        }
                        out.put_slice(&prog.eval(&vals[..k]).to_le_bytes());
                    }
                }
                progs.len().max(1)
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

    let cf_single_lut: Option<[u8; 256]> = if custom_lut.is_none()
        && matches!(
            bands,
            BandRequest::Single { .. }
                | BandRequest::Dem { .. }
                | BandRequest::CustomExpr { .. }
                | BandRequest::Spectral1 { .. }
                | BandRequest::Spectral2 { .. }
                | BandRequest::Spectral3 { .. }
        )
    {
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
        BandRequest::CustomExprMulti { progs } => {
            return Err(EngineError::BadRequest(format!(
                "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                progs.len()
            )));
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

