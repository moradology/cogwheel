use super::*;

/// Render a mosaic tile in a non-WebMercator TileMatrixSet (geographic TMSs delegate to
/// `part_image_of`; projected TMSs composite scenes via `render_crs_bbox`).
#[allow(clippy::too_many_arguments)]
pub(super) async fn render_non_mercator_tms_tile(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    id: &str,
    tms_def: &'static crate::tms::TmsDef,
    addr: TmsTileAddr,
    scale: Option<u32>,
    ext: &str,
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    if scale.is_some_and(|s| !(1..=8).contains(&s)) {
        return detail(StatusCode::BAD_REQUEST, "a tile's scale is from @1x to @8x");
    }
    if (pq.buffer * 2.0).fract() != 0.0 {
        return detail(StatusCode::BAD_REQUEST, "`buffer` must be a multiple of `0.5`");
    }
    let side = scale.map_or(pq.tile_size, |s| s * 256);
    let grown = (side as f32 + 2.0 * pq.buffer).round() as u32;
    let Some(bbox) = tms_def.tile_bounds_for_matrix(addr.matrix, addr.x, addr.y, pq.buffer, side) else {
        return no_content();
    };
    let Some(under) = tms_tile_polygon(tms_def, addr) else {
        return no_content();
    };
    if tms_def.epsg == 4326 {
        pq.buffer = 0.0;
        pq.coord_crs_3857 = false;
        pq.coord_crs = None;
        pq.coord_crs_epsg = None;
        const OF_A_BOX: [&str; 6] = ["buffer", "coord_crs", "dst_crs", "width", "height", "max_size"];
        let filtered: Vec<(String, String)> = pairs
            .iter()
            .filter(|(k, _)| !OF_A_BOX.contains(&k.as_str()))
            .cloned()
            .collect();
        let ext_opt = (!ext.is_empty()).then_some(ext);
        let resp = Box::pin(part_image_of(
            ctx,
            pg,
            id,
            &under,
            Some(&under),
            bbox,
            None,
            PartSize::Exact(grown, grown),
            ext_opt,
            pq,
            &filtered,
        ))
        .await;
        let spelled = if tms_def.id == "WorldCRS84Quad" { Some("crs84") } else { Some("epsg:4326") };
        return placed_as(resp, bbox, 4326, spelled);
    }

    let dst_crs = match tms_def.crs_model() {
        Ok(c) => c,
        Err(e) => return engine_error(&e),
    };
    let ext_opt = (!ext.is_empty()).then_some(ext);
    let _index = normalized_index_as_band_math(&mut pq, ext_opt);
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    let custom_lut = match pq
        .custom_colormap_json
        .as_deref()
        .map(|j| {
            cogzilla_kernel::CustomColormap::from_json_str_with_type(
                j,
                !pq.rescale.is_empty(),
                pq.colormap_type_linear,
            )
        })
        .transpose()
    {
        Ok(m) => m,
        Err(e) => return detail(StatusCode::BAD_REQUEST, e),
    };
    let names = match renderer_assets(&mut pq, &query) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let explicit = ext_opt.map(str::to_string).or_else(|| pq.format_param.clone());
    let tiff = matches!(explicit.as_deref(), Some("tif" | "tiff"));
    let format = match explicit.as_deref() {
        None => None,
        Some(e) => match parse_ext_format(e, pq.webp_quality, pq.jpeg_quality) {
            Ok(f) => Some(f),
            Err(e) => return engine_error(&e),
        },
    };
    if matches!(format, Some((OutputFormat::RawF32, _))) {
        return detail(
            StatusCode::BAD_REQUEST,
            "bare arrays are not written on non-WebMercator pgSTAC tiles; ask for `.npy` or `.tif`",
        );
    }
    let items = match pixel_items(ctx, pg, &under, id, query.limits, &names, None, query.lookup).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }
    let scenes: Result<Vec<Vec<String>>, Resp> = items
        .iter()
        .take(MAX_SCENES_PER_TILE)
        .map(|item| asset_hrefs(item, names.iter().map(String::as_str)))
        .collect();
    let scenes = match scenes {
        Ok(s) => s,
        Err(r) => return r,
    };

    let dim = grown as usize;
    let engine = ctx.engine;
    let want_array = matches!(format, Some((OutputFormat::Npy, _)));
    let rendered: Result<(Bytes, &'static str), EngineError> = async {
        let formula = pq
            .color_formula
            .as_deref()
            .map(cogzilla_kernel::ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)?;
        let mut canvas = vec![0u8; dim * dim * 4];
        let mut selecting: Option<(Fold, Option<Paint>, usize)> = None;
        let visual_array = pq.colormap_explicit || formula.is_some() || !pq.rescale.is_empty();
        let mut map_after: Option<MapAfterFormula> = None;
        let mut one_grey = false;
        for (n, scene) in scenes.iter().enumerate() {
            let mut cogs = Vec::with_capacity(scene.len());
            for href in scene {
                cogs.push(engine.store.open_uri(href).await.map(|c| engine.store.with_nodata(&c, pq.nodata))?);
            }
            let full = cogs[0].index.full_res();
            if query.asset_as_band && full.samples_per_pixel > 1 {
                return Err(EngineError::BadRequest(
                    "Can't use `asset_as_band` for multibands asset".into(),
                ));
            }
            let default_bands = if cogs.len() > 1 { cogs.len() as u16 } else { full.samples_per_pixel };
            if pq.rescale.len() > 1 && pq.rescale.len() != 3 {
                pq.rescale.truncate(1);
            }
            let mut bands = build_band_request(&pq, default_bands, full.sample_type)?;
            let taken = match &formula {
                Some(formula) if pq.colormap_explicit => MapAfterFormula::take(formula, &mut bands),
                _ => None,
            };
            if n == 0 {
                map_after = taken;
                one_grey = !pq.colormap_explicit
                    && (one_grey_band(&bands) || query.selection == Some(Selection::Count))
                    && !crate::http::has_embedded_palette(&pq, &cogs[0], &bands);
            }
            let exact = formula.is_some() && one_grey_band(&bands) && (map_after.is_some() || one_grey);
            let same_crs = cogs[0].index.geo_keys.projected_cs_type == Some(tms_def.epsg);
            let resample = if same_crs { query.resampling } else { query.reproject };
            let raw_values = ((want_array && !visual_array) || (tiff && pq.algorithm.is_none()))
                && !matches!(bands, BandRequest::Terrain { .. });
            if let Some(method) = query.selection.or((exact || raw_values).then_some(Selection::First)) {
                let count = TileEngine::value_bands(&bands);
                if n == 0 {
                    let paint = if raw_values {
                        None
                    } else {
                        Some(Paint::of(&bands).ok_or_else(|| {
                            EngineError::BadRequest(
                                "pixel_selection other than 'first' cannot be used with a terrain encoding".into(),
                            )
                        })?)
                    };
                    let integer = crate::http::reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                    selecting = Some((
                        fold_within_limits(method, scenes.len(), count, dim * dim, integer, false)?,
                        paint,
                        count,
                    ));
                }
                let values = render_crs_bbox(
                    engine,
                    &cogs,
                    bbox,
                    &dst_crs,
                    grown,
                    grown,
                    None,
                    &bands,
                    resample,
                    OutputFormat::RawF32,
                    None,
                    custom_lut.as_ref(),
                    None,
                    true,
                    pq.unscale,
                )
                .await?;
                if let Some((fold, _, _)) = &mut selecting
                    && values.len() == count * dim * dim * 4
                {
                    fold.add_le_bytes(&values);
                    if fold.is_done() {
                        break;
                    }
                }
                continue;
            }
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let rgba = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                &dst_crs,
                grown,
                grown,
                None,
                &bands,
                resample,
                OutputFormat::RawRgba,
                None,
                custom_lut.as_ref(),
                None,
                true,
                pq.unscale,
            )
            .await?;
            if rgba.len() != canvas.len() {
                return Err(EngineError::BadRequest("unexpected image size from the renderer".into()));
            }
            let mut empty = 0usize;
            for (dst, src) in canvas.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
                if dst[3] == 0 {
                    if src[3] != 0 {
                        dst.copy_from_slice(src);
                    } else {
                        empty += 1;
                    }
                }
            }
            if empty == 0 {
                break;
            }
        }
        let raw_output = (want_array && !visual_array) || (tiff && pq.algorithm.is_none());
        if let Some((fold, paint, count)) = selecting {
            if let Some(selected) = fold.finish() {
                if raw_output {
                    let npy = npy_array(
                        &selected.values,
                        selected.bands,
                        dim,
                        dim,
                        SampleType::Float32,
                        pq.return_mask,
                        None,
                    );
                    let made = (npy, "application/x-binary");
                    return Ok(as_geotiff(made, tiff, bbox, tms_def.epsg, pq.return_mask));
                }
                if let Some(paint) = paint {
                    canvas = paint.rgba(&selected, dim * dim);
                }
            } else if raw_output {
                let empty = vec![f32::NAN; count.saturating_mul(dim).saturating_mul(dim)];
                let made = (
                    npy_array(&empty, count, dim, dim, SampleType::Float32, pq.return_mask, None),
                    "application/x-binary",
                );
                return Ok(as_geotiff(made, tiff, bbox, tms_def.epsg, pq.return_mask));
            }
        }
        match (&map_after, formula) {
            (Some(map_after), _) => map_after.colour(&mut canvas),
            (None, Some(formula)) => {
                let grey = one_grey && canvas.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
                formula.apply_buffer_in_place(&mut canvas, 4);
                if grey {
                    canvas.chunks_exact_mut(4).for_each(|p| (p[1], p[2]) = (p[0], p[0]));
                }
            }
            (None, None) => {}
        }
        let grey = one_grey
            && canvas
                .chunks_exact(4)
                .all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
        if want_array {
            return Ok((rgba_to_npy_u8(&canvas, dim, dim, grey, pq.return_mask), "application/x-binary"));
        }
        let asked = format.map(|(f, ct)| (f, if explicit.as_deref() == Some("jpg") { "image/jpg" } else { ct }));
        let made = encode_picture(
            canvas,
            dim,
            dim,
            None,
            asked,
            pq.jpeg_quality,
            pq.return_mask,
            grey,
        )?;
        if made.1 == crate::http::GEOTIFF_CONTENT_TYPE && made.0.len() == dim * dim * 4 {
            let body = crate::geotiff::encode_geotiff_rgba(
                grown,
                grown,
                &made.0,
                bbox,
                tms_def.epsg,
                pq.return_mask,
            );
            return Ok((body, made.1));
        }
        Ok(as_geotiff(made, tiff, bbox, tms_def.epsg, pq.return_mask))
    }
    .await;
    match rendered {
        Ok(made) => placed(crate::http::to_http_response(engine, Ok(made), ctx.if_none_match), bbox, tms_def.epsg),
        Err(e) => engine_error(&e),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn tile(
    ctx: &Ctx<'_>,
    pg: &PgStac,
    id: &str,
    tile: TileCoord,
    scale: Option<u32>,
    ext: &str,
    mut pq: ParsedQuery,
    pairs: &[(String, String)],
) -> Resp {
    // A tile with a `buffer` is the tile's box grown by that many pixels on every side,
    // at a size grown as much: to rio-tiler, and here, a box in Web Mercator like any other.
    // The renderer draws tiles of 256 and 512 pixels, and planes no algorithm: a tile of
    // another size under `algorithm`, or with band math only the renderer reads, is drawn
    // as its box too, with no buffer.
    let odd_for_renderer = !matches!(scale.map_or(pq.tile_size, |s| s * 256), 256 | 512)
        && (pq.algorithm.is_some() || check_mosaic_query(&pq, pairs).is_ok_and(|q| !planes_read_all(&pq, &q)));
    if pq.buffer > 0.0 || odd_for_renderer {
        let buffer_asked = pq.buffer;
        if scale.is_some_and(|s| !(1..=8).contains(&s)) {
            return detail(StatusCode::BAD_REQUEST, "a tile's scale is from @1x to @8x");
        }
        pq.buffer = 0.0;
        // A box has no bare array: said here, in a tile's words, sooner than further on in a box's.
        let raw = matches!(ext, "bin" | "raw" | "f32") || matches!(query_value(pairs, "format"), Some("bin" | "raw" | "f32"));
        if let Err(r) = check_mosaic_query(&pq, pairs) {
            return r;
        }
        if raw {
            return detail(
                StatusCode::BAD_REQUEST,
                "a tile with a `buffer`, or of a size other than 256 and 512 drawn by the renderer, is drawn as a picture, `.npy` or `.tif`",
            );
        }
        let buffer = buffer_asked;
        // As in rio-tiler, which grows the tile by whole pixels: halves, since both sides grow.
        if (buffer * 2.0).fract() != 0.0 {
            return detail(StatusCode::BAD_REQUEST, "`buffer` must be a multiple of `0.5`");
        }
        let side = scale.map_or(pq.tile_size, |s| s * 256);
        let grown = (side as f32 + 2.0 * buffer).round() as u32;
        let bbox = crate::http::web_mercator_buffered_3857_bbox(tile.z, tile.x, tile.y, buffer, side);
        pq.coord_crs_3857 = true;
        // What a box reads and a tile passes over: the system is the tile's, and so is the size.
        const OF_A_BOX: [&str; 6] = ["buffer", "coord_crs", "dst_crs", "width", "height", "max_size"];
        let pairs: Vec<(String, String)> = pairs.iter().filter(|(k, _)| !OF_A_BOX.contains(&k.as_str())).cloned().collect();
        let ext = (!ext.is_empty()).then_some(ext);
        let size = PartSize::Exact(grown, grown);
        let under = tile_polygon(tile);
        return Box::pin(part_image_of(ctx, pg, id, &under, Some(&under), bbox, None, size, ext, pq, &pairs)).await;
    }
    // (A buffered tile is a box, which does this for itself.)
    let index = normalized_index_as_band_math(&mut pq, Some(ext).filter(|e| !e.is_empty()));
    let query = match check_mosaic_query(&pq, pairs) {
        Ok(q) => q,
        Err(r) => return r,
    };
    // What the renderer cannot draw in one pass is made from planes instead. So is a tile
    // with nearest-neighbour resampling, whose planes are read the way GDAL's warp reads
    // and are TiTiler's pixels; the renderer draws the other methods, the algorithms, and
    // band math only its own engine reads.
    // A colour map given as JSON is applied to planes; the renderer on this road has none.
    let custom = match pq.custom_colormap_json.as_deref().map(MosaicColormap::parse).transpose() {
        Ok(map) => map.flatten(),
        Err(e) => return detail(StatusCode::BAD_REQUEST, e),
    };
    // So is a `nodata` of the request's own: the renderer on this road reads the raster's.
    let nodata = pq.nodata;
    let planes_wanted = wants_planes(&pq, &query) || custom.is_some() || nodata.is_some();
    let planes_can = planes_read_all(&pq, &query) && pq.algorithm.is_none();
    let planes_preferred =
        planes_can && query.resampling == ResampleMode::Nearest && query.reproject == ResampleMode::Nearest;
    // The request's own band math, before an asset's is put in its place for the renderer.
    let expression = pq.expr_raw.clone();
    let names = if planes_wanted { expression_names(&pq, &query) } else { renderer_assets(&mut pq, &query) };
    let names = match names {
        Ok(names) => names,
        Err(r) => return r,
    };
    // `@{n}x` is a tile of n times 256 pixels, within the sizes `tilesize` takes.
    if scale.is_some_and(|s| !(1..=8).contains(&s)) {
        return detail(StatusCode::BAD_REQUEST, "a tile's scale is from @1x to @8x");
    }
    // A `nodata` of the request's own leaves the listing to pgSTAC's search: the lean
    // lookup passes over an item whose footprint adds nothing, and its raster may not.
    let grid = if pq.nodata.is_some() { None } else { tile_grid(scale, &pq) };
    let items = match pixel_items(ctx, pg, &tile_polygon(tile), id, query.limits, &names, grid, query.lookup).await {
        Ok(items) => items,
        Err(e) => return db_error(&e),
    };
    if items.is_empty() {
        return no_content();
    }

    let scenes: Result<Vec<Vec<String>>, Resp> =
        items.iter().take(MAX_SCENES_PER_TILE).map(|item| asset_hrefs(item, names.iter().map(String::as_str))).collect();
    let scenes = match scenes {
        Ok(scenes) => scenes,
        Err(resp) => return resp,
    };

    // An explicit extension, then `format=`, then TiTiler's rule: JPEG when every pixel is
    // valid and PNG when some are not.
    let explicit = if ext.is_empty() {
        pq.format_param.clone().unwrap_or_default()
    } else {
        ext.to_string()
    };
    // A GeoTIFF is the array with a place on the ground: it is made as the array is, and
    // wrapped at the end.
    let tiff = matches!(explicit.as_str(), "tif" | "tiff");
    let explicit = if tiff { "npy".to_string() } else { explicit };
    let auto_format = explicit.is_empty();
    let (resampling, reproject) = (query.resampling, query.reproject);
    let asset_as_band = query.asset_as_band;
    let selection = query.selection;
    let return_mask = pq.return_mask;
    let jpeg_quality = pq.jpeg_quality;
    let engine = ctx.engine;
    let specs = query.assets;
    // The tile is of these rasters: a search that comes to list others, or an item whose
    // asset has moved, is another tile, not this one served again.
    let cache_key = format!("{}?{}#{:x}", ctx.path, ctx.raw_query, source_identity(&scenes));
    let rendered = engine
        .tile_cache
        .get_or_render(cache_key, || async {
            // A tile asked for as a GeoTIFF is one, placed where the tile lies.
            let (west, south, east, north) = tile_bounds_3857(tile);
            let lies = [west, south, east, north];
            let made: Result<(Bytes, &'static str), EngineError> = async {
            // The first scene is opened here, to learn what the tile's bands are. The scenes
            // after it are put on their way meanwhile, so the mosaic does not start their
            // headers only once this one has come back. The list holds only scenes whose
            // footprints add to the tile, so these are opens the tile is going to need.
            for uri in scenes.iter().skip(1).take(MOSAIC_SCENES_AHEAD - 1).flatten() {
                let (store, uri) = (engine.store.clone(), uri.clone());
                tokio::spawn(async move {
                    let _ = store.open_uri(&uri).await;
                });
            }
            let first = engine.store.open_uri(&scenes[0][0]).await?;
            let full = first.index.full_res();
            let formula = pq
                .color_formula
                .as_deref()
                .map(cogzilla_kernel::ColorFormula::parse)
                .transpose()
                .map_err(EngineError::BadRequest)?;
            // Assets that do not lie on one grid cannot be read as the bands of one image.
            let mut one_grid = true;
            if !planes_wanted && scenes[0].len() > 1 {
                let mut cogs = vec![first.clone()];
                for uri in &scenes[0][1..] {
                    cogs.push(engine.store.open_uri(uri).await?);
                }
                one_grid = TileEngine::on_one_grid(&cogs);
            }
            // Left to the renderer: a raster with an alpha band, which it reads as the mask,
            // and raw arrays.
            let dim = scale.map_or(pq.tile_size, |s| s.saturating_mul(256));
            let raw = matches!(explicit.as_str(), "bin" | "raw" | "f32");
            // A NumPy array is written from planes, in the data's own type and with its
            // mask, whatever the number of bands.
            let array = explicit == "npy";
            // And a raster of two bands, or four and more: the renderer draws the first, or
            // the first three, where planes would stack them all and TiTiler refuse most.
            let preferred = planes_preferred
                && !raw
                && full.extra_samples.is_empty()
                && (array || scenes[0].len() > 1 || matches!(full.samples_per_pixel, 1 | 3));
            // A plain request that takes this road only because it is the better reader keeps
            // the bands of its first raster for every item, as the renderer does: a later
            // item with more bands gives the same ones, not a tile of another shape.
            let mut specs = specs.clone();
            if !planes_wanted && one_grid && preferred && specs.len() == 1 && scenes[0].len() == 1 {
                let spec = &mut specs[0];
                if spec.indexes.is_none() && spec.expression.is_none() {
                    spec.indexes = Some((1..=full.samples_per_pixel).collect());
                }
            }
            // The renderer draws tiles of 256 and 512 pixels: one of another size is read
            // as planes whatever its resampling, where planes can make it.
            // So is one of a method that gathers a pixel's footprint, at a zoom where the
            // renderer would gather too little of it.
            let first_crs = engine.get_or_build_crs(&first)?;
            let first_resample = if matches!(first_crs, CrsModel::Epsg3857) { resampling } else { reproject };
            let too_little = engine_gathers_too_little(&first_crs, &first, first_resample, tile.z);
            let odd_size = (!matches!(dim, 256 | 512) || too_little) && planes_can && !raw && full.extra_samples.is_empty();
            if planes_wanted || !one_grid || preferred || odd_size {
                if pq.algorithm.is_some() {
                    return Err(EngineError::BadRequest(
                        "`algorithm` is not supported with these assets and options yet".into(),
                    ));
                }
                let asked = if auto_format {
                    None
                } else {
                    let (format, content_type) = parse_ext_format(&explicit, pq.webp_quality, jpeg_quality)?;
                    if matches!(format, OutputFormat::RawF32) {
                        return Err(EngineError::BadRequest(
                            "raw array formats are not supported with these assets and options yet".into(),
                        ));
                    }
                    Some((format, if explicit == "jpg" { "image/jpg" } else { content_type }))
                };
                let job = PlanesTile {
                    assets: specs.clone(),
                    as_band: asset_as_band,
                    expression: expression.clone(),
                    selection: selection.unwrap_or(Selection::First),
                    resampling,
                    reproject,
                    nodata,
                    array,
                    window: Window::Tile(crate::engine::TileRequest {
                        tile,
                        tile_size: dim,
                        bands: BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Grayscale },
                        resample: reproject,
                        format: OutputFormat::RawF32,
                        unscale: pq.unscale,
                    }),
                    pool: scenes.len() * specs.len() > 1,
                    index,
                };
                let look = Look {
                    rescale: &pq.rescale,
                    colormap: pq.colormap,
                    mapped: pq.colormap_explicit,
                    custom: custom.as_ref(),
                    formula: formula.as_ref(),
                    asked,
                    jpeg_quality,
                    return_mask,
                };
                let made = job.picture(engine, &scenes, dim as usize, dim as usize, &look).await?;
                return Ok(made);
            }
            // On planes the bands are counted after an asset's own options; here they are the raster's.
            if asset_as_band && full.samples_per_pixel > 1 {
                return Err(EngineError::BadRequest("Can't use `asset_as_band` for multibands asset".into()));
            }
            let default_bands = if scenes[0].len() > 1 {
                scenes[0].len() as u16
            } else {
                full.samples_per_pixel
            };
            // The renderer's bands are one or three: ranges that are not one each are the
            // first for all, as TiTiler has it (the shared routes refuse two for one band).
            let pq = if pq.rescale.len() > 1 && pq.rescale.len() != 3 {
                let mut first = pq.clone();
                first.rescale.truncate(1);
                std::borrow::Cow::Owned(first)
            } else {
                std::borrow::Cow::Borrowed(&pq)
            };
            let pq: &ParsedQuery = &pq;
            let (mut treq, content_type) = build_tile_request(
                Some(engine),
                tile,
                scale,
                Some(explicit.as_str()),
                pq,
                default_bands,
                full.sample_type,
                false,
                Some(&first),
            )?;
            // As in TiTiler, `resampling` applies to a source already in the tile's projection
            // and `reproject` to one that has to be warped into it.
            treq.resample = match engine.get_or_build_crs(&first)? {
                CrsModel::Epsg3857 => resampling,
                _ => reproject,
            };
            // What planes could not make of such a tile is refused, not drawn wrong.
            if engine_gathers_too_little(&first_crs, &first, treq.resample, treq.tile.z) {
                return Err(EngineError::BadRequest(
                    "this resampling method is drawn at a zoom this far below the raster's coarsest overview without `algorithm`, and with band math planes can read; use nearest, bilinear or cubic, or a `bbox`".into(),
                ));
            }
            // The renderer draws tiles of 256 and 512 pixels; any other size is read as
            // planes, which is where a request with nearest-neighbour pixels has gone.
            if treq.tile_size != 256 && treq.tile_size != 512 {
                return Err(EngineError::BadRequest(format!(
                    "tilesize={} is drawn without `algorithm`, as a picture or `.npy`, and with band math planes can read; use 256 or 512 otherwise",
                    treq.tile_size
                )));
            }
            let dim = treq.tile_size as usize;
            let raw_array = matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy);
            // TiTiler names the media type after the extension that was asked for.
            let content_type = if explicit == "jpg" { "image/jpg" } else { content_type };
            let asked = treq.format;
            // One band under a colour map, with a colour formula: drawn grey here, then
            // the formula, then the map.
            let map_after = match &formula {
                Some(formula) if pq.colormap_explicit && !raw_array => MapAfterFormula::take(formula, &mut treq.bands),
                _ => None,
            };
            // An array of a raster's own bands, read with another resampling than nearest: the
            // renderer's values, written in the raster's type with the mask as a band, as
            // an array from planes is. What makes a picture's bytes of them, band math and
            // `algorithm` are still the renderer's own float32.
            let own_bands = matches!(treq.bands, BandRequest::Single { .. } | BandRequest::Rgb { .. });
            // One band of band math is the renderer's too, in the type the math gives.
            let band_math = pq.expr_raw.as_ref().filter(|_| {
                matches!(
                    treq.bands,
                    BandRequest::Spectral1 { .. } | BandRequest::Spectral2 { .. } | BandRequest::Spectral3 { .. } | BandRequest::CustomExpr { .. }
                )
            });
            if matches!(treq.format, OutputFormat::Npy)
                && (own_bands && pq.expr_raw.is_none() || band_math.is_some())
                && selection.is_none()
                && !pq.colormap_explicit
                && formula.is_none()
                && pq.algorithm.is_none()
                && full.extra_samples.is_empty()
                && !crate::http::has_embedded_palette(pq, &first, &treq.bands)
            {
                let bands = TileEngine::value_bands(&treq.bands);
                treq.format = OutputFormat::RawF32;
                let raw = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                if raw.len() == bands * dim * dim * 4 {
                    // Several assets read as bands: the type that holds them all.
                    let mut kind = full.sample_type;
                    for uri in scenes[0].iter().skip(1) {
                        kind = wider_type(kind, engine.store.open_uri(uri).await?.index.full_res().sample_type);
                    }
                    let kind = if pq.unscale { SampleType::Float32 } else { kind };
                    // Where the math makes doubles the values are still the renderer's 32-bit
                    // ones: written as float32, half the bytes of TiTiler's float64 and no less told.
                    let kind = band_math.map_or(kind, |math| match band_math_type(std::iter::once(math), kind) {
                        SampleType::Float64 => SampleType::Float32,
                        other => other,
                    });
                    let whole = band_math.is_none() && !matches!(kind, SampleType::Float32 | SampleType::Float64);
                    // A whole-number raster's values between pixels are rounded to its type.
                    let read = raw
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                        .map(|v| if whole { v.round() } else { v });
                    // With `rescale` the array is of the bytes a picture would show, band by
                    // band, in the ranges the picture's bands were given.
                    if !pq.rescale.is_empty() {
                        let range = |band: usize| match &treq.bands {
                            BandRequest::Rgb { rescale_min, rescale_max, .. } => (rescale_min[band.min(2)], rescale_max[band.min(2)]),
                            BandRequest::Single { rescale, .. } => *rescale,
                            _ => pq.rescale[0],
                        };
                        let values: Vec<f32> = read
                            .enumerate()
                            .map(|(i, v)| {
                                let (min, max) = range(i / (dim * dim));
                                if v.is_nan() { v } else { f32::from(rescale_byte(v, min, max)) }
                            })
                            .collect();
                        return Ok((npy_array(&values, bands, dim, dim, SampleType::Uint8, return_mask, None), content_type));
                    }
                    let values: Vec<f32> = read.collect();
                    return Ok((npy_array(&values, bands, dim, dim, kind, return_mask, None), content_type));
                }
                treq.format = OutputFormat::Npy;
            }
            // A formula on one band works on the band's byte, made here from the values.
            let exact = formula.is_some()
                && !raw_array
                && one_grey_band(&treq.bands)
                && !crate::http::has_embedded_palette(pq, &first, &treq.bands);
            let mut rgba = if let (None, true) = (selection, exact) {
                let paint = Paint::of(&treq.bands).ok_or_else(|| EngineError::BadRequest("not a band of values".into()))?;
                treq.format = OutputFormat::RawF32;
                let raw = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                let values = f32s(&raw);
                if values.len() == dim * dim {
                    paint.rgba(&Selected { bands: 1, values }, dim * dim)
                } else {
                    vec![0u8; dim * dim * 4]
                }
            } else if let Some(method) = selection {
                // Every scene's values, then rio-tiler's arithmetic over them, then colour.
                let paint = Paint::of(&treq.bands).ok_or_else(|| {
                    EngineError::BadRequest("pixel_selection other than 'first' cannot be used with a terrain encoding".into())
                })?;
                // An array of what the method leaves, when it is of the raster's own bands
                // and nothing makes a picture's bytes of them.
                let array = matches!(treq.format, OutputFormat::Npy)
                    && own_bands
                    && !pq.colormap_explicit
                    && formula.is_none()
                    && pq.expr_raw.is_none()
                    && pq.algorithm.is_none()
                    && full.extra_samples.is_empty()
                    && !crate::http::has_embedded_palette(pq, &first, &treq.bands);
                if raw_array && !array {
                    return Err(EngineError::BadRequest(
                        "raw array formats are not supported with pixel_selection other than 'first' yet".into(),
                    ));
                }
                // `mean` and `median` are cast back to the source's type, which only loses
                // anything when the values are the raster's own whole numbers.
                let integer = crate::http::reads_whole_numbers(pq.unscale, &treq.bands, full.sample_type);
                let bands = TileEngine::value_bands(&treq.bands);
                // The median has to hold every scene, so it is refused before anything is
                // read when that would be too much.
                let mut fold = fold_within_limits::<f32>(method, scenes.len(), bands, dim * dim, integer, true)?;
                // For an array each scene's values are brought to whole numbers first, where
                // the raster's are: rio-tiler's scenes are in the raster's type before the
                // method sees them, and a mean cut off after is otherwise half a level low.
                if array && integer {
                    let pixels = bands * dim * dim * 4;
                    engine
                        .mosaic_scene_values(&scenes, &treq, |scene| {
                            if scene.len() == pixels {
                                fold.add(scene.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]).round()));
                            }
                        })
                        .await?;
                } else {
                    engine.mosaic_scene_values(&scenes, &treq, |scene| fold.add_le_bytes(scene)).await?;
                }
                let selected = fold.finish();
                if array {
                    // A count is of bytes and a deviation of doubles; the rest are in the
                    // type that holds every asset read, rounded where that is of whole numbers.
                    let mut kind = full.sample_type;
                    for uri in scenes[0].iter().skip(1) {
                        kind = wider_type(kind, engine.store.open_uri(uri).await?.index.full_res().sample_type);
                    }
                    let kind = match method {
                        Selection::Count => SampleType::Uint8,
                        Selection::Stdev => SampleType::Float64,
                        _ if pq.unscale => SampleType::Float32,
                        _ => kind,
                    };
                    let whole = !matches!(kind, SampleType::Float32 | SampleType::Float64);
                    let values: Vec<f32> = match selected {
                        Some(selected) => selected.values.iter().map(|v| if whole { v.round() } else { *v }).collect(),
                        // No scene reached the tile: nothing, in the method's own shape.
                        None => vec![f32::NAN; if method == Selection::Count { 1 } else { bands } * dim * dim],
                    };
                    let shown = values.len() / (dim * dim);
                    // With `rescale` the array is of the bytes a picture would show, in the
                    // ranges the picture's bands were given.
                    if !pq.rescale.is_empty() {
                        let range = |band: usize| match &treq.bands {
                            BandRequest::Rgb { rescale_min, rescale_max, .. } => (rescale_min[band.min(2)], rescale_max[band.min(2)]),
                            BandRequest::Single { rescale, .. } => *rescale,
                            _ => pq.rescale[0],
                        };
                        let bytes: Vec<f32> = values
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                let (min, max) = range(i / (dim * dim));
                                if v.is_nan() { *v } else { f32::from(rescale_byte(*v, min, max)) }
                            })
                            .collect();
                        return Ok((npy_array(&bytes, shown, dim, dim, SampleType::Uint8, return_mask, None), content_type));
                    }
                    return Ok((npy_array(&values, shown, dim, dim, kind, return_mask, None), content_type));
                }
                match selected {
                    Some(selected) => paint.rgba(&selected, dim * dim),
                    None => vec![0u8; dim * dim * 4],
                }
            } else {
                // Nothing to do to the pixels after the mosaic: the renderer encodes them itself.
                if !auto_format && (raw_array || (formula.is_none() && return_mask)) {
                    let made = engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?;
                    // The renderer's bare pixels or values, written as the GeoTIFF asked for:
                    // floats when the tile is of numbers, not of a picture.
                    let made = crate::http::wrap_geotiff_fmt(made, content_type, treq.format, dim as u32, dim as u32, lies, 3857, return_mask);
                    return Ok((made, content_type));
                }
                treq.format = OutputFormat::RawRgba;
                engine.render_mosaic_scenes_ahead(&scenes, &treq, MOSAIC_SCENES_AHEAD).await?.to_vec()
            };
            let asked = (!auto_format).then_some((asked, content_type));
            // A count is one band whatever was counted.
            // So is one band drawn in grey, whichever scene it came from: a raster with a
            // palette of its own is one band too, and is not grey.
            let one_grey = matches!(Paint::of(&treq.bands), Some(Paint::Gray { colormap: ColormapName::Grayscale, .. }))
                && !crate::http::has_embedded_palette(pq, &first, &treq.bands)
                && rgba.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
            let grey = (one_grey || selection.is_some_and(|method| method == Selection::Count || TileEngine::value_bands(&treq.bands) == 1))
                && !pq.colormap_explicit;
            let formula = match &map_after {
                Some(map_after) => {
                    map_after.colour(&mut rgba);
                    None
                }
                None => formula.as_ref(),
            };
            encode_picture(rgba, dim, dim, formula, asked, jpeg_quality, return_mask, grey)
            }
            .await;
            made.map(|made| as_geotiff(made, tiff, lies, 3857, return_mask))
        })
        .await;
    match rendered {
        Ok(made) => {
            let (west, south, east, north) = tile_bounds_3857(tile);
            placed(to_http_response_cached(engine, Ok(made), ctx.if_none_match), [west, south, east, north], 3857)
        }
        Err(e) => engine_error(&e),
    }
}

