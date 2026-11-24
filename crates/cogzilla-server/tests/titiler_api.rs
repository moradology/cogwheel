use std::path::PathBuf;
use std::sync::Arc;

use cogzilla_server::{TileEngine, serve_listener};
use tokio::net::TcpListener;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures")
        .join(name)
}

#[tokio::test]
async fn full_titiler_cog_api_surface_works_end_to_end() {
    let sat_path = fixture("satellite_utm10n_rgb_zstd_p3.tif");
    if !sat_path.exists() {
        eprintln!("fixtures not generated; skipping HTTP integration test");
        return;
    }
    let sat_url = sat_path.canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b4 = fixture("landsat_c2_B4.tif").canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b3 = fixture("landsat_c2_B3.tif").canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b2 = fixture("landsat_c2_B2.tif").canonicalize().unwrap().to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(128 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });

    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    async fn get_json(client: &reqwest::Client, url: String) -> serde_json::Value {
        let resp = client.get(url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap()
    }

    // 0. GET / serves the landing page with the crate version filled in
    let landing = client.get(format!("{base}/")).send().await.unwrap();
    assert_eq!(landing.status().as_u16(), 200);
    assert!(landing.headers()["content-type"].to_str().unwrap().starts_with("text/html"));
    let landing = landing.text().await.unwrap();
    assert!(landing.contains(concat!("v", env!("CARGO_PKG_VERSION"))));
    assert!(!landing.contains("{{VERSION}}"));

    // 0b. Options this server does not implement are rejected, not ignored.
    let tile = format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.png?url={sat_url}");
    for (what, url) in [
        ("unknown algorithm", format!("{tile}&algorithm=contours")),
        ("unsupported resampling", format!("{tile}&resampling=average")),
        ("padding", format!("{tile}&padding=4")),
        ("unsupported reproject", format!("{tile}&reproject=average")),
        ("bad rescale", format!("{tile}&rescale=abc")),
        ("unknown colormap", format!("{tile}&bidx=1&colormap_name=nope")),
        ("bad tilesize", format!("{tile}&tilesize=7")),
        ("unknown tms tile", format!("{base}/cog/tiles/UTM31WGS84Quad/10/163/395.png?url={sat_url}")),
        ("unknown tms tilejson", format!("{base}/cog/UTM31WGS84Quad/tilejson.json?url={sat_url}")),
        ("statistics algorithm", format!("{base}/cog/statistics?url={sat_url}&algorithm=hillshade")),
        ("non-stats percentiles", format!("{base}/cog/info?url={sat_url}&p=5")),
        ("point algorithm", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&algorithm=hillshade")),
        ("point unsupported coord_crs", format!("{base}/cog/point/1,1?url={sat_url}&coord_crs=epsg:4267")),
        ("preview dst_crs", format!("{base}/cog/preview.png?url={sat_url}&dst_crs=epsg:4326")),
        ("preview explicit dims width", format!("{base}/cog/preview/256x256.png?url={sat_url}&width=300")),
        ("bbox explicit dims max_size", format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.8/256x256.png?url={sat_url}&max_size=2000")),
        ("info.geojson crs", format!("{base}/cog/info.geojson?url={sat_url}&crs=epsg:32610")),
        ("non-colormap orientation", format!("{base}/cog/info?url={sat_url}&orientation=vertical")),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 422, "{what} should be rejected with 422: {url}");
        let body: serde_json::Value =
            serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
        assert!(body["detail"][0]["msg"].is_string(), "422 response should match FastAPI JSON schema");
    }
    // Supported spellings and parameters this server does not know are still accepted.
    for (what, url) in [
        ("tilesize alias", format!("{tile}&tilesize=512&rescale=0,1")),
        ("cubic_spline", format!("{tile}&resampling=cubic_spline&rescale=0,1")),
        ("reproject=bilinear", format!("{tile}&reproject=bilinear&rescale=0,1")),
        ("padding=0", format!("{tile}&padding=0&rescale=0,1")),
        ("unrelated parameter", format!("{tile}&rescale=0,1&_cachebust=1&access_token=x")),
        ("hillshade", format!("{tile}&bidx=1&algorithm=hillshade")),
        ("WorldCRS84Quad", format!("{base}/cog/WorldCRS84Quad/tilejson.json?url={sat_url}")),
        ("point coord_crs 4326", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&coord_crs=epsg:4326")),
        ("point coord_crs 32610 UTM", format!("{base}/cog/point/535280,4172640?url={sat_url}&coord_crs=epsg:32610")),
        ("preview width query", format!("{base}/cog/preview.png?url={sat_url}&width=300")),
        ("colormap vertical orientation", format!("{base}/colorMaps/viridis?f=png&orientation=vertical")),
        ("algorithms list", format!("{base}/algorithms")),
        ("algorithm hillshade", format!("{base}/algorithms/hillshade")),
        ("cog validate", format!("{base}/cog/validate?url={sat_url}")),
        ("cog wmts", format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}")),
        ("statistics custom percentiles & bins & bidx", format!("{base}/cog/statistics?url={sat_url}&bidx=1&p=5&p=95&histogram_bins=20&categorical=true")),
        ("point expression", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2")),
        ("point bidx", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&bidx=2")),
        ("mosaic tile buffer+nodata", format!("{base}/mosaic/tiles/WebMercatorQuad/10/163/395.png?url={sat_url}&rescale=0,1&buffer=4&nodata=0")),
        ("algorithm grayscale", format!("{tile}&algorithm=grayscale")),
        ("algorithm normalizedIndex", format!("{tile}&algorithm=normalizedIndex&bidx=1,2")),
        ("algorithm median", format!("{tile}&algorithm=median")),
        ("algorithm bitonal", format!("{tile}&algorithm=bitonal&algorithm_params=%7B%22threshold%22:0.5%7D")),
        ("statistics expression", format!("{base}/cog/statistics?url={sat_url}&expression=b1-b2;b2-b3")),
        ("point multi-expression", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3")),
        ("cog tile geotiff", format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1")),
        ("cog preview geotiff", format!("{base}/cog/preview/64x64.tif?url={sat_url}&rescale=0,1")),
        ("cog tile 3-band rgb expr", format!("{tile}&expression=b3*1.0;b2*1.0;b1*1.0&rescale=0,1")),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{what} should be accepted: {url}");
    }
    let st_custom = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&bidx=1&p=5&p=95&histogram_bins=20&categorical=true"),
    )
    .await;
    assert_eq!(st_custom.as_object().unwrap().len(), 1);
    assert!(st_custom["b1"]["percentile_5"].is_number());
    assert!(st_custom["b1"]["percentile_95"].is_number());
    assert_eq!(st_custom["b1"]["histogram"][0].as_array().unwrap().len(), 20);
    assert!(st_custom["b1"]["categories"].is_object());

    let pt_expr = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2"),
    )
    .await;
    assert_eq!(pt_expr["values"].as_array().unwrap().len(), 1);
    assert_eq!(pt_expr["band_names"][0], "b1-b2");

    let pt_b2 = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&bidx=2"),
    )
    .await;
    assert_eq!(pt_b2["values"].as_array().unwrap().len(), 1);
    assert_eq!(pt_b2["band_names"][0], "b2");

    // Multi-expression /cog/statistics and /cog/point + GeoTIFF .tif output validation
    let st_expr = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&expression=b1-b2;b2-b3"),
    )
    .await;
    assert!(st_expr["b1-b2"]["mean"].is_number());
    assert!(st_expr["b2-b3"]["mean"].is_number());

    let pt_multi = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3"),
    )
    .await;
    assert_eq!(pt_multi["values"].as_array().unwrap().len(), 2);
    assert_eq!(pt_multi["band_names"], serde_json::json!(["b1-b2", "b2-b3"]));

    let tif_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap();
    assert_eq!(tif_resp.headers().get("content-type").unwrap(), "image/tiff; application=geotiff");
    let tif_bytes = tif_resp.bytes().await.unwrap();
    assert_eq!(&tif_bytes[0..4], b"II\x2a\x00", "valid little-endian Classic TIFF magic");
    let parsed_tif = cogzilla_tiff::parse_cog(&tif_bytes).unwrap();
    assert_eq!(parsed_tif.full_res().width, 256);
    assert_eq!(parsed_tif.full_res().height, 256);
    assert_eq!(parsed_tif.geo_keys.projected_cs_type, Some(3857));

    // Out-of-bounds /cog/tiles returns 404 Not Found (matching TiTiler TileOutsideBounds)
    let oob_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/0/0.png?url={sat_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(oob_resp.status().as_u16(), 404);
    let prev_300 = client
        .get(format!("{base}/cog/preview.png?url={sat_url}&width=300"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(u32::from_be_bytes(prev_300[16..20].try_into().unwrap()), 300, "width=300 yields 300 px PNG");
    let vcm = client
        .get(format!("{base}/colorMaps/viridis?f=png&orientation=vertical"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(
        (
            u32::from_be_bytes(vcm[16..20].try_into().unwrap()),
            u32::from_be_bytes(vcm[20..24].try_into().unwrap()),
        ),
        (20, 256),
        "orientation=vertical yields 20x256 PNG"
    );
    let big = client.get(format!("{tile}&tilesize=512&rescale=0,1")).send().await.unwrap().bytes().await.unwrap();
    assert_eq!(u32::from_be_bytes(big[16..20].try_into().unwrap()), 512, "tilesize=512 yields a 512 px PNG");

    // 1. GET /cog/info
    let info = get_json(&client, format!("{base}/cog/info?url={sat_url}")).await;
    assert_eq!(info["dtype"], "float32");
    assert_eq!(info["count"], 3);
    assert_eq!(info["width"], 2048);
    let bounds = info["wgs84_bounds"].as_array().unwrap();
    assert!(bounds[0].as_f64().unwrap() < -122.5 && bounds[2].as_f64().unwrap() > -122.5);

    // 2. GET /cog/info.geojson
    let gj = get_json(&client, format!("{base}/cog/info.geojson?url={sat_url}")).await;
    assert_eq!(gj["type"], "Feature");
    assert_eq!(gj["geometry"]["type"], "Polygon");

    // 3. GET /cog/WebMercatorQuad/tilejson.json
    let tj = get_json(&client, format!("{base}/cog/WebMercatorQuad/tilejson.json?url={sat_url}")).await;
    assert_eq!(tj["tilejson"], "3.0.0");
    assert!(tj["tiles"][0].as_str().unwrap().contains("/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"));

    // 4. GET /cog/tiles & /cog/tiles/WebMercatorQuad
    assert_eq!(
        client.get(format!("{base}/cog/tiles")).send().await.unwrap().status().as_u16(),
        200
    );
    assert_eq!(
        client.get(format!("{base}/cog/tiles/WebMercatorQuad")).send().await.unwrap().status().as_u16(),
        200
    );

    // 5. GET /cog/WebMercatorQuad/map.html
    let html = client
        .get(format!("{base}/cog/WebMercatorQuad/map.html?url={sat_url}"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("maplibregl"));

    // 6. GET /cog/tiles/WebMercatorQuad/{z}/{x}/{y}@1x.png & @2x.png & arbitrary ?expression=
    let png_1x = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583@1x.png?url={sat_url}&bidx=1&bidx=2&bidx=3&rescale=0,0.35"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&png_1x[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    let png_2x = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583@2x.png?url={sat_url}&expression=(b1-b2)/(b1+b2)&rescale=-0.5,0.5&colormap_name=ndvi"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&png_2x[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 7. Multi-COG Landsat C2 uint16 3-file RGB via TiTiler route
    let lsat_png = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={lsat_b4}&url={lsat_b3}&url={lsat_b2}&rescale=7500,22500"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&lsat_png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 8. GET /cog/point/{lon},{lat}
    let pt = get_json(&client, format!("{base}/cog/point/-122.65,37.773?url={sat_url}")).await;
    assert_eq!(pt["band_names"].as_array().unwrap().len(), 3);
    assert!(pt["values"][0].as_f64().unwrap() > 0.0);

    // 9. GET /cog/statistics
    let stats = get_json(&client, format!("{base}/cog/statistics?url={sat_url}&max_size=512")).await;
    assert!(stats["b1"]["valid_pixels"].as_u64().unwrap() > 1000);
    assert!(stats["b1"]["mean"].as_f64().unwrap() > 0.0);

    // 10. GET /cog/preview/256x256.png
    let prev = client
        .get(format!("{base}/cog/preview/256x256.png?url={sat_url}&rescale=0,0.35"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&prev[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 11. GET /cog/bbox/{minx},{miny},{maxx},{maxy}/256x256.png
    let bbox_img = client
        .get(format!(
            "{base}/cog/bbox/-122.8,37.6,-122.5,37.9/256x256.png?url={sat_url}&rescale=0,0.35"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&bbox_img[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 12. POST /cog/feature/256x256.png
    let feat = serde_json::json!({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.8, 37.6],
                [-122.5, 37.6],
                [-122.5, 37.9],
                [-122.8, 37.9],
                [-122.8, 37.6]
            ]]
        },
        "properties": {}
    });
    let feat_img = client
        .post(format!("{base}/cog/feature/256x256.png?url={sat_url}&rescale=0,0.35"))
        .body(feat.to_string())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&feat_img[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // ------------------------------------------------------------------------
    // 13. /colorMaps and /colorMaps/{colorMapId}
    // ------------------------------------------------------------------------
    let cmaps: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/colorMaps"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    let cmap_list = cmaps["colormaps"].as_array().unwrap();
    assert!(cmap_list.iter().any(|v| v["id"] == "viridis"));
    assert!(cmap_list.iter().any(|v| v["id"] == "rdylgn"));

    let viridis_json: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/colorMaps/viridis"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(viridis_json.as_object().unwrap().len(), 256);
    assert_eq!(viridis_json["0"], serde_json::json!([68, 1, 84, 255]));

    let viridis_png = client
        .get(format!("{base}/colorMaps/viridis?format=png"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&viridis_png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // ------------------------------------------------------------------------
    // 14. /tileMatrixSets and /tileMatrixSets/{tileMatrixSetId}
    // ------------------------------------------------------------------------
    let tms_list: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/tileMatrixSets"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    let tms_arr = tms_list["tileMatrixSets"].as_array().unwrap();
    assert!(tms_arr.iter().any(|t| t["id"] == "WebMercatorQuad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "WorldCRS84Quad"));

    let tms_wm: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/tileMatrixSets/WebMercatorQuad"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(tms_wm["id"], "WebMercatorQuad");
    assert_eq!(tms_wm["tileMatrices"].as_array().unwrap().len(), 25);

    // ------------------------------------------------------------------------
    // 15. /stac/* (MultiBaseTilerFactory(reader=STACReader))
    // ------------------------------------------------------------------------
    let stac_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/landsat_c2_stac.json");
    if stac_path.exists() {
        let stac_url = stac_path.to_string_lossy();

        // /stac/assets
        let assets: serde_json::Value = serde_json::from_str(
            &client
                .get(format!("{base}/stac/assets?url={stac_url}"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        let arr = assets.as_array().unwrap();
        assert!(arr.iter().any(|v| v == "red"));
        assert!(arr.iter().any(|v| v == "green"));
        assert!(arr.iter().any(|v| v == "blue"));
        assert!(!arr.iter().any(|v| v == "thumbnail"));

        // /stac/info
        let stac_info: serde_json::Value = serde_json::from_str(
            &client
                .get(format!("{base}/stac/info?url={stac_url}&assets=red&assets=green"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_info.get("red").is_some());
        assert!(stac_info.get("green").is_some());

        // /stac/tiles/WebMercatorQuad/12/652/1583.png?assets=red&assets=green&assets=blue
        let stac_tile = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&assets=red&assets=green&assets=blue&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_tile[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // /stac/tiles/... with asset-named expression
        let stac_expr_tile = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&expression=(red-green)/(red+green)&rescale=-0.5,0.5&colormap_name=rdylgn"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_expr_tile[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // /stac/WebMercatorQuad/tilejson.json
        let stac_tj: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/WebMercatorQuad/tilejson.json?url={stac_url}&assets=red&assets=green&assets=blue"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stac_tj["tilejson"], "3.0.0");

        // /stac/point/{lon},{lat}
        let stac_pt: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/point/-122.65,37.77?url={stac_url}&assets=red&assets=green&assets=blue"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stac_pt["values"].as_array().unwrap().len(), 3);
        assert_eq!(stac_pt["band_names"].as_array().unwrap().len(), 3);

        // /stac/asset_statistics & /stac/statistics
        let stac_astats: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/asset_statistics?url={stac_url}&assets=red&assets=green&max_size=256"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_astats.get("red").is_some());
        assert!(stac_astats.get("green").is_some());

        let stac_stats: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/statistics?url={stac_url}&assets=red&assets=green&max_size=256"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_stats.get("b1").is_some());
        assert!(stac_stats.get("b2").is_some());

        // /stac/preview/256x256.png
        let stac_prev = client
            .get(format!(
                "{base}/stac/preview/256x256.png?url={stac_url}&assets=red&assets=green&assets=blue&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_prev[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    // ------------------------------------------------------------------------
    // 16. Dynamic early-exit COG mosaic (/mosaic/tiles/...)
    // ------------------------------------------------------------------------
    let merc_path = fixture("mercator_4k_zstd_p3.tif");
    if merc_path.exists() {
        let merc_url = merc_path.canonicalize().unwrap().to_string_lossy().to_string();
        // Proof 1: When the front scene covers 100% of the tile, background URLs are NEVER opened!
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            let hits = Arc::new(AtomicU64::new(0));
            let hits_srv = hits.clone();
            let bg_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let bg_addr = bg_listener.local_addr().unwrap();
            tokio::spawn(async move {
                loop {
                    let Ok((_stream, _)) = bg_listener.accept().await else {
                        break;
                    };
                    hits_srv.fetch_add(1, Ordering::Relaxed);
                }
            });
            let bg_url = format!("http://{bg_addr}/never_touched.tif");
            let mosaic_png = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&url={bg_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mosaic_png.status(), 200);
            let bytes = mosaic_png.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            assert_eq!(
                hits.load(Ordering::Relaxed),
                0,
                "background scene should never be opened when front scene covers 100% of pixels"
            );
        }

        // Proof 2: When the front scene has partial nodata holes (coastal_4k_nodata.tif),
        // the mosaic fills those holes from the background scene!
        let coastal_path = fixture("coastal_4k_nodata.tif");
        if coastal_path.exists() {
            let coastal_url = coastal_path.canonicalize().unwrap().to_string_lossy().to_string();
            let resp = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url={coastal_url}&url={merc_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // PNG IHDR color_type is at byte offset 25: 2 = RGB (100% opaque, all holes filled!)
            assert_eq!(
                bytes[25], 2,
                "expected opaque RGB PNG (color_type=2) after filling coastal nodata holes from background scene"
            );
        }

        // --------------------------------------------------------------------
        // 17. Indexed spatial catalog mosaic (/mosaic/{name}/tiles/...)
        // --------------------------------------------------------------------
        {
            use cogzilla_proj::{TileCoord, tile_bounds_3857};
            use cogzilla_server::{InMemoryCatalog, SceneEntry};
            let (tx0, ty0, tx1, ty1) = tile_bounds_3857(TileCoord {
                z: 12,
                x: 655,
                y: 1583,
            });
            let cat = InMemoryCatalog::new(vec![
                SceneEntry {
                    uri: merc_url.clone(),
                    bbox_3857: [tx0 - 500.0, ty0 - 500.0, tx1 + 500.0, ty1 + 500.0],
                    priority: 100,
                },
                SceneEntry {
                    uri: "/nonexistent/low_priority_scene.tif".into(),
                    bbox_3857: [tx0 - 500.0, ty0 - 500.0, tx1 + 500.0, ty1 + 500.0],
                    priority: 10,
                },
            ]);
            engine.register_catalog("demo", Arc::new(cat));

            // In-bounds: hits high-priority merc_url and early-exits before touching nonexistent scene!
            let resp = client
                .get(format!(
                    "{base}/mosaic/demo/tiles/WebMercatorQuad/12/655/1583.png?rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Out-of-bounds: spatial index returns 0 scenes -> instant transparent PNG!
            let empty_resp = client
                .get(format!(
                    "{base}/mosaic/demo/tiles/WebMercatorQuad/12/0/0.png"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(empty_resp.status(), 200);
            let empty_bytes = empty_resp.bytes().await.unwrap();
            assert_eq!(&empty_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Unknown catalog -> 404
            let missing_resp = client
                .get(format!(
                    "{base}/mosaic/nope/tiles/WebMercatorQuad/12/655/1583.png"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(missing_resp.status(), 404);
        }

        // --------------------------------------------------------------------
        // 18. HTTP Caching (ETag, Cache-Control, If-None-Match -> 304) & /metrics
        // --------------------------------------------------------------------
        {
            let tile_url = format!(
                "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&rescale=0,2500&colormap_name=viridis"
            );
            let r1 = client.get(&tile_url).send().await.unwrap();
            assert_eq!(r1.status(), 200);
            let etag = r1
                .headers()
                .get("etag")
                .expect("ETag header missing")
                .to_str()
                .unwrap()
                .to_string();
            let cc = r1
                .headers()
                .get("cache-control")
                .expect("Cache-Control header missing")
                .to_str()
                .unwrap();
            assert!(cc.contains("public"), "unexpected Cache-Control: {cc}");
            let _ = r1.bytes().await.unwrap();

            // Conditional GET with If-None-Match -> 304 Not Modified with empty body
            let r2 = client
                .get(&tile_url)
                .header("If-None-Match", &etag)
                .send()
                .await
                .unwrap();
            assert_eq!(r2.status(), 304);
            assert!(r2.bytes().await.unwrap().is_empty());

            // Prometheus /metrics exposition
            let m = client.get(format!("{base}/metrics")).send().await.unwrap();
            assert_eq!(m.status(), 200);
            let ct = m.headers().get("content-type").unwrap().to_str().unwrap();
            assert!(ct.starts_with("text/plain"));
            let body = m.text().await.unwrap();
            assert!(body.contains("cogzilla_tiles_served_total "));
            assert!(body.contains("cogzilla_http_304_not_modified_total 1"));
            assert!(body.contains("cogzilla_tile_cache_coalesced_total "));
        }

        // --------------------------------------------------------------------
        // 19. MosaicJSON (/mosaicjson/*) & Dynamic Registration (POST /mosaic/register)
        // --------------------------------------------------------------------
        {
            let qk = cogzilla_server::tile_to_quadkey(12, 655, 1583);
            let mj_doc = serde_json::json!({
                "mosaicjson": "0.0.3",
                "minzoom": 10,
                "maxzoom": 16,
                "quadkey_zoom": 12,
                "bounds": [-123.0, 37.0, -122.0, 38.0],
                "tiles": {
                    qk: [merc_url.clone(), "/nonexistent_bg.tif"]
                }
            });
            let mj_path = std::env::temp_dir().join(format!(
                "cogzilla_test_mosaic_{}.json",
                std::process::id()
            ));
            std::fs::write(&mj_path, mj_doc.to_string()).unwrap();
            let mj_url = mj_path.to_string_lossy().to_string();

            let mj_info = get_json(&client, format!("{base}/mosaicjson/info?url={mj_url}")).await;
            assert_eq!(mj_info["quadkey_zoom"], 12);
            assert_eq!(mj_info["tiles_count"], 1);

            let mj_tile = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_tile.status(), 200);
            let bytes = mj_tile.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Dynamic registration via POST /mosaic/register
            let reg_resp = client
                .post(format!("{base}/mosaic/register"))
                .header("content-type", "application/json")
                .body(
                    serde_json::json!({
                        "name": "live_mj",
                        "catalog": mj_doc,
                    })
                    .to_string(),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(reg_resp.status(), 200);
            let reg_tile = client
                .get(format!(
                    "{base}/mosaic/live_mj/tiles/WebMercatorQuad/12/655/1583.png?rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(reg_tile.status(), 200);
            let _ = std::fs::remove_file(&mj_path);
        }

        // --------------------------------------------------------------------
        // 20. Terrain Tiles: Terrarium & Mapbox Terrain-RGB (?algorithm=...)
        // --------------------------------------------------------------------
        for alg in ["terrarium", "terrainrgb"] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm={alg}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "algorithm={alg} failed");
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // Opaque RGB PNG (color_type=2) for clean DEM tile
            assert_eq!(bytes[25], 2, "expected RGB PNG for algorithm={alg}");
        }

        // --------------------------------------------------------------------
        // 21. On-the-Fly DEM Hillshade, Slope, Aspect (?algorithm=...)
        // --------------------------------------------------------------------
        for alg in ["hillshade", "slope", "aspect"] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm={alg}&azimuth=315&altitude=45&z_factor=2.0"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "algorithm={alg} failed");
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // Indexed-palette PNG (color_type=3)
            assert_eq!(bytes[25], 3, "expected indexed PNG for algorithm={alg}");
        }

        // Verify TiTiler-compatible JSON ?algorithm_params={"azimuth":180,"angle_altitude":30}
        {
            let r_default = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm=hillshade"
                ))
                .send()
                .await
                .unwrap();
            let b_default = r_default.bytes().await.unwrap();

            let r_json = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A180%2C%22angle_altitude%22%3A30%2C%22z_factor%22%3A2.5%7D"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(r_json.status(), 200);
            let b_json = r_json.bytes().await.unwrap();
            assert_eq!(&b_json[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            assert_ne!(
                b_default, b_json,
                "algorithm_params JSON should alter hillshade lighting"
            );
        }
    }
}
