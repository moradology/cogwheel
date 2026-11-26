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
        ("unknown algorithm", format!("{tile}&algorithm=does_not_exist")),
        ("unsupported resampling", format!("{tile}&resampling=bogus")),
        ("excessive padding", format!("{tile}&padding=999")),
        ("unsupported reproject", format!("{tile}&reproject=bogus")),
        ("bad rescale", format!("{tile}&rescale=abc")),
        ("unknown colormap", format!("{tile}&bidx=1&colormap_name=nope")),
        ("bad tilesize", format!("{tile}&tilesize=0")),
        ("unknown tms tile", format!("{base}/cog/tiles/FakeQuad/10/163/395.png?url={sat_url}")),
        ("unknown tms tilejson", format!("{base}/cog/FakeQuad/tilejson.json?url={sat_url}")),
        ("non-stats percentiles", format!("{base}/cog/info?url={sat_url}&p=5")),
        ("point algorithm", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&algorithm=hillshade")),
        ("point unsupported coord_crs", format!("{base}/cog/point/1,1?url={sat_url}&coord_crs=epsg:4267")),
        ("preview explicit dims width", format!("{base}/cog/preview/256x256.png?url={sat_url}&width=300")),
        ("bbox explicit dims width", format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.8/256x256.png?url={sat_url}&width=300")),
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
        ("lanczos", format!("{tile}&resampling=lanczos&rescale=0,1")),
        ("average", format!("{tile}&resampling=average&rescale=0,1")),
        ("mode", format!("{tile}&resampling=mode&rescale=0,1")),
        ("gauss", format!("{tile}&resampling=gauss&rescale=0,1")),
        ("rms", format!("{tile}&resampling=rms&rescale=0,1")),
        ("med", format!("{tile}&resampling=med&rescale=0,1")),
        ("q1", format!("{tile}&resampling=q1&rescale=0,1")),
        ("q3", format!("{tile}&resampling=q3&rescale=0,1")),
        ("sum", format!("{tile}&resampling=sum&rescale=0,1")),
        ("reproject=average", format!("{tile}&reproject=average&rescale=0,1")),
        ("algorithm contours", format!("{tile}&bidx=1&algorithm=contours&algorithm_params=%7B%22increment%22:0.1,%22thickness%22:2%7D")),
        ("algorithm contours meta", format!("{base}/algorithms/contours")),
        ("reproject=bilinear", format!("{tile}&reproject=bilinear&rescale=0,1")),
        ("padding=0", format!("{tile}&padding=0&rescale=0,1")),
        ("padding=4", format!("{tile}&padding=4&rescale=0,1")),
        ("unrelated parameter", format!("{tile}&rescale=0,1&_cachebust=1&access_token=x")),
        ("hillshade", format!("{tile}&bidx=1&algorithm=hillshade")),
        ("WorldCRS84Quad", format!("{base}/cog/WorldCRS84Quad/tilejson.json?url={sat_url}")),
        ("UTM31WGS84Quad tilejson", format!("{base}/cog/UTM31WGS84Quad/tilejson.json?url={sat_url}")),
        ("WorldMercatorWGS84Quad tilejson", format!("{base}/cog/WorldMercatorWGS84Quad/tilejson.json?url={sat_url}")),
        ("WGS1984Quad tile", format!("{base}/cog/tiles/WGS1984Quad/2/1/1.png?url={sat_url}&rescale=0,1")),
        ("point coord_crs 4326", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&coord_crs=epsg:4326")),
        ("point coord_crs 32610 UTM", format!("{base}/cog/point/535280,4172640?url={sat_url}&coord_crs=epsg:32610")),
        ("preview width query", format!("{base}/cog/preview.png?url={sat_url}&width=300")),
        ("preview dst_crs 4326", format!("{base}/cog/preview/64x64.png?url={sat_url}&dst_crs=epsg:4326")),
        ("preview dst_crs 3857", format!("{base}/cog/preview/64x64.tif?url={sat_url}&dst_crs=epsg:3857")),
        ("info.geojson crs 32610", format!("{base}/cog/info.geojson?url={sat_url}&crs=epsg:32610")),
        ("bounds crs 3857", format!("{base}/cog/bounds?url={sat_url}&crs=epsg:3857")),
        ("bbox geotiff without dims", format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.8.tif?url={sat_url}&max_size=64&rescale=0,1")),
        ("preview max_size=0", format!("{base}/cog/preview.png?url={sat_url}&max_size=0&rescale=0,1")),
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
        ("algorithm median 4-band", format!("{tile}&algorithm=median&bidx=1&bidx=2&bidx=3&bidx=1")),
        ("algorithm std 1-band", format!("{tile}&algorithm=std&bidx=1")),
        ("algorithm cast uint8 3-band", format!("{tile}&algorithm=cast&algorithm_params=%7B%22dtype%22:%22uint8%22%7D")),
        ("algorithm ceil 3-band", format!("{tile}&algorithm=ceil")),
        ("algorithm floor 3-band", format!("{tile}&algorithm=floor")),
        ("fractional buffer=0.5", format!("{tile}&buffer=0.5&rescale=0,1")),
        ("algorithm bitonal", format!("{tile}&algorithm=bitonal&algorithm_params=%7B%22threshold%22:0.5%7D")),
        ("statistics expression", format!("{base}/cog/statistics?url={sat_url}&expression=b1-b2;b2-b3")),
        ("point multi-expression", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3")),
        ("cog tile geotiff", format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1")),
        ("cog preview geotiff", format!("{base}/cog/preview/64x64.tif?url={sat_url}&rescale=0,1")),
        ("cog tile 3-band rgb expr", format!("{tile}&expression=b3*1.0;b2*1.0;b1*1.0&rescale=0,1")),
        ("ogc conformance", format!("{base}/conformance")),
        ("cog map", format!("{base}/cog/map?url={sat_url}&width=64&height=64&rescale=0,1")),
        ("cog map with bbox and crs", format!("{base}/cog/map?url={sat_url}&bbox=-122.8,37.6,-122.5,37.8&width=64&height=64&crs=epsg:3857&rescale=0,1")),
        ("color_formula numeric channels 123", format!("{tile}&rescale=0,1&color_formula=Gamma%20123%201.5")),
        ("trailing slash normalized", format!("{base}/cog/info/?url={sat_url}")),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{what} should be accepted: {url}");
        assert!(
            resp.headers().get("server-timing").is_some(),
            "{what} should include Server-Timing header"
        );
    }
    let st_custom = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&bidx=1&p=5&p=95&histogram_bins=20"),
    )
    .await;
    assert_eq!(st_custom.as_object().unwrap().len(), 1);
    assert!(st_custom["b1"]["percentile_5"].is_number());
    assert!(st_custom["b1"]["percentile_95"].is_number());
    assert_eq!(st_custom["b1"]["histogram"][0].as_array().unwrap().len(), 20);
    let st_cat = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&bidx=1&max_size=64&categorical=true&c=0&c=1"),
    )
    .await;
    assert_eq!(st_cat["b1"]["histogram"][0].as_array().unwrap().len(), 2);
    assert_eq!(st_cat["b1"]["histogram"][1], serde_json::json!([0, 1]));
    assert!(st_cat["b1"]["categories"].is_object());

    let pt_expr = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2"),
    )
    .await;
    assert_eq!(pt_expr["values"].as_array().unwrap().len(), 1);
    assert_eq!(pt_expr["band_names"][0], "b1");
    assert_eq!(pt_expr["band_descriptions"][0], "b1-b2");

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
    assert!(st_expr["b1"]["mean"].is_number());
    assert_eq!(st_expr["b1"]["description"], "b1-b2");
    assert!(st_expr["b2"]["mean"].is_number());
    assert_eq!(st_expr["b2"]["description"], "b2-b3");

    let pt_multi = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3"),
    )
    .await;
    assert_eq!(pt_multi["values"].as_array().unwrap().len(), 2);
    assert_eq!(pt_multi["band_names"], serde_json::json!(["b1", "b2"]));
    assert_eq!(pt_multi["band_descriptions"], serde_json::json!(["b1-b2", "b2-b3"]));

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
    assert_eq!(tms_arr.len(), 13, "expected all 13 morecantile TileMatrixSets");
    assert!(tms_arr.iter().any(|t| t["id"] == "WebMercatorQuad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "WorldCRS84Quad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "UTM31WGS84Quad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "WorldMercatorWGS84Quad"));

    // Verify GeoTIFF wrapping on /mosaic/tiles and native EPSG on /cog/preview.tif
    let mosaic_tif = client
        .get(format!("{base}/mosaic/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap();
    assert_eq!(mosaic_tif.status().as_u16(), 200);
    let mosaic_tif_bytes = mosaic_tif.bytes().await.unwrap();
    assert_eq!(&mosaic_tif_bytes[0..4], b"II\x2a\x00", "/mosaic/tiles .tif must be a valid GeoTIFF");

    let preview_tif = client
        .get(format!("{base}/cog/preview/64x64.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let parsed_preview = cogzilla_tiff::parse_cog(&preview_tif).expect("valid preview GeoTIFF");
    assert_eq!(
        parsed_preview.geo_keys.projected_cs_type,
        Some(32610),
        "native preview GeoTIFF should preserve source UTM EPSG:32610"
    );

    // Verify .npy Content-Type matches TiTiler (application/x-binary)
    let npy_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.npy?url={sat_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(npy_resp.status().as_u16(), 200);
    assert_eq!(
        npy_resp.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/x-binary"
    );

    // Verify WMTSCapabilities ScaleDenominator scales inversely with tile_scale
    let wmts_1x = client
        .get(format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}&tile_scale=1"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let wmts_2x = client
        .get(format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}&tile_scale=2"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let first_sd = |xml: &str| -> f64 {
        let s = xml.split("<ScaleDenominator>").nth(1).unwrap();
        s.split("</ScaleDenominator>").next().unwrap().parse::<f64>().unwrap()
    };
    let sd1 = first_sd(&wmts_1x);
    let sd2 = first_sd(&wmts_2x);
    assert!(
        ((sd1 / sd2) - 2.0).abs() < 1e-6,
        "WMTS ScaleDenominator should halve for tile_scale=2 (got {sd1} vs {sd2})"
    );

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

        // /stac/tiles with per-asset asset_bidx=red|1&asset_bidx=green|1&asset_bidx=blue|1 must match assets=red,green,blue
        let stac_ab_resp = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&asset_bidx=red|1&asset_bidx=green|1&asset_bidx=blue|1&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(
            stac_ab_resp.headers().get("x-assets").and_then(|v| v.to_str().ok()),
            Some("red,green,blue")
        );
        let stac_ab_tile = stac_ab_resp.bytes().await.unwrap();
        assert_eq!(stac_ab_tile, stac_tile, "per-asset asset_bidx should render identical 3-asset RGB");

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
            assert_eq!(mj_info["mosaic_tilematrixset"], "WebMercatorQuad");
            assert!(mj_info["quadkeys"].as_array().is_some_and(|a| !a.is_empty()));
            assert!(mj_info["crs"].as_str().is_some());

            // Non-WebMercatorQuad TMS on /mosaicjson/tiles (WorldCRS84Quad overlapping 12/655/1583)
            let mj_crs84 = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WorldCRS84Quad/10/327/297.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_crs84.status(), 200);
            let crs84_bytes = mj_crs84.bytes().await.unwrap();
            assert_eq!(&crs84_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

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

            // /mosaicjson/info.geojson, /mosaicjson/assets, /mosaicjson/point, /mosaicjson/bbox, /mosaicjson/statistics
            let mj_geo = get_json(&client, format!("{base}/mosaicjson/info.geojson?url={mj_url}")).await;
            assert_eq!(mj_geo["type"], "Feature");
            let mj_assets = get_json(&client, format!("{base}/mosaicjson/assets?url={mj_url}")).await;
            assert_eq!(mj_assets.as_array().unwrap().len(), 2);
            let mj_pt = get_json(&client, format!("{base}/mosaicjson/point/-122.4,37.75?url={mj_url}")).await;
            assert!(!mj_pt["assets"].as_array().unwrap().is_empty());
            let mj_pt_expr = get_json(
                &client,
                format!("{base}/mosaicjson/point/-122.4,37.75?url={mj_url}&expression=b1*2"),
            )
            .await;
            assert_eq!(mj_pt_expr["assets"][0]["band_names"][0], "b1");
            assert_eq!(mj_pt_expr["assets"][0]["band_descriptions"][0], "b1*2");
            let mj_stats = get_json(&client, format!("{base}/mosaicjson/statistics?url={mj_url}&max_size=128")).await;
            assert!(mj_stats["b1"]["mean"].is_number());
            let mj_bbox = client
                .get(format!("{base}/mosaicjson/bbox/-122.5,37.7,-122.3,37.8/64x64.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_bbox.status(), 200);
            let mj_bbox_bytes = mj_bbox.bytes().await.unwrap();
            assert_eq!(&mj_bbox_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // /mosaicjson/tiles outside mosaic bounds returns 404 Not Found with JSON detail
            let mj_outside = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/0/0.png?url={mj_url}&buffer=8"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_outside.status(), 404);

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

            // Terrain NPY must emit 3-band RGB + mask (4, 256, 256), not raw 1-band elevation
            let npy_resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.npy?url={merc_url}&algorithm={alg}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(npy_resp.status(), 200);
            let npy_bytes = npy_resp.bytes().await.unwrap();
            let hdr_str = std::str::from_utf8(&npy_bytes[10..128]).unwrap_or("");
            assert!(
                hdr_str.contains("(4, 256, 256)"),
                "expected 3-band RGB + mask NPY for algorithm={alg}, got header: {hdr_str}"
            );
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

            // Mixed-resolution / mixed-CRS multi-COG composite (UTM10N RGB + WebMercator f32 DEM)
            let mixed_resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={sat_url}&url={merc_url}&expression=b1+b2&rescale=0,500"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(
                mixed_resp.status(),
                200,
                "mixed-resolution/mixed-CRS multi-COG composite should succeed"
            );
            let mixed_bytes = mixed_resp.bytes().await.unwrap();
            assert_eq!(&mixed_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Audit c40 regression checks:
            // 1. Path {w}x{h} > 4096 must be rejected with 400 (no OOM in GeoTIFF/bbox)
            let oob_dim = client
                .get(format!("{base}/cog/preview/5000x5000.tif?url={merc_url}"))
                .send()
                .await
                .unwrap();
            assert_eq!(oob_dim.status(), 400);

            // 2. Non-4326 geographic GeoTIFF (NAD83 EPSG:4269) must write ModelTypeGeographic (2)
            //    and unstyled float32 DEM .tif must emit native IEEE Float32 GeoTIFF (SampleType::Float32)
            let nad83_resp = client
                .get(format!(
                    "{base}/cog/bbox/-122.5,37.7,-122.4,37.8/16x16.tif?url={merc_url}&dst_crs=EPSG:4269"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(nad83_resp.status(), 200);
            let nad83_tif = nad83_resp.bytes().await.unwrap();
            let parsed_nad83 = cogzilla_tiff::parse_cog(&nad83_tif).unwrap();
            assert_eq!(parsed_nad83.geo_keys.model_type, Some(2));
            assert_eq!(parsed_nad83.geo_keys.geographic_type, Some(4269));
            assert_eq!(parsed_nad83.geo_keys.projected_cs_type, None);
            assert_eq!(parsed_nad83.full_res().sample_type, cogzilla_tiff::SampleType::Float32);

            // Styled .tif with rescale+colormap must still emit 8-bit visual GeoTIFF
            let styled_tif_resp = client
                .get(format!(
                    "{base}/cog/bbox/-122.5,37.7,-122.4,37.8/16x16.tif?url={merc_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(styled_tif_resp.status(), 200);
            let parsed_styled = cogzilla_tiff::parse_cog(&styled_tif_resp.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_styled.full_res().sample_type, cogzilla_tiff::SampleType::Uint8);

            // Audit c42 regression checks:
            // Unstyled float32 DEM .tif on /mosaic/tiles must emit Float32 GeoTIFF (not corrupt RGBA8)
            let mosaic_f32_tif = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.tif?url={merc_url}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mosaic_f32_tif.status(), 200);
            let parsed_mosaic_f32 = cogzilla_tiff::parse_cog(&mosaic_f32_tif.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_mosaic_f32.full_res().sample_type, cogzilla_tiff::SampleType::Float32);

            // ?algorithm=contours .tif must emit 8-bit visual GeoTIFF (not Float32)
            let contours_tif = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.tif?url={merc_url}&algorithm=contours"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(contours_tif.status(), 200);
            let parsed_contours = cogzilla_tiff::parse_cog(&contours_tif.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_contours.full_res().sample_type, cogzilla_tiff::SampleType::Uint8);

            // Audit c45 regression check: unknown/malformed TMS on WMTSCapabilities.xml must return 404/422 (never 200 XML)
            let bad_wmts = client
                .get(format!("{base}/cog/a&b/WMTSCapabilities.xml?url={merc_url}"))
                .send()
                .await
                .unwrap();
            assert!(
                bad_wmts.status() == 404 || bad_wmts.status() == 422,
                "unknown WMTS TMS should be rejected, got {}",
                bad_wmts.status()
            );

            // Audit c47 regression check: buffered WebMercatorQuad x=0 tile must not false-404
            let buf_x0 = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/1/0/0.png?url={sat_url}&buffer=2&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(buf_x0.status(), 200, "buffered x=0 WebMercatorQuad tile must not false-404");

            // Audit c48 regression check: constant & where() expression statistics
            let const_stats = get_json(
                &client,
                format!("{base}/cog/statistics?url={merc_url}&expression=42&max_size=64"),
            )
            .await;
            assert!((const_stats["b1"]["mean"].as_f64().unwrap() - 42.0).abs() < 1e-4);
            assert_eq!(const_stats["b1"]["description"], "42");

            // 3. 16-bit Tag 320 palette entries >= 256 must be preserved in CustomColormap
            let mut pal = vec![[10u8, 20, 30, 255]; 300];
            pal[299] = [200, 150, 100, 255];
            let cmap = cogzilla_kernel::CustomColormap::from_rgba_palette(&pal);
            assert_eq!(cmap.lookup(299.0, 0.0, 1.0), [200, 150, 100, 255]);

            // Audit c69 regression check: antimeridian (w > e) and inverted (s > n) STAC Item
            // bboxes on /stac/map must not panic in render_bbox_impl and must preserve 5:4 aspect.
            let am_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_antimeridian_{}.json",
                std::process::id()
            ));
            let am_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "fiji-antimeridian",
                "bbox": [177.0, -20.0, -178.0, -16.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "b1": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            std::fs::write(&am_stac_path, am_stac.to_string()).unwrap();
            let am_stac_url = am_stac_path.to_str().unwrap();
            let am_resp = client
                .get(format!(
                    "{base}/stac/map?url={am_stac_url}&assets=b1&max_size=128&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(
                am_resp.status(),
                404,
                "antimeridian /stac/map outside asset bounds must return 404 without panicking"
            );

            let inv_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "inverted-bbox",
                "bbox": [0.0, 10.0, 1.0, -10.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "b1": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            let inv_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_inverted_{}.json",
                std::process::id()
            ));
            std::fs::write(&inv_stac_path, inv_stac.to_string()).unwrap();
            let inv_stac_url = inv_stac_path.to_str().unwrap();
            let inv_resp = client
                .get(format!(
                    "{base}/stac/map?url={inv_stac_url}&assets=b1&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(inv_resp.status(), 400, "inverted s > n STAC bbox must return 400");
            let _ = std::fs::remove_file(&am_stac_path);
            let _ = std::fs::remove_file(&inv_stac_path);

            // Audit c75 regression checks:
            // 1. GeometryCollection accepted in PolygonMask::from_geojson + constant expression
            //    respects shape_mask (outside-polygon pixels must be NaN, not 1.0).
            let gc_poly = serde_json::json!({
                "type": "GeometryCollection",
                "geometries": [{
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.65, 37.76],
                        [-122.60, 37.76],
                        [-122.65, 37.80],
                        [-122.65, 37.76]
                    ]]
                }]
            });
            let gc_resp = client
                .post(format!("{base}/cog/feature/32x32.f32?url={sat_url}&expression=1"))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(gc_resp.status(), 200, "GeometryCollection /cog/feature must succeed");
            let gc_bytes = gc_resp.bytes().await.unwrap();
            assert_eq!(gc_bytes.len(), 32 * 32 * 4);
            // Top-right pixel (row 0, col 31) is outside the lower-left triangle -> must be NaN
            let tr_val = f32::from_le_bytes([
                gc_bytes[31 * 4],
                gc_bytes[31 * 4 + 1],
                gc_bytes[31 * 4 + 2],
                gc_bytes[31 * 4 + 3],
            ]);
            assert!(tr_val.is_nan(), "outside-polygon pixel for constant expression must be NaN, got {tr_val}");
            // Bottom-left pixel (row 30, col 1) is inside the triangle -> must be 1.0
            let bl_off = (30 * 32 + 1) * 4;
            let bl_val = f32::from_le_bytes([
                gc_bytes[bl_off],
                gc_bytes[bl_off + 1],
                gc_bytes[bl_off + 2],
                gc_bytes[bl_off + 3],
            ]);
            assert_eq!(bl_val, 1.0, "inside-polygon pixel for expression=1 must be 1.0");

            // 2. 2-vertex degenerate ring must be rejected with 400
            let two_vtx = serde_json::json!({
                "type": "Polygon",
                "coordinates": [[[-122.65, 37.76], [-122.60, 37.80]]]
            });
            let two_resp = client
                .post(format!("{base}/cog/feature/32x32.png?url={merc_url}&rescale=0,2500"))
                .header("content-type", "application/json")
                .body(two_vtx.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(two_resp.status(), 400, "2-vertex GeoJSON ring must return 400");

            // Audit c77 regression check: mixed-grid multi-COG /cog/feature with constant expr
            // must still mask outside-polygon pixels to NaN.
            let mg_resp = client
                .post(format!(
                    "{base}/cog/feature/32x32.f32?url={sat_url}&url={merc_url}&expression=1"
                ))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(mg_resp.status(), 200);
            let mg_bytes = mg_resp.bytes().await.unwrap();
            assert_eq!(mg_bytes.len(), 32 * 32 * 4);
            let mg_tr = f32::from_le_bytes(mg_bytes[31 * 4..32 * 4].try_into().unwrap());
            let mg_bl = f32::from_le_bytes(mg_bytes[bl_off..bl_off + 4].try_into().unwrap());
            assert!(mg_tr.is_nan(), "mixed-grid outside-polygon pixel must be NaN, got {mg_tr}");
            assert_eq!(mg_bl, 1.0, "mixed-grid inside-polygon pixel must be 1.0");

            // Audit c78 regression checks:
            // 1. Mixed-grid where(b2>=0,b2,0) must NOT clip b2 (sat_url) to cogs[0] (merc_url)'s footprint
            let mg_b2 = client
                .post(format!(
                    "{base}/cog/feature/32x32.f32?url={merc_url}&url={sat_url}&expression=where(b2%3E=0,b2,0)"
                ))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(mg_b2.status(), 200);
            let mg_b2_bytes = mg_b2.bytes().await.unwrap();
            let mg_b2_tr = f32::from_le_bytes(mg_b2_bytes[31 * 4..32 * 4].try_into().unwrap());
            let mg_b2_bl = f32::from_le_bytes(mg_b2_bytes[bl_off..bl_off + 4].try_into().unwrap());
            assert!(mg_b2_tr.is_nan(), "outside-polygon pixel must remain NaN, got {mg_b2_tr}");
            assert!(mg_b2_bl.is_finite(), "b2 pixel inside sat_url must not be clipped to cogs[0], got {mg_b2_bl}");

            // 2. /mosaic/tiles must skip an unreadable urls[0] and render urls[1]
            let resil_resp = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url=/nonexistent_cogzilla_scene_0.tif&url={merc_url}&rescale=0,2500"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resil_resp.status(), 200, "mosaic tile must skip unreadable urls[0]");
            let resil_bytes = resil_resp.bytes().await.unwrap();
            assert_eq!(&resil_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // 3. GET /stac/statistics must preserve ?bidx=3,1 order
            let bidx_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_bidx_order_{}.json",
                std::process::id()
            ));
            let bidx_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "bidx-order",
                "bbox": [-122.8, 37.6, -122.5, 37.9],
                "geometry": null,
                "properties": {},
                "assets": {
                    "visual": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            std::fs::write(&bidx_stac_path, bidx_stac.to_string()).unwrap();
            let bidx_stats = get_json(
                &client,
                format!(
                    "{base}/stac/statistics?url={}&assets=visual&bidx=3&bidx=1&max_size=64",
                    bidx_stac_path.to_str().unwrap()
                ),
            )
            .await;
            assert_eq!(bidx_stats["b1"]["description"].as_str(), Some("visual_b3"));
            assert_eq!(bidx_stats["b2"]["description"].as_str(), Some("visual_b1"));
            let _ = std::fs::remove_file(&bidx_stac_path);

            // 4. MosaicJSON tile assets & pixel_selection (highest, lowest, mean, count)
            let mj_ps_path = std::env::temp_dir().join(format!(
                "cogzilla_mj_ps_{}.json",
                std::process::id()
            ));
            let mj_ps = serde_json::json!({
                "mosaicjson": "0.0.2",
                "minzoom": 8,
                "maxzoom": 14,
                "bounds": [-123.0, 37.0, -121.5, 38.5],
                "quadkey_zoom": 8,
                "tiles": {
                    "02301020": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301021": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301022": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301023": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()]
                }
            });
            std::fs::write(&mj_ps_path, mj_ps.to_string()).unwrap();
            let tile_assets = get_json(
                &client,
                format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583/assets?url={}",
                    mj_ps_path.to_str().unwrap()
                ),
            )
            .await;
            assert!(tile_assets.is_array() && !tile_assets.as_array().unwrap().is_empty());
            let pt_assets = get_json(
                &client,
                format!(
                    "{base}/mosaicjson/point/-122.4,37.75/assets?url={}",
                    mj_ps_path.to_str().unwrap()
                ),
            )
            .await;
            assert!(pt_assets.is_array() && !pt_assets.as_array().unwrap().is_empty());
            let _ = std::fs::remove_file(&mj_ps_path);

            // Compare pixel_selection=highest vs lowest vs mean vs count on two scenes (merc vs sat b1)
            let read_f32_first = |b: &[u8]| -> f32 {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .find(|v| v.is_finite())
                    .unwrap_or(f32::NAN)
            };
            let hi_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={sat_url}&bidx=1&pixel_selection=highest"))
                .send().await.unwrap().bytes().await.unwrap();
            let lo_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={sat_url}&bidx=1&pixel_selection=lowest"))
                .send().await.unwrap().bytes().await.unwrap();
            let cnt_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={merc_url}&bidx=1&pixel_selection=count"))
                .send().await.unwrap().bytes().await.unwrap();
            let hi_v = read_f32_first(&hi_bytes);
            let lo_v = read_f32_first(&lo_bytes);
            let cnt_v = read_f32_first(&cnt_bytes);
            assert!(hi_v >= lo_v, "highest ({hi_v}) must be >= lowest ({lo_v})");
            assert!((cnt_v - 2.0).abs() < 1e-4, "count on 2 identical scenes must be 2.0, got {cnt_v}");

            let ps_tile = format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582");
            let f32s = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            let sat_tile = format!("{base}/mosaic/tiles/WebMercatorQuad/12/652/1583");
            let fetch = |tile: &str, query: String| {
                let client = client.clone();
                let url = format!("{tile}.f32?{query}");
                async move {
                    let resp = client.get(url).send().await.unwrap();
                    (resp.status(), resp.bytes().await.unwrap())
                }
            };

            // count is a number everywhere, zero where no scene has a value.
            let counts = f32s(&cnt_bytes);
            assert_eq!(counts.len(), 256 * 256);
            assert!(counts.iter().all(|v| *v == 0.0 || *v == 2.0), "count is 0 or 2 over two identical scenes");

            // Names: any case, and `std` for `stdev`; anything else is refused.
            let (status, stdev) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=stdev")).await;
            assert_eq!(status, 200);
            for name in ["std", "STD", "Stdev"] {
                let (status, body) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection={name}")).await;
                assert_eq!(status, 200, "{name}");
                assert_eq!(body, stdev, "{name}");
            }
            let (status, _) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=sum")).await;
            assert_eq!(status, 422);

            // The last-band methods leave the deciding band out, and need a band beside it.
            for name in ["lastbandhigh", "lastbandlow"] {
                let three = format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection={name}");
                let (status, body) = fetch(&sat_tile, three).await;
                assert_eq!(status, 200, "{name}");
                assert_eq!(body.len(), 2 * 256 * 256 * 4, "{name} of three bands leaves two");
                let (status, _) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection={name}")).await;
                assert_eq!(status, 400, "{name} of one band");
            }
            // Over one scene the bands left are that scene's own.
            let (_, kept) = fetch(&sat_tile, format!("url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection=lastbandhigh")).await;
            let (_, own) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&pixel_selection=highest")).await;
            assert_eq!(kept, own);

            // median holds its scenes, within 64 MiB: two three-band scenes at 2048 px are over it.
            let over = format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection=median&tilesize=2048");
            let (status, body) = fetch(&sat_tile, over).await;
            assert_eq!(status, 400);
            assert!(
                String::from_utf8_lossy(&body).contains("pixel_selection=median exceeds memory budget (64 MiB limit)"),
                "{}",
                String::from_utf8_lossy(&body)
            );
            let (status, body) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection=median")).await;
            assert_eq!(status, 200);
            let (_, one) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection=highest")).await;
            assert_eq!(body, one, "the median of a scene taken twice is the scene");

            // mean and median of whole numbers are whole numbers, the fraction dropped;
            // of floats they keep it.
            let (b2, b3) = (fixture("landsat_c2_B2.tif"), fixture("landsat_c2_B3.tif"));
            if b2.exists() && b3.exists() {
                let pair = format!(
                    "url={}&url={}&bidx=1",
                    b2.canonicalize().unwrap().to_string_lossy(),
                    b3.canonicalize().unwrap().to_string_lossy()
                );
                let (_, hi) = fetch(&sat_tile, format!("{pair}&pixel_selection=highest")).await;
                let (_, lo) = fetch(&sat_tile, format!("{pair}&pixel_selection=lowest")).await;
                let (hi, lo) = (f32s(&hi), f32s(&lo));
                let halves = hi.iter().zip(&lo).filter(|(h, l)| h.is_finite() && (**h + **l) % 2.0 == 1.0).count();
                assert!(halves > 0, "some pixel's mean of two scenes has a half to drop");
                for name in ["mean", "median"] {
                    let (status, got) = fetch(&sat_tile, format!("{pair}&pixel_selection={name}")).await;
                    assert_eq!(status, 200, "{name}");
                    let got = f32s(&got);
                    assert_eq!(got.len(), hi.len());
                    for (i, v) in got.iter().enumerate() {
                        if hi[i].is_finite() {
                            let exact = (f64::from(hi[i]) + f64::from(lo[i])) / 2.0;
                            assert_eq!(f64::from(*v), exact.trunc(), "{name} at {i}: of {} and {}", lo[i], hi[i]);
                        } else {
                            assert!(v.is_nan(), "{name} at {i}");
                        }
                    }
                }
            }
            let (_, mean) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=mean")).await;
            assert!(f32s(&mean).iter().any(|v| v.is_finite() && v.fract() != 0.0), "a float mean keeps its fraction");

            // What a pixel selection leaves is painted and encoded like any other picture:
            // every format, with and without the mask, under a colour map and a formula.
            let formats: [(&str, &[u8]); 3] = [("png", &[0x89, b'P', b'N', b'G']), ("jpg", &[0xff, 0xd8]), ("webp", b"RIFF")];
            for (ext, magic) in formats {
                for look in ["", "&return_mask=false", "&rescale=0,100&colormap_name=viridis", "&color_formula=gamma%20rgb%202"] {
                    for bands in ["bidx=1", "bidx=1&bidx=2&bidx=3"] {
                        // A colour map is for one band; it is refused over three.
                        if look.contains("colormap_name") && bands != "bidx=1" {
                            continue;
                        }
                        let url = format!("{sat_tile}.{ext}?url={sat_url}&url={sat_url}&{bands}&pixel_selection=highest{look}");
                        let resp = client.get(&url).send().await.unwrap();
                        assert_eq!(resp.status(), 200, "{url}");
                        let body = resp.bytes().await.unwrap();
                        assert!(body.starts_with(magic), "{url}");
                    }
                }
            }

            // Sweep #1 & #7 C1-C4, D2(validate), G1-G3 regression checks
            let cog_stac = get_json(
                &client,
                format!("{base}/cog/stac?url={sat_url}&id=my-sat&collection=c1&collection_url=https://example.com/c1&asset_name=rgb&asset_href=https://example.com/sat.tif&asset_roles=visual&extensions=https://example.com/ext.json"),
            )
            .await;
            assert_eq!(cog_stac["type"], "Feature");
            assert_eq!(cog_stac["stac_version"], "1.1.0");
            assert_eq!(cog_stac["id"], "my-sat");
            assert_eq!(cog_stac["collection"], "c1");
            assert_eq!(cog_stac["properties"]["proj:epsg"], 32610);
            assert!(cog_stac["properties"]["proj:geometry"].is_object());
            assert_eq!(cog_stac["assets"]["rgb"]["href"], "https://example.com/sat.tif");
            assert_eq!(cog_stac["assets"]["rgb"]["roles"], serde_json::json!(["visual"]));
            assert_eq!(cog_stac["assets"]["rgb"]["type"], "image/tiff; application=geotiff");
            assert!(cog_stac["assets"]["rgb"]["raster:bands"][0]["statistics"].is_object());
            assert!(cog_stac["assets"]["rgb"]["raster:bands"][0]["histogram"].is_object());
            assert_eq!(cog_stac["assets"]["rgb"]["raster:bands"][0]["sampling"], "area");
            assert!(cog_stac["assets"]["rgb"]["eo:bands"].is_array());
            assert_eq!(cog_stac["links"][0]["rel"], "collection");
            assert!(cog_stac["stac_extensions"].as_array().unwrap().iter().any(|v| v == "https://example.com/ext.json"));

            let viewer_cog = client
                .get(format!("{base}/cog/viewer?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(viewer_cog.status(), 200);
            assert!(viewer_cog.text().await.unwrap().contains("tilejson.json"));

            let stac_renders_path = std::env::temp_dir().join(format!(
                "cogzilla_test_stac_renders_{}.json",
                std::process::id()
            ));
            let stac_renders_doc = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "renders-test",
                "bbox": [-122.9, 37.5, -122.4, 38.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "red": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                },
                "renders": {
                    "visual": { "assets": ["red"], "rescale": [[0, 1]] }
                }
            });
            std::fs::write(&stac_renders_path, stac_renders_doc.to_string()).unwrap();
            let stac_item_url = stac_renders_path.to_str().unwrap();
            let viewer_stac = client
                .get(format!("{base}/stac/viewer?url={stac_item_url}&assets=red"))
                .send().await.unwrap();
            assert_eq!(viewer_stac.status(), 200);

            let renders_json = get_json(&client, format!("{base}/stac/renders?url={stac_item_url}")).await;
            assert!(renders_json["renders"]["visual"]["valid"] == true);
            assert!(renders_json["renders"]["visual"]["params"]["assets"][0] == "red");
            assert!(renders_json["links"].is_array());
            let visual_render = get_json(&client, format!("{base}/stac/renders/visual?url={stac_item_url}")).await;
            assert_eq!(visual_render["valid"], true);
            assert_eq!(visual_render["params"]["assets"][0], "red");
            assert_eq!(visual_render["links"].as_array().unwrap().len(), 3);
            let missing_render = client
                .get(format!("{base}/stac/renders/nonexistent?url={stac_item_url}"))
                .send().await.unwrap();
            assert_eq!(missing_render.status(), 404);
            let _ = std::fs::remove_file(&stac_renders_path);

            let mj_doc = serde_json::json!({
                "mosaicjson": "0.0.3",
                "minzoom": 10,
                "maxzoom": 12,
                "quadkey_zoom": 10,
                "bounds": [-123.0, 37.0, -122.0, 38.0],
                "name": null,
                "attribution": null,
                "tiles": { "0230102033": [merc_url.clone()] }
            });
            let mj_val = client
                .post(format!("{base}/mosaicjson/validate"))
                .header("content-type", "application/json")
                .body(mj_doc.to_string())
                .send().await.unwrap();
            assert_eq!(mj_val.status(), 200);
            let mj_val_json: serde_json::Value =
                serde_json::from_slice(&mj_val.bytes().await.unwrap()).unwrap();
            assert_eq!(mj_val_json["mosaicjson"], "0.0.3");
            assert!(mj_val_json.get("name").is_none(), "null optional fields must be omitted");
            assert!(mj_val_json["tiles"].is_object());

            let mj_root_path = std::env::temp_dir().join(format!(
                "cogzilla_test_mj_root_{}.json",
                std::process::id()
            ));
            std::fs::write(&mj_root_path, mj_doc.to_string()).unwrap();
            let mj_root = get_json(
                &client,
                format!("{base}/mosaicjson/?url={}", mj_root_path.to_str().unwrap()),
            )
            .await;
            assert_eq!(mj_root["mosaicjson"], "0.0.3");
            assert!(mj_root.get("name").is_none());
            assert!(mj_root.get("attribution").is_none());
            assert!(mj_root["tiles"].is_object());

            // Sweep #8 H3: color_formula on single-band mosaic stays grayscale PNG (no red tint)
            // Quadkey "0230102033" corresponds to WebMercatorQuad tile 10/163/395.
            let mj_cf_resp = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395.png?url={}&rescale=0,1000&color_formula=gamma%20r%201.6",
                    mj_root_path.to_str().unwrap()
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_cf_resp.status(), 200);
            let mj_cf_bytes = mj_cf_resp.bytes().await.unwrap();
            assert!(mj_cf_bytes.len() > 26 && &mj_cf_bytes[..8] == b"\x89PNG\r\n\x1a\n");
            // PNG IHDR color_type byte at offset 25: 0 = Grayscale (L), 4 = Grayscale+Alpha (LA)
            let color_type = mj_cf_bytes[25];
            assert!(
                color_type == 0 || color_type == 4,
                "single-band mosaic with color_formula must encode grayscale PNG (color_type 0 or 4), got {color_type}"
            );
            // Mosaic point outside coverage returns 404 with JSON detail (not empty 204)
            let mj_pt_oob = client
                .get(format!(
                    "{base}/mosaicjson/point/0,0?url={}",
                    mj_root_path.to_str().unwrap()
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_pt_oob.status(), 404);
            let mj_pt_json: serde_json::Value =
                serde_json::from_slice(&mj_pt_oob.bytes().await.unwrap()).unwrap();
            assert!(
                mj_pt_json["detail"]
                    .as_str()
                    .is_some_and(|s| s.to_ascii_lowercase().contains("no asset")),
                "expected 404 JSON detail naming missing coverage, got {mj_pt_json}"
            );
            let _ = std::fs::remove_file(&mj_root_path);

            // Sweep #1 D4, D6, E2 regression checks
            let ts_one = get_json(&client, format!("{base}/cog/tiles/WebMercatorQuad?url={sat_url}")).await;
            assert!(ts_one["links"].as_array().is_some_and(|l| l.len() >= 3));
            assert!(ts_one["boundingBox"]["lowerLeft"].is_array());
            assert!(ts_one["tileMatrixSetLimits"].as_array().is_some_and(|l| !l.is_empty()));
            assert_eq!(ts_one["accessConstraints"], "unclassified");

            let tms_one = get_json(&client, format!("{base}/tileMatrixSets/WebMercatorQuad")).await;
            assert!(tms_one["uri"].as_str().is_some_and(|s| s.contains("WebMercatorQuad")));
            assert!(tms_one["wellKnownScaleSet"].as_str().is_some_and(|s| s.contains("GoogleMapsCompatible")));

            let conf = get_json(&client, format!("{base}/conformance")).await;
            assert_eq!(conf["conformsTo"].as_array().unwrap().len(), 23);

            let mna1 = client.post(format!("{base}/cog/info?url={sat_url}")).send().await.unwrap();
            assert_eq!(mna1.status(), 405);
            assert!(mna1.headers().get("allow").is_some());
            let mna2 = client.get(format!("{base}/cog/feature?url={sat_url}")).send().await.unwrap();
            assert_eq!(mna2.status(), 405);
            let mna3 = client.delete(format!("{base}/healthz")).send().await.unwrap();
            assert_eq!(mna3.status(), 405);

            // Sweep #1 E1 & D5 regression checks: gzip + Vary + Cache-Control on metadata, and 13-TMS WMTS
            let gz_resp = client
                .get(format!("{base}/tileMatrixSets/WebMercatorQuad"))
                .header("accept-encoding", "gzip")
                .send()
                .await
                .unwrap();
            assert_eq!(gz_resp.status(), 200);
            assert_eq!(
                gz_resp.headers().get("content-encoding").and_then(|v| v.to_str().ok()),
                Some("gzip")
            );
            assert!(
                gz_resp
                    .headers()
                    .get("vary")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| v.to_ascii_lowercase().contains("accept-encoding"))
            );
            assert!(gz_resp.headers().get("cache-control").is_some());
            let gz_bytes = gz_resp.bytes().await.unwrap();
            let decomp = cogzilla_codec::decompress_gzip_limited(&gz_bytes, 1024 * 1024).unwrap();
            let tms_decomp: serde_json::Value = serde_json::from_slice(&decomp).unwrap();
            assert_eq!(tms_decomp["id"], "WebMercatorQuad");

            let wmts_all = client
                .get(format!("{base}/cog/WMTSCapabilities.xml?url={sat_url}"))
                .send()
                .await
                .unwrap();
            assert_eq!(wmts_all.status(), 200);
            assert!(wmts_all.headers().get("cache-control").is_some());
            let wmts_body = wmts_all.text().await.unwrap();
            assert_eq!(wmts_body.matches("<Layer>").count(), 13);
            assert_eq!(wmts_body.matches("<TileMatrixSet>").count(), 26); // 13 links + 13 definitions
            // Sweep #10 K1 & K3: TileMatrixSetLimits, ServiceMetadataURL, EPSG:xxxx on use_epsg=true, zooms stripped
            assert_eq!(wmts_body.matches("<TileMatrixSetLimits>").count(), 13);
            assert!(wmts_body.contains("<MinTileRow>"));
            assert!(wmts_body.contains("<ServiceMetadataURL"));
            let wmts_epsg = client
                .get(format!("{base}/cog/WMTSCapabilities.xml?url={sat_url}&use_epsg=true&zooms=WebMercatorQuad::9,10"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert!(wmts_epsg.contains("<ows:SupportedCRS>EPSG:3857</ows:SupportedCRS>"));
            assert!(wmts_epsg.contains("<ows:SupportedCRS>EPSG:4326</ows:SupportedCRS>"));
            assert!(
                wmts_epsg
                    .lines()
                    .filter(|l| l.contains("<ResourceURL"))
                    .all(|l| !l.contains("zooms=")),
                "ResourceURL templates must not include zooms="
            );

            // Sweep #10 K2: binary .npy tile gzip-compressed when Accept-Encoding: gzip or *
            let npy_gz = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={sat_url}"))
                .header("accept-encoding", "gzip")
                .send()
                .await
                .unwrap();
            assert_eq!(npy_gz.status(), 200);
            assert_eq!(
                npy_gz.headers().get("content-encoding").and_then(|v| v.to_str().ok()),
                Some("gzip")
            );
            let npy_gz_bytes = npy_gz.bytes().await.unwrap();
            let npy_raw = cogzilla_codec::decompress_gzip_limited(&npy_gz_bytes, 8 * 1024 * 1024).unwrap();
            assert!(npy_raw.starts_with(b"\x93NUMPY"));

            // Sweep #9 J1-J6 regression checks
            // J1: empty expression= is treated as absent
            let j1 = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&expression="))
                .send().await.unwrap();
            assert_eq!(j1.status(), 200);
            // J2 & J3: float-int params (bidx=1.0, tilesize=16.0, z=12.0) and tilesize < 64
            let j23 = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12.0/652/1583.png?url={sat_url}&bidx=1.0&tilesize=16.0"))
                .send().await.unwrap();
            assert_eq!(j23.status(), 200);
            // J4: space after comma in /point and /bbox path coordinates
            let j4p = client
                .get(format!("{base}/cog/point/-122.65,%2037.773?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j4p.status(), 200);
            let j4b = client
                .get(format!("{base}/cog/bbox/-122.8,%2037.6,-122.5,37.9/32x32.png?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j4b.status(), 200);
            // J5: EPSG:4979, ESRI:102100, and PROJ string CRS forms
            for crs in [
                "EPSG:4979",
                "ESRI:102100",
                "+proj=longlat%20+datum=WGS84",
            ] {
                let r = client
                    .get(format!("{base}/cog/preview/32x32.png?url={sat_url}&dst_crs={crs}"))
                    .send().await.unwrap();
                assert_eq!(r.status(), 200, "dst_crs={crs} should succeed");
            }
            // J6: /cog/preview/5000x5000.png error message states size exceeds limit
            let j6 = client
                .get(format!("{base}/cog/preview/5000x5000.png?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j6.status(), 400);
            let j6_body = j6.text().await.unwrap();
            assert!(
                j6_body.contains("4096") && j6_body.to_ascii_lowercase().contains("exceed"),
                "expected size-limit message, got: {j6_body}"
            );

            // N4: mismatched rescale count (2 ranges for 3 bands) returns 400 naming the mismatch
            let n4_bad = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_bad.status(), 400);
            let n4_body = n4_bad.text().await.unwrap();
            assert!(
                n4_body.contains('2') && n4_body.contains('3'),
                "expected rescale count mismatch message naming 2 and 3, got: {n4_body}"
            );
            // 1 range (broadcast) and 3 ranges (per-band) remain valid
            let n4_one = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_one.status(), 200);
            let n4_three = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3&rescale=0,0.3&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_three.status(), 200);
        }
    }
}

/// Write a minimal uncompressed single-band float32 GeoTIFF with `ProjectedCSTypeGeoKey = epsg`
/// so CI tests never depend on gitignored `benches/fixtures/`.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_f32_projected_geotiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    epsg: u16,
    origin_x: f64,
    origin_y: f64,
    pixel_size: f64,
    pixels: &[f32],
) {
    assert_eq!(pixels.len(), (width * height) as usize);
    let mut buf = Vec::<u8>::new();
    // Little-endian classic TIFF header
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // IFD at offset 8

    let num_entries: u16 = 13;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let data_start = 8 + ifd_size;

    let scale_off = data_start as u32;
    let tiepoint_off = scale_off + 24;
    let geokey_off = tiepoint_off + 48;
    let pixel_off = geokey_off + 24;
    let pixel_bytes = (pixels.len() * 4) as u32;

    let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    };

    buf.extend_from_slice(&num_entries.to_le_bytes());
    push_entry(&mut buf, 256, 4, 1, width); // ImageWidth
    push_entry(&mut buf, 257, 4, 1, height); // ImageLength
    push_entry(&mut buf, 258, 3, 1, 32); // BitsPerSample = 32
    push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
    push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
    push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
    push_entry(&mut buf, 277, 3, 1, 1); // SamplesPerPixel = 1
    push_entry(&mut buf, 278, 4, 1, height); // RowsPerStrip
    push_entry(&mut buf, 279, 4, 1, pixel_bytes); // StripByteCounts
    push_entry(&mut buf, 339, 3, 1, 3); // SampleFormat = IEEEFP (3)
    push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
    push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
    push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
    buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0

    assert_eq!(buf.len(), scale_off as usize);
    for v in [pixel_size, pixel_size, 0.0f64] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), tiepoint_off as usize);
    for v in [0.0f64, 0.0, 0.0, origin_x, origin_y, 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), geokey_off as usize);
    // GeoKeyDirectory: 1,1,0,2 keys: GTModelTypeGeoKey(1024)=1 (Projected), ProjectedCSTypeGeoKey(3072)=epsg
    for v in [1u16, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), pixel_off as usize);
    for &p in pixels {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &buf).unwrap();
}

fn decode_npy_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.starts_with(b"\x93NUMPY"), "not an npy payload");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let payload = &bytes[10 + hlen..];
    payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::test]
async fn sweep11_l1_l2_and_sweep12_m1_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_l1_l2_m1_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. Synthetic 64x64 float32 UTM 10N GeoTIFF (no overviews) for L1 average + L2 CRS tests
    let w = 64u32;
    let h = 64u32;
    let mut grad = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            grad.push(((y * 7 + x * 3) % 97) as f32 / 100.0 + (x as f32) * 0.01);
        }
    }
    let grad_path = tmp.join("grad_64_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &grad_path,
        w,
        h,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &grad,
    );

    // 2. Synthetic 64x64 categorical float32 GeoTIFF for L1 mode test:
    // Each 4x4 block has 11 pixels of class `c` and 5 pixels of other classes.
    let mut cat = vec![0.0f32; (w * h) as usize];
    for by in 0..16u32 {
        for bx in 0..16u32 {
            let dominant = ((by * 16 + bx) % 10 + 1) as f32;
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let v = if dy == 0 && dx < 2 {
                        99.0
                    } else if dy == 3 && dx >= 1 {
                        88.0
                    } else {
                        dominant
                    };
                    cat[((by * 4 + dy) * w + (bx * 4 + dx)) as usize] = v;
                }
            }
        }
    }
    let cat_path = tmp.join("cat_64_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &cat_path,
        w,
        h,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &cat,
    );

    // 3. Synthetic 16x16 GeoTIFF in unsupported CRS EPSG:54030 (Robinson) for M1 test
    let unsup_path = tmp.join("robinson_54030.tif");
    write_synthetic_f32_projected_geotiff(
        &unsup_path,
        16,
        16,
        54030,
        0.0,
        100_000.0,
        1000.0,
        &vec![42.0f32; 256],
    );

    // 4. Synthetic 32x32 GeoTIFFs in EPSG:6933 (EASE-2 Global), EPSG:3410, and EPSG:8857 (Equal Earth)
    let ease6933_path = tmp.join("ease_6933.tif");
    write_synthetic_f32_projected_geotiff(
        &ease6933_path,
        32,
        32,
        6933,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &(0..1024).map(|i| (i % 50) as f32 + 10.0).collect::<Vec<_>>(),
    );
    let ease3410_path = tmp.join("ease_3410.tif");
    write_synthetic_f32_projected_geotiff(
        &ease3410_path,
        32,
        32,
        3410,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &vec![7.0f32; 1024],
    );
    let ee8857_path = tmp.join("equal_earth_8857.tif");
    write_synthetic_f32_projected_geotiff(
        &ee8857_path,
        32,
        32,
        8857,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &vec![13.0f32; 1024],
    );

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let grad_url = grad_path.to_string_lossy().to_string();
    let cat_url = cat_path.to_string_lossy().to_string();
    let unsup_url = unsup_path.to_string_lossy().to_string();
    let ease6933_url = ease6933_path.to_string_lossy().to_string();
    let ease3410_url = ease3410_path.to_string_lossy().to_string();
    let ee8857_url = ee8857_path.to_string_lossy().to_string();

    // --- L1: resampling=average at 1x, 2x, 4x, 8x equals exact full-res block mean ---
    for factor in [1u32, 2, 4, 8] {
        let out_dim = w / factor;
        let resp = client
            .get(format!(
                "{base}/cog/preview.npy?url={grad_url}&max_size={out_dim}&resampling=average&return_mask=false"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "average preview factor={factor}");
        let vals = decode_npy_f32(&resp.bytes().await.unwrap());
        assert_eq!(vals.len(), (out_dim * out_dim) as usize);
        let mut max_err = 0.0f64;
        for by in 0..out_dim {
            for bx in 0..out_dim {
                let mut block_sum = 0.0f64;
                for dy in 0..factor {
                    for dx in 0..factor {
                        block_sum += grad[((by * factor + dy) * w + (bx * factor + dx)) as usize] as f64;
                    }
                }
                let expected = block_sum / ((factor * factor) as f64);
                let got = vals[(by * out_dim + bx) as usize] as f64;
                max_err = max_err.max((got - expected).abs());
            }
        }
        assert!(
            max_err < 1e-6,
            "L1 average at {factor}x reduction max_err={max_err} (expected < 1e-6)"
        );
    }

    // --- L1: resampling=mode at 4x matches exact 4x4 block mode ---
    let mode_resp = client
        .get(format!(
            "{base}/cog/preview.npy?url={cat_url}&max_size=16&resampling=mode&return_mask=false"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(mode_resp.status(), 200);
    let mode_vals = decode_npy_f32(&mode_resp.bytes().await.unwrap());
    assert_eq!(mode_vals.len(), 256);
    for by in 0..16u32 {
        for bx in 0..16u32 {
            let expected = ((by * 16 + bx) % 10 + 1) as f32;
            let got = mode_vals[(by * 16 + bx) as usize];
            assert_eq!(got, expected, "L1 mode mismatch at ({bx},{by})");
        }
    }

    // --- L2: legacy GeoJSON `crs` member honoured on /cog/statistics and /cog/feature ---
    let utm_feature_with_crs = serde_json::json!({
        "type": "Feature",
        "crs": { "type": "name", "properties": { "name": "EPSG:32610" } },
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_000.0, 4_200_000.0],
                [500_300.0, 4_200_000.0],
                [500_300.0, 4_199_700.0],
                [500_000.0, 4_199_700.0],
                [500_000.0, 4_200_000.0]
            ]]
        }
    });
    let st_crs = client
        .post(format!("{base}/cog/statistics?url={grad_url}&max_size=0"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_feature_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_crs.status(), 200, "legacy GeoJSON crs should be honoured");
    let st_crs_json: serde_json::Value =
        serde_json::from_slice(&st_crs.bytes().await.unwrap()).unwrap();
    let count = st_crs_json["properties"]["statistics"]["b1"]["count"]
        .as_f64()
        .unwrap();
    assert!(
        (count - 100.0).abs() < 1e-4,
        "10x10 UTM pixel box with legacy GeoJSON crs should have count=100, got {count}"
    );

    // FeatureCollection with top-level `crs`
    let utm_fc_with_crs = serde_json::json!({
        "type": "FeatureCollection",
        "crs": { "type": "name", "properties": { "name": "EPSG:32610" } },
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": utm_feature_with_crs["geometry"].clone()
        }]
    });
    let fc_crs = client
        .post(format!("{base}/cog/statistics?url={grad_url}&max_size=0"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_fc_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(fc_crs.status(), 200);
    let fc_crs_json: serde_json::Value =
        serde_json::from_slice(&fc_crs.bytes().await.unwrap()).unwrap();
    let fc_count = fc_crs_json["features"][0]["properties"]["statistics"]["b1"]["count"]
        .as_f64()
        .unwrap();
    assert!((fc_count - 100.0).abs() < 1e-4, "FC legacy crs count={fc_count}");

    // /cog/feature with legacy `crs`
    let feat_crs = client
        .post(format!("{base}/cog/feature.npy?url={grad_url}&max_size=64"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_feature_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(feat_crs.status(), 200, "/cog/feature should honour legacy GeoJSON crs");

    // --- L2: projected coordinates without coord_crs or `crs` return 400 with hint ---
    let utm_feature_no_crs = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": utm_feature_with_crs["geometry"].clone()
    });
    for ep in [
        format!("{base}/cog/statistics?url={grad_url}&max_size=0"),
        format!("{base}/cog/feature.npy?url={grad_url}&max_size=64"),
    ] {
        let bad = client
            .post(&ep)
            .header("content-type", "application/json")
            .body(serde_json::to_string(&utm_feature_no_crs).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 400, "projected coords without coord_crs must 400 on {ep}");
        let body = bad.text().await.unwrap();
        assert!(
            body.contains("coord_crs"),
            "400 message should hint at coord_crs, got: {body}"
        );
    }

    // --- M1: unsupported CRS (EPSG:54030) returns 400 on /cog/tiles (matching /cog/info) ---
    let info_unsup = client
        .get(format!("{base}/cog/info?url={unsup_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(info_unsup.status(), 400);
    let info_body = info_unsup.text().await.unwrap();
    assert!(info_body.to_ascii_lowercase().contains("unsupported crs"), "got: {info_body}");

    let tile_unsup = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/0/0/0.png?url={unsup_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        tile_unsup.status(),
        400,
        "unsupported CRS tile must return 400 (not 404 outside image bounds)"
    );
    let tile_body = tile_unsup.text().await.unwrap();
    assert!(
        tile_body.to_ascii_lowercase().contains("unsupported crs"),
        "expected unsupported CRS message on tile route, got: {tile_body}"
    );

    // --- EPSG:6933, EPSG:3410 (EASE-Grid Global) and EPSG:8857 (Equal Earth) ---
    for (label, u) in [
        ("EPSG:6933", &ease6933_url),
        ("EPSG:3410", &ease3410_url),
        ("EPSG:8857", &ee8857_url),
    ] {
        let info = client.get(format!("{base}/cog/info?url={u}")).send().await.unwrap();
        assert_eq!(info.status(), 200, "{label} /cog/info should return 200");
        let tile = client
            .get(format!("{base}/cog/tiles/WebMercatorQuad/2/1/1.png?url={u}"))
            .send()
            .await
            .unwrap();
        assert_eq!(tile.status(), 200, "{label} /cog/tiles should return 200");
    }

    // Equal-area zonal statistics on EPSG:6933: every 25km x 25km pixel covers exactly 625,000,000 m^2,
    // so area == count * pixel_area and area_weighted=true/false produce identical mean.
    let ease_poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-1_000_000.0, 4_000_000.0],
                [-750_000.0,   4_000_000.0],
                [-750_000.0,   3_750_000.0],
                [-1_000_000.0, 3_750_000.0],
                [-1_000_000.0, 4_000_000.0]
            ]]
        }
    });
    let st_aw = client
        .post(format!("{base}/cog/statistics?url={ease6933_url}&max_size=0&coord_crs=EPSG:6933&area_weighted=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&ease_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_aw.status(), 200);
    let st_aw_j: serde_json::Value = serde_json::from_slice(&st_aw.bytes().await.unwrap()).unwrap();
    let st_un = client
        .post(format!("{base}/cog/statistics?url={ease6933_url}&max_size=0&coord_crs=EPSG:6933&area_weighted=false"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&ease_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_un.status(), 200);
    let st_un_j: serde_json::Value = serde_json::from_slice(&st_un.bytes().await.unwrap()).unwrap();
    let b1_aw = &st_aw_j["properties"]["statistics"]["b1"];
    let b1_un = &st_un_j["properties"]["statistics"]["b1"];
    let count_aw = b1_aw["count"].as_f64().unwrap();
    let area_aw = b1_aw["area"].as_f64().unwrap();
    let mean_aw = b1_aw["mean"].as_f64().unwrap();
    let mean_un = b1_un["mean"].as_f64().unwrap();
    assert!((count_aw - 100.0).abs() < 1e-4, "expected 100 px, got {count_aw}");
    let expected_area = 100.0 * 25_000.0 * 25_000.0;
    let rel_area_err = (area_aw - expected_area).abs() / expected_area;
    assert!(
        rel_area_err < 1e-5,
        "EPSG:6933 equal-area should match 100 * 25km^2 = {expected_area}, got {area_aw} (rel_err={rel_area_err})"
    );
    assert!(
        (mean_aw - mean_un).abs() < 1e-12,
        "equal-area EPSG:6933 area_weighted=true ({mean_aw}) and false ({mean_un}) must match"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn uniform_semantic_http_statuses_across_all_tms() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_statuses_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Small UTM 10N raster in Northern California (~ -122.99..-122.97, 37.94..37.95)
    let utm_path = tmp.join("small_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &utm_path,
        32,
        32,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![123.0f32; 1024],
    );

    // All-NaN (nodata) raster at the exact same location
    let nodata_path = tmp.join("all_nodata_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &nodata_path,
        32,
        32,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![f32::NAN; 1024],
    );

    // MosaicJSON with wide advertised bounds [-125, 35, -120, 40] but only one tile covered
    let mj_path = tmp.join("gap_mosaic.json");
    let utm_url = utm_path.to_string_lossy().to_string();
    let nodata_url = nodata_path.to_string_lossy().to_string();
    let mj_json = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 8,
        "maxzoom": 12,
        "bounds": [-125.0, 35.0, -120.0, 40.0],
        "quadkey_zoom": 8,
        "tiles": {
            "02301020": [utm_url]
        }
    });
    std::fs::write(&mj_path, mj_json.to_string()).unwrap();
    let mj_url = mj_path.to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. Out-of-matrix x returns 400 Bad Request
    let bad_x = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/4/99/0.png?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_x.status(), 400, "out-of-matrix x must return 400");

    // 2. Every single one of the 13 TileMatrixSets returns 404 for a valid tile outside the UTM file
    let outside_tiles = [
        ("WebMercatorQuad", "4/15/15"),
        ("WorldCRS84Quad", "4/31/15"),
        ("WorldMercatorWGS84Quad", "4/15/15"),
        ("WGS1984Quad", "4/31/15"),
        ("UPSArcticWGS84Quad", "4/0/0"),
        ("UPSAntarcticWGS84Quad", "4/0/0"),
        ("CanadianNAD83_LCC", "4/0/0"),
        ("EuropeanETRS89_LAEAQuad", "4/0/0"),
        ("NZTM2000Quad", "4/0/0"),
        ("LINZAntarticaMapTilegrid", "4/0/0"),
        ("UTM31WGS84Quad", "4/0/0"),
        ("CDB1GlobalGrid", "4/0/0"),
        ("GNOSISGlobalGrid", "4/0/0"),
    ];
    for (tms, zxy) in outside_tiles {
        let r = client
            .get(format!("{base}/cog/tiles/{tms}/{zxy}.png?url={utm_url}"))
            .send()
            .await
            .unwrap();
        let status = r.status();
        let body = r.text().await.unwrap();
        assert_eq!(
            status, 404,
            "TMS {tms}/{zxy} outside UTM file must return 404, got {status}: {body}"
        );
        assert!(
            body.contains("outside") && body.contains(tms),
            "404 body for {tms} should name tile & bounds, got: {body}"
        );
    }

    // 3. All-nodata tile INSIDE dataset bounds returns 200 OK
    // UTM 10N (500_000, 4_200_000) is (-123.000, 37.947) -> WebMercator tile 10/162/395
    let nd_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/162/395.png?url={nodata_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        nd_resp.status(),
        200,
        "all-nodata tile inside footprint must return 200"
    );

    // 4. Mosaic gap INSIDE advertised mj.bounds returns 204; tile OUTSIDE mj.bounds returns 404
    // (-121.0, 36.0) is inside [-125, 35, -120, 40] -> WebMercator z=8 x=41, y=99 (uncovered gap)
    let gap_resp = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/8/41/99.png?url={mj_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(gap_resp.status(), 204, "mosaic gap inside bounds must return 204");
    let out_mj = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/8/0/0.png?url={mj_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(out_mj.status(), 404, "mosaic tile outside bounds must return 404");

    // 5. /cog/point, /cog/bbox, and /cog/feature outside footprint all return 404 with detail
    let pt_out = client
        .get(format!("{base}/cog/point/0,0?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(pt_out.status(), 404);
    let bbox_out = client
        .get(format!("{base}/cog/bbox/0,0,1,1/32x32.png?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(bbox_out.status(), 404);
    let feat_out = client
        .post(format!("{base}/cog/feature.png?url={utm_url}&max_size=32"))
        .header("content-type", "application/json")
        .body(r#"{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(feat_out.status(), 404);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn s1_cylindrical_equal_area_exact_in_worldcrs84_and_worldmercator() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_s1_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Global 512x256 ramp rasters in EPSG:6933 and EPSG:3410 (~67.8 km pixels):
    // pixel value = row * 512 + col so any 1-row shift shows up immediately as delta=512.
    let w = 512u32;
    let h = 256u32;
    let ramp: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();

    let g6933_path = tmp.join("g6933_ramp.tif");
    write_synthetic_f32_projected_geotiff(
        &g6933_path,
        w,
        h,
        6933,
        -17_367_520.0,
        7_314_540.0,
        34_735_040.0 / (w as f64),
        &ramp,
    );
    let g3410_path = tmp.join("g3410_ramp.tif");
    write_synthetic_f32_projected_geotiff(
        &g3410_path,
        w,
        h,
        3410,
        -17_334_200.0,
        7_344_784.0,
        34_668_400.0 / (w as f64),
        &ramp,
    );

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    for (epsg, path, ox, oy, px_sz) in [
        (6933u16, &g6933_path, -17_367_520.0, 7_314_540.0, 34_735_040.0 / (w as f64)),
        (3410u16, &g3410_path, -17_334_200.0, 7_344_784.0, 34_668_400.0 / (w as f64)),
    ] {
        let crs = cogzilla_proj::CrsModel::from_epsg(epsg).unwrap();
        let u = path.to_string_lossy().to_string();

        // Check WorldCRS84Quad z2/2/1 (lon 0..45, lat 0..45) and WorldMercatorWGS84Quad z2/2/1
        for (tms, z, x, y, dst_crs, tb) in [
            (
                "WorldCRS84Quad",
                2u8,
                2u32,
                1u32,
                cogzilla_proj::CrsModel::Geographic,
                [-90.0, 0.0, -45.0, 45.0],
            ),
            (
                "WorldMercatorWGS84Quad",
                2u8,
                2u32,
                1u32,
                cogzilla_proj::CrsModel::WorldMercator,
                [0.0, 0.0, 10_018_754.171_394_622, 10_018_754.171_394_622],
            ),
        ] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/{tms}/{z}/{x}/{y}.npy?url={u}&bidx=1&resampling=nearest&reproject=nearest&return_mask=false"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "EPSG:{epsg} {tms} {z}/{x}/{y}");
            let vals = decode_npy_f32(&resp.bytes().await.unwrap());
            assert_eq!(vals.len(), 256 * 256);

            let dlx = (tb[2] - tb[0]) / 256.0;
            let dly = (tb[3] - tb[1]) / 256.0;
            let mut mismatches = 0usize;
            for py in 0..256usize {
                let dy = tb[3] - (py as f64 + 0.5) * dly;
                for px in 0..256usize {
                    let dx = tb[0] + (px as f64 + 0.5) * dlx;
                    let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                    let (sx, sy) = crs.from_lonlat_deg(lon, lat);
                    let col = ((sx - ox) / px_sz).floor() as i64;
                    let row = ((oy - sy) / px_sz).floor() as i64;
                    if col >= 0 && col < w as i64 && row >= 0 && row < h as i64 {
                        let expected = (row as u32 * w + col as u32) as f32;
                        let got = vals[py * 256 + px];
                        if (got - expected).abs() > 0.5 {
                            mismatches += 1;
                        }
                    }
                }
            }
            assert_eq!(
                mismatches, 0,
                "S1 EPSG:{epsg} in {tms} {z}/{x}/{y} had {mismatches} mismatched pixels (expected 0)"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep15_t1_t2_t3_t4_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep15_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // 64x64 float32 UTM 10N raster with extreme outliers outside [10..13, 10..13] and
    // 9 distinct float values ~0.17..0.25 inside [10..13, 10..13].
    let w = 64u32;
    let h = 64u32;
    let ox = 500_000.0_f64;
    let oy = 4_200_000.0_f64;
    let px = 30.0_f64;
    let mut data = vec![999.0f32; (w * h) as usize];
    for r in 0..h {
        for c in 0..w {
            if r == 9 || r == 13 || c == 9 || c == 13 {
                data[(r * w + c) as usize] = -999.0;
            }
        }
    }
    for r in 10..13u32 {
        for c in 10..13u32 {
            let v = 0.17_f32 + ((r - 10) * 3 + (c - 10)) as f32 * 0.01;
            data[(r * w + c) as usize] = v;
        }
    }
    let utm_path = tmp.join("t1_t2_t3_utm.tif");
    write_synthetic_f32_projected_geotiff(&utm_path, w, h, 32610, ox, oy, px, &data);

    // 16x16 Web Mercator categorical raster spanning ~20N..50N (strong cos^2(lat) area variation)
    let mut merc_cat = vec![1.0f32; 256];
    for r in 8..16 {
        for c in 0..16 {
            merc_cat[r * 16 + c] = 2.0;
        }
    }
    let merc_path = tmp.join("t4_merc_cat.tif");
    write_synthetic_f32_projected_geotiff(
        &merc_path,
        16,
        16,
        3857,
        0.0,
        6_446_275.841_017_158,
        180_000.0,
        &merc_cat,
    );

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let utm_url = utm_path.to_string_lossy().to_string();
    let merc_url = merc_path.to_string_lossy().to_string();

    // --- T1 & T2: aligned 3x3 UTM polygon through real coord_crs=EPSG:32610 round-trip ---
    let sq_3x3 = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [ox + 10.0 * px, oy - 10.0 * px],
                [ox + 13.0 * px, oy - 10.0 * px],
                [ox + 13.0 * px, oy - 13.0 * px],
                [ox + 10.0 * px, oy - 13.0 * px],
                [ox + 10.0 * px, oy - 10.0 * px]
            ]]
        }
    });
    let r_t12 = client
        .post(format!("{base}/cog/statistics?url={utm_url}&bidx=1&max_size=0&coord_crs=EPSG:32610&categorical=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&sq_3x3).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_t12.status(), 200);
    let j_t12: serde_json::Value = serde_json::from_slice(&r_t12.bytes().await.unwrap()).unwrap();
    let b1 = &j_t12["properties"]["statistics"]["b1"];
    assert_eq!(b1["valid_pixels"].as_f64().unwrap(), 9.0, "T1 valid_pixels must be 9, got {b1}");
    assert_eq!(b1["count"].as_f64().unwrap(), 9.0);
    assert!(
        (b1["min"].as_f64().unwrap() - 0.17).abs() < 1e-5,
        "T1 min polluted by outlier neighbor: {}",
        b1["min"]
    );
    assert!(
        (b1["max"].as_f64().unwrap() - 0.25).abs() < 1e-5,
        "T1 max polluted by outlier neighbor: {}",
        b1["max"]
    );
    // T2: categorical=true on float data preserves all 9 distinct float values (not truncated to 0)
    let hist_counts = b1["histogram"][0].as_array().unwrap();
    let hist_vals = b1["histogram"][1].as_array().unwrap();
    assert_eq!(hist_vals.len(), 9, "T2 float categorical must preserve 9 distinct values, got {hist_vals:?}");
    assert_eq!(hist_counts.len(), 9);
    assert!((hist_vals[0].as_f64().unwrap() - 0.17).abs() < 1e-5);

    // --- T3: POST /cog/feature/32x32.npy matches GET /cog/bbox/.../32x32.npy identically ---
    let full_rect = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [ox, oy],
                [ox + 64.0 * px, oy],
                [ox + 64.0 * px, oy - 64.0 * px],
                [ox, oy - 64.0 * px],
                [ox, oy]
            ]]
        }
    });
    let bbox_str = format!("{},{},{},{}", ox, oy - 64.0 * px, ox + 64.0 * px, oy);
    let r_feat = client
        .post(format!("{base}/cog/feature/32x32.npy?url={utm_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&resampling=average&return_mask=false"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&full_rect).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_feat.status(), 200);
    let r_bbox = client
        .get(format!("{base}/cog/bbox/{bbox_str}/32x32.npy?url={utm_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&resampling=average&return_mask=false"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bbox.status(), 200);
    let v_feat = decode_npy_f32(&r_feat.bytes().await.unwrap());
    let v_bbox = decode_npy_f32(&r_bbox.bytes().await.unwrap());
    assert_eq!(v_feat.len(), v_bbox.len());
    let max_diff = v_feat
        .iter()
        .zip(v_bbox.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "T3 /feature vs /bbox max_diff={max_diff}");

    // --- T4: histogram & categories in coverage units even with area_weighted=true + category_areas ---
    let merc_poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 6_446_275.841_017_158],
                [16.0 * 180_000.0, 6_446_275.841_017_158],
                [16.0 * 180_000.0, 6_446_275.841_017_158 - 16.0 * 180_000.0],
                [0.0, 6_446_275.841_017_158 - 16.0 * 180_000.0],
                [0.0, 6_446_275.841_017_158]
            ]]
        }
    });
    let r_t4 = client
        .post(format!("{base}/cog/statistics?url={merc_url}&bidx=1&max_size=0&coord_crs=EPSG:3857&categorical=true&area_weighted=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&merc_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_t4.status(), 200);
    let j_t4: serde_json::Value = serde_json::from_slice(&r_t4.bytes().await.unwrap()).unwrap();
    let b1_t4 = &j_t4["properties"]["statistics"]["b1"];
    assert_eq!(b1_t4["categories"]["1"].as_f64().unwrap(), 128.0);
    assert_eq!(b1_t4["categories"]["2"].as_f64().unwrap(), 128.0);
    assert_eq!(b1_t4["histogram"], serde_json::json!([[128.0, 128.0], [1.0, 2.0]]));
    let a1 = b1_t4["category_areas"]["1"].as_f64().unwrap();
    let a2 = b1_t4["category_areas"]["2"].as_f64().unwrap();
    let total_area = b1_t4["area"].as_f64().unwrap();
    assert!(a2 > a1, "southern class 2 covers more ground area than northern class 1 in EPSG:3857");
    assert!(
        ((a1 + a2) - total_area).abs() / total_area < 1e-9,
        "category_areas must sum to total area"
    );
    assert!(b1_t4["mean"].as_f64().unwrap() > 1.55);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep16_u1_stac_post_expr_and_u2a_gcp_message() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep16_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Two aligned 64x64 UTM 10N single-band assets (`red` = 30.0, `green` = 10.0) in a STAC item
    let w = 64u32;
    let h = 64u32;
    let ox = 500_000.0_f64;
    let oy = 4_200_000.0_f64;
    let px = 30.0_f64;
    let red_path = tmp.join("red.tif");
    let green_path = tmp.join("green.tif");
    write_synthetic_f32_projected_geotiff(&red_path, w, h, 32610, ox, oy, px, &vec![30.0f32; 4096]);
    write_synthetic_f32_projected_geotiff(&green_path, w, h, 32610, ox, oy, px, &vec![10.0f32; 4096]);

    let stac_path = tmp.join("item.json");
    let stac = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "u1-test",
        "bbox": [-123.0, 37.9, -122.9, 38.0],
        "geometry": null,
        "properties": {},
        "assets": {
            "red": { "href": red_path.to_string_lossy(), "type": "image/tiff; application=geotiff" },
            "green": { "href": green_path.to_string_lossy(), "type": "image/tiff; application=geotiff" }
        }
    });
    std::fs::write(&stac_path, stac.to_string()).unwrap();

    // Synthetic GCP-only TIFF (ModelTiepointTag with 4 GCPs = 24 f64s, no ModelPixelScaleTag)
    let gcp_path = tmp.join("gcp_only.tif");
    {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let num_entries: u16 = 12;
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let tp_off = data_start;
        let gkd_off = tp_off + 24 * 8;
        let pix_off = gkd_off + 12 * 2;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, 8);
        push(&mut buf, 257, 4, 1, 8);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, 8);
        push(&mut buf, 279, 4, 1, 8 * 8 * 4);
        push(&mut buf, 339, 3, 1, 3);
        push(&mut buf, 33922, 12, 24, tp_off); // 4 GCPs
        push(&mut buf, 34735, 3, 12, gkd_off); // GeoKeyDirectory: EPSG:4326
        buf.extend_from_slice(&0u32.to_le_bytes());
        for gcp in [
            [0.0f64, 0.0, 0.0, -122.5, 37.9, 0.0],
            [8.0, 0.0, 0.0, -122.4, 37.9, 0.0],
            [0.0, 8.0, 0.0, -122.5, 37.8, 0.0],
            [8.0, 8.0, 0.0, -122.4, 37.8, 0.0],
        ] {
            for v in gcp {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for _ in 0..64 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        std::fs::write(&gcp_path, &buf).unwrap();
    }

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let stac_url = stac_path.to_string_lossy().to_string();
    let gcp_url = gcp_path.to_string_lossy().to_string();

    // U1: Tilted quad in pixel space -> exact shoelace area = 305.5 px (definitely not 4096 or 16M!)
    let pts = [(10.0, 10.0), (30.0, 12.0), (28.0, 27.0), (9.0, 25.0)];
    let mut ring: Vec<[f64; 2]> = pts.iter().map(|&(c, r)| [ox + c * px, oy - r * px]).collect();
    ring.push(ring[0]);
    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Polygon", "coordinates": [ring] }
    });
    // Exact shoelace area in pixel space:
    let mut area2 = 0.0_f64;
    for i in 0..pts.len() {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % pts.len()];
        area2 += x0 * y1 - x1 * y0;
    }
    let true_px = 0.5 * area2.abs();

    for (label, query, expected_mean) in [
        ("single-asset red/2", "assets=red&asset_as_band=true&expression=red/2", 15.0),
        ("two-asset red_b1-green_b1", "expression=(red_b1-green_b1)/(red_b1%2Bgreen_b1)", 0.5),
        ("two-asset asset_as_band", "assets=red&assets=green&asset_as_band=true&expression=(red-green)/(red%2Bgreen)", 0.5),
    ] {
        let r = client
            .post(format!("{base}/stac/statistics?url={stac_url}&{query}&max_size=0&coord_crs=EPSG:32610"))
            .header("content-type", "application/json")
            .body(serde_json::to_string(&poly).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "U1 {label}");
        let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
        let b1 = &j["properties"]["statistics"]["b1"];
        let cnt = b1["count"].as_f64().unwrap();
        let mean = b1["mean"].as_f64().unwrap();
        assert!(
            (cnt - true_px).abs() < 1e-4,
            "U1 {label}: expected count={true_px}, got {cnt}"
        );
        assert!(
            (mean - expected_mean).abs() < 1e-5,
            "U1 {label}: expected mean={expected_mean}, got {mean}"
        );
    }

    // U2(b): 4-GCP affine TIFF now succeeds with 200, gcp_residual_px ~ 0, and exact /cog/point!
    let r_gcp = client
        .get(format!("{base}/cog/info?url={gcp_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_gcp.status(), 200);
    let gcp_info: serde_json::Value = serde_json::from_slice(&r_gcp.bytes().await.unwrap()).unwrap();
    let res_px = gcp_info["gcp_residual_px"].as_f64().expect("gcp_residual_px present");
    assert!(res_px < 1e-9, "expected exact affine residual < 1e-9, got {res_px}");
    let r_pt = client
        .get(format!("{base}/cog/point/-122.45,37.85?url={gcp_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_pt.status(), 200);
    let pt_json: serde_json::Value = serde_json::from_slice(&r_pt.bytes().await.unwrap()).unwrap();
    assert_eq!(pt_json["values"][0].as_f64().unwrap(), 1.0);

    let _ = std::fs::remove_dir_all(&tmp);
}


#[tokio::test]
async fn upstream_http_faults_and_uri_schemes_return_semantic_statuses() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Spawn a minimal raw TCP mock origin supporting fault routes:
    // /e401, /e403, /e404, /e500, /e503, /norange, /trunc, /redirloop
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("/");
                let resp: Vec<u8> = if path.starts_with("/e401") {
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e403") {
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e404") {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e500") {
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e503") {
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/norange") {
                    // Origin ignores Range and advertises a 500 KiB 200 OK body.
                    b"HTTP/1.1 200 OK\r\nContent-Length: 512000\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/trunc") {
                    // Claims 4096 bytes in Content-Length, sends only 16 bytes, then closes.
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n0123456789abcdef".to_vec()
                } else if path.starts_with("/redirloop") {
                    format!(
                        "HTTP/1.1 302 Found\r\nLocation: http://{mock_addr}/redirloop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .into_bytes()
                } else {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                };
                let _ = sock.write_all(&resp).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    // Reserve a closed port for connection-refused testing.
    let closed_port = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap().port()
    };

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // V1: COG upstream statuses
    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e401/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let body = r.text().await.unwrap();
    assert!(body.contains("origin refused access") && body.contains("401"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e403/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let body = r.text().await.unwrap();
    assert!(body.contains("origin refused access") && body.contains("403"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e404/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e500/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(body.contains("500") && body.contains("/e500/a.tif"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e503/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(body.contains("503") && body.contains("/e503/a.tif"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/trunc/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/redirloop/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    let r = client
        .get(format!("{base}/cog/info?url=http://127.0.0.1:{closed_port}/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    // V2: Origin ignores Range header (200 OK with full file > 128 KiB) -> 502 with clear message
    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/norange/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(
        body.contains("origin does not support HTTP range requests"),
        "body={body}"
    );

    // V3: Unsupported URI schemes -> 400 naming the scheme and supported ones
    for route in ["/cog/info", "/stac/info", "/mosaicjson/info"] {
        let r = client
            .get(format!("{base}{route}?url=ftp://127.0.0.1/x.tif"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "route={route}");
        let body = r.text().await.unwrap();
        assert!(body.contains("ftp://") && body.contains("supported"), "route={route} body={body}");
    }

    // V1 on STAC & MosaicJSON routes too
    for route in ["/stac/info", "/mosaicjson/info"] {
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e403/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 403, "route={route}");
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e500/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 502, "route={route}");
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e404/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404, "route={route}");
    }

    mock_task.abort();
    srv_task.abort();
}


#[tokio::test]
async fn sweep17_v4_non_cog_ifd_seeking_and_v5_tight_block_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Build a 512x512 float32 tiled (256x256 blocks -> 2x2 = 4 blocks) GeoTIFF in non-COG layout:
    // header (8 B) -> 4 blocks of 256 KiB (1 MiB) -> 1 MiB padding -> IFD at offset > 2 MiB!
    let mut tif: Vec<u8> = Vec::new();
    tif.extend_from_slice(b"II");
    tif.extend_from_slice(&42u16.to_le_bytes());
    let first_ifd_patch_pos = tif.len();
    tif.extend_from_slice(&0u32.to_le_bytes()); // patched later

    let block_bytes = 256usize * 256 * 4; // 262,144 bytes per block
    let mut tile_offsets = [0u32; 4];
    let tile_counts = [block_bytes as u32; 4];
    for (b, off_slot) in tile_offsets.iter_mut().enumerate() {
        *off_slot = tif.len() as u32;
        let val = (b as f32 + 1.0) * 10.0;
        let vb = val.to_le_bytes();
        for _ in 0..(256 * 256) {
            tif.extend_from_slice(&vb);
        }
    }
    // Pad to > 2 MiB before writing IFD so a contiguous header read would have to fetch > 2 MiB.
    tif.resize(2 * 1024 * 1024 + 4096, 0u8);

    // Out-of-line tag payloads placed right next to the IFD.
    let offsets_pos = tif.len() as u32;
    for o in tile_offsets {
        tif.extend_from_slice(&o.to_le_bytes());
    }
    let counts_pos = tif.len() as u32;
    for c in tile_counts {
        tif.extend_from_slice(&c.to_le_bytes());
    }
    let scale_pos = tif.len() as u32;
    for v in [0.001f64, 0.001, 0.0] {
        tif.extend_from_slice(&v.to_le_bytes());
    }
    let tie_pos = tif.len() as u32;
    for v in [0.0f64, 0.0, 0.0, 10.0, 45.0, 0.0] {
        tif.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_pos = tif.len() as u32;
    // GeoKeyDirectory: WGS84 Geographic (EPSG:4326)
    for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
        tif.extend_from_slice(&k.to_le_bytes());
    }

    let ifd_off = tif.len() as u32;
    tif[first_ifd_patch_pos..first_ifd_patch_pos + 4].copy_from_slice(&ifd_off.to_le_bytes());

    let entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, 512),         // ImageWidth
        (257, 4, 1, 512),         // ImageLength
        (258, 3, 1, 32),          // BitsPerSample
        (259, 3, 1, 1),           // Compression = None
        (262, 3, 1, 1),           // Photometric = MinIsBlack
        (277, 3, 1, 1),           // SamplesPerPixel
        (284, 3, 1, 1),           // PlanarConfiguration = Chunky
        (322, 4, 1, 256),         // TileWidth
        (323, 4, 1, 256),         // TileLength
        (324, 4, 4, offsets_pos), // TileOffsets
        (325, 4, 4, counts_pos),  // TileByteCounts
        (339, 3, 1, 3),           // SampleFormat = IEEEFP
        (33550, 12, 3, scale_pos),
        (33922, 12, 6, tie_pos),
        (34735, 3, 12, gkd_pos),
    ];
    tif.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        tif.extend_from_slice(&tag.to_le_bytes());
        tif.extend_from_slice(&ty.to_le_bytes());
        tif.extend_from_slice(&count.to_le_bytes());
        tif.extend_from_slice(&val.to_le_bytes());
    }
    tif.extend_from_slice(&0u32.to_le_bytes()); // next_ifd = 0

    let tif_arc = Arc::new(tif);
    let req_count = Arc::new(AtomicUsize::new(0));
    let bytes_sent = Arc::new(AtomicUsize::new(0));

    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let tif_srv = tif_arc.clone();
    let rc_srv = req_count.clone();
    let bs_srv = bytes_sent.clone();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            let data = tif_srv.clone();
            let rc = rc_srv.clone();
            let bs = bs_srv.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let range_line = req
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                    .unwrap_or("");
                let total = data.len();
                let (s, e) = if let Some(spec) = range_line.split("bytes=").nth(1) {
                    let mut parts = spec.trim().split('-');
                    let s: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                    let e: usize = parts
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(total - 1)
                        .min(total - 1);
                    (s, e)
                } else {
                    (0, total - 1)
                };
                let slice = &data[s..=e];
                rc.fetch_add(1, Ordering::Relaxed);
                bs.fetch_add(slice.len(), Ordering::Relaxed);
                let hdr = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = sock.write_all(hdr.as_bytes()).await;
                let _ = sock.write_all(slice).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let url = format!("http://{mock_addr}/noncog.tif");

    // V4: Opening non-COG with IFD > 2 MiB should jump directly to IFD (~160 KiB total, not > 2 MiB)
    let r = client
        .get(format!("{base}/cog/info?url={url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let open_reqs = req_count.load(Ordering::Relaxed);
    let open_bytes = bytes_sent.load(Ordering::Relaxed);
    assert!(
        open_bytes < 256 * 1024,
        "V4 failed: non-COG open fetched {open_bytes} bytes in {open_reqs} reqs (expected < 256 KiB)"
    );

    // V5: A 1-block 256x256 native window (top-left block: lon 10.0..10.256, lat 44.744..45.0)
    // must fetch ONLY 1 block (262,144 bytes in 1 range request), not 2x2 blocks!
    req_count.store(0, Ordering::Relaxed);
    bytes_sent.store(0, Ordering::Relaxed);
    let r = client
        .get(format!(
            "{base}/cog/bbox/10.0,44.744,10.256,45.0/256x256.npy?url={url}&coord_crs=EPSG:4326&dst_crs=EPSG:4326"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let win_reqs = req_count.load(Ordering::Relaxed);
    let win_bytes = bytes_sent.load(Ordering::Relaxed);
    assert_eq!(win_reqs, 1, "V5 failed: expected 1 range request, got {win_reqs}");
    assert_eq!(
        win_bytes, block_bytes,
        "V5 failed: expected exactly 1 block ({block_bytes} bytes), got {win_bytes} bytes"
    );

    mock_task.abort();
    srv_task.abort();
}


#[tokio::test]
async fn sweep16_u3_ungeoreferenced_and_u2b_nonlinear_gcp_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_u3_u2b_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: write a tiny stripped float32 TIFF with optional ModelTiepointTag GCPs (no scale).
    let write_tiff = |path: &std::path::Path, w: u32, h: u32, gcps: Option<&[[f64; 6]]>, pixels: &[f32]| {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let has_gcp = gcps.is_some();
        let num_entries: u16 = if has_gcp { 12 } else { 10 };
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let tp_off = data_start;
        let tp_bytes = gcps.map_or(0, |g| (g.len() * 6 * 8) as u32);
        let gkd_off = tp_off + tp_bytes;
        let gkd_bytes = if has_gcp { 12 * 2 } else { 0 };
        let pix_off = gkd_off + gkd_bytes;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, w);
        push(&mut buf, 257, 4, 1, h);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, h);
        push(&mut buf, 279, 4, 1, w * h * 4);
        push(&mut buf, 339, 3, 1, 3);
        if let Some(g) = gcps {
            push(&mut buf, 33922, 12, (g.len() * 6) as u32, tp_off);
            push(&mut buf, 34735, 3, 12, gkd_off);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        if let Some(g) = gcps {
            for row in g {
                for v in row {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
                buf.extend_from_slice(&k.to_le_bytes());
            }
        }
        for &p in pixels {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        std::fs::write(path, &buf).unwrap();
    };

    // 1. Ungeoreferenced 32x32 TIFF (U3)
    let nogeo_path = tmp.join("nogeo.tif");
    let nogeo_px: Vec<f32> = (0..(32 * 32)).map(|i| (i % 32) as f32 * 10.0).collect();
    write_tiff(&nogeo_path, 32, 32, None, &nogeo_px);

    // 2. Nonlinear quadratic 9-GCP 64x64 TIFF (U2b order-2 polynomial)
    let gcp_nl_path = tmp.join("gcp_nonlinear.tif");
    let mut nl_gcps = Vec::new();
    for &row in &[0.0f64, 32.0, 64.0] {
        for &col in &[0.0f64, 32.0, 64.0] {
            let u = col / 64.0;
            let v = row / 64.0;
            let x = -122.5 + 0.25 * u + 0.01 * u * v;
            let y = 37.9 - 0.25 * v - 0.01 * u * u;
            nl_gcps.push([col, row, 0.0, x, y, 0.0]);
        }
    }
    let mut nl_px = vec![0.0f32; 64 * 64];
    for r in 0..64 {
        for c in 0..64 {
            nl_px[r * 64 + c] = (r * 100 + c) as f32;
        }
    }
    write_tiff(&gcp_nl_path, 64, 64, Some(&nl_gcps), &nl_px);

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let nogeo_url = nogeo_path.to_string_lossy().to_string();
    let nl_url = gcp_nl_path.to_string_lossy().to_string();

    // U3: /cog/info returns 200 with crs=null, bounds=null, width=32, height=32
    let r = client.get(format!("{base}/cog/info?url={nogeo_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let info: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert!(info["crs"].is_null(), "expected crs: null, got {info}");
    assert!(info["bounds"].is_null(), "expected bounds: null, got {info}");
    assert_eq!(info["width"], 32);
    assert_eq!(info["height"], 32);

    // U3: /cog/preview and GET /cog/statistics succeed in pixel space
    let r = client.get(format!("{base}/cog/preview.png?url={nogeo_url}&rescale=0,310")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = client.get(format!("{base}/cog/preview/16x16.png?url={nogeo_url}&rescale=0,310")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = client.get(format!("{base}/cog/statistics?url={nogeo_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let stats: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert!((stats["b1"]["mean"].as_f64().unwrap() - 155.0).abs() < 1e-4);

    // U3: Geographic routes return 400 Bad Request
    for path in [
        format!("/cog/bounds?url={nogeo_url}"),
        format!("/cog/point/0,0?url={nogeo_url}"),
        format!("/cog/tiles/WebMercatorQuad/0/0/0.png?url={nogeo_url}"),
        format!("/cog/bbox/0,0,1,1/16x16.png?url={nogeo_url}"),
        format!("/cog/WebMercatorQuad/tilejson.json?url={nogeo_url}"),
    ] {
        let r = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 400, "U3 expected 400 for {path}");
    }

    // U2(b): Nonlinear 9-GCP TIFF succeeds on /cog/info (with gcp_residual_px) and /cog/point
    let r = client.get(format!("{base}/cog/info?url={nl_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let nl_info: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    let res = nl_info["gcp_residual_px"].as_f64().expect("gcp_residual_px present");
    assert!(res < 0.05, "expected order-2 poly residual < 0.05 px, got {res}");

    // Query center pixel (col=32.5, row=32.5) mapped through the exact quadratic formula:
    let (u, v) = (32.5 / 64.0, 32.5 / 64.0);
    let qx = -122.5 + 0.25 * u + 0.01 * u * v;
    let qy = 37.9 - 0.25 * v - 0.01 * u * u;
    let r = client.get(format!("{base}/cog/point/{qx},{qy}?url={nl_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let pt: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(pt["values"][0].as_f64().unwrap(), (32 * 100 + 32) as f64);

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}


#[tokio::test]
async fn sweep18_w1_source_freshness_and_w2_referenced_asset_errors() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let tmp = std::env::temp_dir().join(format!("cogzilla_w1_w2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: build a stripped EPSG:4326 float32 GeoTIFF covering [-122.5..-122.244, 37.644..37.9]
    let build_geo_tiff = |w: u32, h: u32, fill_val: f32| -> Vec<u8> {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let num_entries: u16 = 13;
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let scale_off = data_start;
        let tp_off = scale_off + 24;
        let gkd_off = tp_off + 48;
        let pix_off = gkd_off + 24;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, w);
        push(&mut buf, 257, 4, 1, h);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, h);
        push(&mut buf, 279, 4, 1, w * h * 4);
        push(&mut buf, 339, 3, 1, 3);
        push(&mut buf, 33550, 12, 3, scale_off);
        push(&mut buf, 33922, 12, 6, tp_off);
        push(&mut buf, 34735, 3, 12, gkd_off);
        buf.extend_from_slice(&0u32.to_le_bytes());
        let px_deg = 0.256 / (w as f64);
        for v in [px_deg, px_deg, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, -122.5, 37.9, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        let vb = fill_val.to_le_bytes();
        for _ in 0..(w * h) {
            buf.extend_from_slice(&vb);
        }
        buf
    };

    // Mutable remote HTTP origin supporting Range, ETag, If-Match (412), and If-None-Match (304)
    let remote_state = Arc::new(std::sync::Mutex::new((
        "\"v1\"".to_string(),
        build_geo_tiff(64, 64, 111.0),
    )));
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let rs_srv = remote_state.clone();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            let rs = rs_srv.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let mut if_match = None;
                let mut if_none_match = None;
                let mut range_spec = None;
                for line in req.lines().skip(1) {
                    if let Some((k, v)) = line.split_once(':') {
                        let kl = k.trim().to_ascii_lowercase();
                        let vt = v.trim().to_string();
                        if kl == "if-match" {
                            if_match = Some(vt);
                        } else if kl == "if-none-match" {
                            if_none_match = Some(vt);
                        } else if kl == "range" {
                            range_spec = Some(vt);
                        }
                    }
                }
                let (etag, data) = rs.lock().unwrap().clone();
                if let Some(im) = if_match
                    && im != etag
                {
                    let hdr = format!(
                        "HTTP/1.1 412 Precondition Failed\r\nETag: {etag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(hdr.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
                if let Some(inm) = if_none_match
                    && inm == etag
                {
                    let hdr = format!(
                        "HTTP/1.1 304 Not Modified\r\nETag: {etag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(hdr.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
                let total = data.len();
                let (s, e) = if let Some(rs) = range_spec
                    && let Some(spec) = rs.strip_prefix("bytes=")
                {
                    let mut parts = spec.split('-');
                    let s: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                    let e: usize = parts
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(total - 1)
                        .min(total - 1);
                    (s, e)
                } else {
                    (0, total - 1)
                };
                let slice = &data[s..=e];
                let hdr = format!(
                    "HTTP/1.1 206 Partial Content\r\nETag: {etag}\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = sock.write_all(hdr.as_bytes()).await;
                let _ = sock.write_all(slice).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    // Use a store with remote_cache_ttl = 0 so we also test immediate conditional revalidation
    let mut store = cogzilla_io::CogStore::new(16 * 1024 * 1024);
    store.remote_cache_ttl = std::time::Duration::ZERO;
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(Arc::new(store), 4 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // --- W1 Local: overwrite (same size), overwrite (larger size), delete ---
    let local_path = tmp.join("mutable.tif");
    let local_url = local_path.to_string_lossy().to_string();
    std::fs::write(&local_path, build_geo_tiff(64, 64, 111.0)).unwrap();

    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 111.0);

    // Overwrite same size via atomic rename (changes inode + mtime even within same second)
    // Also exercise encoded TileCache hit invalidation on /cog/tiles!
    let tile_url = format!("{base}/cog/tiles/WebMercatorQuad/12/655/1583.npy?url={local_url}");
    let t1 = client.get(&tile_url).send().await.unwrap();
    assert_eq!(t1.status(), 200);
    let t1_bytes = t1.bytes().await.unwrap();
    let tmp_swap = tmp.join("mutable_v2.tif");
    std::fs::write(&tmp_swap, build_geo_tiff(64, 64, 222.0)).unwrap();
    std::fs::rename(&tmp_swap, &local_path).unwrap();
    let t2 = client.get(&tile_url).send().await.unwrap();
    assert_eq!(t2.status(), 200);
    assert_ne!(t2.bytes().await.unwrap(), t1_bytes, "W1 TileCache hit must invalidate after overwrite");
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 222.0, "W1 local overwrite failed");

    // Overwrite with larger 128x128 file (changes size)
    std::fs::write(&local_path, build_geo_tiff(128, 128, 333.0)).unwrap();
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 333.0, "W1 local resize failed");

    // Delete file -> 404
    std::fs::remove_file(&local_path).unwrap();
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 404, "W1 deleted local file must return 404");

    // --- W1 Remote: ETag replacement detected and re-read ---
    let remote_url = format!("http://{mock_addr}/live.tif");
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={remote_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 111.0);

    // Replace remote object with new ETag "v2" and value 222.0
    *remote_state.lock().unwrap() = ("\"v2\"".to_string(), build_geo_tiff(64, 64, 222.0));
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={remote_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 222.0, "W1 remote ETag replacement failed");

    // --- W2: Referenced STAC & MosaicJSON asset failures -> 502; partial mosaic -> X-Failed-Assets ---
    let gone_path = tmp.join("gone_asset.tif");
    let valid_path = tmp.join("valid_asset.tif");
    std::fs::write(&valid_path, build_geo_tiff(64, 64, 42.0)).unwrap();

    // STAC item referencing missing asset -> 502 naming gone_asset.tif
    let stac_path = tmp.join("w2_item.json");
    std::fs::write(
        &stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "w2-stac",
            "bbox": [-122.5, 37.644, -122.244, 37.9],
            "geometry": null,
            "properties": {},
            "assets": {
                "red": { "href": gone_path.to_string_lossy(), "type": "image/tiff; application=geotiff" }
            }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/stac/info?url={}&assets=red", stac_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 missing STAC asset must return 502");
    let body = r.text().await.unwrap();
    assert!(body.contains("gone_asset.tif"), "W2 502 body should name asset: {body}");

    // MosaicJSON where every asset is missing -> 502 naming gone_asset.tif
    let mj_all_gone = tmp.join("mj_all_gone.json");
    // Quadkey for WebMercatorQuad z=12, x=655, y=1583 -> "023010203333"
    let qk = "023010203333";
    std::fs::write(
        &mj_all_gone,
        serde_json::json!({
            "mosaicjson": "0.0.2",
            "minzoom": 10,
            "maxzoom": 14,
            "quadkey_zoom": 12,
            "bounds": [-122.5, 37.644, -122.244, 37.9],
            "tiles": { qk: [gone_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={}", mj_all_gone.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 all-missing mosaic assets must return 502");
    let body = r.text().await.unwrap();
    assert!(body.contains("gone_asset.tif"), "W2 mosaic 502 body should name asset: {body}");

    // MosaicJSON /point where every covering asset is missing -> 502 (not 404)
    // (-122.4, 37.75) lies inside WebMercatorQuad z=12, x=655, y=1583 (quadkey 023010203333)
    let r = client
        .get(format!("{base}/mosaicjson/point/-122.4,37.75?url={}", mj_all_gone.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 all-missing mosaicjson/point must return 502");

    // MosaicJSON with [gone_asset.tif, valid_asset.tif] -> 200 with X-Failed-Assets header!
    let mj_partial = tmp.join("mj_partial.json");
    std::fs::write(
        &mj_partial,
        serde_json::json!({
            "mosaicjson": "0.0.2",
            "minzoom": 10,
            "maxzoom": 14,
            "quadkey_zoom": 12,
            "bounds": [-122.5, 37.644, -122.244, 37.9],
            "tiles": { qk: [gone_path.to_string_lossy(), valid_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={}", mj_partial.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "W2 partial mosaic should render 200 from remaining asset");
    let failed_hdr = r
        .headers()
        .get("x-failed-assets")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        failed_hdr.contains("gone_asset.tif"),
        "W2 partial mosaic must report failed asset in X-Failed-Assets, got: '{failed_hdr}'"
    );

    mock_task.abort();
    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep23_cc1_null_geometry_and_indexed_fc_errors() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep23_cc1_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Synthetic 64x64 EPSG:32610 GeoTIFF at (500_000, 4_200_000), 30m px, filled with 42.0
    let cog_path = tmp.join("scene.tif");
    write_synthetic_f32_projected_geotiff(
        &cog_path,
        64,
        64,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![42.0f32; 64 * 64],
    );
    let cog_url = cog_path.to_string_lossy().to_string();

    let stac_path = tmp.join("item.json");
    std::fs::write(
        &stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "cc1-item",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.5, 37.5], [-122.5, 37.5], [-122.5, 38.5], [-123.5, 38.5], [-123.5, 37.5]]]
            },
            "bbox": [-123.5, 37.5, -122.5, 38.5],
            "properties": {"datetime": "2026-01-01T00:00:00Z"},
            "assets": {
                "data": {"href": cog_url, "type": "image/tiff; application=geotiff"}
            }
        })
        .to_string(),
    )
    .unwrap();
    let stac_url = stac_path.to_string_lossy().to_string();

    let mj_path = tmp.join("mosaic.json");
    std::fs::write(
        &mj_path,
        serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 8,
            "maxzoom": 12,
            "quadkey_zoom": 8,
            "bounds": [-124.0, 37.0, -122.0, 39.0],
            "tiles": { "02301020": [cog_url] }
        })
        .to_string(),
    )
    .unwrap();
    let mj_url = mj_path.to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(serve_listener(engine.clone(), listener));
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let poly_a = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[[500_300.0, 4_199_100.0], [500_900.0, 4_199_100.0], [500_900.0, 4_199_700.0], [500_300.0, 4_199_700.0], [500_300.0, 4_199_100.0]]]
    });
    let poly_c = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[[501_000.0, 4_198_500.0], [501_600.0, 4_198_500.0], [501_600.0, 4_199_000.0], [501_000.0, 4_199_000.0], [501_000.0, 4_198_500.0]]]
    });
    let fc_with_null = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "f0", "properties": {"name": "first"}, "geometry": poly_a},
            {"type": "Feature", "id": "f1-null", "properties": {"name": "middle"}, "geometry": null},
            {"type": "Feature", "id": "f2", "properties": {"name": "last"}, "geometry": poly_c}
        ]
    });
    let fc_malformed = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "ok-0", "properties": {}, "geometry": poly_a},
            {
                "type": "Feature",
                "id": "bad-parcel-42",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[[500_300.0, 4_199_100.0], [500_900.0, 4_199_100.0]]]}
            },
            {"type": "Feature", "id": "ok-2", "properties": {}, "geometry": poly_c}
        ]
    });

    let endpoints = [
        ("cog", format!("{base}/cog/statistics?url={cog_url}&coord_crs=EPSG:32610"), "b1"),
        (
            "stac",
            format!("{base}/stac/statistics?url={stac_url}&assets=data&coord_crs=EPSG:32610"),
            "b1",
        ),
        (
            "mosaicjson",
            format!("{base}/mosaicjson/statistics?url={mj_url}&coord_crs=EPSG:32610"),
            "b1",
        ),
    ];

    for (label, url, band_key) in &endpoints {
        // 1. 3-feature collection with null geometry in middle returns 3 features in order
        let resp = client
            .post(url)
            .header("content-type", "application/json")
            .body(fc_with_null.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{label}: null-geom FC must return 200");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let feats = body["features"].as_array().unwrap();
        assert_eq!(feats.len(), 3, "{label}: expected 3 features");
        assert_eq!(feats[0]["id"], "f0");
        assert_eq!(feats[1]["id"], "f1-null");
        assert_eq!(feats[2]["id"], "f2");
        assert!(
            feats[0]["properties"]["statistics"][band_key]["count"]
                .as_f64()
                .unwrap()
                > 0.0,
            "{label}: feature 0 must have positive count"
        );
        let mid_stats = &feats[1]["properties"]["statistics"][band_key];
        assert_eq!(
            mid_stats["count"].as_f64().unwrap(),
            0.0,
            "{label}: null-geometry feature must have count 0, got {mid_stats}"
        );
        assert!(
            mid_stats["mean"].is_null(),
            "{label}: null-geometry feature must have null mean, got {mid_stats}"
        );
        assert!(
            feats[2]["properties"]["statistics"][band_key]["count"]
                .as_f64()
                .unwrap()
                > 0.0,
            "{label}: feature 2 must have positive count"
        );

        // 2. Malformed second feature returns 400 naming features[1] and its id
        let bad = client
            .post(url)
            .header("content-type", "application/json")
            .body(fc_malformed.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 400, "{label}: malformed feature must return 400");
        let bad_text = bad.text().await.unwrap();
        assert!(
            bad_text.contains("features[1]") && bad_text.contains("bad-parcel-42"),
            "{label}: 400 detail must name features[1] and id, got: {bad_text}"
        );
    }

    // 3. STAC DEM algorithm + resized expression with null-geometry FC must succeed (200)
    for extra in ["&algorithm=hillshade", "&expression=data*2&width=32"] {
        let r = client
            .post(format!("{base}/stac/statistics?url={stac_url}&assets=data&coord_crs=EPSG:32610{extra}"))
            .header("content-type", "application/json")
            .body(fc_with_null.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "STAC {extra} with null-geom FC must return 200");
        let b: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        assert_eq!(b["features"][1]["properties"]["statistics"]["b1"]["count"].as_f64().unwrap(), 0.0);
    }

    // 4. MosaicJSON null-geometry with invalid bidx=99 must return 400 (not swallowed 200)
    let null_feat = serde_json::json!({"type": "Feature", "properties": {}, "geometry": null});
    let r_bad_bidx = client
        .post(format!("{base}/mosaicjson/statistics?url={mj_url}&bidx=99"))
        .header("content-type", "application/json")
        .body(null_feat.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_bad_bidx.status(), 400, "mosaicjson null-geom with bidx=99 must return 400");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit53_color_formula_with_custom_colormap_and_where_nan_zonal() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_audit53_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // Two 64x64 f32 GeoTIFFs in EPSG:32610 (10 m pixels, origin (500000, 4200640)):
    // a.tif = 10 + col, b.tif = NaN everywhere.
    let w = 64usize;
    let h = 64usize;
    let mut b1 = vec![0.0f32; w * h];
    for (i, px) in b1.iter_mut().enumerate() {
        *px = 10.0 + (i % 64) as f32;
    }
    let b2 = vec![f32::NAN; w * h];
    let a_path = tmp.join("a.tif");
    let b_path = tmp.join("b.tif");
    write_synthetic_f32_projected_geotiff(
        &a_path,
        w as u32,
        h as u32,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &b1,
    );
    write_synthetic_f32_projected_geotiff(
        &b_path,
        w as u32,
        h as u32,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &b2,
    );
    let stac_path = tmp.join("item.json");
    let stac_json = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "audit53",
        "geometry": null,
        "bbox": [-123.0, 37.9, -122.9, 38.0],
        "properties": {},
        "assets": {
            "a": { "href": a_path.to_string_lossy() },
            "b": { "href": b_path.to_string_lossy() }
        }
    });
    std::fs::write(&stac_path, stac_json.to_string()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let a_url = a_path.to_string_lossy().to_string();
    let b_url = b_path.to_string_lossy().to_string();
    let stac_url = stac_path.to_string_lossy().to_string();

    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_100.0, 4_200_100.0],
                [500_200.0, 4_200_100.0],
                [500_200.0, 4_200_200.0],
                [500_100.0, 4_200_200.0],
                [500_100.0, 4_200_100.0]
            ]]
        }
    });

    // 1a. Single-COG POST /cog/statistics with where(1, 42, b1) when untaken b1 is NaN must
    // evaluate the taken branch (42) instead of aborting on NaN.
    let r_cog = client
        .post(format!(
            "{base}/cog/statistics?url={b_url}&coord_crs=EPSG:32610&max_size=0&expression=where(1,42,b1)"
        ))
        .header("content-type", "application/json")
        .body(poly.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_cog.status(), 200);
    let cog_stats: serde_json::Value = serde_json::from_str(&r_cog.text().await.unwrap()).unwrap();
    let cog_b1 = &cog_stats["properties"]["statistics"]["b1"];
    assert_eq!(
        cog_b1["valid_pixels"].as_f64().unwrap(),
        100.0,
        "where(1,42,b1) on NaN band must keep all 100 valid pixels: {cog_b1}"
    );
    assert!((cog_b1["mean"].as_f64().unwrap() - 42.0).abs() < 1e-4);

    // 1b. Multi-COG POST /stac/statistics with where(a>0, a, b) when untaken b is NaN must keep
    // all 100 valid pixels from a.
    let r_stac = client
        .post(format!(
            "{base}/stac/statistics?url={stac_url}&assets=a&assets=b&coord_crs=EPSG:32610&max_size=0&expression=where(a%3E0,a,b)"
        ))
        .header("content-type", "application/json")
        .body(poly.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_stac.status(), 200);
    let stac_stats: serde_json::Value = serde_json::from_str(&r_stac.text().await.unwrap()).unwrap();
    let stac_b1 = &stac_stats["properties"]["statistics"]["b1"];
    assert_eq!(
        stac_b1["valid_pixels"].as_f64().unwrap(),
        100.0,
        "STAC where(a>0,a,b) must keep all 100 valid pixels even when untaken b is NaN: {stac_b1}"
    );
    // cols 10..20 have values 20..29 -> mean = 24.5
    assert!((stac_b1["mean"].as_f64().unwrap() - 24.5).abs() < 1e-4);

    // 2. ?color_formula= with custom JSON ?colormap= must actually transform the output pixels.
    let cmap = "%5B%5B%5B0%2C100%5D%2C%5B120%2C120%2C120%2C255%5D%5D%5D";
    let r_plain = client
        .get(format!(
            "{base}/cog/bbox/500000,4200000,500640,4200640/32x32.png?url={a_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&colormap={cmap}"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_plain.status(), 200);
    let png_plain = r_plain.bytes().await.unwrap();

    let r_cf = client
        .get(format!(
            "{base}/cog/bbox/500000,4200000,500640,4200640/32x32.png?url={a_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&colormap={cmap}&color_formula=Gamma+RGB+2.0"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_cf.status(), 200);
    let png_cf = r_cf.bytes().await.unwrap();
    assert_ne!(
        png_plain, png_cf,
        "color_formula must not be silently dropped when custom JSON colormap is active"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit55_admin_clear_cache_and_mosaic_tile_cache_freshness() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_audit55_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // 64x64 f32 GeoTIFF in EPSG:3857 covering WebMercator tile 2/1/1 (x in [-10018754..0], y in [0..10018754]).
    let w = 64u32;
    let h = 64u32;
    let tile_span = 20_037_508.342789244 / 2.0; // z=2 tile width in EPSG:3857
    let px = tile_span / (w as f64);
    let tif_path = tmp.join("fresh_mosaic.tif");
    let v1 = vec![10.0f32; (w * h) as usize];
    write_synthetic_f32_projected_geotiff(
        &tif_path,
        w,
        h,
        3857,
        -tile_span,
        tile_span,
        px,
        &v1,
    );

    // Enable encoded tile_cache (16 MiB) to exercise warm tile_cache freshness on /mosaic/tiles!
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(
        Arc::new(cogzilla_io::CogStore::new(16 * 1024 * 1024)),
        16 * 1024 * 1024,
    ));
    let engine_check = engine.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let tif_url = tif_path.to_string_lossy().to_string();

    // Warm /mosaic/tiles cache
    let r1 = client
        .get(format!("{base}/mosaic/tiles/2/1/1.png?url={tif_url}&rescale=0,100"))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);
    let b1 = r1.bytes().await.unwrap();

    // Overwrite local file with different values -> warm tile_cache MUST detect mtime/inode change!
    let tmp_swap = tmp.join("fresh_mosaic_v2.tif");
    let v2 = vec![90.0f32; (w * h) as usize];
    write_synthetic_f32_projected_geotiff(
        &tmp_swap,
        w,
        h,
        3857,
        -tile_span,
        tile_span,
        px,
        &v2,
    );
    std::fs::rename(&tmp_swap, &tif_path).unwrap();

    let r2 = client
        .get(format!("{base}/mosaic/tiles/2/1/1.png?url={tif_url}&rescale=0,100"))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let b2 = r2.bytes().await.unwrap();
    assert_ne!(
        b1, b2,
        "warm tile_cache on /mosaic/tiles must invalidate when backing file is replaced"
    );

    // POST /admin/clear-cache must bump invalidation_epoch (clearing datasets_by_uri)
    let ep_before = engine_check
        .store
        .invalidation_epoch
        .load(std::sync::atomic::Ordering::Acquire);
    let rc = client
        .post(format!("{base}/admin/clear-cache"))
        .send()
        .await
        .unwrap();
    assert_eq!(rc.status(), 200);
    assert!(
        engine_check
            .store
            .invalidation_epoch
            .load(std::sync::atomic::Ordering::Acquire)
            > ep_before,
        "POST /admin/clear-cache must clear CogStore open URIs and bump invalidation_epoch"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep26_dd1_antimeridian_bbox_and_dd2_partial_polygon_valid_percent() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep26_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // --- DD1 fixture: EPSG:32701 (UTM 1S, central meridian 177W) spanning ~179E..176W, 16S..18S ---
    // At 179E, 16S -> UTM 1S easting ~ 72_000, northing ~ 8_230_000.
    // 500 m pixels x 1024 cols = 512 km (~4.8 deg) -> crosses 180 deg into 176W.
    let am_w = 1024u32;
    let am_h = 512u32;
    let mut am_px = Vec::with_capacity((am_w * am_h) as usize);
    for _r in 0..am_h {
        for c in 0..am_w {
            am_px.push((c + 1) as f32);
        }
    }
    let am_path = tmp.join("am32701.tif");
    write_synthetic_f32_projected_geotiff(
        &am_path,
        am_w,
        am_h,
        32701,
        70_000.0,
        8_230_000.0,
        500.0,
        &am_px,
    );

    // --- DD2 fixture: 64x64 EPSG:32610 (10 m pixels, x in 500000..500640, y in 4200000..4200640) ---
    let z_path = tmp.join("dd2_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &z_path,
        64,
        64,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &vec![42.0f32; 64 * 64],
    );

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let am_url = am_path.to_string_lossy().to_string();
    let z_url = z_path.to_string_lossy().to_string();

    // DD1a: /cog/bbox/{west>east}.npy must match {west,south,east+360,north}.npy byte-for-byte
    let r_wrap = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_wrap.status(), 200, "DD1: geographic west>east bbox must succeed");
    let b_wrap = r_wrap.bytes().await.unwrap();
    let r_unwrap = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,181.5,-16.5.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_unwrap.status(), 200);
    let b_unwrap = r_unwrap.bytes().await.unwrap();
    assert_eq!(b_wrap, b_unwrap, "DD1: wrapped and +360 unwrapped /cog/bbox must match identically");

    // DD1b: /cog/bbox/{west>east}/200x100.npy must match {west,south,east+360,north}/200x100.npy
    let r_wrap_wh = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5/200x100.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_wrap_wh.status(), 200);
    let b_wrap_wh = r_wrap_wh.bytes().await.unwrap();
    let r_unwrap_wh = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,181.5,-16.5/200x100.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_unwrap_wh.status(), 200);
    let b_unwrap_wh = r_unwrap_wh.bytes().await.unwrap();
    assert_eq!(b_wrap_wh, b_unwrap_wh, "DD1: wrapped and +360 /cog/bbox/200x100 must match identically");

    // DD1c: /cog/map?bbox=179.5,-17.5,-178.5,-16.5 must match 181.5 form
    let r_map_wrap = client
        .get(format!(
            "{base}/cog/map?url={am_url}&bbox=179.5,-17.5,-178.5,-16.5&width=64&height=32&format=npy"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_map_wrap.status(), 200);
    let b_map_wrap = r_map_wrap.bytes().await.unwrap();
    let r_map_unwrap = client
        .get(format!(
            "{base}/cog/map?url={am_url}&bbox=179.5,-17.5,181.5,-16.5&width=64&height=32&format=npy"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_map_unwrap.status(), 200);
    assert_eq!(b_map_wrap, r_map_unwrap.bytes().await.unwrap());

    // DD1d: projected coord_crs with minx > maxx must return 400 with one clear message
    for path in [
        format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5.npy?url={am_url}&coord_crs=EPSG:3857"),
        format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5/200x100.npy?url={am_url}&coord_crs=EPSG:3857"),
    ] {
        let r_proj = client.get(&path).send().await.unwrap();
        assert_eq!(r_proj.status(), 400, "projected minx>maxx must return 400 for {path}");
        let msg = r_proj.text().await.unwrap();
        assert!(
            msg.contains("minx") && msg.contains("maxx"),
            "projected minx>maxx 400 must clearly name minx and maxx, got: {msg}"
        );
    }

    // DD2a: fully inside polygon stays at valid_percent == 100.0, masked_pixels == 0.0
    let poly_inside = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_100.0, 4_200_100.0],
                [500_300.0, 4_200_100.0],
                [500_300.0, 4_200_300.0],
                [500_100.0, 4_200_300.0],
                [500_100.0, 4_200_100.0]
            ]]
        }
    });
    let r_in = client
        .post(format!("{base}/cog/statistics?url={z_url}&coord_crs=EPSG:32610&max_size=0"))
        .header("content-type", "application/json")
        .body(poly_inside.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_in.status(), 200);
    let j_in: serde_json::Value = serde_json::from_str(&r_in.text().await.unwrap()).unwrap();
    let b_in = &j_in["properties"]["statistics"]["b1"];
    assert_eq!(b_in["valid_percent"].as_f64().unwrap(), 100.0);
    assert_eq!(b_in["masked_pixels"].as_f64().unwrap(), 0.0);

    // DD2b: half-inside, half-off-file polygon (40x20 = 800 px, 400 px inside file):
    // native path must report valid_percent == 50.0 and masked_pixels == 400.0 (matching width/height).
    let poly_half = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_440.0, 4_200_100.0],
                [500_840.0, 4_200_100.0],
                [500_840.0, 4_200_300.0],
                [500_440.0, 4_200_300.0],
                [500_440.0, 4_200_100.0]
            ]]
        }
    });
    let r_half = client
        .post(format!("{base}/cog/statistics?url={z_url}&coord_crs=EPSG:32610&max_size=0"))
        .header("content-type", "application/json")
        .body(poly_half.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_half.status(), 200);
    let j_half: serde_json::Value = serde_json::from_str(&r_half.text().await.unwrap()).unwrap();
    let b_half = &j_half["properties"]["statistics"]["b1"];
    assert!(
        (b_half["count"].as_f64().unwrap() - 400.0).abs() < 1e-4,
        "expected count 400, got {b_half}"
    );
    assert!(
        (b_half["valid_percent"].as_f64().unwrap() - 50.0).abs() < 1e-4,
        "DD2: expected valid_percent 50.0 for half-off-file polygon, got {b_half}"
    );
    assert!(
        (b_half["masked_pixels"].as_f64().unwrap() - 400.0).abs() < 1e-4,
        "DD2: expected masked_pixels 400.0 for half-off-file polygon, got {b_half}"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

fn write_synthetic_u16_2band_4326_geotiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    bounds: [f64; 4], // [west, south, east, north]
    band1_val: u16,
    band2_val: u16,
    hole: Option<&str>, // "left" | "top"
) {
    let mut pixels = Vec::<u16>::with_capacity((width * height * 2) as usize);
    for row in 0..height {
        for col in 0..width {
            let masked = match hole {
                Some("left") => col < width / 2,
                Some("top") => row < height / 2,
                _ => false,
            };
            if masked {
                pixels.push(0);
                pixels.push(0);
            } else {
                pixels.push(band1_val);
                pixels.push(band2_val);
            }
        }
    }
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());

    let num_entries: u16 = 14;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let data_start = 8 + ifd_size;

    let scale_off = data_start as u32;
    let tiepoint_off = scale_off + 24;
    let geokey_off = tiepoint_off + 48;
    let pixel_off = geokey_off + 24;
    let pixel_bytes = (pixels.len() * 2) as u32;

    let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    };

    buf.extend_from_slice(&num_entries.to_le_bytes());
    push_entry(&mut buf, 256, 4, 1, width); // ImageWidth
    push_entry(&mut buf, 257, 4, 1, height); // ImageLength
    // BitsPerSample = [16, 16] packed inline in 4 bytes
    push_entry(&mut buf, 258, 3, 2, 16u32 | (16u32 << 16));
    push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
    push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
    push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
    push_entry(&mut buf, 277, 3, 1, 2); // SamplesPerPixel = 2
    push_entry(&mut buf, 278, 4, 1, height); // RowsPerStrip
    push_entry(&mut buf, 279, 4, 1, pixel_bytes); // StripByteCounts
    push_entry(&mut buf, 284, 3, 1, 1); // PlanarConfiguration = Chunky
    push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
    push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
    push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
    // GDAL_NODATA = "0\0" (2 bytes inline in 4-byte value field)
    push_entry(&mut buf, 42113, 2, 2, u32::from_le_bytes([b'0', 0, 0, 0]));
    buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0

    let sx = (bounds[2] - bounds[0]) / width as f64;
    let sy = (bounds[3] - bounds[1]) / height as f64;
    assert_eq!(buf.len(), scale_off as usize);
    for v in [sx, sy, 0.0f64] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), tiepoint_off as usize);
    for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), geokey_off as usize);
    // GTModelTypeGeoKey(1024)=2 (Geographic), GeographicTypeGeoKey(2048)=4326
    for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), pixel_off as usize);
    for &p in &pixels {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &buf).unwrap();
}

/// Parity Sweep #30 FF1:
/// - /mosaicjson/bbox honours all 9 pixel_selection methods and returns X-Assets
/// - GET & POST /mosaicjson/statistics mosaic all intersecting assets through Fold,
///   honour all 9 pixel_selection methods, and return X-Assets
/// - Unsupported routes refuse ?pixel_selection= with 400 (while bogus values stay 422)
#[tokio::test]
async fn sweep30_ff1_mosaic_bbox_statistics_pixel_selection_and_400_refusal() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep30_ff1_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // WebMercator tile 10/163/395 WGS84 bounds (quadkey "0230102033")
    let bounds = [-122.6953125, 37.71859032558816, -122.34375, 37.996162679728116];
    let a_path = tmp.join("a.tif");
    let b_path = tmp.join("b.tif");
    let c_path = tmp.join("c.tif");
    write_synthetic_u16_2band_4326_geotiff(&a_path, 64, 64, bounds, 10, 300, Some("left"));
    write_synthetic_u16_2band_4326_geotiff(&b_path, 64, 64, bounds, 20, 100, None);
    write_synthetic_u16_2band_4326_geotiff(&c_path, 64, 64, bounds, 30, 200, Some("top"));

    let mj_path = tmp.join("m.json");
    let mj = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 8,
        "maxzoom": 12,
        "quadkey_zoom": 10,
        "bounds": bounds,
        "center": [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, 10],
        "tiles": {
            "0230102033": [
                a_path.to_str().unwrap(),
                b_path.to_str().unwrap(),
                c_path.to_str().unwrap()
            ]
        }
    });
    std::fs::write(&mj_path, mj.to_string()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let mp = mj_path.to_string_lossy().to_string();
    let ap = a_path.to_string_lossy().to_string();
    let bbox_str = format!("{},{},{},{}", bounds[0], bounds[1], bounds[2], bounds[3]);

    // Expected [TL, TR, BL, BR] for band 1, plus (expected_bands, min, max, mean) for stats
    // Over 64x64 = 4096 pixels with equal 32x32 quadrants.
    let cases = [
        ("first",        [20.0f32, 10.0, 20.0, 10.0], 2usize, 10.0f64, 20.0f64, 15.0f64),
        ("highest",      [20.0, 20.0, 30.0, 30.0], 2, 20.0, 30.0, 25.0),
        ("lowest",       [20.0, 10.0, 20.0, 10.0], 2, 10.0, 20.0, 15.0),
        ("mean",         [20.0, 15.0, 25.0, 20.0], 2, 15.0, 25.0, 20.0),
        ("median",       [20.0, 15.0, 25.0, 20.0], 2, 15.0, 25.0, 20.0),
        ("stdev",        [0.0,  5.0,  5.0,  8.164966], 2, 0.0, 8.164966, 4.541241),
        ("count",        [1.0,  2.0,  2.0,  3.0],  1, 1.0,  3.0,  2.0),
        ("lastbandhigh", [20.0, 10.0, 30.0, 10.0], 1, 10.0, 30.0, 17.5),
        ("lastbandlow",  [20.0, 20.0, 20.0, 20.0], 1, 20.0, 20.0, 20.0),
    ];

    // 1. FF1(a): /mosaicjson/bbox/{bbox}/64x64.npy across all 9 methods + X-Assets
    for (ps, expected_q, expected_nb, _, _, _) in cases {
        let r = client
            .get(format!(
                "{base}/mosaicjson/bbox/{bbox_str}/64x64.npy?url={mp}&pixel_selection={ps}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "bbox {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "bbox {ps} missing X-Assets: {x_assets:?}"
        );
        let body = r.bytes().await.unwrap();
        let vals = decode_npy_f32(&body);
        assert_eq!(
            vals.len(),
            (expected_nb + 1) * 64 * 64,
            "bbox {ps} unexpected npy plane count (data+mask)"
        );
        let b1 = &vals[..64 * 64];
        let got_q = [
            b1[16 * 64 + 16],
            b1[16 * 64 + 48],
            b1[48 * 64 + 16],
            b1[48 * 64 + 48],
        ];
        for (i, (&g, &e)) in got_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "bbox {ps} quadrant {i}: got {got_q:?}, expected {expected_q:?}"
            );
        }
    }

    // 2. FF1(b): GET /mosaicjson/statistics across all 9 methods + X-Assets
    for (ps, _, expected_nb, exp_min, exp_max, exp_mean) in cases {
        let r = client
            .get(format!(
                "{base}/mosaicjson/statistics?url={mp}&pixel_selection={ps}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "GET statistics {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "GET statistics {ps} missing X-Assets: {x_assets:?}"
        );
        if ps != "first" {
            assert!(
                x_assets.contains("c.tif"),
                "GET statistics {ps} should include c.tif in X-Assets: {x_assets:?}"
            );
        }
        let j: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        let obj = j.as_object().unwrap();
        assert_eq!(
            obj.len(),
            expected_nb,
            "GET statistics {ps}: expected {expected_nb} bands, got {j}"
        );
        let b1 = &obj["b1"];
        assert!(
            (b1["count"].as_f64().unwrap() - 4096.0).abs() < 1e-4,
            "GET statistics {ps}: expected full mosaic count 4096, got {b1}"
        );
        assert!(
            (b1["min"].as_f64().unwrap() - exp_min).abs() < 1e-3,
            "GET statistics {ps}: min got {}, expected {exp_min}",
            b1["min"]
        );
        assert!(
            (b1["max"].as_f64().unwrap() - exp_max).abs() < 1e-3,
            "GET statistics {ps}: max got {}, expected {exp_max}",
            b1["max"]
        );
        assert!(
            (b1["mean"].as_f64().unwrap() - exp_mean).abs() < 1e-3,
            "GET statistics {ps}: mean got {}, expected {exp_mean}",
            b1["mean"]
        );
    }

    // 3. FF1(b): POST /mosaicjson/statistics with polygon covering the mosaic
    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]]
            ]]
        }
    });
    for (ps, _, expected_nb, exp_min, exp_max, exp_mean) in cases {
        let r = client
            .post(format!(
                "{base}/mosaicjson/statistics?url={mp}&pixel_selection={ps}"
            ))
            .header("content-type", "application/json")
            .body(poly.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "POST statistics {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "POST statistics {ps} missing X-Assets: {x_assets:?}"
        );
        let j: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        let stats = j["properties"]["statistics"].as_object().unwrap();
        assert_eq!(
            stats.len(),
            expected_nb,
            "POST statistics {ps}: expected {expected_nb} bands, got {j}"
        );
        let b1 = &stats["b1"];
        assert!(
            (b1["count"].as_f64().unwrap() - 4096.0).abs() < 1e-4,
            "POST statistics {ps}: expected full mosaic count 4096, got {b1}"
        );
        assert!(
            (b1["min"].as_f64().unwrap() - exp_min).abs() < 1e-3,
            "POST statistics {ps}: min got {}, expected {exp_min}",
            b1["min"]
        );
        assert!(
            (b1["max"].as_f64().unwrap() - exp_max).abs() < 1e-3,
            "POST statistics {ps}: max got {}, expected {exp_max}",
            b1["max"]
        );
        assert!(
            (b1["mean"].as_f64().unwrap() - exp_mean).abs() < 1e-3,
            "POST statistics {ps}: mean got {}, expected {exp_mean}",
            b1["mean"]
        );
    }

    // 4. Unsupported routes refuse ?pixel_selection= with 400 (bogus value stays 422)
    let cx = (bounds[0] + bounds[2]) / 2.0 + 0.01;
    let cy = (bounds[1] + bounds[3]) / 2.0 - 0.01;
    for unsupported in [
        format!("{base}/mosaicjson/point/{cx},{cy}?url={mp}&pixel_selection=highest"),
        format!("{base}/mosaicjson/info?url={mp}&pixel_selection=highest"),
        format!("{base}/cog/bbox/{bbox_str}/32x32.npy?url={ap}&pixel_selection=highest"),
        format!("{base}/cog/statistics?url={ap}&pixel_selection=highest"),
    ] {
        let r = client.get(&unsupported).send().await.unwrap();
        assert_eq!(
            r.status(),
            400,
            "expected 400 for unsupported pixel_selection on {unsupported}, got {}",
            r.status()
        );
    }
    let r_bogus = client
        .get(format!(
            "{base}/mosaicjson/bbox/{bbox_str}/32x32.npy?url={mp}&pixel_selection=bogus"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bogus.status(), 422, "bogus pixel_selection must stay 422");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}
