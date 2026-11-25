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
        ("bad tilesize", format!("{tile}&tilesize=7")),
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

            // Empty /mosaicjson/tiles returns 204 No Content (matching TiTiler 2.4 EmptyMosaicError)
            let mj_empty_buf = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/0/0.png?url={mj_url}&buffer=8"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_empty_buf.status(), 204);

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
            assert_eq!(am_resp.status(), 200, "antimeridian /stac/map must not panic");
            let am_bytes = am_resp.bytes().await.unwrap();
            assert_eq!(&am_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            let am_w = u32::from_be_bytes([am_bytes[16], am_bytes[17], am_bytes[18], am_bytes[19]]);
            let am_h = u32::from_be_bytes([am_bytes[20], am_bytes[21], am_bytes[22], am_bytes[23]]);
            assert_eq!((am_w, am_h), (128, 102));

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
        }
    }
}
