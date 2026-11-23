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
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });

    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    async fn get_json(client: &reqwest::Client, url: String) -> serde_json::Value {
        let resp = client.get(url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap()
    }

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
}
