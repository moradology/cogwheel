//! Integration tests for `/zarr/*` HTTP metadata routes (#372), error status table (#388),
//! and pre-fetch safety caps (#386).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use cogzilla_server::engine::TileEngine;
use cogzilla_server::http::handle_request;
use http_body_util::{BodyExt, Full};
use hyper::{Request, StatusCode};
use serde_json::Value;

fn zarr_fixtures_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benches/fixtures/zarr");
    dir.is_dir().then_some(dir)
}

fn test_engine() -> Arc<TileEngine> {
    Arc::new(TileEngine::new(64 * 1024 * 1024))
}

async fn get_json(engine: &Arc<TileEngine>, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(Arc::clone(engine), req).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let val: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, val)
}

#[tokio::test]
async fn zarr_metadata_routes_answer_for_v2_v3_sharded_and_3d_time() {
    let Some(dir) = zarr_fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    let engine = test_engine();

    // Test v2 consolidated, v3 consolidated, and v3 sharded stores
    for (store_name, var_name, expected_ver, expected_sharded) in [
        ("v2_consolidated.zarr", "temp", 2, false),
        ("v3_consolidated.zarr", "temp", 3, false),
        ("v3_sharded.zarr", "data", 3, true),
    ] {
        let url = dir.join(store_name).to_string_lossy().to_string();

        // 1. /zarr/info
        let (st, info) = get_json(&engine, &format!("/zarr/info?url={url}")).await;
        assert_eq!(st, StatusCode::OK, "info failed for {store_name}: {info}");
        assert_eq!(info["variable"], var_name);
        assert_eq!(info["width"], 128);
        assert_eq!(info["height"], 128);
        assert_eq!(info["dtype"], "float32");

        // 2. /zarr/info.geojson
        let (st, gj) = get_json(&engine, &format!("/zarr/info.geojson?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(gj["type"], "Feature");
        assert_eq!(gj["geometry"]["type"], "Polygon");
        assert_eq!(gj["properties"]["variable"], var_name);

        // 3. /zarr/bounds
        let (st, b) = get_json(&engine, &format!("/zarr/bounds?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(b["bounds"].as_array().unwrap().len(), 4);

        // 4. /zarr/validate
        let (st, val) = get_json(&engine, &format!("/zarr/validate?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(val["valid"], true);
        assert_eq!(val["version"], expected_ver);
        assert_eq!(val["consolidated"], true);
        assert_eq!(val["sharded"], expected_sharded);
        assert_eq!(val["regular_grid"], true);

        // 5. /zarr/dataset/ & /zarr/dataset/keys
        let (st, keys) = get_json(&engine, &format!("/zarr/dataset/keys?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert!(keys.as_array().unwrap().contains(&Value::String(var_name.into())));

        // 6. /zarr/dataset/dimensions
        let (st, dims) = get_json(&engine, &format!("/zarr/dataset/dimensions?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(dims["x"], 128);
        assert_eq!(dims["y"], 128);

        // 7. /zarr/dataset/dict
        let (st, dict) = get_json(&engine, &format!("/zarr/dataset/dict?url={url}")).await;
        assert_eq!(st, StatusCode::OK);
        assert!(dict["data_vars"].get(var_name).is_some());
    }

    // 8. /zarr/dataset/coordinates/time & show_times=true on 3D noleap store
    let t_url = dir.join("time_3d_noleap.zarr").to_string_lossy().to_string();
    let (st, t_coord) = get_json(
        &engine,
        &format!("/zarr/dataset/coordinates/time?url={t_url}&decode_times=true"),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let times = t_coord["data"].as_array().unwrap();
    assert_eq!(times[0], "2000-01-01T00:00:00");
    assert_eq!(times[2], "2000-03-01T00:00:00");

    let (st, t_raw) = get_json(
        &engine,
        &format!("/zarr/dataset/coordinates/time?url={t_url}&decode_times=false"),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(t_raw["data"][2], 59.0);

    // 9. Irregular grid refused with 400 naming dimension & index
    let irr_url = dir.join("irregular_coords.zarr").to_string_lossy().to_string();
    let (st, err) = get_json(&engine, &format!("/zarr/info?url={irr_url}")).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let detail = err["detail"].as_str().unwrap();
    assert!(detail.contains("\"x\"") && detail.contains("index 2"), "{detail}");

    // 10. Non-existent store returns 404
    let (st, _) = get_json(&engine, "/zarr/info?url=/nonexistent/store.zarr").await;
    assert_eq!(st, StatusCode::NOT_FOUND);
}

async fn get_bytes(
    engine: &Arc<TileEngine>,
    uri: &str,
    if_none_match: Option<&str>,
) -> (StatusCode, Option<String>, Bytes) {
    let mut b = Request::builder().method("GET").uri(uri);
    if let Some(inm) = if_none_match {
        b = b.header("If-None-Match", inm);
    }
    let req = b.body(Full::new(Bytes::new())).unwrap();
    let resp = handle_request(Arc::clone(engine), req).await.unwrap();
    let status = resp.status();
    let etag = resp
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, etag, body)
}

#[tokio::test]
async fn zarr_raster_routes_and_geotiff_twin_http_tile_parity() {
    let Some(dir) = zarr_fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    let engine = test_engine();
    let zarr_url = dir.join("geotiff_twin.zarr").to_string_lossy().to_string();
    let tif_url = dir.join("geotiff_twin.tif").to_string_lossy().to_string();

    // 1. TileJSON, Map HTML, WMTS, and OGC Tilesets
    let (st, tj) = get_json(
        &engine,
        &format!("/zarr/WebMercatorQuad/tilejson.json?url={zarr_url}"),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "tilejson failed: {tj}");
    assert_eq!(tj["tilejson"], "3.0.0");
    assert!(tj["tiles"][0].as_str().unwrap().contains("/zarr/tiles/WebMercatorQuad/"));

    let (st, _, html_body) = get_bytes(
        &engine,
        &format!("/zarr/WebMercatorQuad/map.html?url={zarr_url}"),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert!(std::str::from_utf8(&html_body).unwrap().contains("tilejson.json"));

    let (st, _, wmts_body) = get_bytes(
        &engine,
        &format!("/zarr/WebMercatorQuad/WMTSCapabilities.xml?url={zarr_url}"),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert!(std::str::from_utf8(&wmts_body).unwrap().contains("Capabilities"));

    let (st, tlist) = get_json(&engine, &format!("/zarr/tiles?url={zarr_url}")).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(tlist["tilesets"].as_array().unwrap().len(), 13);

    // 2. HTTP tile parity: /zarr/tiles/WebMercatorQuad/10/163/395.png vs /cog/tiles/...
    let zarr_tile_uri = format!(
        "/zarr/tiles/WebMercatorQuad/10/163/395.png?url={zarr_url}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
    );
    let cog_tile_uri = format!(
        "/cog/tiles/WebMercatorQuad/10/163/395.png?url={tif_url}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
    );
    let (z_st, z_etag, z_png) = get_bytes(&engine, &zarr_tile_uri, None).await;
    let (c_st, _, c_png) = get_bytes(&engine, &cog_tile_uri, None).await;
    assert_eq!(z_st, StatusCode::OK);
    assert_eq!(c_st, StatusCode::OK);
    assert_eq!(z_png, c_png, "HTTP /zarr/tiles PNG must be byte-identical to /cog/tiles PNG");

    // 3. ETag + If-None-Match -> 304 Not Modified
    let etag = z_etag.expect("expected ETag header on /zarr/tiles");
    let (st304, _, body304) = get_bytes(&engine, &zarr_tile_uri, Some(&etag)).await;
    assert_eq!(st304, StatusCode::NOT_MODIFIED);
    assert!(body304.is_empty());

    // 4. Out-of-bounds tile returns 404
    let (st404, _, _) = get_bytes(
        &engine,
        &format!("/zarr/tiles/WebMercatorQuad/10/0/0.png?url={zarr_url}"),
        None,
    )
    .await;
    assert_eq!(st404, StatusCode::NOT_FOUND);

    // 5. Foreign TMS (WorldCRS84Quad) works too
    let (st_crs84, _, body_crs84) = get_bytes(
        &engine,
        &format!("/zarr/tiles/WorldCRS84Quad/8/81/74.png?url={zarr_url}&rescale=100,900"),
        None,
    )
    .await;
    assert_eq!(st_crs84, StatusCode::OK);
    assert!(!body_crs84.is_empty());

    // 6. /zarr/point/{lon},{lat} matches /cog/point/{lon},{lat}
    let (zp_st, zp) = get_json(&engine, &format!("/zarr/point/-122.2,37.8?url={zarr_url}")).await;
    let (cp_st, cp) = get_json(&engine, &format!("/cog/point/-122.2,37.8?url={tif_url}")).await;
    assert_eq!(zp_st, StatusCode::OK);
    assert_eq!(cp_st, StatusCode::OK);
    let zv = zp["values"][0].as_f64().unwrap();
    let cv = cp["values"][0].as_f64().unwrap();
    assert!((zv - cv).abs() < 1e-4, "point mismatch: zarr={zv} cog={cv}");

    // 7. /zarr/preview and /zarr/bbox
    let (st_prev, _, prev_png) = get_bytes(
        &engine,
        &format!("/zarr/preview/128x128.png?url={zarr_url}&rescale=100,900"),
        None,
    )
    .await;
    assert_eq!(st_prev, StatusCode::OK);
    assert_eq!(&prev_png[0..4], b"\x89PNG");

    let (st_bbox, _, bbox_png) = get_bytes(
        &engine,
        &format!("/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.png?url={zarr_url}&rescale=100,900"),
        None,
    )
    .await;
    assert_eq!(st_bbox, StatusCode::OK);
    assert_eq!(&bbox_png[0..4], b"\x89PNG");

    // 8. GET /zarr/statistics matches GET /cog/statistics
    let (zs_st, zstats) = get_json(&engine, &format!("/zarr/statistics?url={zarr_url}")).await;
    let (cs_st, cstats) = get_json(&engine, &format!("/cog/statistics?url={tif_url}&max_size=0")).await;
    assert_eq!(zs_st, StatusCode::OK);
    assert_eq!(cs_st, StatusCode::OK);
    assert_eq!(zstats["b1"]["count"], cstats["b1"]["count"]);
    assert!((zstats["b1"]["mean"].as_f64().unwrap() - cstats["b1"]["mean"].as_f64().unwrap()).abs() < 1e-4);

    // 9. POST /zarr/statistics matches POST /cog/statistics on the exact same polygon (#374)
    let poly = serde_json::json!({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.4, 37.6],
                [-122.0, 37.6],
                [-122.0, 37.9],
                [-122.4, 37.9],
                [-122.4, 37.6]
            ]]
        },
        "properties": {}
    });
    let poly_bytes = Bytes::from(serde_json::to_vec(&poly).unwrap());

    let post_json = |uri: String, body: Bytes| {
        let eng = Arc::clone(&engine);
        async move {
            let req = Request::builder()
                .method("POST")
                .uri(&uri)
                .header("Content-Type", "application/json")
                .body(Full::new(body))
                .unwrap();
            let resp = handle_request(eng, req).await.unwrap();
            let st = resp.status();
            let b = resp.into_body().collect().await.unwrap().to_bytes();
            let v: Value = serde_json::from_slice(&b).unwrap_or(Value::Null);
            (st, v)
        }
    };

    let (zps_st, zps) = post_json(format!("/zarr/statistics?url={zarr_url}"), poly_bytes.clone()).await;
    let (cps_st, cps) = post_json(format!("/cog/statistics?url={tif_url}&max_size=0"), poly_bytes.clone()).await;
    assert_eq!(zps_st, StatusCode::OK, "POST /zarr/statistics failed: {zps}");
    assert_eq!(cps_st, StatusCode::OK);
    let zb1 = &zps["properties"]["statistics"]["b1"];
    let cb1 = &cps["properties"]["statistics"]["b1"];
    assert!((zb1["count"].as_f64().unwrap() - cb1["count"].as_f64().unwrap()).abs() < 1e-6);
    assert!((zb1["mean"].as_f64().unwrap() - cb1["mean"].as_f64().unwrap()).abs() < 1e-4);
    assert!((zb1["area"].as_f64().unwrap() - cb1["area"].as_f64().unwrap()).abs() < 1e-2);

    // 10. POST /zarr/feature/64x64.png returns a valid PNG
    let feat_req = Request::builder()
        .method("POST")
        .uri(format!("/zarr/feature/64x64.png?url={zarr_url}&rescale=100,900"))
        .header("Content-Type", "application/json")
        .body(Full::new(poly_bytes))
        .unwrap();
    let feat_resp = handle_request(Arc::clone(&engine), feat_req).await.unwrap();
    assert_eq!(feat_resp.status(), StatusCode::OK);
    let feat_png = feat_resp.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&feat_png[0..4], b"\x89PNG");

    // 11. /metrics exposes Zarr counters (#386)
    let (mst, _, mbody) = get_bytes(&engine, "/metrics", None).await;
    assert_eq!(mst, StatusCode::OK);
    let mtxt = std::str::from_utf8(&mbody).unwrap();
    assert!(mtxt.contains("cogzilla_zarr_chunks_fetched_total"));
    assert!(mtxt.contains("cogzilla_zarr_chunk_holes_total"));
    assert!(mtxt.contains("cogzilla_zarr_bytes_decoded_total"));
    assert!(mtxt.contains("cogzilla_zarr_cap_refusals_total"));

    // 12. GeoZarr / OME-Zarr multiscales fixture serves low-zoom & high-zoom tiles (#375)
    let ms_url = dir.join("multiscale.zarr").to_string_lossy().to_string();
    let (st_lo, _, body_lo) = get_bytes(
        &engine,
        &format!("/zarr/tiles/WebMercatorQuad/9/81/197.png?url={ms_url}&rescale=100,900"),
        None,
    )
    .await;
    assert_eq!(st_lo, StatusCode::OK);
    assert_eq!(&body_lo[0..4], b"\x89PNG");

    let (st_hi, _, body_hi) = get_bytes(
        &engine,
        &format!("/zarr/tiles/WebMercatorQuad/11/327/791.png?url={ms_url}&rescale=100,900"),
        None,
    )
    .await;
    assert_eq!(st_hi, StatusCode::OK);
    assert_eq!(&body_hi[0..4], b"\x89PNG");

    // 13. Freshness & validator ETag (#387): mutating a local store invalidates ETag (200 not 304)
    let tmp_dir = std::env::temp_dir().join(format!("cogzilla_zarr_freshness_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    copy_dir_all(&dir.join("v3_consolidated.zarr"), &tmp_dir);
    let tmp_url = tmp_dir.to_string_lossy().to_string();
    let tile_uri = format!(
        "/zarr/tiles/WebMercatorQuad/9/81/197.png?url={tmp_url}&rescale=100,900"
    );
    let (st1, etag1, _) = get_bytes(&engine, &tile_uri, None).await;
    assert_eq!(st1, StatusCode::OK);
    let etag1 = etag1.expect("expected ETag");
    // Revalidate before mutation -> 304
    let (st304, _, _) = get_bytes(&engine, &tile_uri, Some(&etag1)).await;
    assert_eq!(st304, StatusCode::NOT_MODIFIED);

    // Mutate root zarr.json (simulating a store append / attribute update)
    let zjson_path = tmp_dir.join("zarr.json");
    let mut zjson = std::fs::read_to_string(&zjson_path).unwrap();
    zjson.push('\n'); // changes file length + mtime
    std::fs::write(&zjson_path, zjson).unwrap();

    // Revalidate with old ETag after mutation -> 200 OK with new ETag (NOT 304!)
    let (st2, etag2, body2) = get_bytes(&engine, &tile_uri, Some(&etag1)).await;
    assert_eq!(st2, StatusCode::OK, "mutated store must return 200, not stale 304");
    assert_ne!(etag2.as_deref(), Some(etag1.as_str()));
    assert_eq!(&body2[0..4], b"\x89PNG");

    // 14. HEAD /zarr/tiles returns 200 + ETag + Content-Length > 0 with empty body
    let head_req = Request::builder()
        .method(hyper::Method::HEAD)
        .uri(&tile_uri)
        .body(Full::new(Bytes::new()))
        .unwrap();
    let head_resp = handle_request(Arc::clone(&engine), head_req).await.unwrap();
    assert_eq!(head_resp.status(), StatusCode::OK);
    assert!(head_resp.headers().contains_key(hyper::header::ETAG));
    let cl: usize = head_resp
        .headers()
        .get(hyper::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(cl > 0, "HEAD must advertise non-zero Content-Length");
    let head_body = head_resp.into_body().collect().await.unwrap().to_bytes();
    assert!(head_body.is_empty(), "HEAD body must be empty");

    // 15. Percent-encoded query key (%75%72%6C = "url") works (#408) & tiles_served incremented
    let pct_uri = format!("/zarr/bounds?%75%72%6C={tmp_url}");
    let (st_pct, v_pct) = get_json(&engine, &pct_uri).await;
    assert_eq!(st_pct, StatusCode::OK);
    assert!(v_pct["bounds"].is_array());
    assert!(
        engine.stats.tiles_served.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "Zarr tile requests must increment engine.stats.tiles_served"
    );

    // 16. Unified TileCache seam (#542): format query/extension never collide in cache
    // and ETag uses the server-wide weak W/"cgz-..." format.
    assert!(
        etag1.starts_with("W/\"cgz-"),
        "expected unified weak ETag format, got {etag1}"
    );
    let png_fmt_uri = format!("/zarr/tiles/WebMercatorQuad/9/81/197?url={tmp_url}&rescale=100,900&format=png");
    let webp_fmt_uri = format!("/zarr/tiles/WebMercatorQuad/9/81/197?url={tmp_url}&rescale=100,900&format=webp");
    let (st_p, _, b_p) = get_bytes(&engine, &png_fmt_uri, None).await;
    let (st_w, _, b_w) = get_bytes(&engine, &webp_fmt_uri, None).await;
    assert_eq!(st_p, StatusCode::OK);
    assert_eq!(st_w, StatusCode::OK);
    assert_eq!(&b_p[0..4], b"\x89PNG");
    assert_eq!(&b_w[0..4], b"RIFF");
    assert!(
        engine.tile_cache.misses.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "Zarr tile requests must go through TileCache::get_or_render (misses > 0)"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn zarr_tile_cache_coalesces_concurrent_requests() {
    let Some(dir) = zarr_fixtures_dir() else {
        return;
    };
    let engine = test_engine();
    let zarr_url = dir.join("geotiff_twin.zarr").to_string_lossy().to_string();
    let uri = format!(
        "/zarr/tiles/WebMercatorQuad/10/163/395.png?url={zarr_url}&rescale=100,900&colormap_name=viridis&resampling=bilinear"
    );

    // Clear caches and launch 32 concurrent requests behind a barrier so they race into
    // `TileCache::get_or_render` on multiple OS threads.
    let barrier = Arc::new(tokio::sync::Barrier::new(32));
    let mut handles = Vec::with_capacity(32);
    for _ in 0..32 {
        let eng = Arc::clone(&engine);
        let u = uri.clone();
        let b = Arc::clone(&barrier);
        handles.push(tokio::spawn(async move {
            b.wait().await;
            get_bytes(&eng, &u, None).await
        }));
    }
    let mut first_body: Option<Bytes> = None;
    for h in handles {
        let (st, etag, body) = h.await.unwrap();
        assert_eq!(st, StatusCode::OK);
        assert!(etag.as_deref().unwrap_or("").starts_with("W/\"cgz-"));
        assert_eq!(&body[0..4], b"\x89PNG");
        if let Some(ref fb) = first_body {
            assert_eq!(&body, fb);
        } else {
            first_body = Some(body);
        }
    }
    let misses = engine.tile_cache.misses.load(std::sync::atomic::Ordering::Relaxed);
    let coalesced = engine.tile_cache.coalesced.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        coalesced > 0 && misses < 32,
        "expected concurrent /zarr/tiles burst to coalesce via get_or_render (misses={misses}, coalesced={coalesced})"
    );
}

fn copy_dir_all(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let ty = entry.file_type().unwrap();
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst.join(entry.file_name()));
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
        }
    }
}
