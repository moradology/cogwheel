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
