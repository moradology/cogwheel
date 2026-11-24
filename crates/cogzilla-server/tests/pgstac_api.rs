//! HTTP tests for the pgSTAC `/searches` routes.
//!
//! The first test needs nothing. The second talks to a real pgSTAC database and runs only
//! when `COGZILLA_TEST_DATABASE_URL` is set, for example against the reference stack in
//! `benches/pgstac/docker-compose.yml`:
//!
//! ```text
//! COGZILLA_TEST_DATABASE_URL=postgresql://username:password@127.0.0.1:5439/postgis \
//!     cargo test -p cogzilla-server --test pgstac_api
//! ```
//!
//! It registers one search over a collection that does not exist, so it reads no rasters
//! and leaves a single row in `pgstac.searches`.

use std::sync::Arc;

use cogzilla_pgstac::{PgStac, PgStacConfig};
use cogzilla_server::{TileEngine, serve_listener};
use serde_json::{Value, json};
use tokio::net::TcpListener;

async fn body_json(resp: reqwest::Response) -> Value {
    serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap()
}

async fn get_json(client: &reqwest::Client, url: String) -> Value {
    body_json(client.get(url).send().await.unwrap()).await
}

async fn post_json(client: &reqwest::Client, url: String, body: &Value) -> reqwest::Response {
    client
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap()
}

async fn start(engine: Arc<TileEngine>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn search_routes_say_so_when_no_database_is_configured() {
    let base = start(Arc::new(TileEngine::new(8 * 1024 * 1024))).await;
    let client = reqwest::Client::new();
    for path in [
        "/searches/",
        "/searches/abc/info",
        "/searches/abc/tiles/WebMercatorQuad/0/0/0?assets=visual",
    ] {
        let resp = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 503, "{path}");
        let body = body_json(resp).await;
        assert!(body["detail"].as_str().unwrap().contains("pgSTAC is not configured"), "{path}");
    }
    // The rest of the server is unaffected.
    let health = client.get(format!("{base}/healthz")).send().await.unwrap();
    assert_eq!(health.status().as_u16(), 200);
}

#[tokio::test]
async fn search_routes_match_titiler_pgstac_shapes() {
    let Ok(url) = std::env::var("COGZILLA_TEST_DATABASE_URL") else {
        eprintln!("COGZILLA_TEST_DATABASE_URL not set; skipping pgSTAC HTTP test");
        return;
    };
    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    let config = PgStacConfig::from_connection_string(&url).unwrap();
    assert!(engine.set_pgstac(Arc::new(PgStac::new(config).unwrap())));
    let base = start(engine.clone()).await;
    let client = reqwest::Client::new();
    assert!(!engine.pgstac().unwrap().version().await.unwrap().is_empty());

    // Register: the id is pgSTAC's hash of the stored search, so registering twice agrees.
    let body = json!({
        "collections": ["cogzilla-test-collection-that-does-not-exist"],
        "metadata": { "name": "cogzilla test", "minzoom": 3, "maxzoom": 9, "bounds": [5.0, 43.5, 12.0, 48.0] }
    });
    let mut ids = Vec::new();
    for _ in 0..2 {
        let resp = post_json(&client, format!("{base}/searches/register"), &body).await;
        assert_eq!(resp.status().as_u16(), 200);
        let reg = body_json(resp).await;
        let rels: Vec<&str> = reg["links"].as_array().unwrap().iter().map(|l| l["rel"].as_str().unwrap()).collect();
        assert_eq!(rels, ["metadata", "tilejson", "map", "wmts"]);
        ids.push(reg["id"].as_str().unwrap().to_string());
    }
    assert_eq!(ids[0], ids[1]);
    let id = &ids[0];
    assert_eq!(id.len(), 32);

    // Info: titiler-pgstac always stores the filter language and defaults the type.
    let info = get_json(&client, format!("{base}/searches/{id}/info")).await;
    assert_eq!(info["search"]["hash"], json!(id));
    assert_eq!(info["search"]["search"]["filter-lang"], "cql2-json");
    assert_eq!(info["search"]["metadata"]["type"], "mosaic");
    assert_eq!(info["search"]["metadata"]["name"], "cogzilla test");
    assert_eq!(info["links"][0]["rel"], "self");

    // Listing includes it.
    let list = get_json(&client, format!("{base}/searches/?limit=1000")).await;
    assert!(list["context"]["matched"].as_i64().unwrap() >= 1);
    assert!(list["searches"].as_array().unwrap().iter().any(|s| s["search"]["hash"] == json!(id)));

    // TileJSON: zooms and bounds come from the search metadata, the tile size is pinned.
    let tj = get_json(
        &client,
        format!("{base}/searches/{id}/WebMercatorQuad/tilejson.json?assets=visual&rescale=0,255"),
    )
    .await;
    assert_eq!(tj["minzoom"], 3);
    assert_eq!(tj["maxzoom"], 9);
    assert_eq!(tj["bounds"], json!([5.0, 43.5, 12.0, 48.0]));
    assert_eq!(tj["center"], json!([8.5, 45.75, 3]));
    assert_eq!(
        tj["tiles"][0],
        json!(format!("{base}/searches/{id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?assets=visual&rescale=0%2C255&tilesize=512"))
    );

    // No item covers anything: an empty list, and an empty tile.
    let tile = format!("{base}/searches/{id}/tiles/WebMercatorQuad/8/133/91");
    let assets = get_json(&client, format!("{tile}/assets")).await;
    assert_eq!(assets, json!([]));
    let empty = client.get(format!("{tile}.png?assets=visual")).send().await.unwrap();
    assert_eq!(empty.status().as_u16(), 204);
    assert!(empty.bytes().await.unwrap().is_empty());
    let past_edge = client.get(format!("{base}/searches/{id}/tiles/WebMercatorQuad/3/99/0?assets=visual")).send().await.unwrap();
    assert_eq!(past_edge.status().as_u16(), 204);

    // Validation and lookups answer the way titiler-pgstac does.
    let missing = client.get(format!("{tile}.png")).send().await.unwrap();
    assert_eq!(missing.status().as_u16(), 422);
    let missing = body_json(missing).await;
    assert_eq!(missing["detail"][0]["loc"], json!(["query", "assets"]));
    assert_eq!(missing["detail"][0]["type"], "missing");

    for path in [
        "/searches/deadbeef/info".to_string(),
        "/searches/deadbeef/WebMercatorQuad/tilejson.json?assets=visual".to_string(),
        "/searches/deadbeef/tiles/WebMercatorQuad/8/133/91?assets=visual".to_string(),
        "/searches/deadbeef/tiles/WebMercatorQuad/8/133/91/assets".to_string(),
    ] {
        let resp = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 404, "{path}");
        let body = body_json(resp).await;
        assert_eq!(body["detail"], "SearchId `deadbeef` not found", "{path}");
    }

    let unknown_tms = client.get(format!("{base}/searches/{id}/tiles/Nope/8/133/91?assets=visual")).send().await.unwrap();
    assert_eq!(unknown_tms.status().as_u16(), 422);
    let unsupported_tms = client.get(format!("{base}/searches/{id}/tiles/WorldCRS84Quad/8/133/91?assets=visual")).send().await.unwrap();
    assert_eq!(unsupported_tms.status().as_u16(), 400);

    for (what, query) in [
        ("pixel selection", "assets=visual&pixel_selection=mean"),
        ("asset options", "assets=visual%7Cbidx%3D1"),
        ("too many assets", "assets=a&assets=b&assets=c&assets=d"),
        ("buffer", "assets=visual&buffer=2"),
        ("bad limit", "assets=visual&items_limit=many"),
    ] {
        let resp = client.get(format!("{tile}.png?{query}")).send().await.unwrap();
        assert!(matches!(resp.status().as_u16(), 400 | 422), "{what}: {}", resp.status());
    }

    let both = json!({ "bbox": [0, 0, 1, 1], "intersects": { "type": "Point", "coordinates": [0, 0] } });
    let resp = post_json(&client, format!("{base}/searches/register"), &both).await;
    assert_eq!(resp.status().as_u16(), 422);
    let resp = client.post(format!("{base}/searches/register")).body("not json").send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 422);
    let resp = client.get(format!("{base}/searches/{id}/nope")).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}
