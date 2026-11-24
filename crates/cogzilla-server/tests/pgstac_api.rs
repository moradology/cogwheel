//! HTTP tests for the pgSTAC `/searches` and `/collections` routes.
//!
//! The first test needs nothing. The others talk to a real pgSTAC database and run only
//! when `COGZILLA_TEST_DATABASE_URL` is set, for example against the reference stack in
//! `benches/pgstac/docker-compose.yml`:
//!
//! ```text
//! COGZILLA_TEST_DATABASE_URL=postgresql://username:password@127.0.0.1:5439/postgis \
//!     cargo test -p cogzilla-server --test pgstac_api
//! ```
//!
//! They read no rasters. They leave one empty collection, `cogzilla-test-collection`, and a
//! few rows in `pgstac.searches`.

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
        "/searches/abc/tiles",
        "/collections/abc/info",
        "/collections/abc/point/1,2/assets",
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

    // Tilesets: the limits follow the search's bounds and zoom range.
    let list = get_json(&client, format!("{base}/searches/{id}/tiles?assets=visual")).await;
    let sets = list["tilesets"].as_array().unwrap();
    assert_eq!(sets.len(), 1);
    assert_eq!(sets[0]["links"][0]["href"], json!(format!("{base}/searches/{id}/tiles/WebMercatorQuad?assets=visual")));
    let set = get_json(&client, format!("{base}/searches/{id}/tiles/WebMercatorQuad?assets=visual")).await;
    assert_eq!(set["links"][0]["href"], json!(format!("{base}/searches/{id}/tiles/WebMercatorQuad")));
    assert_eq!(
        set["links"][1]["href"],
        json!(format!("{base}/searches/{id}/tiles/WebMercatorQuad/%7Bz%7D/%7Bx%7D/%7By%7D?assets=visual"))
    );
    assert_eq!(set["boundingBox"]["lowerLeft"], json!([5.0, 43.5]));
    let limits = set["tileMatrixSetLimits"].as_array().unwrap();
    assert_eq!(limits.len(), 7);
    assert_eq!(
        limits[5],
        json!({ "tileMatrix": "8", "minTileRow": 88, "maxTileRow": 93, "minTileCol": 131, "maxTileCol": 136 })
    );
    let narrowed = get_json(&client, format!("{base}/searches/{id}/tiles/WebMercatorQuad?minzoom=6&maxzoom=6")).await;
    assert_eq!(narrowed["tileMatrixSetLimits"].as_array().unwrap().len(), 1);

    // Item lists for a box and a point: nothing is there, and malformed paths are not routes.
    for path in ["bbox/6,44,8,46/assets", "point/8.5,45.75/assets", "point/8.5,45.75/assets?items_limit=3"] {
        let items = get_json(&client, format!("{base}/searches/{id}/{path}")).await;
        assert_eq!(items, json!([]), "{path}");
    }
    for (path, status) in [
        ("bbox/6,44,8/assets", 404),
        ("point/8.5/assets", 404),
        ("bbox/6,44,8,46/assets?coord_crs=epsg:32632", 400),
        ("bbox/6,44,8,46/assets?items_limit=many", 422),
        ("tiles/Nope", 422),
    ] {
        let resp = client.get(format!("{base}/searches/{id}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), status, "{path}");
    }
}

#[tokio::test]
async fn collection_routes_match_titiler_pgstac_shapes() {
    let Ok(url) = std::env::var("COGZILLA_TEST_DATABASE_URL") else {
        eprintln!("COGZILLA_TEST_DATABASE_URL not set; skipping pgSTAC collection test");
        return;
    };
    // An empty collection to describe: no items, so nothing is rendered.
    let cid = "cogzilla-test-collection";
    let collection = json!({
        "id": cid, "type": "Collection", "stac_version": "1.0.0",
        "description": "cogzilla HTTP test", "license": "proprietary", "links": [],
        "extent": {
            "spatial": { "bbox": [[5.0, 43.5, 12.0, 48.0]] },
            "temporal": { "interval": [[null, null]] }
        },
        "item_assets": { "visual": {}, "thumbnail": {}, "red": {}, "nir08": {} }
    });
    let (db, conn) = tokio_postgres::connect(&url, tokio_postgres::NoTls).await.unwrap();
    tokio::spawn(conn);
    db.execute("SELECT pgstac.upsert_collection($1::text::jsonb)", &[&collection.to_string()])
        .await
        .unwrap();

    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    let config = PgStacConfig::from_connection_string(&url).unwrap();
    assert!(engine.set_pgstac(Arc::new(PgStac::new(config).unwrap())));
    let base = start(engine).await;
    let client = reqwest::Client::new();
    let col = format!("{base}/collections/{cid}");

    // Info: the search and metadata titiler-pgstac derives, asset names in jsonb key order.
    let info = get_json(&client, format!("{col}/info")).await;
    let search = &info["search"];
    assert_eq!(search["search"], json!({ "collections": [cid], "filter-lang": "cql2-json" }));
    assert_eq!(
        search["metadata"],
        json!({
            "type": "mosaic",
            "name": format!("Mosaic for '{cid}' Collection"),
            "bounds": [5.0, 43.5, 12.0, 48.0],
            "assets": ["red", "nir08", "visual", "thumbnail"]
        })
    );
    assert!(search.get("_where").is_none());
    assert_eq!(info["links"][0]["href"], json!(format!("{col}/info")));
    assert_eq!(info["links"][1]["href"], json!(format!("{col}/{{tileMatrixSetId}}/tilejson.json")));
    let hash = search["hash"].as_str().unwrap().to_string();
    assert_eq!(hash.len(), 32);
    let again = get_json(&client, format!("{col}/info")).await;
    assert_eq!(again["search"]["hash"], json!(hash));

    // Search parameters in the query string narrow the mosaic and change its id.
    let narrowed = get_json(
        &client,
        format!("{col}/info?bbox=6,44,8,46&ids=a,b&datetime=2024-06-01T00:00:00Z/..&sortby=%2Bproperties.eo:cloud_cover,-datetime&query=%7B%22eo%3Acloud_cover%22%3A%7B%22lt%22%3A20%7D%7D"),
    )
    .await;
    let s = &narrowed["search"]["search"];
    assert_eq!(s["bbox"], json!([6.0, 44.0, 8.0, 46.0]));
    assert_eq!(s["ids"], json!(["a", "b"]));
    assert_eq!(s["datetime"], "2024-06-01T00:00:00Z/..");
    assert_eq!(
        s["sortby"],
        json!([
            { "field": "properties.eo:cloud_cover", "direction": "asc" },
            { "field": "datetime", "direction": "desc" }
        ])
    );
    assert_eq!(s["query"], json!({ "eo:cloud_cover": { "lt": 20 } }));
    assert_eq!(narrowed["search"]["metadata"]["bounds"], json!([6.0, 44.0, 8.0, 46.0]));
    assert_ne!(narrowed["search"]["hash"], json!(hash));

    // TileJSON and tilesets point back at the collection.
    let tj = get_json(&client, format!("{col}/WebMercatorQuad/tilejson.json?assets=visual")).await;
    assert_eq!(tj["bounds"], json!([5.0, 43.5, 12.0, 48.0]));
    assert_eq!(
        tj["tiles"][0],
        json!(format!("{col}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?assets=visual&tilesize=512"))
    );
    let set = get_json(&client, format!("{col}/tiles/WebMercatorQuad?minzoom=8&maxzoom=8")).await;
    assert_eq!(
        set["tileMatrixSetLimits"],
        json!([{ "tileMatrix": "8", "minTileRow": 88, "maxTileRow": 93, "minTileCol": 131, "maxTileCol": 136 }])
    );
    let list = get_json(&client, format!("{col}/tiles")).await;
    assert_eq!(list["tilesets"][0]["links"][0]["href"], json!(format!("{col}/tiles/WebMercatorQuad")));
    let map = client.get(format!("{col}/WebMercatorQuad/map.html?assets=visual")).send().await.unwrap();
    assert_eq!(map.status().as_u16(), 200);

    // The collection is empty: no items anywhere, and an empty tile.
    let tile = format!("{col}/tiles/WebMercatorQuad/8/133/91");
    for url in [format!("{tile}/assets"), format!("{col}/bbox/6,44,8,46/assets"), format!("{col}/point/8.5,45.75/assets")] {
        assert_eq!(get_json(&client, url.clone()).await, json!([]), "{url}");
    }
    let empty = client.get(format!("{tile}.png?assets=visual")).send().await.unwrap();
    assert_eq!(empty.status().as_u16(), 204);
    let missing = client.get(format!("{tile}.png")).send().await.unwrap();
    assert_eq!(missing.status().as_u16(), 422);

    // Lookups and validation.
    for path in ["info", "WebMercatorQuad/tilejson.json?assets=visual", "tiles/WebMercatorQuad/8/133/91?assets=visual", "tiles"] {
        let resp = client.get(format!("{base}/collections/cogzilla-no-such-collection/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 404, "{path}");
        let body = body_json(resp).await;
        assert_eq!(body["detail"], "CollectionId `cogzilla-no-such-collection` not found", "{path}");
    }
    for (path, status) in [
        ("info?bbox=6,44,8", 422),
        ("info?query=not-json", 422),
        ("info?filter=eo:cloud_cover%3C20", 400),
        ("tiles/Nope/8/133/91?assets=visual", 422),
        ("items/some-item/info", 404),
        ("nope", 404),
    ] {
        let resp = client.get(format!("{col}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), status, "{path}");
    }
}
