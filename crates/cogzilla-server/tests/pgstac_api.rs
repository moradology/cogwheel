//! HTTP tests for the pgSTAC `/searches`, `/collections` and item routes.
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
//! They read no rasters. They leave two collections (`cogzilla-test-collection`, empty, and
//! `cogzilla-test-items`, with one item whose assets do not exist) and a few rows in
//! `pgstac.searches`. The lean lookup's test makes and removes a collection of its own.

use std::sync::Arc;

use cogzilla_pgstac::{LeanMode, PgStac, PgStacConfig};
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

/// A connection for a test that writes to the catalog. Tests run side by side on one
/// database, and pgSTAC's item writes take locks on partitions and their bookkeeping in an
/// order that two writers at once can deadlock on; a session lock, held until the
/// connection closes with the test, lets one write at a time.
async fn writer(url: &str) -> tokio_postgres::Client {
    let (db, conn) = tokio_postgres::connect(url, tokio_postgres::NoTls).await.unwrap();
    tokio::spawn(conn);
    db.execute("SELECT pg_advisory_lock(7203117)", &[]).await.unwrap();
    db
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
        "/collections/abc/items/def/assets",
        "/collections/abc/items/def/tiles/WebMercatorQuad/0/0/0?assets=visual",
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
async fn openapi_lists_every_pgstac_operation() {
    let base = start(Arc::new(TileEngine::new(8 * 1024 * 1024))).await;
    let spec = get_json(&reqwest::Client::new(), format!("{base}/openapi.json")).await;
    let paths = spec["paths"].as_object().unwrap();
    // A path with two methods lists both.
    let statistics = &paths["/collections/{collection_id}/items/{item_id}/statistics"];
    assert!(statistics["get"]["summary"].is_string() && statistics["post"]["summary"].is_string(), "{statistics}");
    for (path, method) in [
        ("/searches/list", "get"),
        ("/searches/{search_id}/tiles/{tileMatrixSetId}/{z}/{x}/{y}", "get"),
        ("/searches/{search_id}/feature", "post"),
        ("/searches/{search_id}/feature/{width}x{height}.{format}", "post"),
        ("/collections/{collection_id}/bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{format}", "get"),
        ("/collections/{collection_id}/items/{item_id}/preview/{width}x{height}.{format}", "get"),
    ] {
        assert!(paths[path][method]["summary"].is_string(), "{method} {path}");
    }
    // titiler-pgstac 3.2 has 63 operations under these two prefixes.
    let listed: usize = paths
        .iter()
        .filter(|(path, _)| path.starts_with("/searches") || path.starts_with("/collections"))
        .map(|(_, methods)| methods.as_object().map_or(0, |m| m.len()))
        .sum();
    assert_eq!(listed, 63);
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
        ("a pixel selection nobody has", "assets=visual&pixel_selection=nope"),
        ("an asset option without a value", "assets=visual%7Cbidx"),
        ("asset names in an expression", "assets=red&assets=nir&expression=nir-red"),
        ("too many assets", "assets=a&assets=b&assets=c&assets=d&assets=e&assets=f&assets=g&assets=h&assets=i&assets=j&assets=k&assets=l&assets=m&assets=n&assets=o&assets=p&assets=q"),
        ("buffer", "assets=visual&buffer=2"),
        ("bad limit", "assets=visual&items_limit=many"),
    ] {
        let resp = client.get(format!("{tile}.png?{query}")).send().await.unwrap();
        assert!(matches!(resp.status().as_u16(), 400 | 422), "{what}: {}", resp.status());
    }

    // The other pixel selections are taken wherever `first` is (here over nothing, so
    // nothing comes back); the two that are not there yet are refused.
    let unknown = client.get(format!("{tile}.png?assets=visual&pixel_selection=nope")).send().await.unwrap();
    assert_eq!(unknown.status().as_u16(), 422);
    assert_eq!(body_json(unknown).await["detail"][0]["loc"], json!(["query", "pixel_selection"]));
    for method in ["first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh"] {
        let resp = client.get(format!("{tile}.png?assets=visual&pixel_selection={method}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 204, "{method}");
    }
    let bbox = format!("{base}/searches/{id}/bbox/6,44,7,45.png?assets=visual&max_size=64");
    let plain = client.get(&bbox).send().await.unwrap().status().as_u16();
    for method in ["highest", "lowest", "mean", "median", "stdev", "count"] {
        let image = client.get(format!("{bbox}&pixel_selection={method}")).send().await.unwrap();
        assert_eq!(image.status().as_u16(), plain, "{method}");
    }
    let image = client.get(format!("{bbox}&pixel_selection=lastbandlow")).send().await.unwrap();
    assert_eq!(image.status().as_u16(), 400);
    assert!(body_json(image).await["detail"].as_str().unwrap().contains("last band"));
    // More than three assets, or two without band math, are tiles made from planes; over
    // nothing they are nothing, and on an image they are refused.
    for assets in ["assets=red&assets=green&assets=blue&assets=nir&asset_as_band=true", "assets=red&assets=nir"] {
        let resp = client.get(format!("{tile}.png?{assets}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 204, "{assets}");
    }
    let image = client
        .get(format!("{base}/searches/{id}/bbox/6,44,7,45.png?assets=red&assets=green&assets=blue&assets=nir&max_size=64"))
        .send()
        .await
        .unwrap();
    assert_eq!(image.status().as_u16(), 400);
    let seventeen: String = (0..17).map(|n| format!("assets=a{n}&")).collect();
    let resp = client.get(format!("{tile}.png?{seventeen}")).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let image = client.get(format!("{bbox}&pixel_selection=nope")).send().await.unwrap();
    assert_eq!(image.status().as_u16(), 422);

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

    // WMTS: one layer from the query string, limited to the search's zooms and bounds.
    let wmts = client.get(format!("{base}/searches/{id}/WMTSCapabilities.xml?assets=visual&tile_format=jpg")).send().await.unwrap();
    assert_eq!(wmts.status().as_u16(), 200);
    assert_eq!(wmts.headers()["content-type"], "application/xml");
    let xml = String::from_utf8(wmts.bytes().await.unwrap().to_vec()).unwrap();
    assert!(xml.contains(&format!("<ows:Identifier>{id}_WebMercatorQuad_default</ows:Identifier>")));
    assert!(xml.contains(&format!("template=\"{base}/searches/{id}/tiles/WebMercatorQuad/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.jpg?assets=visual\"")));
    assert_eq!(xml.matches("<TileMatrixLimits>").count(), 7);
    assert!(xml.contains("<ows:LowerCorner>5.0 43.5</ows:LowerCorner>"));
    let no_layer = client.get(format!("{base}/searches/{id}/WMTSCapabilities.xml")).send().await.unwrap();
    assert_eq!(no_layer.status().as_u16(), 400);

    // Item lists for a box and a point: nothing is there, and malformed paths are not routes.
    for path in ["bbox/6,44,8,46/assets", "point/8.5,45.75/assets", "point/8.5,45.75/assets?items_limit=3"] {
        let items = get_json(&client, format!("{base}/searches/{id}/{path}")).await;
        assert_eq!(items, json!([]), "{path}");
    }
    // bbox and feature images: nothing to draw, and the parameters are checked first.
    for (path, status) in [
        ("bbox/6,44,8,46.png?assets=visual&max_size=64", 204),
        ("bbox/6,44,8,46/64x64.png?assets=visual", 204),
        ("bbox/6,44,8,46.png", 422),
        ("bbox/6,44,8,46?assets=visual", 422),
        ("bbox/6,44,8,46.png?assets=visual&width=wide", 422),
        ("bbox/6,44,8,46.png?assets=visual&dst_crs=epsg:3857", 400),
        ("bbox/6,44,8,46.npy?assets=visual", 400),
        ("bbox/6,44,8.png?assets=visual", 404),
        ("feature.png?assets=visual", 405),
    ] {
        let resp = client.get(format!("{base}/searches/{id}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), status, "{path}");
    }
    let triangle = json!({ "type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[6.0, 44.0], [8.0, 44.0], [7.0, 46.0], [6.0, 44.0]]] } });
    for (path, status) in [
        ("feature.png?assets=visual&max_size=64", 204),
        ("feature?assets=visual", 204),
        ("feature/64x32.png?assets=visual", 204),
        ("feature.png", 422),
        ("info", 404),
    ] {
        let resp = post_json(&client, format!("{base}/searches/{id}/{path}"), &triangle).await;
        assert_eq!(resp.status().as_u16(), status, "POST {path}");
    }
    let not_geojson = client.post(format!("{base}/searches/{id}/feature.png?assets=visual")).body("nope").send().await.unwrap();
    assert_eq!(not_geojson.status().as_u16(), 422);
    // The body is a Feature: a bare geometry or an empty object is not one.
    for body in [&triangle["geometry"], &json!({})] {
        let resp = post_json(&client, format!("{base}/searches/{id}/feature.png?assets=visual"), body).await;
        assert_eq!(resp.status().as_u16(), 422, "{body}");
    }

    // Statistics under features: nothing to read, and the parameters are checked first.
    let collection = json!({ "type": "FeatureCollection", "features": [triangle.clone()] });
    for (path, body, status) in [
        ("statistics?assets=visual", &triangle, 204),
        ("statistics?assets=visual&max_size=64&categorical=true&c=1&p=5&histogram_bins=4&cover_scale=2", &triangle, 204),
        ("statistics?assets=visual", &collection, 204),
        ("statistics", &triangle, 422),
        ("statistics?assets=visual", &triangle["geometry"], 422),
        ("statistics?assets=visual&histogram_bins=ten", &triangle, 422),
        ("statistics?assets=visual&cover_scale=0", &triangle, 422),
        ("statistics?assets=visual&width=wide", &triangle, 422),
        ("statistics?assets=visual&dst_crs=epsg:3857", &triangle, 400),
        ("statistics?assets=visual&algorithm=hillshade", &triangle, 400),
        ("statistics?assets=visual&pixel_selection=mean", &triangle, 204),
        ("statistics?assets=visual&pixel_selection=lastbandhigh", &triangle, 204),
        ("statistics?assets=visual&pixel_selection=nope", &triangle, 422),
        ("statistics/extra?assets=visual", &triangle, 404),
    ] {
        let resp = post_json(&client, format!("{base}/searches/{id}/{path}"), body).await;
        assert_eq!(resp.status().as_u16(), status, "POST {path}");
    }
    let get = client.get(format!("{base}/searches/{id}/statistics?assets=visual")).send().await.unwrap();
    assert_eq!(get.status().as_u16(), 405);

    // Point values: nothing is under the point, and the parameters are checked first.
    for (path, status) in [
        ("point/8.5,45.75?assets=visual", 204),
        ("point/945000,5740000?assets=visual&coord_crs=epsg:3857", 204),
        ("point/8.5,45.75", 422),
        ("point/8.5,45.75?assets=visual&expression=b1*2", 204),
        ("point/8.5,45.75?assets=visual&asset_as_band=true", 204),
        ("point/8.5,45.75?assets=visual%7Cbidx%3D3,1", 204),
        ("point/8.5,45.75?assets=visual%7Cbidx%3Dx", 422),
        ("point/8.5,45.75?assets=visual&asset_as_band=perhaps", 422),
        ("point/8.5,45.75?assets=,", 422),
        ("point/8.5,45.75?assets=visual&coord_crs=epsg:32632", 400),
        ("point/8.5,45.75?assets=visual&items_limit=many", 422),
        ("point/8.5?assets=visual", 404),
    ] {
        let resp = client.get(format!("{base}/searches/{id}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), status, "{path}");
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
    let db = writer(&url).await;
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
        ("info?filter=eo:cloud_cover%3C", 422),
        ("info?filter=eo:cloud_cover", 422),
        ("info?filter=a%3D1&filter-lang=cql2-nope", 422),
        ("tiles/Nope/8/133/91?assets=visual", 422),
        ("items/some-item/info", 404),
        ("nope", 404),
    ] {
        let resp = client.get(format!("{col}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), status, "{path}");
    }
}

#[tokio::test]
async fn item_routes_match_titiler_pgstac_shapes() {
    let Ok(url) = std::env::var("COGZILLA_TEST_DATABASE_URL") else {
        eprintln!("COGZILLA_TEST_DATABASE_URL not set; skipping pgSTAC item test");
        return;
    };
    // One item with no projection metadata, so its extent is its bbox and no raster is opened.
    let cid = "cogzilla-test-items";
    let iid = "cogzilla-test-item";
    let tiff = "image/tiff; application=geotiff; profile=cloud-optimized";
    let collection = json!({
        "id": cid, "type": "Collection", "stac_version": "1.0.0",
        "description": "cogzilla HTTP test", "license": "proprietary", "links": [],
        "extent": {
            "spatial": { "bbox": [[5.0, 43.5, 12.0, 48.0]] },
            "temporal": { "interval": [[null, null]] }
        }
    });
    let item = json!({
        "type": "Feature", "stac_version": "1.0.0", "id": iid, "collection": cid,
        "bbox": [6.0, 44.0, 8.0, 46.0],
        "geometry": { "type": "Polygon", "coordinates": [[[6.0, 44.0], [8.0, 44.0], [8.0, 46.0], [6.0, 46.0], [6.0, 44.0]]] },
        "properties": {
            "datetime": "2024-06-01T00:00:00Z",
            "renders": {
                "rgb": { "title": "True colour", "assets": ["visual"] },
                "math": { "expression": "b1*2" }
            }
        },
        "links": [],
        "assets": {
            "visual": { "href": "https://example.invalid/cogzilla-test/visual.tif", "type": tiff },
            "red": { "href": "https://example.invalid/cogzilla-test/red.tif", "type": tiff },
            "red-jp2": { "href": "https://example.invalid/cogzilla-test/red.jp2", "type": "image/jp2" },
            "untyped": { "href": "https://example.invalid/cogzilla-test/untyped.tif" },
            "thumbnail": { "href": "https://example.invalid/cogzilla-test/thumb.jpg", "type": "image/jpeg" },
            "metadata": { "href": "https://example.invalid/cogzilla-test/meta.xml", "type": "application/xml" }
        }
    });
    let db = writer(&url).await;
    db.execute("SELECT pgstac.upsert_collection($1::text::jsonb)", &[&collection.to_string()])
        .await
        .unwrap();
    db.execute("SELECT pgstac.upsert_item($1::text::jsonb)", &[&item.to_string()]).await.unwrap();

    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    let config = PgStacConfig::from_connection_string(&url).unwrap();
    assert!(engine.set_pgstac(Arc::new(PgStac::new(config).unwrap())));
    let base = start(engine).await;
    let client = reqwest::Client::new();
    let it = format!("{base}/collections/{cid}/items/{iid}");

    // The assets rio-tiler would read, in jsonb key order: typed rasters and untyped assets.
    let assets = get_json(&client, format!("{it}/assets")).await;
    assert_eq!(assets, json!(["red", "visual", "red-jp2", "untyped"]));

    // Renders: one that can make a tile, with its TileJSON and WMTS links, and one that cannot.
    let renders = get_json(&client, format!("{it}/renders")).await;
    assert_eq!(renders["links"][0]["href"], json!(format!("{it}/renders")));
    let rgb = &renders["renders"]["rgb"];
    assert_eq!(rgb["valid"], true);
    assert_eq!(rgb["params"], json!({ "title": "True colour", "assets": ["visual"] }));
    assert_eq!(rgb["links"].as_array().unwrap().len(), 3);
    assert_eq!(
        rgb["links"][1]["href"],
        json!(format!("{it}/%7BtileMatrixSetId%7D/tilejson.json?url=%3CItem+id%3D{iid}%3E&title=True+colour&assets=visual"))
    );
    assert_eq!(renders["renders"]["math"]["valid"], false);
    assert_eq!(renders["renders"]["math"]["links"].as_array().unwrap().len(), 1);
    let one = get_json(&client, format!("{it}/renders/rgb")).await;
    assert_eq!(&one, rgb);
    let missing = client.get(format!("{it}/renders/nope")).send().await.unwrap();
    assert_eq!(missing.status().as_u16(), 404);
    assert_eq!(body_json(missing).await["detail"], "Render not found");
    // The render that can make a tile is a WMTS layer even with no `assets` in the query.
    let wmts = client.get(format!("{it}/WMTSCapabilities.xml")).send().await.unwrap();
    assert_eq!(wmts.status().as_u16(), 200);
    let xml = String::from_utf8(wmts.bytes().await.unwrap().to_vec()).unwrap();
    assert!(xml.contains("<ows:Identifier>TiTiler_WebMercatorQuad_rgb</ows:Identifier>"));
    assert_eq!(xml.matches("<Layer>").count(), 1);

    // Without projection metadata the tileset covers the item's bbox at every zoom.
    let set = get_json(&client, format!("{it}/tiles/WebMercatorQuad?assets=visual")).await;
    assert_eq!(set["boundingBox"]["lowerLeft"], json!([6.0, 44.0]));
    assert_eq!(set["boundingBox"]["upperRight"], json!([8.0, 46.0]));
    assert_eq!(set["tileMatrixSetLimits"].as_array().unwrap().len(), 25);
    assert_eq!(
        set["tileMatrixSetLimits"][7],
        json!({ "tileMatrix": "7", "minTileRow": 45, "maxTileRow": 46, "minTileCol": 66, "maxTileCol": 66 })
    );
    assert_eq!(set["links"][0]["href"], json!(format!("{it}/tiles/WebMercatorQuad")));
    assert_eq!(
        set["links"][1]["href"],
        json!(format!("{it}/tiles/WebMercatorQuad/%7Bz%7D/%7Bx%7D/%7By%7D?assets=visual"))
    );
    let list = get_json(&client, format!("{it}/tiles")).await;
    assert_eq!(list["tilesets"][0]["links"][0]["href"], json!(format!("{it}/tiles/WebMercatorQuad")));
    let map = client.get(format!("{it}/WebMercatorQuad/map.html?assets=visual")).send().await.unwrap();
    assert_eq!(map.status().as_u16(), 200);
    assert!(String::from_utf8_lossy(&map.bytes().await.unwrap())
        .contains(&format!("/collections/{cid}/items/{iid}/WebMercatorQuad/tilejson.json?assets=visual")));

    // WMTS for the item: its bbox at every zoom, and a 400 when nothing can make a tile.
    let wmts = client.get(format!("{it}/WMTSCapabilities.xml?assets=visual")).send().await.unwrap();
    assert_eq!(wmts.status().as_u16(), 200);
    let xml = String::from_utf8(wmts.bytes().await.unwrap().to_vec()).unwrap();
    assert!(xml.contains("<ows:Identifier>TiTiler_WebMercatorQuad_default</ows:Identifier>"));
    assert!(xml.contains(&format!("template=\"{it}/tiles/WebMercatorQuad/{{TileMatrix}}/{{TileCol}}/{{TileRow}}.png?assets=visual\"")));
    // Two layers, the render and the one from the query, each with limits at 25 zooms.
    assert_eq!(xml.matches("<Layer>").count(), 2);
    assert_eq!(xml.matches("<TileMatrixLimits>").count(), 50);
    assert!(xml.contains("<ows:UpperCorner>8.0 46.0</ows:UpperCorner>"));

    // Lookups.
    for (path, message) in [
        (format!("/collections/{cid}/items/nope/assets"), format!("No item 'nope' found in '{cid}' collection")),
        (format!("/collections/nope/items/{iid}/info?assets=visual"), format!("No item '{iid}' found in 'nope' collection")),
    ] {
        let resp = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 404, "{path}");
        assert_eq!(body_json(resp).await["detail"], json!(message), "{path}");
    }

    // `assets` is required, and has to name an asset of the item.
    for path in [
        "info", "info.geojson", "statistics", "asset_statistics", "point/7,45", "preview.png",
        "bbox/6,44,7,45.png", "WebMercatorQuad/tilejson.json", "tiles/WebMercatorQuad/7/66/45.png",
        "tiles/WebMercatorQuad/7/66/45?expression=visual_b1",
    ] {
        let resp = client.get(format!("{it}/{path}")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 422, "{path}");
        let body = body_json(resp).await;
        assert_eq!(body["detail"][0]["loc"], json!(["query", "assets"]), "{path}");
    }
    let unknown = client.get(format!("{it}/info?assets=nope")).send().await.unwrap();
    assert_eq!(unknown.status().as_u16(), 404);
    assert_eq!(
        body_json(unknown).await["detail"],
        "'nope' is not valid, should be one of ['red', 'visual', 'red-jp2', 'untyped']"
    );

    // A tile that does not touch the item is refused before anything is read.
    let outside = client.get(format!("{it}/tiles/WebMercatorQuad/5/0/0.png?assets=visual")).send().await.unwrap();
    assert_eq!(outside.status().as_u16(), 404);
    assert_eq!(body_json(outside).await["detail"], "Tile(x=0, y=0, z=5) is outside bounds");

    // Anything else is the /stac router's answer, as JSON.
    let unreadable = client.get(format!("{it}/tiles/WebMercatorQuad/7/66/45.png?assets=visual")).send().await.unwrap();
    assert!(unreadable.status().as_u16() >= 400);
    assert!(body_json(unreadable).await["detail"].is_string());
    let nope = client.get(format!("{it}/nope")).send().await.unwrap();
    assert_eq!(nope.status().as_u16(), 404);
    assert!(body_json(nope).await["detail"].is_string());
    let bare = client.get(format!("{base}/collections/{cid}/items/{iid}")).send().await.unwrap();
    assert_eq!(bare.status().as_u16(), 404);
}

/// The lean lookup against `geojsonsearch` on four items made for the purpose: two halves
/// of a box, a sliver between sample points, and one item under everything.
#[tokio::test]
async fn lean_lookup_lists_the_items_a_picture_samples() {
    let Ok(url) = std::env::var("COGZILLA_TEST_DATABASE_URL") else {
        eprintln!("COGZILLA_TEST_DATABASE_URL not set; skipping lean lookup test");
        return;
    };
    // The same items twice: in a collection with a partition per month, so the walk crosses
    // chunks, and in one left as pgSTAC makes it, a single partition whose chunk runs
    // from minus infinity to infinity.
    let cid = "cogzilla-test-lean";
    let flat = "cogzilla-test-lean-flat";
    let db = writer(&url).await;
    for id in [cid, flat] {
        let collection = json!({
            "id": id, "type": "Collection", "stac_version": "1.0.0",
            "description": "cogzilla lean lookup test", "license": "proprietary", "links": [],
            "extent": { "spatial": { "bbox": [[-1.0, -1.0, 11.0, 11.0]] }, "temporal": { "interval": [[null, null]] } }
        });
        db.execute("SELECT pgstac.delete_collection($1)", &[&id]).await.ok();
        db.execute("SELECT pgstac.upsert_collection($1::text::jsonb)", &[&collection.to_string()]).await.unwrap();
    }
    db.execute("UPDATE pgstac.collections SET partition_trunc = 'month' WHERE id = $1", &[&cid]).await.unwrap();
    let item = |cid: &str, id: &str, day: &str, [w, s, e, n]: [f64; 4], assets: Value| {
        json!({
            "type": "Feature", "stac_version": "1.0.0", "id": id, "collection": cid,
            "bbox": [w, s, e, n],
            "geometry": { "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
            "properties": { "datetime": format!("{day}T00:00:00Z") },
            "links": [], "assets": assets
        })
    };
    let visual = |name: &str| json!({ "visual": { "href": format!("https://example.invalid/lean/{name}.tif") } });
    for id in [cid, flat] {
        for doc in [
            item(id, "lean-a", "2024-06-05", [-1.0, -1.0, 5.0, 11.0], visual("a")),
            // With 50 sample columns over ten degrees the samples sit at 0.1, 0.3, ...: this
            // strip holds none of them, and pgSTAC keeps it for its area.
            item(id, "lean-b", "2024-06-04", [5.12, 0.0, 5.28, 10.0], visual("b")),
            item(id, "lean-c", "2024-05-20", [5.0, -1.0, 11.0, 11.0], visual("c")),
            item(id, "lean-d", "2024-04-01", [-1.0, -1.0, 11.0, 11.0], json!({ "other": { "href": "https://example.invalid/lean/d.tif" } })),
        ] {
            db.execute("SELECT pgstac.upsert_item($1::text::jsonb)", &[&doc.to_string()]).await.unwrap();
        }
    }

    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    let mut config = PgStacConfig::from_connection_string(&url).unwrap();
    config.lean = LeanMode::On;
    assert!(engine.set_pgstac(Arc::new(PgStac::new(config).unwrap())));
    let base = start(engine).await;
    let client = reqwest::Client::new();
    let ids = |v: &Value| -> Vec<String> {
        v.as_array().unwrap().iter().map(|i| i["id"].as_str().unwrap().to_string()).collect()
    };
    let boxed = format!("{base}/collections/{cid}/bbox/0,0,10,10/assets");

    // pgSTAC's own list keeps the sliver; the lean list for a 10 by 10 picture does not.
    let reference = get_json(&client, boxed.clone()).await;
    assert_eq!(ids(&reference), ["lean-a", "lean-b", "lean-c"]);
    assert!(reference[0]["bbox"].is_array());
    let lean = get_json(&client, format!("{boxed}?item_lookup=lean&width=10&height=10&assets=visual")).await;
    assert_eq!(ids(&lean), ["lean-a", "lean-c"]);
    assert_eq!(
        lean[1],
        json!({ "id": "lean-c", "collection": cid, "assets": { "visual": { "href": "https://example.invalid/lean/c.tif" } } })
    );
    // A picture with samples close enough together to land in the sliver keeps it.
    let fine = get_json(&client, format!("{boxed}?item_lookup=lean&width=100&height=100")).await;
    assert_eq!(ids(&fine), ["lean-a", "lean-b", "lean-c"]);
    // Without skipping, everything under the box is listed until it is full.
    let all = get_json(&client, format!("{boxed}?item_lookup=lean&width=10&height=10&skipcovered=false")).await;
    assert_eq!(ids(&all), ["lean-a", "lean-b", "lean-c"]);
    let every = get_json(&client, format!("{boxed}?item_lookup=lean&width=10&height=10&skipcovered=false&exitwhenfull=false")).await;
    assert_eq!(ids(&every), ["lean-a", "lean-b", "lean-c", "lean-d"]);
    let one = get_json(&client, format!("{boxed}?item_lookup=lean&width=10&height=10&items_limit=1")).await;
    assert_eq!(ids(&one), ["lean-a"]);
    // An item kept that lacks an asset asked for: the lean lookup stands aside, and the
    // answer is pgSTAC's, whole items and all.
    let aside = get_json(&client, format!("{boxed}?item_lookup=lean&width=10&height=10&assets=other")).await;
    assert_eq!(ids(&aside), ["lean-a", "lean-b", "lean-c"]);
    assert!(aside[0]["bbox"].is_array());

    // The collection in one partition answers the same.
    let flat_box = format!("{base}/collections/{flat}/bbox/0,0,10,10/assets");
    assert_eq!(ids(&get_json(&client, flat_box.clone()).await), ["lean-a", "lean-b", "lean-c"]);
    let flat_lean = get_json(&client, format!("{flat_box}?item_lookup=lean&width=10&height=10&assets=visual")).await;
    assert_eq!(ids(&flat_lean), ["lean-a", "lean-c"]);
    assert_eq!(flat_lean[0]["assets"]["visual"]["href"], "https://example.invalid/lean/a.tif");
    let flat_point = get_json(&client, format!("{base}/collections/{flat}/point/2,2/assets?item_lookup=lean")).await;
    assert_eq!(ids(&flat_point), ["lean-a", "lean-d"]);
    // An odd number of rows puts samples on the line through the middle of the box, and
    // footprints that meet there still fill it: two halves, top and bottom.
    for (name, day, bounds) in [("lean-n", "2024-07-02", [-1.0, 5.0, 11.0, 11.0]), ("lean-s", "2024-07-01", [-1.0, -1.0, 11.0, 5.0])] {
        let doc = item(flat, name, day, bounds, visual(name));
        db.execute("SELECT pgstac.upsert_item($1::text::jsonb)", &[&doc.to_string()]).await.unwrap();
    }
    let halves = get_json(&client, format!("{flat_box}?item_lookup=lean&width=9&height=9")).await;
    assert_eq!(ids(&halves), ["lean-n", "lean-s"]);

    // A point has no area to cover: every item under it, newest first.
    let point = get_json(&client, format!("{base}/collections/{cid}/point/2,2/assets?item_lookup=lean")).await;
    assert_eq!(ids(&point), ["lean-a", "lean-d"]);
    let reference_point = get_json(&client, format!("{base}/collections/{cid}/point/2,2/assets")).await;
    assert_eq!(ids(&reference_point), ["lean-a", "lean-d"]);

    // A search that does not sort by time is one query with no chunks.
    for (direction, expect) in [("asc", vec!["lean-a", "lean-c"]), ("desc", vec!["lean-d"])] {
        let body = json!({ "collections": [cid], "sortby": [{ "field": "id", "direction": direction }] });
        let registered = body_json(post_json(&client, format!("{base}/searches/register"), &body).await).await;
        let sid = registered["id"].as_str().unwrap();
        let url = format!("{base}/searches/{sid}/bbox/0,0,10,10/assets");
        assert_eq!(ids(&get_json(&client, format!("{url}?item_lookup=lean&width=10&height=10")).await), expect, "{direction}");
        // pgSTAC agrees, but for the sliver.
        let theirs = ids(&get_json(&client, url).await);
        assert_eq!(theirs.iter().filter(|i| *i != "lean-b").collect::<Vec<_>>(), expect, "{direction}");
    }

    // A tile, sampled on the tile's own rows.
    let tile = get_json(&client, format!("{base}/collections/{cid}/tiles/WebMercatorQuad/6/32/31/assets?item_lookup=lean")).await;
    let theirs = get_json(&client, format!("{base}/collections/{cid}/tiles/WebMercatorQuad/6/32/31/assets")).await;
    assert!(!ids(&tile).is_empty());
    assert!(ids(&tile).iter().all(|i| ids(&theirs).contains(i)), "{tile} {theirs}");

    let unknown = client.get(format!("{boxed}?item_lookup=nope")).send().await.unwrap();
    assert_eq!(unknown.status().as_u16(), 400);
    for path in ["tiles/WebMercatorQuad/6/32/31.png", "bbox/0,0,10,10.png", "point/2,2"] {
        let resp = client.get(format!("{base}/collections/{cid}/{path}?assets=visual&item_lookup=nope")).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 400, "{path}");
    }

    for id in [cid, flat] {
        db.execute("SELECT pgstac.delete_collection($1)", &[&id]).await.unwrap();
    }
}
