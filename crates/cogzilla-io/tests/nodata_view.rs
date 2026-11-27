use std::path::PathBuf;
use std::sync::Arc;

use cogzilla_io::CogStore;

/// A `?nodata=` view of an opened raster: the file itself when nothing changes, else one view
/// per value with an id of its own, its parent's source id, and the parent's IFD tables shared.
#[tokio::test]
async fn nodata_view_shares_the_parent_and_keeps_its_source() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benches/fixtures/landsat_c2_B4.tif");
    if !path.exists() {
        return;
    }
    let uri = path.canonicalize().unwrap().to_string_lossy().to_string();
    let store = CogStore::new(16 * 1024 * 1024);
    let cog = store.open_uri(&uri).await.unwrap();
    assert_eq!(cog.index.nodata, Some(0.0), "the fixture's own nodata");
    assert_eq!(cog.source_id, cog.id);

    assert!(Arc::ptr_eq(&store.with_nodata(&cog, None), &cog), "no override");
    assert!(Arc::ptr_eq(&store.with_nodata(&cog, Some(0.0)), &cog), "the file's own value");

    let view = store.with_nodata(&cog, Some(1000.0));
    assert_ne!(view.id, cog.id, "decoded apart from the file's blocks");
    assert_eq!(view.source_id, cog.id, "a change found through the view evicts the parent");
    assert_eq!(view.index.nodata, Some(1000.0));
    assert_eq!(view.index.nodata_f64, Some(1000.0));
    assert!(Arc::ptr_eq(&view.index.ifds, &cog.index.ifds), "the tile tables are not copied");
    assert!(Arc::ptr_eq(&store.with_nodata(&cog, Some(1000.0)), &view), "one view per value");
    assert!(!Arc::ptr_eq(&store.with_nodata(&cog, Some(7.0)), &view));

    store.clear_open_uris();
    let after = store.with_nodata(&cog, Some(1000.0));
    assert!(!Arc::ptr_eq(&after, &view), "cleared with the open rasters");
    assert_eq!(after.source_id, cog.id);
}
