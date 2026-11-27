//! Smoke tests against `benches/fixtures/zarr`.
//!
//! Skips cleanly when `benches/fixtures/zarr` is absent, matching the existing COG integration
//! tests.

use std::fs;
use std::path::{Path, PathBuf};

use cogzilla_zarr::{
    ArrayMeta, DataType, Endianness, FsStore, OpenMode, ZarrHierarchy, ZarrStore, ZarrVersion,
    decode_chunk_bytes,
};

fn fixtures_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/zarr");
    dir.is_dir().then_some(dir)
}

#[test]
fn zarr_fixtures_parse_with_array_meta() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    // v2 consolidated and unconsolidated
    assert!(dir.join("v2_consolidated.zarr/.zmetadata").is_file());
    assert!(!dir.join("v2_unconsolidated.zarr/.zmetadata").exists());
    for store in ["v2_consolidated.zarr", "v2_unconsolidated.zarr"] {
        let zarray = fs::read(dir.join(store).join("temp/.zarray")).unwrap();
        let zattrs = fs::read(dir.join(store).join("temp/.zattrs")).unwrap();
        let m = ArrayMeta::from_v2(&zarray, Some(&zattrs)).unwrap();
        assert_eq!(m.version, ZarrVersion::V2);
        assert_eq!(m.shape, vec![128, 128]);
        assert_eq!(m.chunk_shape, vec![64, 64]);
        assert_eq!(m.dtype, DataType::Float32);
        assert!(m.fill_value.unwrap().is_nan());
    }

    // v3 consolidated and unconsolidated
    let v3_cons: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("v3_consolidated.zarr/zarr.json")).unwrap())
            .unwrap();
    assert!(v3_cons.get("consolidated_metadata").is_some());
    let v3_uncons: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("v3_unconsolidated.zarr/zarr.json")).unwrap())
            .unwrap();
    assert!(v3_uncons.get("consolidated_metadata").is_none());

    for store in ["v3_consolidated.zarr", "v3_unconsolidated.zarr"] {
        let meta = fs::read(dir.join(store).join("temp/zarr.json")).unwrap();
        let m = ArrayMeta::from_v3(&meta).unwrap();
        assert_eq!(m.version, ZarrVersion::V3);
        assert_eq!(m.shape, vec![128, 128]);
        assert_eq!(m.dtype, DataType::Float32);
    }

    // v3 sharded vs unsharded
    let sharded = ArrayMeta::from_v3(&fs::read(dir.join("v3_sharded.zarr/data/zarr.json")).unwrap())
        .unwrap();
    assert!(sharded.codecs.iter().any(|c| c.name == "sharding_indexed"));
    let unsharded =
        ArrayMeta::from_v3(&fs::read(dir.join("v3_unsharded.zarr/data/zarr.json")).unwrap())
            .unwrap();
    assert!(!unsharded.codecs.iter().any(|c| c.name == "sharding_indexed"));

    // Big-endian v3 and v2
    let be3 = ArrayMeta::from_v3(&fs::read(dir.join("big_endian_v3.zarr/f32_be/zarr.json")).unwrap())
        .unwrap();
    assert_eq!(be3.endianness, Endianness::Big);
    let be2 = ArrayMeta::from_v2(
        &fs::read(dir.join("big_endian_v2.zarr/f32_be/.zarray")).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(be2.endianness, Endianness::Big);

    // 3D time with non-standard calendar
    let t3d = ArrayMeta::from_v3(&fs::read(dir.join("time_3d_noleap.zarr/tas/zarr.json")).unwrap())
        .unwrap();
    assert_eq!(t3d.shape, vec![6, 64, 64]);
    assert_eq!(t3d.rank(), 3);

    // Sparse holes: 3x3 chunk grid, only 3 diagonal chunks present on disk
    let sparse =
        ArrayMeta::from_v3(&fs::read(dir.join("sparse_holes.zarr/sparse/zarr.json")).unwrap())
            .unwrap();
    assert_eq!(sparse.chunks_per_dim(), vec![3, 3]);
    let mut present = 0usize;
    for r in 0..3 {
        for c in 0..3 {
            let key = sparse.chunk_key(&[r, c]);
            if dir.join("sparse_holes.zarr/sparse").join(key).is_file() {
                present += 1;
            }
        }
    }
    assert_eq!(present, 3, "expected exactly 3 of 9 chunks written");

    // GeoTIFF twin exists alongside Zarr twin
    assert!(dir.join("geotiff_twin.tif").is_file());
    assert!(dir.join("geotiff_twin.zarr/data/zarr.json").is_file());

    // Remaining fixtures exist
    for expected in [
        "codecs_v2.zarr/.zmetadata",
        "codecs_v3.zarr/zarr.json",
        "lon_0_360_desc_y.zarr/sst/zarr.json",
        "irregular_coords.zarr/warped/zarr.json",
        "crs_cf_grid_mapping.zarr/band/zarr.json",
        "crs_spatial_ref_wkt.zarr/band/zarr.json",
        "crs_none.zarr/band/zarr.json",
        "multiscale.zarr/0/elevation/zarr.json",
        "multiscale.zarr/1/elevation/zarr.json",
    ] {
        assert!(dir.join(expected).is_file(), "missing fixture {expected}");
    }
}

#[tokio::test]
async fn hierarchy_opens_generated_fixtures() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    // v2 consolidated vs unconsolidated
    let v2c = ZarrHierarchy::open(&FsStore::new(dir.join("v2_consolidated.zarr")))
        .await
        .unwrap();
    assert_eq!(v2c.open_mode, OpenMode::V2Consolidated);
    assert_eq!(v2c.group(None).unwrap().data_variables(), vec!["temp"]);

    let v2u = ZarrHierarchy::open(&FsStore::new(dir.join("v2_unconsolidated.zarr")))
        .await
        .unwrap();
    assert_eq!(v2u.open_mode, OpenMode::V2Walked);
    assert_eq!(v2u.group(None).unwrap().data_variables(), vec!["temp"]);

    // v3 consolidated vs unconsolidated
    let v3c = ZarrHierarchy::open(&FsStore::new(dir.join("v3_consolidated.zarr")))
        .await
        .unwrap();
    assert_eq!(v3c.open_mode, OpenMode::V3Consolidated);
    assert_eq!(v3c.group(None).unwrap().data_variables(), vec!["temp"]);

    let v3u = ZarrHierarchy::open(&FsStore::new(dir.join("v3_unconsolidated.zarr")))
        .await
        .unwrap();
    assert_eq!(v3u.open_mode, OpenMode::V3Walked);
    assert_eq!(v3u.group(None).unwrap().data_variables(), vec!["temp"]);

    // Multiscale nested groups
    let ms = ZarrHierarchy::open(&FsStore::new(dir.join("multiscale.zarr")))
        .await
        .unwrap();
    assert_eq!(ms.group(None).unwrap().subgroups(), vec!["0", "1"]);
    assert_eq!(
        ms.group(Some("0")).unwrap().array("elevation").unwrap().shape,
        vec![256, 256]
    );
    assert_eq!(
        ms.group(Some("1")).unwrap().array("elevation").unwrap().shape,
        vec![128, 128]
    );
}

#[tokio::test]
async fn codec_chain_decodes_v2_v3_and_big_endian_fixtures_identically() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    // Reference: codecs_v3.zarr/uncompressed chunk (0, 0)
    let v3_store = FsStore::new(dir.join("codecs_v3.zarr"));
    let v3_h = ZarrHierarchy::open(&v3_store).await.unwrap();
    let v3_root = v3_h.group(None).unwrap();
    let ref_meta = v3_root.array("uncompressed").unwrap();
    let ref_key = format!("uncompressed/{}", ref_meta.chunk_key(&[0, 0]));
    let ref_raw = v3_store.get(&ref_key).await.unwrap().unwrap();
    let expected = decode_chunk_bytes(ref_meta, &ref_raw, &ref_key).unwrap();
    assert_eq!(expected.len(), 64 * 64 * 4);

    // All v3 codecs (including all Blosc variants) must match `expected` bit-for-bit
    for name in [
        "gzip",
        "zstd",
        "transpose_f_order",
        "blosc_zstd_shuffle",
        "blosc_zstd_bitshuffle",
        "blosc_lz4_shuffle",
        "blosc_lz4_noshuffle",
        "blosc_blosclz_shuffle",
    ] {
        let meta = v3_root.array(name).unwrap();
        let key = format!("{name}/{}", meta.chunk_key(&[0, 0]));
        let raw = v3_store.get(&key).await.unwrap().unwrap();
        let got = decode_chunk_bytes(meta, &raw, &key).unwrap();
        assert_eq!(got, expected, "v3 codec {name} diverged from uncompressed");
    }

    // All v2 codecs (including all Blosc variants) must match `expected` bit-for-bit
    let v2_store = FsStore::new(dir.join("codecs_v2.zarr"));
    let v2_h = ZarrHierarchy::open(&v2_store).await.unwrap();
    let v2_root = v2_h.group(None).unwrap();
    for name in [
        "uncompressed",
        "gzip",
        "zstd",
        "shuffle_zstd",
        "blosc_zstd_shuffle",
        "blosc_zstd_bitshuffle",
        "blosc_lz4_shuffle",
        "blosc_lz4_noshuffle",
        "blosc_blosclz_shuffle",
    ] {
        let meta = v2_root.array(name).unwrap();
        let key = format!("{name}/{}", meta.chunk_key(&[0, 0]));
        let raw = v2_store.get(&key).await.unwrap().unwrap();
        let got = decode_chunk_bytes(meta, &raw, &key).unwrap();
        assert_eq!(got, expected, "v2 codec {name} diverged from uncompressed");
    }

    // Big-endian v3 and v2 f32 arrays must also match `expected` bit-for-bit
    for store_name in ["big_endian_v3.zarr", "big_endian_v2.zarr"] {
        let store = FsStore::new(dir.join(store_name));
        let h = ZarrHierarchy::open(&store).await.unwrap();
        let meta = h.group(None).unwrap().array("f32_be").unwrap();
        let key = format!("f32_be/{}", meta.chunk_key(&[0, 0]));
        let raw = store.get(&key).await.unwrap().unwrap();
        let got = decode_chunk_bytes(meta, &raw, &key).unwrap();
        assert_eq!(got, expected, "{store_name}/f32_be diverged from LE reference");
    }
}
