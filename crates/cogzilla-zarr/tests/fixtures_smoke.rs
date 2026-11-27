//! Smoke tests against `benches/fixtures/zarr`.
//!
//! Skips cleanly when `benches/fixtures/zarr` is absent, matching the existing COG integration
//! tests.

use std::fs;
use std::path::{Path, PathBuf};

use cogzilla_zarr::{ArrayMeta, DataType, Endianness, ZarrVersion};

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
