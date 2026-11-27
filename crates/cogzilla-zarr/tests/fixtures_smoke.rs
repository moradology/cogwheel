//! Smoke tests against `benches/fixtures/zarr`.
//!
//! Skips cleanly when `benches/fixtures/zarr` is absent, matching the existing COG integration
//! tests.

use std::fs;
use std::path::{Path, PathBuf};

use cogzilla_zarr::{
    ArrayMeta, DataType, Endianness, FsStore, OpenMode, ZarrError, ZarrHierarchy, ZarrStore,
    ZarrVersion, decode_chunk_bytes, read_chunk_window, resolve_geo,
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

#[tokio::test]
async fn sparse_holes_fixture_maps_missing_chunks_to_none() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    let store = FsStore::new(dir.join("sparse_holes.zarr"));
    let h = ZarrHierarchy::open(&store).await.unwrap();
    let meta = h.group(None).unwrap().array("sparse").unwrap();

    let wb = read_chunk_window(&store, "sparse", meta, 0, 0, 192, 192)
        .await
        .unwrap();
    assert_eq!(wb.blocks.len(), 9);
    let some_count = wb.blocks.iter().filter(|b| b.is_some()).count();
    let none_count = wb.blocks.iter().filter(|b| b.is_none()).count();
    assert_eq!(some_count, 3, "expected 3 diagonal chunks present");
    assert_eq!(none_count, 6, "expected 6 unwritten chunks as None holes");

    let refs = wb.block_refs();
    let src = wb.as_band_source(&refs);
    // Chunk (0, 0) has data; chunk (0, 1) is an unwritten hole -> NaN
    assert!(src.sample_checked(10, 10).is_finite());
    assert!(src.sample_checked(80, 10).is_nan());
    assert!(src.sample_checked(80, 80).is_finite());
}

/// Keystone test (#367): a tile rendered from `geotiff_twin.zarr` through `BandSource` and
/// `cogzilla_kernel::render_single_band` must come out **byte-identical** to the same tile
/// rendered from `geotiff_twin.tif` written from the same samples on the same grid.
#[tokio::test]
async fn keystone_zarr_tile_is_byte_identical_to_geotiff_twin() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    use cogzilla_kernel::{
        BandSource, ColormapName, OutputFormat, ResampleMode, SingleBandStyle, render_single_band,
    };
    use cogzilla_proj::CoordPlan;

    // 1. Open GeoTIFF twin and decode all 4 (256x256) tiles
    let tif_bytes = fs::read(dir.join("geotiff_twin.tif")).unwrap();
    let cog = cogzilla_tiff::parse_cog(&tif_bytes).unwrap();
    let ifd = &cog.ifds[0];
    assert_eq!((ifd.width, ifd.height, ifd.tile_width, ifd.tile_height), (512, 512, 256, 256));
    let mut tiff_decoded = Vec::with_capacity(4);
    for tidx in 0..4 {
        let off = ifd.tile_offsets[tidx] as usize;
        let len = ifd.tile_byte_counts[tidx] as usize;
        let dt = cogzilla_codec::decode_tile(
            &tif_bytes[off..off + len],
            ifd,
            cog.byte_order,
            cog.nodata,
        )
        .unwrap();
        tiff_decoded.push(dt);
    }
    let tiff_refs: Vec<Option<&cogzilla_codec::DecodedBand>> =
        tiff_decoded.iter().map(|t| Some(&t.bands[0])).collect();
    let tiff_src = BandSource::new(512, 512, 256, 256, 0, 0, 2, 2, &tiff_refs);

    // 2. Open Zarr twin and read the same 4 (256x256) chunks into BandSource
    let zarr_store = FsStore::new(dir.join("geotiff_twin.zarr"));
    let zarr_h = ZarrHierarchy::open(&zarr_store).await.unwrap();
    let zarr_meta = zarr_h.group(None).unwrap().array("data").unwrap();
    let zarr_wb = read_chunk_window(&zarr_store, "data", zarr_meta, 0, 0, 512, 512)
        .await
        .unwrap();
    let zarr_refs = zarr_wb.block_refs();
    let zarr_src = zarr_wb.as_band_source(&zarr_refs);

    // Build two 256x256 pixel plans:
    // (a) chunk-aligned window [0..256, 0..256] (includes the 16x16 NaN nodata corner!)
    // (b) 4-chunk straddling window [128..384, 128..384] with subpixel offset for bilinear
    let make_plan = |x0: f32, y0: f32| -> CoordPlan<256> {
        let mut u = Box::new([0.0f32; 256]);
        let mut v = Box::new([0.0f32; 256]);
        for i in 0..256 {
            u[i] = x0 + i as f32;
            v[i] = y0 + i as f32;
        }
        CoordPlan::Separable1D {
            u,
            v,
            src_min_col: x0.floor() as i32,
            src_max_col: (x0 + 255.0).ceil() as i32,
            src_min_row: y0.floor() as i32,
            src_max_row: (y0 + 255.0).ceil() as i32,
            all_in_bounds: true,
        }
    };

    let plans = [
        ("chunk_aligned_0_0", make_plan(0.0, 0.0)),
        ("straddling_4_chunks", make_plan(128.25, 128.75)),
    ];

    for (label, plan) in &plans {
        for (resample, format) in [
            (ResampleMode::Nearest, OutputFormat::Npy),
            (ResampleMode::Bilinear, OutputFormat::Npy),
            (ResampleMode::Bilinear, OutputFormat::Png),
        ] {
            let style = SingleBandStyle {
                rescale_min: 100.0,
                rescale_max: 900.0,
                colormap: ColormapName::Viridis,
                resample,
                format,
            };
            let tiff_out = render_single_band::<256>(plan, &tiff_src, &style);
            let zarr_out = render_single_band::<256>(plan, &zarr_src, &style);
            assert_eq!(
                zarr_out, tiff_out,
                "keystone parity mismatch on {label} ({resample:?}, {format:?})"
            );
        }
    }
}

#[tokio::test]
async fn v3_sharded_reads_identically_to_unsharded_via_ranged_reads() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    use cogzilla_zarr::{MemoryStore, read_chunk_window_for_band};

    // Read full 128x128 unsharded reference
    let u_store = FsStore::new(dir.join("v3_unsharded.zarr"));
    let u_h = ZarrHierarchy::open(&u_store).await.unwrap();
    let u_meta = u_h.group(None).unwrap().array("data").unwrap();
    let u_wb = read_chunk_window(&u_store, "data", u_meta, 0, 0, 128, 128)
        .await
        .unwrap();
    let u_refs = u_wb.block_refs();
    let u_src = u_wb.as_band_source(&u_refs);

    // Load v3_sharded.zarr into MemoryStore to assert exact ranged-read count & bytes!
    let s_dir = dir.join("v3_sharded.zarr");
    let mut mem = MemoryStore::new();
    mem.insert("zarr.json", fs::read(s_dir.join("zarr.json")).unwrap());
    for sy in 0..2 {
        for sx in 0..2 {
            let rel = format!("data/c/{sy}/{sx}");
            mem.insert(&rel, fs::read(s_dir.join(&rel)).unwrap());
        }
    }
    let s_h = ZarrHierarchy::open(&mem).await.unwrap();
    let s_meta = s_h.group(None).unwrap().array("data").unwrap();

    // 1. Read a single 64x64 shard (4 inner 32x32 chunks) with concurrency=1 so the first inner
    //    chunk populates the shard index cache and the remaining 3 reuse it:
    //    -> 0 whole-shard get()s (only the 1 zarr.json open GET), 1 index range + 4 chunk ranges!
    let gets_before = mem.get_count();
    let shard0_wb = read_chunk_window_for_band(&mem, "data", s_meta, None, 0, 0, 64, 64, 1)
        .await
        .unwrap();
    assert_eq!(mem.get_count(), gets_before, "sharded reader must not issue whole-shard get()");
    assert_eq!(
        mem.range_count(),
        5,
        "expected 1 shard-index range read + 4 inner-chunk range reads"
    );

    // 2. Read full 128x128 sharded array and verify every single pixel is bit-identical to unsharded
    let s_wb = read_chunk_window(&mem, "data", s_meta, 0, 0, 128, 128)
        .await
        .unwrap();
    let s_refs = s_wb.block_refs();
    let s_src = s_wb.as_band_source(&s_refs);
    for row in 0..128 {
        for col in 0..128 {
            assert_eq!(
                s_src.sample_checked(col, row).to_bits(),
                u_src.sample_checked(col, row).to_bits(),
                "sharded vs unsharded mismatch at ({col}, {row})"
            );
        }
    }
    drop(shard0_wb);
}

#[tokio::test]
async fn crs_and_geo_resolve_correctly_across_all_fixtures() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping: benches/fixtures/zarr not generated");
        return;
    };

    use cogzilla_proj::CrsModel;

    // 1. CF grid_mapping (UTM 10N Transverse Mercator)
    let cf_store = FsStore::new(dir.join("crs_cf_grid_mapping.zarr"));
    let cf_h = ZarrHierarchy::open(&cf_store).await.unwrap();
    let cf_g = cf_h.group(None).unwrap();
    let cf_geo = resolve_geo(&cf_store, &cf_g, cf_g.array("band").unwrap())
        .await
        .unwrap();
    assert!(matches!(cf_geo.crs.model, CrsModel::TransverseMercator(_)));
    assert!((cf_geo.bounds[0] - 500_000.0).abs() < 1e-6);
    assert!((cf_geo.bounds[1] - 4_150_000.0).abs() < 1e-6);
    assert!((cf_geo.bounds[2] - 561_440.0).abs() < 1e-6);
    assert!((cf_geo.bounds[3] - 4_211_440.0).abs() < 1e-6);

    // 2. spatial_ref WKT (EPSG:3857)
    let wkt_store = FsStore::new(dir.join("crs_spatial_ref_wkt.zarr"));
    let wkt_h = ZarrHierarchy::open(&wkt_store).await.unwrap();
    let wkt_g = wkt_h.group(None).unwrap();
    let wkt_geo = resolve_geo(&wkt_store, &wkt_g, wkt_g.array("band").unwrap())
        .await
        .unwrap();
    assert_eq!(wkt_geo.crs.epsg, Some(3857));
    assert!(matches!(wkt_geo.crs.model, CrsModel::Epsg3857));

    // 3. No CRS -> defaults to EPSG:4326
    let none_store = FsStore::new(dir.join("crs_none.zarr"));
    let none_h = ZarrHierarchy::open(&none_store).await.unwrap();
    let none_g = none_h.group(None).unwrap();
    let none_geo = resolve_geo(&none_store, &none_g, none_g.array("band").unwrap())
        .await
        .unwrap();
    assert_eq!(none_geo.crs.epsg, Some(4326));
    assert!(matches!(none_geo.crs.model, CrsModel::Geographic));
    assert!((none_geo.bounds[0] - (-123.0)).abs() < 1e-9);
    assert!((none_geo.bounds[1] - 37.0).abs() < 1e-9);
    assert!((none_geo.bounds[2] - (-121.5)).abs() < 1e-9);
    assert!((none_geo.bounds[3] - 38.5).abs() < 1e-9);

    // 4. GeoTIFF twin affine matches geotiff_twin.tif to < 1e-9
    let tif_bytes = fs::read(dir.join("geotiff_twin.tif")).unwrap();
    let cog = cogzilla_tiff::parse_cog(&tif_bytes).unwrap();
    let tif_aff = cog.ifds[0].pixel_to_crs;
    let twin_store = FsStore::new(dir.join("geotiff_twin.zarr"));
    let twin_h = ZarrHierarchy::open(&twin_store).await.unwrap();
    let twin_g = twin_h.group(None).unwrap();
    let twin_geo = resolve_geo(&twin_store, &twin_g, twin_g.array("data").unwrap())
        .await
        .unwrap();
    assert!((twin_geo.affine.a - tif_aff.a).abs() < 1e-9);
    assert!((twin_geo.affine.c - tif_aff.c).abs() < 1e-9);
    assert!((twin_geo.affine.e - tif_aff.e).abs() < 1e-9);
    assert!((twin_geo.affine.f - tif_aff.f).abs() < 1e-9);

    // 5. 0..360 longitudes + descending Y
    let l360_store = FsStore::new(dir.join("lon_0_360_desc_y.zarr"));
    let l360_h = ZarrHierarchy::open(&l360_store).await.unwrap();
    let l360_g = l360_h.group(None).unwrap();
    let l360_geo = resolve_geo(&l360_store, &l360_g, l360_g.array("sst").unwrap())
        .await
        .unwrap();
    assert!(l360_geo.lon_0_360);
    assert!(l360_geo.affine.e < 0.0);
    assert_eq!(l360_geo.bounds, [-180.0, -90.0, 180.0, 90.0]);

    // 6. Irregular coordinate spacing must be refused with dim name & index
    let irr_store = FsStore::new(dir.join("irregular_coords.zarr"));
    let irr_h = ZarrHierarchy::open(&irr_store).await.unwrap();
    let irr_g = irr_h.group(None).unwrap();
    let err = resolve_geo(&irr_store, &irr_g, irr_g.array("warped").unwrap())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ZarrError::IrregularGrid { dim, index, .. } if dim == "x" && *index == 2),
        "{err}"
    );
}
