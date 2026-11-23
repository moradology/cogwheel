use std::path::PathBuf;

use cogwheel_codec::decode_tile;
use cogwheel_tiff::parse_cog;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures")
}

fn decode_all_tiles_band0(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read fixture");
    let cog = parse_cog(&bytes).expect("parse cog");
    let ifd = cog.full_res();
    let mut out = Vec::new();
    for ty in 0..ifd.tiles_down {
        for tx in 0..ifd.tiles_across {
            let (off, len) = ifd.tile_byte_range(tx, ty, 0).expect("non-sparse tile");
            let slice = &bytes[off as usize..(off as usize + len as usize)];
            let decoded = decode_tile(slice, ifd, cog.byte_order, cog.nodata).expect("decode");
            out.extend_from_slice(&decoded.bands[0].data);
        }
    }
    out
}

#[test]
fn all_codecs_and_predictors_are_bit_exact_against_uncompressed() {
    let dir = fixtures_dir();
    let ref_path = dir.join("codec_none_p1.tif");
    if !ref_path.exists() {
        eprintln!("fixtures not generated; skipping");
        return;
    }
    let reference = decode_all_tiles_band0(&ref_path);
    assert_eq!(reference.len(), 1024 * 1024);

    let cases = [
        "codec_deflate_p1.tif",
        "codec_deflate_p2.tif",
        "codec_deflate_p3.tif",
        "codec_zstd_p1.tif",
        "codec_zstd_p3.tif",
        "codec_lzw_p1.tif",
        "codec_lzw_p3.tif",
    ];
    for name in cases {
        let decoded = decode_all_tiles_band0(&dir.join(name));
        assert_eq!(decoded.len(), reference.len(), "{name}: length mismatch");
        for (i, (&a, &b)) in decoded.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: bit mismatch at idx {i}: got {a} ({:#010x}), expected {b} ({:#010x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

#[test]
fn satellite_multiband_chunky_zstd_p3_and_coastal_sparse() {
    let dir = fixtures_dir();
    let sat_path = dir.join("satellite_utm10n_rgb_zstd_p3.tif");
    if !sat_path.exists() {
        return;
    }
    let bytes = std::fs::read(&sat_path).unwrap();
    let cog = parse_cog(&bytes).unwrap();
    assert_eq!(cog.ifds.len(), 4); // full res + 2, 4, 8 overviews
    assert_eq!(cog.geo_keys.projected_cs_type, Some(32610));
    let ifd = cog.full_res();
    assert_eq!(ifd.samples_per_pixel, 3);
    // Center tile (1, 1) should have valid reflectance values in [0, 1] and no nodata
    let (off, len) = ifd.tile_byte_range(1, 1, 0).unwrap();
    let decoded = decode_tile(&bytes[off as usize..off as usize + len as usize], ifd, cog.byte_order, cog.nodata).unwrap();
    assert_eq!(decoded.bands.len(), 3);
    for b in 0..3 {
        assert!(!decoded.bands[b].stats.all_nodata);
        assert!(decoded.bands[b].stats.min >= 0.0 && decoded.bands[b].stats.max <= 1.0);
    }

    // Coastal sparse COG: western tiles (tx < 4) are all NaN or sparse
    let coast_bytes = std::fs::read(dir.join("coastal_sparse_4k_zstd_p3.tif")).unwrap();
    let coast_cog = parse_cog(&coast_bytes).unwrap();
    let cifd = coast_cog.full_res();
    if let Some((off, len)) = cifd.tile_byte_range(0, 0, 0) {
        let dec = decode_tile(
            &coast_bytes[off as usize..off as usize + len as usize],
            cifd,
            coast_cog.byte_order,
            coast_cog.nodata,
        )
        .unwrap();
        assert!(dec.bands[0].stats.all_nodata, "western ocean tile must be all_nodata");
    }
}
