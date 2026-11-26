//! GeoTIFF output.

#[cfg(test)]
mod golden_tests {
    //! Byte-exact pins of `encode_geotiff_rgba` / `encode_geotiff_f32` output: every case records
    //! the exact length and a 128-bit FNV-1a digest of the encoded file.

    use bytes::Bytes;

    use crate::titiler::{encode_geotiff_f32, encode_geotiff_rgba};

    const MERC: [f64; 4] = [-20037508.342789244, -7.081154551613622e-10, 0.0, 20037508.342789244];
    const UTM: [f64; 4] = [399960.0, 4890240.0, 509760.0, 5000040.0];
    const GEO: [f64; 4] = [8.25, 45.5, 9.75, 46.125];
    const ANTIMERIDIAN: [f64; 4] = [170.0, -10.5, -170.0, 10.25];

    fn fnv1a_128(bytes: &[u8]) -> u128 {
        let mut hash: u128 = 0x6c62272e07bb014262b821756295c58d;
        for &b in bytes {
            hash ^= u128::from(b);
            hash = hash.wrapping_mul(0x0000000001000000000000000000013b);
        }
        hash
    }

    /// Deterministic byte stream (64-bit LCG, high byte of each step).
    fn noise(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    enum Alpha {
        Opaque,
        Mixed,
        Clear,
    }

    fn rgba(seed: u64, w: usize, h: usize, gray: bool, alpha: Alpha) -> Vec<u8> {
        let mut px = noise(seed, w * h * 4);
        for (i, p) in px.chunks_exact_mut(4).enumerate() {
            if gray {
                p[1] = p[0];
                p[2] = p[0];
            }
            p[3] = match alpha {
                Alpha::Opaque => 255,
                Alpha::Mixed => [0, 255, 0, 128, 255][i % 5],
                Alpha::Clear => 0,
            };
        }
        px
    }

    /// Band-major little-endian `f32` planes; with `holes`, some pixels are NaN in every band
    /// (mask 0), some in one band only (mask stays valid), and one sample is infinite.
    fn f32_chw(seed: u64, w: usize, h: usize, bands: usize, holes: bool) -> Vec<u8> {
        let n_px = w * h;
        let raw = noise(seed, n_px * bands * 2);
        let mut out = Vec::with_capacity(n_px * bands * 4);
        for b in 0..bands {
            for px in 0..n_px {
                let i = (b * n_px + px) * 2;
                let mut v = f32::from(u16::from_le_bytes([raw[i], raw[i + 1]])) / 97.0 - 300.0;
                if holes {
                    if px % 4 == 1 || (px % 7 == 3 && b == 0) {
                        v = f32::NAN;
                    } else if px % 11 == 5 && b + 1 == bands {
                        v = f32::INFINITY;
                    }
                }
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn cases() -> Vec<(&'static str, Bytes)> {
        let mut out: Vec<(&'static str, Bytes)> = Vec::new();
        let mut rgba_case = |name, w: u32, h: u32, px: &[u8], bbox, epsg, mask| {
            out.push((name, encode_geotiff_rgba(w, h, px, bbox, epsg, mask)));
        };
        let colour = rgba(1, 8, 8, false, Alpha::Opaque);
        rgba_case("rgba colour opaque 8x8 3857", 8, 8, &colour, MERC, 3857, false);
        rgba_case("rgba colour opaque 8x8 3857 mask", 8, 8, &colour, MERC, 3857, true);
        let colour_holes = rgba(2, 8, 8, false, Alpha::Mixed);
        rgba_case("rgba colour transparent 8x8 3857", 8, 8, &colour_holes, MERC, 3857, false);
        rgba_case("rgba colour transparent 8x8 3857 mask", 8, 8, &colour_holes, MERC, 3857, true);
        let gray = rgba(3, 8, 8, true, Alpha::Opaque);
        rgba_case("rgba gray opaque 8x8 4326", 8, 8, &gray, GEO, 4326, false);
        rgba_case("rgba gray opaque 8x8 4326 mask", 8, 8, &gray, GEO, 4326, true);
        // Gray detection only looks at pixels with alpha > 0: colour under transparent pixels.
        let mut gray_holes = rgba(4, 8, 8, true, Alpha::Mixed);
        for p in gray_holes.chunks_exact_mut(4).filter(|p| p[3] == 0) {
            p[1] = p[0].wrapping_add(17);
        }
        rgba_case("rgba gray transparent 8x8 4326", 8, 8, &gray_holes, GEO, 4326, false);
        rgba_case("rgba gray transparent 8x8 4326 mask", 8, 8, &gray_holes, GEO, 4326, true);
        let clear = rgba(5, 8, 8, true, Alpha::Clear);
        rgba_case("rgba all transparent 8x8 3857", 8, 8, &clear, MERC, 3857, false);
        rgba_case("rgba all transparent 8x8 3857 mask", 8, 8, &clear, MERC, 3857, true);
        let wide = rgba(6, 7, 3, false, Alpha::Mixed);
        rgba_case("rgba colour 7x3 4326", 7, 3, &wide, GEO, 4326, false);
        rgba_case("rgba colour 7x3 4326 mask", 7, 3, &wide, GEO, 4326, true);
        rgba_case("rgba colour 7x3 antimeridian 4326 mask", 7, 3, &wide, ANTIMERIDIAN, 4326, true);
        rgba_case("rgba colour 7x3 antimeridian 3857 mask", 7, 3, &wide, ANTIMERIDIAN, 3857, true);
        rgba_case("rgba colour 7x3 32633", 7, 3, &wide, UTM, 32633, false);
        rgba_case("rgba colour 7x3 4269 geographic", 7, 3, &wide, GEO, 4269, false);
        rgba_case("rgba colour 7x3 unknown epsg", 7, 3, &wide, GEO, 1, true);
        let tall = rgba(7, 3, 9, true, Alpha::Mixed);
        rgba_case("rgba gray 3x9 3857 mask", 3, 9, &tall, MERC, 3857, true);
        // Buffer shorter than width*height*4: the strip is zero-padded.
        rgba_case("rgba colour short buffer", 8, 8, &colour[..100], MERC, 3857, false);
        rgba_case("rgba colour short buffer mask", 8, 8, &colour_holes[..101], MERC, 3857, true);
        rgba_case("rgba gray short buffer", 8, 8, &gray[..100], MERC, 3857, false);
        rgba_case("rgba gray short buffer mask", 8, 8, &gray[..102], MERC, 3857, true);
        // Buffer longer than width*height*4: the tail is ignored.
        rgba_case("rgba colour long buffer", 4, 8, &colour, MERC, 3857, false);
        rgba_case("rgba colour long buffer mask", 4, 8, &colour, MERC, 3857, true);
        rgba_case("rgba empty buffer", 2, 2, &[], MERC, 3857, true);
        // Dimensions clamp to 1..=4096.
        rgba_case("rgba zero width", 0, 4, &colour, MERC, 3857, false);
        rgba_case("rgba zero size mask", 0, 0, &colour, GEO, 4326, true);
        rgba_case("rgba 5000x2 clamps to 4096", 5000, 2, &colour, MERC, 3857, true);
        rgba_case("rgba 2x5000 clamps to 4096", 2, 5000, &gray, GEO, 4326, false);
        rgba_case("rgba degenerate bbox", 8, 8, &colour, [5.0, 5.0, 5.0, 5.0], 4326, false);

        let mut f32_case = |name, w: u32, h: u32, data: &[u8], bbox, epsg, mask| {
            out.push((name, encode_geotiff_f32(w, h, data, bbox, epsg, mask)));
        };
        let one = f32_chw(11, 8, 8, 1, false);
        f32_case("f32 1 band 8x8 3857", 8, 8, &one, MERC, 3857, false);
        f32_case("f32 1 band 8x8 3857 mask", 8, 8, &one, MERC, 3857, true);
        let one_holes = f32_chw(12, 8, 8, 1, true);
        f32_case("f32 1 band holes 8x8 4326", 8, 8, &one_holes, GEO, 4326, false);
        f32_case("f32 1 band holes 8x8 4326 mask", 8, 8, &one_holes, GEO, 4326, true);
        let two = f32_chw(13, 8, 8, 2, true);
        f32_case("f32 2 bands 8x8 3857", 8, 8, &two, MERC, 3857, false);
        f32_case("f32 2 bands 8x8 3857 mask", 8, 8, &two, MERC, 3857, true);
        let three = f32_chw(14, 7, 3, 3, true);
        f32_case("f32 3 bands 7x3 4326", 7, 3, &three, GEO, 4326, false);
        f32_case("f32 3 bands 7x3 4326 mask", 7, 3, &three, GEO, 4326, true);
        f32_case("f32 3 bands 7x3 antimeridian 4326", 7, 3, &three, ANTIMERIDIAN, 4326, false);
        f32_case("f32 3 bands 7x3 antimeridian 3857", 7, 3, &three, ANTIMERIDIAN, 3857, false);
        f32_case("f32 3 bands 7x3 32633 mask", 7, 3, &three, UTM, 32633, true);
        f32_case("f32 3 bands 7x3 4269 geographic", 7, 3, &three, GEO, 4269, false);
        f32_case("f32 3 bands 7x3 unknown epsg", 7, 3, &three, GEO, 1, true);
        let four = f32_chw(15, 3, 9, 4, true);
        f32_case("f32 4 bands 3x9 3857", 3, 9, &four, MERC, 3857, false);
        f32_case("f32 4 bands 3x9 3857 mask", 3, 9, &four, MERC, 3857, true);
        let five = f32_chw(16, 4, 4, 5, true);
        f32_case("f32 5 bands 4x4 3857", 4, 4, &five, MERC, 3857, false);
        f32_case("f32 5 bands 4x4 3857 mask", 4, 4, &five, MERC, 3857, true);
        // More than 15 planes: the band count clamps to 15.
        let many = f32_chw(17, 2, 2, 17, false);
        f32_case("f32 17 bands 2x2 clamps to 15", 2, 2, &many, MERC, 3857, false);
        f32_case("f32 17 bands 2x2 clamps to 15 mask", 2, 2, &many, MERC, 3857, true);
        // Buffer shorter than one plane: one band, NaN-padded.
        f32_case("f32 short buffer", 8, 8, &one[..100], MERC, 3857, false);
        f32_case("f32 short ragged buffer", 8, 8, &one[..103], MERC, 3857, false);
        f32_case("f32 short buffer mask", 8, 8, &one_holes[..100], MERC, 3857, true);
        f32_case("f32 empty buffer", 2, 2, &[], MERC, 3857, false);
        f32_case("f32 empty buffer mask", 2, 2, &[], GEO, 4326, true);
        // Trailing partial plane: the band count rounds down and the tail is ignored.
        f32_case("f32 2.5 planes", 8, 8, &two[..two.len() - 30], MERC, 3857, false);
        f32_case("f32 1 band ragged tail", 8, 8, &two[..one.len() + 6], MERC, 3857, false);
        f32_case("f32 2.5 planes mask", 4, 8, &two[..320], MERC, 3857, true);
        // Dimensions clamp to 1..=4096.
        f32_case("f32 zero width", 0, 4, &one, MERC, 3857, false);
        f32_case("f32 zero size mask", 0, 0, &one, GEO, 4326, true);
        f32_case("f32 5000x1 clamps to 4096", 5000, 1, &one, MERC, 3857, true);
        f32_case("f32 1x5000 clamps to 4096", 1, 5000, &one, GEO, 4326, false);
        f32_case("f32 degenerate bbox", 8, 8, &one, [5.0, 5.0, 5.0, 5.0], 4326, false);
        out
    }

    /// `(case, exact byte length, FNV-1a 128 of the bytes)`.
    #[rustfmt::skip]
    const GOLDEN: &[(&str, usize, u128)] = &[
        ("rgba colour opaque 8x8 3857", 472, 0x40b39d95e1dd8242cd4e61314b2c883e),
        ("rgba colour opaque 8x8 3857 mask", 550, 0xf2788e72fef190c2149992bd0ad51142),
        ("rgba colour transparent 8x8 3857", 472, 0x9ea75746088952c6fd1c2110644f8033),
        ("rgba colour transparent 8x8 3857 mask", 550, 0x0daaeb6cb274306c0bb60d18c0313598),
        ("rgba gray opaque 8x8 4326", 338, 0x2794fc34fb532e35d94759148881cdea),
        ("rgba gray opaque 8x8 4326 mask", 414, 0x8db51fc3e8a7f4d72a4c1972c2b997e4),
        ("rgba gray transparent 8x8 4326", 338, 0x7493896440638fc67fb76cef663c8efd),
        ("rgba gray transparent 8x8 4326 mask", 414, 0x9319f6be7aa1a7b185529ec0ac966d8e),
        ("rgba all transparent 8x8 3857", 472, 0x2b4daaa07709ef679a011a6f8c8bcc58),
        ("rgba all transparent 8x8 3857 mask", 550, 0xa94bbefbb0a18655d44f2a5190253fc6),
        ("rgba colour 7x3 4326", 343, 0x076b1662af032024a2adeb99dadc0ca5),
        ("rgba colour 7x3 4326 mask", 378, 0x1e3ca78cdc500be2e43a2db7d16c7e51),
        ("rgba colour 7x3 antimeridian 4326 mask", 378, 0x7978095b925f0cb102652db90ecc241e),
        ("rgba colour 7x3 antimeridian 3857 mask", 378, 0x6020e3a1afdc59f9484f145772bbb8b2),
        ("rgba colour 7x3 32633", 343, 0xe73936a6b077f9bd9848725a02c50344),
        ("rgba colour 7x3 4269 geographic", 343, 0x6a0ec148dad417d07ff62d403dc9a944),
        ("rgba colour 7x3 unknown epsg", 378, 0x590b512ee96a0a1eedb1333cfaf81a39),
        ("rgba gray 3x9 3857 mask", 340, 0xf88e601763d18fab5dd4cb008a765fd8),
        ("rgba colour short buffer", 472, 0xe1df2d707d22aaccc0a616b02d35bc34),
        ("rgba colour short buffer mask", 550, 0x41fc6a168ec981eae90169704fb6fd16),
        ("rgba gray short buffer", 338, 0x16b6a1de3f464a033cc7b15f5ec0a1c5),
        ("rgba gray short buffer mask", 414, 0x19d4b5c05f9ce6a4e6150105b0cf3c70),
        ("rgba colour long buffer", 376, 0x6dfd48bae9edc6882312c2e69452dd5d),
        ("rgba colour long buffer mask", 422, 0xcfdc98d6088fd5459184a0b51dc54faa),
        ("rgba empty buffer", 310, 0xaa88dbae3e02844be81b95f5a3674e63),
        ("rgba zero width", 292, 0xd385a0fd37e237113d972b6fdff74d24),
        ("rgba zero size mask", 298, 0x1c3546afc2f5571eed355c28ee240086),
        ("rgba 5000x2 clamps to 4096", 33062, 0x37a63c2171ea4e25e7c3486ea9dbbd1a),
        ("rgba 2x5000 clamps to 4096", 8466, 0x91a0617ef64144c3b603af220996a770),
        ("rgba degenerate bbox", 472, 0x3355ba441ba3ada133eafe2c30eba037),
        ("f32 1 band 8x8 3857", 554, 0xdde4d748674ef2bc81d021240a98ebcf),
        ("f32 1 band 8x8 3857 mask", 822, 0x08db91822f415632d017d03b0bdf0c44),
        ("f32 1 band holes 8x8 4326", 554, 0x73fa21f219cfcb212fd0c1e6cca7c510),
        ("f32 1 band holes 8x8 4326 mask", 822, 0xe58604386c30557b1ce5dbc5dfdfa66f),
        ("f32 2 bands 8x8 3857", 822, 0xfd4ce31b6df242eeb08d326262699eac),
        ("f32 2 bands 8x8 3857 mask", 1090, 0x11265b727a01fe9311caf16178e1ac38),
        ("f32 3 bands 7x3 4326", 562, 0x632fddbcee08f0ad83c0c80e621bb1a6),
        ("f32 3 bands 7x3 4326 mask", 662, 0xa11fd9350bf4823646aedcd728e2b2f1),
        ("f32 3 bands 7x3 antimeridian 4326", 562, 0xaf6e112717a280ed5655be47dc8bea25),
        ("f32 3 bands 7x3 antimeridian 3857", 562, 0xeacca17411ba330b65b2c907c25b1bb1),
        ("f32 3 bands 7x3 32633 mask", 662, 0xb88f10ed14fcd11e3bf56d9be6716abe),
        ("f32 3 bands 7x3 4269 geographic", 562, 0xe41253ffd99587bff9bfca6a8d041235),
        ("f32 3 bands 7x3 unknown epsg", 662, 0x2753918e75e330db5a7e0db6bd8b1949),
        ("f32 4 bands 3x9 3857", 764, 0xc4bf027ac462ad26a47c425ad6cb078a),
        ("f32 4 bands 3x9 3857 mask", 878, 0x5cb9dbbfdb3238bd60657adf68bcb841),
        ("f32 5 bands 4x4 3857", 658, 0x205b3b62431759c71f590198d3831d3c),
        ("f32 5 bands 4x4 3857 mask", 728, 0x74f1828bfabd673cde86a4fc34f5d3ed),
        ("f32 17 bands 2x2 clamps to 15", 638, 0x2a738330881c87d8e93c49f6ade4e1e8),
        ("f32 17 bands 2x2 clamps to 15 mask", 660, 0x61e6dcc95dccf30856b256f2c8cc804a),
        ("f32 short buffer", 554, 0x1be98b50444e6b34ec3b712059699c72),
        ("f32 short ragged buffer", 554, 0x1be98b50444e6b34ec3b712059699c72),
        ("f32 short buffer mask", 822, 0xdafcc46b7bc612559c77c3783340626a),
        ("f32 empty buffer", 314, 0xa26b64f76a69388a8bcb8436089db1bf),
        ("f32 empty buffer mask", 342, 0xdde2eb1040e5b226fb4653f317f95ae7),
        ("f32 2.5 planes", 554, 0x0b296f29fb92d4025d623bdc68dca791),
        ("f32 1 band ragged tail", 554, 0x0b296f29fb92d4025d623bdc68dca791),
        ("f32 2.5 planes mask", 706, 0x15259173be79ae4d8dcd26193187e3ac),
        ("f32 zero width", 638, 0xe1a00199c0d3744b183d8994904236e8),
        ("f32 zero size mask", 468, 0x9a8bb8bbb1d3f5199de9019207180a83),
        ("f32 5000x1 clamps to 4096", 33078, 0xd67aadf53f8cbde3572bd480f11453c9),
        ("f32 1x5000 clamps to 4096", 16682, 0xc2d46d38acc6ae0e082da46b4bf4970b),
        ("f32 degenerate bbox", 554, 0x3b22078a2854a2e35ee1684eeb246636),
    ];

    #[test]
    fn existing_geotiff_encoders_are_byte_stable() {
        let cases = cases();
        if std::env::var_os("COGZILLA_PRINT_GEOTIFF_GOLDEN").is_some() {
            for (name, bytes) in &cases {
                println!("        ({name:?}, {}, 0x{:032x}),", bytes.len(), fnv1a_128(bytes));
            }
        }
        assert_eq!(cases.len(), GOLDEN.len(), "golden table and case list differ in length");
        for ((name, bytes), (want_name, want_len, want_hash)) in cases.iter().zip(GOLDEN) {
            assert_eq!(name, want_name, "golden table order");
            assert_eq!(bytes.len(), *want_len, "{name}: length changed");
            assert_eq!(fnv1a_128(bytes), *want_hash, "{name}: bytes changed");
        }
    }

    /// The digest is only a pin if it distinguishes neighbouring files.
    #[test]
    fn golden_digest_is_sensitive_to_single_bytes() {
        let a = noise(1, 512);
        let mut b = a.clone();
        b[511] ^= 1;
        assert_ne!(fnv1a_128(&a), fnv1a_128(&b));
        assert_eq!(fnv1a_128(b""), 0x6c62272e07bb014262b821756295c58d);
        assert_eq!(fnv1a_128(b"a"), 0xd228cb696f1a8caf78912b704e4a8964);
    }
}
