//! Colours for the values a pixel selection leaves, as rio-tiler makes them.
//!
//! A mosaic made by a pixel selection method is an array of values, not a picture. What
//! turns it into one is the request's rescaling, its colour formula and its colour map,
//! in that order (`ImageData.post_process`, then `ImageData.render`):
//!
//! - a value becomes a byte by `rio_tiler.utils.linear_rescale`, in double precision;
//! - a colour formula works on those bytes;
//! - one band is then looked up in the colour map, or is the grey itself without one;
//! - a pixel is transparent where no band of it has a value, and a band without one
//!   among bands that have is 0 (`rescale_image` and `ImageData._mask`).

use cogzilla_kernel::{ColorFormula, ColormapName};

use super::pixel_selection::Selected;
use crate::engine::BandRequest;

/// A value as a byte, by rio-tiler's `linear_rescale` step for step and in its double
/// precision: clip to the range, take the share of it, scale to 255, drop the fraction.
/// Done in another order or in single precision, a value on a boundary lands a level off.
///
/// The cast is numpy's and the clamp besides: it drops the fraction, and takes what is
/// under 0 (or not a number) as 0 and what is over 255 as 255.
///
/// A range given upside down (`rescale=255,0`) is the one place this leaves rio-tiler,
/// which has no such stretch: numpy's `clip` with its bounds crossed answers the upper
/// one, so every pixel comes out 255. Here it is the ramp inverted, as the renderer
/// draws it for a single scene (`safe_rescale_denom`).
#[inline(always)]
pub(super) fn rescale_byte(v: f32, min: f32, max: f32) -> u8 {
    let (v, min, max) = (f64::from(v), f64::from(min), f64::from(max));
    let span = if (max - min).abs() < 1e-12 { 1e-12 } else { max - min };
    // Either way, what is not a number lands on `min` and so on 0.
    let within = if min <= max { v.max(min).min(max) } else { v.min(min).max(max) };
    ((within - min) / span * 255.0) as u8
}

/// What a colour formula makes of each grey level.
pub(super) fn formula_levels(formula: &ColorFormula) -> [u8; 256] {
    let mut levels = [0u8; 256];
    for (level, out) in levels.iter_mut().enumerate() {
        let mut px = [level as u8, level as u8, level as u8, 255];
        formula.apply_buffer_in_place(&mut px, 4);
        *out = px[0];
    }
    levels
}

/// How the values a pixel selection leaves become colours: the request's own rescaling
/// and colour map, applied as the renderer applies them to a single scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Paint {
    Gray { min: f32, max: f32, colormap: ColormapName },
    Rgb { min: [f32; 3], max: [f32; 3] },
}

impl Paint {
    /// `None` for a request whose pixels are not values put through a rescale (a terrain encoding).
    pub(super) fn of(bands: &BandRequest) -> Option<Self> {
        Some(match bands {
            BandRequest::Single { rescale, colormap, .. }
            | BandRequest::Spectral1 { rescale, colormap, .. }
            | BandRequest::Spectral2 { rescale, colormap, .. }
            | BandRequest::Spectral3 { rescale, colormap, .. }
            | BandRequest::CustomExpr { rescale, colormap, .. }
            | BandRequest::Dem { rescale, colormap, .. } => Self::Gray { min: rescale.0, max: rescale.1, colormap: *colormap },
            BandRequest::Rgb { rescale_min, rescale_max, .. }
            | BandRequest::SarRgb { rescale_min, rescale_max, .. } => Self::Rgb { min: *rescale_min, max: *rescale_max },
            BandRequest::Terrain { .. }
            | BandRequest::CustomExprRgb { .. }
            | BandRequest::CustomExprMulti { .. }
            | BandRequest::MultiBand { .. } => return None,
        })
    }

    /// RGBA for `pixels` pixels. A pixel is transparent when no band of it has a value.
    pub(super) fn rgba(&self, selected: &Selected, pixels: usize) -> Vec<u8> {
        self.paint(selected, pixels, |byte| byte)
    }

    /// As [`Paint::rgba`], with a colour formula's `levels` (see [`formula_levels`]) between
    /// the rescale and the colour map: the formula works on the band's byte and the map is
    /// looked up with what it leaves, which is rio-tiler's order. This is for one band;
    /// a formula for red, green and blue is applied to the finished picture.
    pub(super) fn rgba_through(&self, selected: &Selected, pixels: usize, levels: &[u8; 256]) -> Vec<u8> {
        self.paint(selected, pixels, |byte| levels[usize::from(byte)])
    }

    /// Every pixel is written once and none is decided by a branch: a pixel without a
    /// value is worked out like any other and then left as nothing. That is what keeps
    /// the double precision arithmetic as quick as the renderer's single.
    fn paint(&self, selected: &Selected, pixels: usize, after: impl Fn(u8) -> u8) -> Vec<u8> {
        const NOTHING: [u8; 4] = [0; 4];
        const OPAQUE: u32 = u32::from_le_bytes([0, 0, 0, 255]);
        let painted: Vec<[u8; 4]> = match (self, selected.bands) {
            (Self::Rgb { min, max }, 3) => {
                let (red, rest) = selected.values.split_at(pixels);
                let (green, blue) = rest.split_at(pixels);
                red.iter()
                    .zip(green)
                    .zip(blue)
                    .map(|((r, g), b)| {
                        // A band without a value among bands that have one is 0.
                        let level = |k: usize, v: f32| if v.is_nan() { 0 } else { rescale_byte(v, min[k], max[k]) };
                        let px = [level(0, *r), level(1, *g), level(2, *b), 255];
                        if r.is_nan() && g.is_nan() && b.is_nan() { NOTHING } else { px }
                    })
                    .collect()
            }
            _ => {
                // One band: the request's colour map, or grey for the count of an RGB request.
                let (min, max, colormap) = match self {
                    Self::Gray { min, max, colormap } => (*min, *max, *colormap),
                    Self::Rgb { min, max } => (min[0], max[0], ColormapName::Grayscale),
                };
                let values = selected.values.iter().take(pixels);
                if colormap == ColormapName::Grayscale {
                    // Without a colour map the byte is the grey, as in TiTiler: written as it
                    // is, with no table to look it up in.
                    values
                        .map(|v| {
                            let level = after(rescale_byte(*v, min, max));
                            if v.is_nan() { NOTHING } else { [level, level, level, 255] }
                        })
                        .collect()
                } else {
                    // The table's colours, opaque whatever the table says: what is absent is
                    // for the values to say.
                    let lut = colormap.lut();
                    values
                        .map(|v| {
                            let colour = lut.rgba_u32[usize::from(after(rescale_byte(*v, min, max)))] | OPAQUE;
                            if v.is_nan() { NOTHING } else { colour.to_le_bytes() }
                        })
                        .collect()
                }
            }
        };
        let mut out = painted.into_flattened();
        // Fewer values than pixels is not a picture of them; the rest is nothing.
        out.resize(pixels * 4, 0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(values: &[f32]) -> Selected {
        Selected { bands: 1, values: values.to_vec() }
    }

    /// The renderer's arithmetic, which this module's replaced for a pixel selection: the
    /// range's reciprocal in single precision, multiplied.
    fn single_precision_byte(v: f32, min: f32, max: f32) -> u8 {
        ((v - min) * (255.0 / (max - min))).clamp(0.0, 255.0) as u8
    }

    #[test]
    fn a_value_becomes_a_byte_in_linear_rescales_order() {
        assert_eq!(rescale_byte(1600.0, 0.0, 4000.0), 102);
        assert_eq!(rescale_byte(51.0, 0.0, 255.0), 51);
        assert_eq!(rescale_byte(1168.0, 0.0, 4000.0), 74);
        // Clipped to the range first.
        assert_eq!(rescale_byte(-5.0, 0.0, 100.0), 0);
        assert_eq!(rescale_byte(500.0, 0.0, 100.0), 255);
        assert_eq!(rescale_byte(f32::INFINITY, 0.0, 100.0), 255);
        assert_eq!(rescale_byte(f32::NEG_INFINITY, 0.0, 100.0), 0);
        // Every byte over its own range is itself.
        for level in 0..=255u8 {
            assert_eq!(rescale_byte(f32::from(level), 0.0, 255.0), level);
        }
        // A range with no width: numpy divides nothing by nothing and the cast makes 0 of it.
        assert_eq!(rescale_byte(3.0, 3.0, 3.0), 0);
        assert_eq!(rescale_byte(9.0, 3.0, 3.0), 0);
        assert_eq!(rescale_byte(f32::NAN, 0.0, 100.0), 0);
        // A range upside down is the ramp inverted, where numpy's clip would answer its
        // upper bound and make everything 255.
        assert_eq!(rescale_byte(3.0, 10.0, 0.0), 178);
        assert_eq!(rescale_byte(0.0, 255.0, 0.0), 255);
        assert_eq!(rescale_byte(255.0, 255.0, 0.0), 0);
        assert_eq!(rescale_byte(-5.0, 10.0, 0.0), 255);
        assert_eq!(rescale_byte(50.0, 10.0, 0.0), 0);
        assert_eq!(rescale_byte(f32::NAN, 10.0, 0.0), 0);
        for level in 0..=255u8 {
            assert_eq!(rescale_byte(f32::from(level), 255.0, 0.0), 255 - level);
        }
    }

    #[test]
    fn a_value_on_a_boundary_lands_where_double_precision_puts_it() {
        // (value, min, max, rio-tiler's byte, the single precision byte)
        let cases: [(f32, f32, f32, u8, u8); 5] = [
            // The top of the range is 255, not a level short of it.
            (0.3, 0.0, 0.3, 255, 254),
            (-0.2, -1.0, 1.0, 101, 102),
            (47.058_823, 0.0, 4000.0, 2, 3),
            (4.313_725_5, 0.0, 100.0, 10, 11),
            (117.647_06, 0.0, 10000.0, 2, 3),
        ];
        for (v, min, max, exact, single) in cases {
            assert_eq!(rescale_byte(v, min, max), exact, "{v} over {min}..{max}");
            assert_eq!(single_precision_byte(v, min, max), single, "{v} over {min}..{max} in single precision");
            let grey = Paint::Gray { min, max, colormap: ColormapName::Grayscale };
            assert_eq!(grey.rgba(&one(&[v]), 1), [exact, exact, exact, 255]);
        }
    }

    #[test]
    fn grey_without_a_colour_map_is_the_byte_itself() {
        let levels: Vec<f32> = (0..=255u8).map(f32::from).collect();
        let grey = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Grayscale };
        let rgba = grey.rgba(&one(&levels), 256);
        for (level, px) in rgba.chunks_exact(4).enumerate() {
            assert_eq!(px, [level as u8, level as u8, level as u8, 255], "level {level}");
        }
        // And what the kernel's grey table holds, which draws the same band on its own road.
        let table = ColormapName::Grayscale.lut();
        for level in 0..256 {
            assert_eq!(usize::from(table.rgb[level][0]), level);
            assert_eq!(rgba[level * 4], level as u8);
        }
    }

    #[test]
    fn one_band_goes_through_the_colour_map_and_is_opaque_where_it_has_a_value() {
        let viridis = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Viridis };
        let table = ColormapName::Viridis.lut();
        let rgba = viridis.rgba(&one(&[0.0, f32::NAN, 64.0, 300.0, f32::INFINITY, f32::NEG_INFINITY]), 6);
        let colour = |level: usize| [table.rgb[level][0], table.rgb[level][1], table.rgb[level][2], 255];
        assert_eq!(rgba[..4], colour(0));
        assert_eq!(rgba[4..8], [0, 0, 0, 0]);
        assert_eq!(rgba[8..12], colour(64));
        assert_eq!(rgba[12..16], colour(255));
        // An infinity is a value: clipped to the range like any other, as numpy clips it.
        assert_eq!(rgba[16..20], colour(255));
        assert_eq!(rgba[20..24], colour(0));
        // Of two bands under one band's paint, the first is the picture.
        let two = Selected { bands: 2, values: vec![64.0, f32::NAN, 9.0, 9.0] };
        assert_eq!(viridis.rgba(&two, 2), [colour(64), [0, 0, 0, 0]].concat());
        // One band left of an RGB request (a count) is grey over the first band's range.
        let rgb = Paint::Rgb { min: [0.0, 5.0, 5.0], max: [2.0, 6.0, 6.0] };
        assert_eq!(rgb.rgba(&one(&[1.0, f32::NAN]), 2), [127, 127, 127, 255, 0, 0, 0, 0]);
    }

    #[test]
    fn a_pixel_is_transparent_only_where_no_band_has_a_value() {
        let nan = f32::NAN;
        let rgb = Paint::Rgb { min: [0.0; 3], max: [255.0, 100.0, 10.0] };
        // Pixels: all three, red absent, red and green absent, none; band-major.
        let values = vec![10.0, nan, nan, nan, 50.0, 50.0, nan, nan, 5.0, 5.0, 5.0, nan];
        let rgba = rgb.rgba(&Selected { bands: 3, values }, 4);
        assert_eq!(rgba[..4], [10, 127, 127, 255]);
        // A band without a value among bands that have one is 0, and the pixel is there.
        assert_eq!(rgba[4..8], [0, 127, 127, 255]);
        assert_eq!(rgba[8..12], [0, 0, 127, 255]);
        assert_eq!(rgba[12..16], [0, 0, 0, 0]);
    }

    #[test]
    fn a_colour_formula_works_on_the_byte_before_the_colour_map() {
        let formula = ColorFormula::parse("gamma rgb 2").unwrap();
        let levels = formula_levels(&formula);
        assert_eq!(levels[0], 0);
        assert_eq!(levels[255], 255);
        assert!(levels[64] > 64, "gamma 2 brightens: {}", levels[64]);
        let table = ColormapName::Viridis.lut();
        let viridis = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Viridis };
        let values = one(&[64.0, f32::NAN]);
        let through = usize::from(levels[64]);
        assert_eq!(
            viridis.rgba_through(&values, 2, &levels),
            [table.rgb[through][0], table.rgb[through][1], table.rgb[through][2], 255, 0, 0, 0, 0]
        );
        // Without a colour map what the formula leaves is the grey.
        let grey = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Grayscale };
        assert_eq!(grey.rgba_through(&values, 2, &levels)[..4], [levels[64], levels[64], levels[64], 255]);
        // The levels as they are change nothing.
        let same: [u8; 256] = std::array::from_fn(|i| i as u8);
        assert_eq!(viridis.rgba_through(&values, 2, &same), viridis.rgba(&values, 2));
    }

    #[test]
    fn a_range_upside_down_is_the_ramp_inverted() {
        let values = one(&[0.0, 51.0, 255.0, f32::NAN]);
        let rgba = Paint::Gray { min: 255.0, max: 0.0, colormap: ColormapName::Grayscale }.rgba(&values, 4);
        assert_eq!([rgba[0], rgba[4], rgba[8]], [255, 204, 0]);
        assert_eq!(rgba[12..], [0, 0, 0, 0]);
        // Only the bands whose range is upside down.
        let three = Selected { bands: 3, values: vec![10.0, 10.0, 10.0] };
        let rgb = Paint::Rgb { min: [0.0, 20.0, 0.0], max: [20.0, 0.0, 20.0] };
        assert_eq!(rgb.rgba(&three, 1), [127, 127, 127, 255]);
    }

    #[test]
    fn a_request_names_its_own_paint() {
        let gray = Paint::of(&BandRequest::Single { band_1based: 1, rescale: (0.0, 100.0), colormap: ColormapName::Grayscale }).unwrap();
        assert_eq!(gray, Paint::Gray { min: 0.0, max: 100.0, colormap: ColormapName::Grayscale });
        let rgb = Paint::of(&BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0, 1.0, 2.0], rescale_max: [255.0, 100.0, 10.0] }).unwrap();
        assert_eq!(rgb, Paint::Rgb { min: [0.0, 1.0, 2.0], max: [255.0, 100.0, 10.0] });
        assert!(Paint::of(&BandRequest::MultiBand { bands_1based: vec![1, 2] }).is_none());
    }

    /// The picture a pixel at a time, by `rescale_byte` and nothing else.
    fn plainly(paint: &Paint, selected: &Selected, pixels: usize) -> Vec<u8> {
        let mut out = vec![0u8; pixels * 4];
        match (paint, selected.bands) {
            (Paint::Rgb { min, max }, 3) => {
                for (i, px) in out.chunks_exact_mut(4).enumerate() {
                    let v = [selected.values[i], selected.values[pixels + i], selected.values[2 * pixels + i]];
                    if v.iter().all(|v| v.is_nan()) {
                        continue;
                    }
                    for k in 0..3 {
                        px[k] = if v[k].is_nan() { 0 } else { rescale_byte(v[k], min[k], max[k]) };
                    }
                    px[3] = 255;
                }
            }
            _ => {
                let (min, max, colormap) = match paint {
                    Paint::Gray { min, max, colormap } => (*min, *max, *colormap),
                    Paint::Rgb { min, max } => (min[0], max[0], ColormapName::Grayscale),
                };
                let lut = (colormap != ColormapName::Grayscale).then(|| colormap.lut());
                for (px, v) in out.chunks_exact_mut(4).zip(&selected.values) {
                    if !v.is_nan() {
                        let level = rescale_byte(*v, min, max);
                        match lut {
                            Some(lut) => px[..3].copy_from_slice(&lut.rgb[usize::from(level)]),
                            None => px[..3].fill(level),
                        }
                        px[3] = 255;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_quick_way_paints_what_the_plain_way_does() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let usual: [(f32, f32); 12] = [
            (0.0, 255.0),
            (0.0, 4000.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (0.0, 0.3),
            (100.0, 3500.0),
            (0.0, 65535.0),
            (255.0, 0.0),
            (3.0, 3.0),
            (-1e30, 1e30),
            (0.0, 1e-13),
            (1e-20, 3e-20),
        ];
        let pixels = 61;
        for round in 0..600 {
            let mut range = || {
                let r = next();
                let any = (f32::from_bits((next() >> 32) as u32), f32::from_bits((next() >> 32) as u32));
                if r % 4 == 0 && any.0.is_finite() && any.1.is_finite() { any } else { usual[(r >> 8) as usize % usual.len()] }
            };
            let ranges = [range(), range(), range()];
            let values: Vec<f32> = (0..3 * pixels)
                .map(|i| {
                    let (min, max) = ranges[i / pixels];
                    let r = next();
                    match r % 16 {
                        0 => f32::NAN,
                        1 => f32::from_bits((r >> 32) as u32),
                        2 => [f32::INFINITY, f32::NEG_INFINITY, min, max, 0.0, -0.0][(r >> 8) as usize % 6],
                        // On a level's boundary, and a step either side of it.
                        3..=8 => {
                            let level = ((r >> 8) % 256) as f64;
                            let v = (f64::from(min) + level * (f64::from(max) - f64::from(min)) / 255.0) as f32;
                            match (r >> 20) % 3 {
                                0 => v,
                                1 => f32::from_bits(v.to_bits().wrapping_add(1)),
                                _ => f32::from_bits(v.to_bits().wrapping_sub(1)),
                            }
                        }
                        9..=11 => (min + ((r >> 8) % 70000) as f32).round(),
                        _ => min + (max - min) * ((r >> 40) as f32 / (1u64 << 24) as f32 * 1.2 - 0.1),
                    }
                })
                .collect();
            // The cast alone is the cast after a clamp, for every value there is.
            for (i, v) in values.iter().enumerate() {
                let (min, max) = ranges[i / pixels];
                let clamped = {
                    let (v, min, max) = (f64::from(*v), f64::from(min), f64::from(max));
                    let span = if (max - min).abs() < 1e-12 { 1e-12 } else { max - min };
                    let (low, high) = if min <= max { (min, max) } else { (max, min) };
                    if v.is_nan() { 0 } else { (((v.max(low).min(high) - min) / span) * 255.0).clamp(0.0, 255.0) as u8 }
                };
                assert_eq!(rescale_byte(*v, min, max), clamped, "{v} over {min}..{max}");
            }
            let colormap = [ColormapName::Grayscale, ColormapName::Viridis, ColormapName::Terrain][round % 3];
            let paints = [
                Paint::Rgb { min: [ranges[0].0, ranges[1].0, ranges[2].0], max: [ranges[0].1, ranges[1].1, ranges[2].1] },
                Paint::Gray { min: ranges[0].0, max: ranges[0].1, colormap },
            ];
            for paint in &paints {
                for bands in 1..=3 {
                    let selected = Selected { bands, values: values[..bands * pixels].to_vec() };
                    assert_eq!(paint.rgba(&selected, pixels), plainly(paint, &selected, pixels), "round {round}: {paint:?} of {bands}");
                }
            }
        }
    }
}
