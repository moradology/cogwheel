/// Whether 2D sample coordinate `(u, v)` lies inside the raster's pixel-center
/// sample extent `[-0.5, src_w - 0.5] x [-0.5, src_h - 0.5]` (inclusive, finite).
///
/// Source pixels are indexed by pixel center (`0.0` = center of pixel 0,
/// `src_w - 1.0` = center of the last pixel), matching `BandSource::contains_sample`
/// on the CPU side.
fn in_sample_extent(u: f32, v: f32) -> bool {
    return is_finite_f32(u) && is_finite_f32(v)
        && u >= -0.5 && v >= -0.5
        && u <= f32(params.src_w) - 0.5
        && v <= f32(params.src_h) - 0.5;
}
