#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDemAlgorithm {
    /// TiTiler's shaded relief in `[0.0, 255.0]`, from the gradient per pixel.
    Hillshade {
        azimuth_deg: f32,
        altitude_deg: f32,
        z_factor: f32,
    },
    /// Terrain slope in degrees `[0.0, 90.0]`.
    Slope { z_factor: f32 },
    /// Compass aspect in degrees `[0.0, 360.0)`.
    Aspect,
    /// Mapzen / AWS Terrarium 24-bit RGB elevation encoding.
    Terrarium,
    /// Mapbox Terrain-RGB 24-bit elevation encoding.
    TerrainRgb,
}
