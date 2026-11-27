//! Shared DEM derivative and terrain RGB encoding algorithm vocabulary.
//!
//! One declaration shared by `cogzilla-kernel`, `cogzilla-gpu`, and `cogzilla-server`.
//! The `alg_mode` wire integers written into `terrain.wgsl` are derived directly from
//! [`DemAlgorithm::gpu_mode`].

/// On-the-fly DEM derivative algorithms and RGB elevation encodings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DemAlgorithm {
    /// TiTiler's shaded relief in `[0.0, 255.0]`: lit from the gradient per pixel.
    Hillshade {
        azimuth_deg: f32,
        altitude_deg: f32,
        z_factor: f32,
    },
    /// Terrain slope in degrees `[0.0, 90.0]`, over the grid's pixel size in its own units.
    Slope { z_factor: f32 },
    /// Compass aspect in degrees `[0.0, 360.0)` clockwise from North (`0=N, 90=E, 180=S, 270=W`).
    Aspect,
    /// Elevation contour lines at multiples of `increment` overlaid on `[minz, maxz]`.
    Contours {
        increment: f32,
        thickness: u32,
        minz: f32,
        maxz: f32,
    },
    /// Mapzen / AWS Terrarium: `elev = (R * 256 + G + B / 256) - 32768` (~3.9 mm precision).
    Terrarium,
    /// Mapbox Terrain-RGB: `elev = -10000 + ((R * 65536 + G * 256 + B) * 0.1)` (0.1 m precision).
    TerrainRgb,
}

impl DemAlgorithm {
    /// The `alg_mode` written into `terrain.wgsl`'s uniform params, or `None` when the
    /// algorithm has no GPU shader path (`Contours`).
    #[inline(always)]
    pub const fn gpu_mode(self) -> Option<u32> {
        match self {
            Self::Hillshade { .. } => Some(0),
            Self::Slope { .. } => Some(1),
            Self::Aspect => Some(2),
            Self::Terrarium => Some(3),
            Self::TerrainRgb => Some(4),
            Self::Contours { .. } => None,
        }
    }

    /// True for direct 24-bit RGB elevation encodings (`Terrarium`, `TerrainRgb`) that run
    /// through the `dem_terrain_rgb` entry point rather than the 3x3 stencil derivative path.
    #[inline(always)]
    pub const fn is_terrain_rgb_encoding(self) -> bool {
        matches!(self, Self::Terrarium | Self::TerrainRgb)
    }
}
