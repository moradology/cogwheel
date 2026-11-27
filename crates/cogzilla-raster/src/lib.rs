#![forbid(unsafe_code)]

//! Shared raster vocabulary across all cogzilla crates (`SampleType`, `Affine2D`, and `DemAlgorithm`).
//!
//! Kept format-agnostic with zero dependencies so TIFF, Zarr, kernel, codec, projection,
//! IO, GPU, and server crates share one definition of pixel sample types, 2D geotransforms,
//! and DEM algorithm descriptors.

pub mod dem;
pub use dem::DemAlgorithm;

/// Pixel sample type shared by TIFF/COG, Zarr, kernel, codec, and server layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    /// Boolean (1 byte per sample, 0 or 1; produced by Zarr stores).
    Bool,
    /// Unsigned 1-bit packed integer (TIFF `NBITS=1`, unpacked to `uint8` values `0`/`1`).
    Uint1,
    /// Unsigned 4-bit packed integer (TIFF `NBITS=4`, unpacked to `uint8` values `0..=15`).
    Uint4,
    /// Unsigned 8-bit integer (Sentinel-2 TCI, visual RGB, QA masks).
    Uint8,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 16-bit integer (Landsat Collection 2, Sentinel-2 L2A).
    Uint16,
    /// Signed 16-bit integer (SRTM / Copernicus DEMs, MODIS).
    Int16,
    /// Unsigned 32-bit integer (classification / counts).
    Uint32,
    /// Signed 32-bit integer.
    Int32,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Signed 64-bit integer.
    Int64,
    /// IEEE-754 16-bit half-precision float (produced by Zarr stores).
    Float16,
    /// IEEE-754 32-bit float (SAR RTC, DEMs, scientific COGs).
    Float32,
    /// IEEE-754 64-bit float (scientific / geodetic rasters).
    Float64,
}

impl SampleType {
    /// Storage width of one unpacked sample in bytes.
    #[inline(always)]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Bool | Self::Uint1 | Self::Uint4 | Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }

    /// Packed byte width of one uncompressed scanline of `width` pixels × `bands_in_tile` samples.
    /// For sub-byte types (`Uint1`, `Uint4`), TIFF pads each row to a byte boundary.
    #[inline(always)]
    pub const fn row_bytes(self, width: usize, bands_in_tile: usize) -> Option<usize> {
        let Some(samples) = width.checked_mul(bands_in_tile) else {
            return None;
        };
        match self {
            Self::Uint1 => Some(samples.div_ceil(8)),
            Self::Uint4 => Some(samples.div_ceil(2)),
            _ => samples.checked_mul(self.bytes_per_sample()),
        }
    }

    /// Canonical NumPy/Zarr/GDAL dtype name (`"uint16"`, `"float32"`, `"bool"`, `"float16"`, …).
    /// Sub-byte TIFF types (`Uint1`, `Uint4`) report `"uint8"` to match GDAL / TiTiler.
    #[inline(always)]
    pub const fn numpy_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Uint1 | Self::Uint4 | Self::Uint8 => "uint8",
            Self::Int8 => "int8",
            Self::Uint16 => "uint16",
            Self::Int16 => "int16",
            Self::Uint32 => "uint32",
            Self::Int32 => "int32",
            Self::Uint64 => "uint64",
            Self::Int64 => "int64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }

    /// Alias for [`Self::numpy_name`] used by GeoTIFF/GDAL metadata builders.
    #[inline(always)]
    pub const fn gdal_name(self) -> &'static str {
        self.numpy_name()
    }

    /// Parse a canonical NumPy dtype name (`"uint8"`, `"float32"`, `"bool"`, `"float16"`, …).
    #[inline]
    pub fn from_numpy_name(s: &str) -> Option<Self> {
        match s {
            "bool" => Some(Self::Bool),
            "uint8" => Some(Self::Uint8),
            "int8" => Some(Self::Int8),
            "uint16" => Some(Self::Uint16),
            "int16" => Some(Self::Int16),
            "uint32" => Some(Self::Uint32),
            "int32" => Some(Self::Int32),
            "uint64" => Some(Self::Uint64),
            "int64" => Some(Self::Int64),
            "float16" => Some(Self::Float16),
            "float32" => Some(Self::Float32),
            "float64" => Some(Self::Float64),
            _ => None,
        }
    }

    /// Little-endian NumPy `.npy` descriptor string (`"|u1"`, `"<u2"`, `"<f4"`, …).
    #[inline(always)]
    pub const fn npy_descr(self) -> &'static str {
        match self {
            Self::Bool => "|b1",
            Self::Uint1 | Self::Uint4 | Self::Uint8 => "|u1",
            Self::Int8 => "|i1",
            Self::Uint16 => "<u2",
            Self::Int16 => "<i2",
            Self::Uint32 => "<u4",
            Self::Int32 => "<i4",
            Self::Uint64 => "<u8",
            Self::Int64 => "<i8",
            Self::Float16 => "<f2",
            Self::Float32 => "<f4",
            Self::Float64 => "<f8",
        }
    }

    /// Representable `(min, max)` range as `f32` for integer / boolean sample types, or `None`
    /// for floating-point types. Used for default TiTiler rescale ranges and integer clamping
    /// after cubic/lanczos resampling.
    #[inline(always)]
    pub const fn value_range(self) -> Option<(f32, f32)> {
        match self {
            Self::Bool => Some((0.0, 1.0)),
            Self::Uint1 | Self::Uint4 | Self::Uint8 => Some((0.0, 255.0)),
            Self::Int8 => Some((-128.0, 127.0)),
            Self::Uint16 => Some((0.0, 65_535.0)),
            Self::Int16 => Some((-32_768.0, 32_767.0)),
            Self::Uint32 => Some((0.0, u32::MAX as f32)),
            Self::Int32 => Some((i32::MIN as f32, i32::MAX as f32)),
            Self::Uint64 => Some((0.0, u64::MAX as f32)),
            Self::Int64 => Some((i64::MIN as f32, i64::MAX as f32)),
            Self::Float16 | Self::Float32 | Self::Float64 => None,
        }
    }

    /// True when the sample type has values that cannot be represented exactly in IEEE-754 `f32`
    /// (`Uint32`, `Int32`, `Uint64`, `Int64`, `Float64`), requiring `f64` storage on data paths.
    #[inline(always)]
    pub const fn needs_f64(self) -> bool {
        matches!(
            self,
            Self::Uint32 | Self::Int32 | Self::Uint64 | Self::Int64 | Self::Float64
        )
    }
}

/// 2D affine transform mapping `(col, row)` -> `(x, y)` in source CRS:
/// `x = a * col + b * row + c`
/// `y = d * col + e * row + f`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2D {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Affine2D {
    #[inline(always)]
    pub fn forward(&self, col: f64, row: f64) -> (f64, f64) {
        (
            self.a * col + self.b * row + self.c,
            self.d * col + self.e * row + self.f,
        )
    }

    pub fn invert(&self) -> Option<Self> {
        if !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c.is_finite()
            || !self.d.is_finite()
            || !self.e.is_finite()
            || !self.f.is_finite()
        {
            return None;
        }
        let det = self.a * self.e - self.b * self.d;
        if !det.is_finite() || det.abs() < 1e-30 {
            return None;
        }
        let inv = 1.0 / det;
        let a = self.e * inv;
        let b = -self.b * inv;
        let d = -self.d * inv;
        let e = self.a * inv;
        let c = -(a * self.c + b * self.f);
        let f = -(d * self.c + e * self.f);
        if !a.is_finite()
            || !b.is_finite()
            || !c.is_finite()
            || !d.is_finite()
            || !e.is_finite()
            || !f.is_finite()
        {
            return None;
        }
        Some(Self { a, b, c, d, e, f })
    }

    #[inline(always)]
    pub fn is_axis_aligned(&self) -> bool {
        self.b == 0.0 && self.d == 0.0
    }
}

/// Complete TiTiler-compatible band statistics computed on the GPU.
///
/// Lives in `cogzilla-raster` (plain data, no `wgpu` types) so both `cogzilla-gpu` and
/// CPU-only `--no-default-features` builds of `cogzilla-server` share the exact same type
/// definition with zero risk of stub field drift.
#[derive(Debug, Clone)]
pub struct GpuBandStats {
    pub total_pixels: usize,
    pub valid_pixels: usize,
    pub masked_pixels: usize,
    pub valid_percent: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub sum: f64,
    pub std: Option<f64>,
    pub median: Option<f64>,
    pub majority: Option<f64>,
    pub minority: Option<f64>,
    pub unique_bins: usize,
    pub percentiles: Vec<(f64, Option<f64>)>,
    pub hist_counts: Vec<u64>,
    pub hist_edges: Vec<f64>,
}

/// Complete coverage-weighted / area-weighted zonal statistics computed on the GPU.
///
/// Lives in `cogzilla-raster` so GPU and CPU-only builds share one definition.
#[derive(Debug, Clone)]
pub struct GpuWeightedBandStats {
    pub valid_pixels: usize,
    pub masked_pixels: usize,
    pub count_cov: f64,
    pub sum_cov: f64,
    pub w_total: f64,
    pub area_valid_m2: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub majority: f64,
    pub minority: f64,
    pub unique_bins: usize,
    pub percentiles: Vec<(f64, f64)>,
    pub hist_counts: Vec<f64>,
    pub hist_edges: Vec<f64>,
}

/// Pixel-selection reduction mode for GPU mosaic compositing (shared by `cogzilla-gpu` and
/// CPU-only builds of `cogzilla-server`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPixelSelection {
    First,
    Highest,
    Lowest,
    Mean,
    Median,
    Stdev,
    Count,
}

impl GpuPixelSelection {
    #[inline(always)]
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::First => 0,
            Self::Highest => 1,
            Self::Lowest => 2,
            Self::Mean => 3,
            Self::Median => 4,
            Self::Stdev => 5,
            Self::Count => 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_type_facts_are_consistent() {
        let all = [
            SampleType::Bool,
            SampleType::Uint1,
            SampleType::Uint4,
            SampleType::Uint8,
            SampleType::Int8,
            SampleType::Uint16,
            SampleType::Int16,
            SampleType::Uint32,
            SampleType::Int32,
            SampleType::Uint64,
            SampleType::Int64,
            SampleType::Float16,
            SampleType::Float32,
            SampleType::Float64,
        ];
        for st in all {
            assert!(matches!(st.bytes_per_sample(), 1 | 2 | 4 | 8));
            assert_eq!(st.gdal_name(), st.numpy_name());
            let parsed = SampleType::from_numpy_name(st.numpy_name()).unwrap();
            if matches!(st, SampleType::Uint1 | SampleType::Uint4) {
                assert_eq!(parsed, SampleType::Uint8);
            } else {
                assert_eq!(parsed, st);
            }
            assert!(st.npy_descr().len() == 3);
        }
        assert_eq!(SampleType::Uint64.value_range(), Some((0.0, u64::MAX as f32)));
        assert_eq!(
            SampleType::Int64.value_range(),
            Some((i64::MIN as f32, i64::MAX as f32))
        );
        assert_eq!(SampleType::Float16.value_range(), None);
        assert_eq!(SampleType::Float32.value_range(), None);
        assert_eq!(SampleType::Float64.value_range(), None);
    }

    #[test]
    fn affine2d_invert_roundtrips() {
        let aff = Affine2D {
            a: 10.0,
            b: 0.0,
            c: 500_000.0,
            d: 0.0,
            e: -10.0,
            f: 4_500_000.0,
        };
        assert!(aff.is_axis_aligned());
        let inv = aff.invert().unwrap();
        let (x, y) = aff.forward(12.5, 34.25);
        let (col, row) = inv.forward(x, y);
        assert!((col - 12.5).abs() < 1e-9);
        assert!((row - 34.25).abs() < 1e-9);
    }

    #[test]
    fn sample_type_row_bytes_and_needs_f64() {
        // Sub-byte TIFF types pad each row to a byte boundary.
        assert_eq!(SampleType::Uint1.row_bytes(0, 1), Some(0));
        assert_eq!(SampleType::Uint1.row_bytes(1, 1), Some(1));
        assert_eq!(SampleType::Uint1.row_bytes(8, 1), Some(1));
        assert_eq!(SampleType::Uint1.row_bytes(9, 1), Some(2));
        assert_eq!(SampleType::Uint4.row_bytes(1, 1), Some(1));
        assert_eq!(SampleType::Uint4.row_bytes(2, 1), Some(1));
        assert_eq!(SampleType::Uint4.row_bytes(3, 1), Some(2));
        assert_eq!(SampleType::Uint4.row_bytes(3, 3), Some(5));
        assert_eq!(SampleType::Uint16.row_bytes(256, 3), Some(256 * 3 * 2));
        assert_eq!(SampleType::Float64.row_bytes(10, 2), Some(160));
        // Overflow returns None rather than wrapping.
        assert_eq!(SampleType::Uint8.row_bytes(usize::MAX, 2), None);
        assert_eq!(SampleType::Float64.row_bytes(usize::MAX / 4, 1), None);

        // Exactly the 5 wide dtypes require f64 precision.
        assert!(SampleType::Uint32.needs_f64());
        assert!(SampleType::Int32.needs_f64());
        assert!(SampleType::Uint64.needs_f64());
        assert!(SampleType::Int64.needs_f64());
        assert!(SampleType::Float64.needs_f64());
        for narrow in [
            SampleType::Bool,
            SampleType::Uint1,
            SampleType::Uint4,
            SampleType::Uint8,
            SampleType::Int8,
            SampleType::Uint16,
            SampleType::Int16,
            SampleType::Float16,
            SampleType::Float32,
        ] {
            assert!(!narrow.needs_f64(), "{narrow:?} should not require f64");
        }
        assert_eq!(SampleType::from_numpy_name("complex64"), None);
        assert_eq!(SampleType::from_numpy_name(""), None);
    }

    #[test]
    fn affine2d_handles_rotated_singular_and_non_finite() {
        let rotated = Affine2D {
            a: 3.0,
            b: 4.0,
            c: 100.0,
            d: -4.0,
            e: 3.0,
            f: -200.0,
        };
        assert!(!rotated.is_axis_aligned());
        let inv = rotated.invert().unwrap();
        let (x, y) = rotated.forward(7.25, -11.5);
        let (col, row) = inv.forward(x, y);
        assert!((col - 7.25).abs() < 1e-12);
        assert!((row - (-11.5)).abs() < 1e-12);

        // Singular (zero determinant) and non-finite matrices return None.
        let singular = Affine2D { a: 2.0, b: 4.0, c: 0.0, d: 1.0, e: 2.0, f: 0.0 };
        assert_eq!(singular.invert(), None);
        let nan_aff = Affine2D { a: f64::NAN, b: 0.0, c: 0.0, d: 0.0, e: 1.0, f: 0.0 };
        assert_eq!(nan_aff.invert(), None);
        let inf_aff = Affine2D { a: 1.0, b: 0.0, c: f64::INFINITY, d: 0.0, e: 1.0, f: 0.0 };
        assert_eq!(inf_aff.invert(), None);
    }

    #[test]
    fn dem_algorithm_and_pixel_selection_wire_codes() {
        assert_eq!(
            DemAlgorithm::Hillshade { azimuth_deg: 315.0, altitude_deg: 45.0, z_factor: 1.0 }.gpu_mode(),
            Some(0)
        );
        assert_eq!(DemAlgorithm::Slope { z_factor: 1.0 }.gpu_mode(), Some(1));
        assert_eq!(DemAlgorithm::Aspect.gpu_mode(), Some(2));
        assert_eq!(DemAlgorithm::Terrarium.gpu_mode(), Some(3));
        assert_eq!(DemAlgorithm::TerrainRgb.gpu_mode(), Some(4));
        assert_eq!(
            DemAlgorithm::Contours { increment: 10.0, thickness: 1, minz: 0.0, maxz: 100.0 }.gpu_mode(),
            None
        );

        assert!(DemAlgorithm::Terrarium.is_terrain_rgb_encoding());
        assert!(DemAlgorithm::TerrainRgb.is_terrain_rgb_encoding());
        assert!(!DemAlgorithm::Aspect.is_terrain_rgb_encoding());
        assert!(!DemAlgorithm::Slope { z_factor: 1.0 }.is_terrain_rgb_encoding());

        let selections = [
            GpuPixelSelection::First,
            GpuPixelSelection::Highest,
            GpuPixelSelection::Lowest,
            GpuPixelSelection::Mean,
            GpuPixelSelection::Median,
            GpuPixelSelection::Stdev,
            GpuPixelSelection::Count,
        ];
        for (i, sel) in selections.into_iter().enumerate() {
            assert_eq!(sel.as_u32(), i as u32);
        }
    }
}
