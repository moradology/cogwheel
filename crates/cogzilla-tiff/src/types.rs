use thiserror::Error;

#[derive(Debug, Error)]
pub enum TiffError {
    #[error("truncated or need more bytes: required offset {required} exceeds available {available}")]
    NeedMoreBytes { required: u64, available: u64 },
    #[error("invalid TIFF header magic or byte order")]
    InvalidHeader,
    #[error("unsupported sample format: cogzilla supports f32, u16, i16, u8, got format={format}, bits={bits}")]
    UnsupportedSampleFormat { format: u16, bits: u16 },
    #[error("unsupported compression code: {0}")]
    UnsupportedCompression(u16),
    #[error("unsupported predictor code: {0}")]
    UnsupportedPredictor(u16),
    #[error("TIFF is not tiled (missing TileWidth/TileLength/TileOffsets)")]
    NotTiled,
    #[error("missing georeferencing tags (ModelPixelScale/Tiepoint or ModelTransformation)")]
    MissingGeoTransform,
    #[error("corrupt TIFF structure: {0}")]
    Corrupt(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    None,
    Lzw,
    Jpeg,
    Deflate,
    Zstd,
    Webp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predictor {
    None,
    HorizontalInt,
    FloatingPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
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
    /// IEEE-754 32-bit float (SAR RTC, DEMs, scientific COGs).
    Float32,
    /// IEEE-754 64-bit float (scientific / geodetic rasters).
    Float64,
}

impl SampleType {
    #[inline(always)]
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }

    pub fn gdal_name(self) -> &'static str {
        match self {
            Self::Uint8 => "uint8",
            Self::Int8 => "int8",
            Self::Uint16 => "uint16",
            Self::Int16 => "int16",
            Self::Uint32 => "uint32",
            Self::Int32 => "int32",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanarConfig {
    /// Pixel-interleaved: `[b0, b1, b2, ...]` per pixel (TIFF value 1).
    Chunky,
    /// Band-separate: separate tile plane per band (TIFF value 2).
    Planar,
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

/// Raw GeoTIFF GeoKeys extracted from IFD 0 for closed-form CRS construction.
#[derive(Debug, Clone, Default)]
pub struct GeoKeys {
    pub model_type: Option<u16>,
    pub raster_type: Option<u16>,
    pub geographic_type: Option<u16>,
    pub projected_cs_type: Option<u16>,
    pub proj_coord_trans: Option<u16>,
    pub proj_linear_units: Option<u16>,
    pub semi_major_axis: Option<f64>,
    /// GeogSemiMinorAxisGeoKey (2058).
    pub semi_minor_axis: Option<f64>,
    pub inv_flattening: Option<f64>,
    pub std_parallel_1: Option<f64>,
    pub std_parallel_2: Option<f64>,
    pub nat_origin_long: Option<f64>,
    pub nat_origin_lat: Option<f64>,
    pub false_easting: Option<f64>,
    pub false_northing: Option<f64>,
    pub center_long: Option<f64>,
    pub center_lat: Option<f64>,
    pub scale_at_nat_origin: Option<f64>,
    /// GeogGeodeticDatumGeoKey (2050).
    pub geodetic_datum: Option<u16>,
    /// GeogPrimeMeridianGeoKey (2051); 8901 is Greenwich.
    pub prime_meridian: Option<u16>,
    /// GeogPrimeMeridianLongGeoKey (2061); 0.0 is Greenwich.
    pub prime_meridian_long: Option<f64>,
    /// GeogAngularUnitsGeoKey (2054); 9102 is degrees.
    pub geog_angular_units: Option<u16>,
    /// GeogEllipsoidGeoKey (2056); 7019 is GRS80, 7030 is WGS84.
    pub ellipsoid: Option<u16>,
    /// ProjFalseOriginLongGeoKey (3084) and ProjFalseOriginLatGeoKey (3085).
    pub false_origin_long: Option<f64>,
    pub false_origin_lat: Option<f64>,
    /// ProjFalseOriginEastingGeoKey (3086) and ProjFalseOriginNorthingGeoKey (3087).
    pub false_origin_easting: Option<f64>,
    pub false_origin_northing: Option<f64>,
    /// ProjCenterEastingGeoKey (3090) and ProjCenterNorthingGeoKey (3091).
    pub center_easting: Option<f64>,
    pub center_northing: Option<f64>,
    /// ProjStraightVertPoleLongGeoKey (3095): the central meridian of a polar stereographic.
    pub straight_vert_pole_long: Option<f64>,
}

/// Compact metadata for an internal transparency mask IFD (`NewSubfileType & 4 != 0`),
/// e.g. GDAL's 1-bit or 8-bit `GDAL_TIFF_INTERNAL_MASK` paired with JPEG/YCbCr COGs.
#[derive(Debug, Clone)]
pub struct MaskIfdMeta {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
    pub bits_per_sample: u16,
    pub compression: Compression,
    pub predictor: Predictor,
    pub is_stripped: bool,
    pub tile_offsets: Box<[u64]>,
    pub tile_byte_counts: Box<[u32]>,
}

impl MaskIfdMeta {
    #[inline(always)]
    pub fn block_height(&self, mask_idx: usize) -> u32 {
        if !self.is_stripped || self.tiles_across == 0 {
            return self.tile_height;
        }
        let ty = (mask_idx as u32) / self.tiles_across;
        let row_start = ty.saturating_mul(self.tile_height);
        self.height.saturating_sub(row_start).min(self.tile_height).max(1)
    }

    #[inline(always)]
    pub fn tile_byte_range(&self, tile_idx: usize) -> Option<(u64, u32)> {
        let count = *self.tile_byte_counts.get(tile_idx)?;
        if count == 0 {
            return None;
        }
        let offset = *self.tile_offsets.get(tile_idx)?;
        Some((offset, count))
    }
}

/// Compact metadata for a single IFD (full resolution or overview) of a COG.
#[derive(Debug, Clone)]
pub struct IfdMeta {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
    pub sample_type: SampleType,
    pub samples_per_pixel: u16,
    pub planar_config: PlanarConfig,
    pub compression: Compression,
    pub predictor: Predictor,
    /// PhotometricInterpretation tag (262): 1=MinIsBlack, 2=RGB, 6=YCbCr, etc.
    pub photometric: u16,
    /// ExtraSamples tag (338): 1=AssociatedAlpha, 2=UnassociatedAlpha.
    pub extra_samples: Vec<u16>,
    /// Shared JPEG quantization/Huffman tables from tag 347 (`JPEGTables`) for abbreviated JPEG tiles.
    pub jpeg_tables: Option<Box<[u8]>>,
    /// Optional internal transparency mask IFD (`NewSubfileType & 4 != 0`) matching this IFD's dimensions.
    pub mask_ifd: Option<Box<MaskIfdMeta>>,
    /// True if this IFD is strip-organized (`StripOffsets`/`RowsPerStrip`) rather than tiled.
    pub is_stripped: bool,
    pub pixel_to_crs: Affine2D,
    pub crs_to_pixel: Affine2D,
    /// Flat array of tile byte offsets.
    /// For Chunky: length = `tiles_across * tiles_down`.
    /// For Planar: length = `samples_per_pixel * tiles_across * tiles_down`.
    pub tile_offsets: Box<[u64]>,
    /// Flat array of compressed tile byte counts. `0` indicates a sparse all-nodata tile.
    pub tile_byte_counts: Box<[u32]>,
}

impl IfdMeta {
    /// Actual row count stored in block `tile_idx`.
    /// Tiled TIFFs always store `tile_height` rows per tile (padded on the bottom edge),
    /// whereas stripped TIFFs store `min(tile_height, height - strip_row * tile_height)`
    /// rows in the final strip of each plane.
    #[inline(always)]
    pub fn block_height(&self, tile_idx: u32) -> u32 {
        if !self.is_stripped || self.tiles_across == 0 || self.tiles_down == 0 {
            return self.tile_height;
        }
        let per_plane = self.tiles_across * self.tiles_down;
        let xy = tile_idx % per_plane;
        let ty = xy / self.tiles_across;
        let y0 = ty.saturating_mul(self.tile_height);
        self.height.saturating_sub(y0).min(self.tile_height).max(1)
    }

    /// Tile linear index in `tile_offsets` / `tile_byte_counts`.
    /// For `Chunky`, `band` is ignored (all bands live in the same tile).
    /// For `Planar`, `band` is 0-based (`0..samples_per_pixel`).
    #[inline(always)]
    pub fn tile_index(&self, tx: u32, ty: u32, band: u16) -> usize {
        let per_plane = (self.tiles_across as usize) * (self.tiles_down as usize);
        let xy = (ty as usize) * (self.tiles_across as usize) + (tx as usize);
        match self.planar_config {
            PlanarConfig::Chunky => xy,
            PlanarConfig::Planar => (band as usize) * per_plane + xy,
        }
    }

    #[inline(always)]
    pub fn tile_byte_range(&self, tx: u32, ty: u32, band: u16) -> Option<(u64, u32)> {
        if tx >= self.tiles_across || ty >= self.tiles_down {
            return None;
        }
        let idx = self.tile_index(tx, ty, band);
        let count = *self.tile_byte_counts.get(idx)?;
        if count == 0 {
            return None;
        }
        let offset = *self.tile_offsets.get(idx)?;
        Some((offset, count))
    }

    /// Resolution magnitude in CRS units per pixel `(abs(scale_x), abs(scale_y))`.
    #[inline(always)]
    pub fn pixel_resolution(&self) -> (f64, f64) {
        let rx = (self.pixel_to_crs.a.powi(2) + self.pixel_to_crs.d.powi(2)).sqrt();
        let ry = (self.pixel_to_crs.b.powi(2) + self.pixel_to_crs.e.powi(2)).sqrt();
        (rx, ry)
    }
}

/// Complete parsed metadata index for a single float32 COG.
#[derive(Debug, Clone)]
pub struct CogIndex {
    pub byte_order: ByteOrder,
    pub is_bigtiff: bool,
    /// `None` means no explicit GDAL_NODATA tag (NaN is still always treated as nodata).
    /// `Some(v)` where `v.is_nan()` means explicit NaN nodata; finite `v` means sentinel nodata.
    pub nodata: Option<f32>,
    /// Per-band scale factors parsed from `GDAL_METADATA` (defaults to `1.0` per band).
    pub scales: Vec<f32>,
    /// Per-band additive offsets parsed from `GDAL_METADATA` (defaults to `0.0` per band).
    pub offsets: Vec<f32>,
    /// Per-band GDAL metadata key/value tags parsed from `GDAL_METADATA` (`sample="k"`).
    pub band_metadata: Vec<Vec<(String, String)>>,
    /// Optional per-band descriptions parsed from `GDAL_METADATA` (`name="DESCRIPTION"`).
    pub band_descriptions: Vec<String>,
    /// Optional embedded TIFF palette (Tag 320 `ColorMap`) as RGBA entries `[r, g, b, 255]`.
    pub color_map: Option<Vec<[u8; 4]>>,
    pub geo_keys: GeoKeys,
    /// IFDs sorted from highest resolution (index 0) to lowest resolution overview.
    pub ifds: Vec<IfdMeta>,
}

impl CogIndex {
    #[inline(always)]
    pub fn full_res(&self) -> &IfdMeta {
        &self.ifds[0]
    }

    /// Select the best overview IFD for a target pixel resolution `(target_rx, target_ry)` in source CRS.
    /// Picks the coarsest overview whose resolution is `<= target * 1.05` (matching GDAL/rio-tiler
    /// so we never alias by picking an overly coarse overview, nor over-fetch full-res when zoomed out).
    pub fn select_ifd(&self, target_rx: f64, target_ry: f64) -> usize {
        let target = target_rx.min(target_ry);
        if !target.is_finite() || target <= 0.0 {
            return self.ifds.len().saturating_sub(1);
        }
        let mut best = 0usize;
        for (i, ifd) in self.ifds.iter().enumerate().skip(1) {
            let (rx, ry) = ifd.pixel_resolution();
            let res = rx.max(ry);
            if res <= target * 1.05 {
                best = i;
            } else {
                break;
            }
        }
        best
    }
}
