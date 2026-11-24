use crate::types::{
    Affine2D, ByteOrder, CogIndex, Compression, GeoKeys, IfdMeta, MaskIfdMeta, PlanarConfig,
    Predictor, SampleType, TiffError,
};

// Defensive parser caps against hostile BigTIFF/TIFF overflows and OOMs
const MAX_IFD_ENTRIES: u64 = 65_536;
const MAX_TAG_ARRAY_BYTES: u64 = 32 * 1024 * 1024;

// TIFF Tag IDs
const TAG_NEW_SUBFILE_TYPE: u16 = 254;
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIG: u16 = 284;
const TAG_PREDICTOR: u16 = 317;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_EXTRA_SAMPLES: u16 = 338;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_JPEG_TABLES: u16 = 347;
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_MODEL_TRANSFORMATION: u16 = 34264;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
const TAG_GDAL_METADATA: u16 = 42112;
const TAG_GDAL_NODATA: u16 = 42113;

// TIFF Field Types
const TYPE_BYTE: u16 = 1;
const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;
const TYPE_SBYTE: u16 = 6;
const TYPE_UNDEFINED: u16 = 7;
const TYPE_SSHORT: u16 = 8;
const TYPE_SLONG: u16 = 9;
const TYPE_SRATIONAL: u16 = 10;
const TYPE_FLOAT: u16 = 11;
const TYPE_DOUBLE: u16 = 12;
const TYPE_IFD: u16 = 13;
const TYPE_LONG8: u16 = 16;
const TYPE_SLONG8: u16 = 17;
const TYPE_IFD8: u16 = 18;

#[inline(always)]
fn type_size(ty: u16) -> Option<u64> {
    match ty {
        TYPE_BYTE | TYPE_ASCII | TYPE_SBYTE | TYPE_UNDEFINED => Some(1),
        TYPE_SHORT | TYPE_SSHORT => Some(2),
        TYPE_LONG | TYPE_SLONG | TYPE_FLOAT | TYPE_IFD => Some(4),
        TYPE_RATIONAL | TYPE_SRATIONAL | TYPE_DOUBLE | TYPE_LONG8 | TYPE_SLONG8 | TYPE_IFD8 => {
            Some(8)
        }
        _ => None,
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    bo: ByteOrder,
}

impl<'a> Reader<'a> {
    #[inline(always)]
    fn need(&self, offset: u64, len: u64) -> Result<&'a [u8], TiffError> {
        let end = offset
            .checked_add(len)
            .ok_or(TiffError::Corrupt("offset overflow"))?;
        if end > self.buf.len() as u64 {
            return Err(TiffError::NeedMoreBytes {
                required: end,
                available: self.buf.len() as u64,
            });
        }
        Ok(&self.buf[offset as usize..end as usize])
    }

    #[inline(always)]
    fn u16_at(&self, offset: u64) -> Result<u16, TiffError> {
        let b = self.need(offset, 2)?;
        let arr = [b[0], b[1]];
        Ok(match self.bo {
            ByteOrder::Little => u16::from_le_bytes(arr),
            ByteOrder::Big => u16::from_be_bytes(arr),
        })
    }

    #[inline(always)]
    fn u32_at(&self, offset: u64) -> Result<u32, TiffError> {
        let b = self.need(offset, 4)?;
        let arr = [b[0], b[1], b[2], b[3]];
        Ok(match self.bo {
            ByteOrder::Little => u32::from_le_bytes(arr),
            ByteOrder::Big => u32::from_be_bytes(arr),
        })
    }

    #[inline(always)]
    fn u64_at(&self, offset: u64) -> Result<u64, TiffError> {
        let b = self.need(offset, 8)?;
        let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Ok(match self.bo {
            ByteOrder::Little => u64::from_le_bytes(arr),
            ByteOrder::Big => u64::from_be_bytes(arr),
        })
    }

    #[inline(always)]
    fn f64_at(&self, offset: u64) -> Result<f64, TiffError> {
        Ok(f64::from_bits(self.u64_at(offset)?))
    }
}

struct TagEntry {
    tag: u16,
    ty: u16,
    count: u64,
    /// Byte offset in file where the tag values reside (either inline inside IFD or out-of-line).
    value_offset: u64,
}

impl TagEntry {
    fn read_u64_scalar(&self, r: &Reader<'_>) -> Result<u64, TiffError> {
        match self.ty {
            TYPE_SHORT => Ok(r.u16_at(self.value_offset)? as u64),
            TYPE_LONG | TYPE_IFD => Ok(r.u32_at(self.value_offset)? as u64),
            TYPE_LONG8 | TYPE_IFD8 => r.u64_at(self.value_offset),
            _ => Err(TiffError::Corrupt("unexpected scalar integer tag type")),
        }
    }

    fn read_u16_vec(&self, r: &Reader<'_>) -> Result<Vec<u16>, TiffError> {
        if self.ty != TYPE_SHORT {
            return Err(TiffError::Corrupt("expected SHORT array"));
        }
        let total = self
            .count
            .checked_mul(2)
            .filter(|&b| b <= MAX_TAG_ARRAY_BYTES)
            .ok_or(TiffError::Corrupt("SHORT array size overflow"))?;
        let n = self.count as usize;
        let _ = r.need(self.value_offset, total)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(r.u16_at(self.value_offset + (i as u64) * 2)?);
        }
        Ok(out)
    }

    fn read_u64_vec(&self, r: &Reader<'_>) -> Result<Box<[u64]>, TiffError> {
        let elem_sz = type_size(self.ty).ok_or(TiffError::Corrupt("unknown tag type"))?;
        let total = self
            .count
            .checked_mul(elem_sz)
            .filter(|&b| b <= MAX_TAG_ARRAY_BYTES)
            .ok_or(TiffError::Corrupt("integer array size overflow"))?;
        let n = self.count as usize;
        let _ = r.need(self.value_offset, total)?;
        let mut out = Vec::with_capacity(n);
        match self.ty {
            TYPE_SHORT => {
                for i in 0..n {
                    out.push(r.u16_at(self.value_offset + (i as u64) * 2)? as u64);
                }
            }
            TYPE_LONG | TYPE_IFD => {
                for i in 0..n {
                    out.push(r.u32_at(self.value_offset + (i as u64) * 4)? as u64);
                }
            }
            TYPE_LONG8 | TYPE_IFD8 => {
                for i in 0..n {
                    out.push(r.u64_at(self.value_offset + (i as u64) * 8)?);
                }
            }
            _ => return Err(TiffError::Corrupt("unexpected integer array type")),
        }
        Ok(out.into_boxed_slice())
    }

    fn read_u32_vec(&self, r: &Reader<'_>) -> Result<Box<[u32]>, TiffError> {
        let elem_sz = type_size(self.ty).ok_or(TiffError::Corrupt("unknown tag type"))?;
        let total = self
            .count
            .checked_mul(elem_sz)
            .filter(|&b| b <= MAX_TAG_ARRAY_BYTES)
            .ok_or(TiffError::Corrupt("byte count array size overflow"))?;
        let n = self.count as usize;
        let _ = r.need(self.value_offset, total)?;
        let mut out = Vec::with_capacity(n);
        match self.ty {
            TYPE_SHORT => {
                for i in 0..n {
                    out.push(r.u16_at(self.value_offset + (i as u64) * 2)? as u32);
                }
            }
            TYPE_LONG | TYPE_IFD => {
                for i in 0..n {
                    out.push(r.u32_at(self.value_offset + (i as u64) * 4)?);
                }
            }
            TYPE_LONG8 | TYPE_IFD8 => {
                for i in 0..n {
                    let v = r.u64_at(self.value_offset + (i as u64) * 8)?;
                    out.push(
                        u32::try_from(v)
                            .map_err(|_| TiffError::Corrupt("tile byte count exceeds u32"))?,
                    );
                }
            }
            _ => return Err(TiffError::Corrupt("unexpected byte count array type")),
        }
        Ok(out.into_boxed_slice())
    }

    fn read_f64_vec(&self, r: &Reader<'_>) -> Result<Vec<f64>, TiffError> {
        if self.ty != TYPE_DOUBLE {
            return Err(TiffError::Corrupt("expected DOUBLE array"));
        }
        let total = self
            .count
            .checked_mul(8)
            .filter(|&b| b <= MAX_TAG_ARRAY_BYTES)
            .ok_or(TiffError::Corrupt("DOUBLE array size overflow"))?;
        let n = self.count as usize;
        let _ = r.need(self.value_offset, total)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(r.f64_at(self.value_offset + (i as u64) * 8)?);
        }
        Ok(out)
    }

    fn read_ascii(&self, r: &Reader<'_>) -> Result<String, TiffError> {
        if self.count > MAX_TAG_ARRAY_BYTES {
            return Err(TiffError::Corrupt("ASCII tag exceeds max size"));
        }
        let bytes = r.need(self.value_offset, self.count)?;
        let trimmed = bytes.split(|&b| b == 0).next().unwrap_or(bytes);
        Ok(String::from_utf8_lossy(trimmed).trim().to_string())
    }

    fn read_bytes(&self, r: &Reader<'_>) -> Result<Box<[u8]>, TiffError> {
        if self.count > MAX_TAG_ARRAY_BYTES {
            return Err(TiffError::Corrupt("byte tag exceeds max size"));
        }
        let bytes = r.need(self.value_offset, self.count)?;
        Ok(bytes.to_vec().into_boxed_slice())
    }
}

fn parse_geo_keys(raw_keys: &[u16], double_params: &[f64]) -> GeoKeys {
    let mut gk = GeoKeys::default();
    if raw_keys.len() < 4 {
        return gk;
    }
    let num_keys = raw_keys[3] as usize;
    for i in 0..num_keys {
        let base = 4 + i * 4;
        if base + 3 >= raw_keys.len() {
            break;
        }
        let key_id = raw_keys[base];
        let tiff_tag_loc = raw_keys[base + 1];
        let _count = raw_keys[base + 2];
        let val_or_offset = raw_keys[base + 3];

        let get_double = || -> Option<f64> {
            if tiff_tag_loc == TAG_GEO_DOUBLE_PARAMS {
                double_params.get(val_or_offset as usize).copied()
            } else {
                None
            }
        };

        match key_id {
            1024 if tiff_tag_loc == 0 => gk.model_type = Some(val_or_offset),
            1025 if tiff_tag_loc == 0 => gk.raster_type = Some(val_or_offset),
            2048 if tiff_tag_loc == 0 => gk.geographic_type = Some(val_or_offset),
            2050 if tiff_tag_loc == 0 => gk.geodetic_datum = Some(val_or_offset),
            2054 if tiff_tag_loc == 0 => gk.geog_angular_units = Some(val_or_offset),
            2056 if tiff_tag_loc == 0 => gk.ellipsoid = Some(val_or_offset),
            2057 => gk.semi_major_axis = get_double(),
            2058 => gk.semi_minor_axis = get_double(),
            2059 => gk.inv_flattening = get_double(),
            3072 if tiff_tag_loc == 0 => gk.projected_cs_type = Some(val_or_offset),
            3075 if tiff_tag_loc == 0 => gk.proj_coord_trans = Some(val_or_offset),
            3076 if tiff_tag_loc == 0 => gk.proj_linear_units = Some(val_or_offset),
            3078 => gk.std_parallel_1 = get_double(),
            3079 => gk.std_parallel_2 = get_double(),
            3080 => gk.nat_origin_long = get_double(),
            3081 => gk.nat_origin_lat = get_double(),
            3082 => gk.false_easting = get_double(),
            3083 => gk.false_northing = get_double(),
            3084 => gk.false_origin_long = get_double(),
            3085 => gk.false_origin_lat = get_double(),
            3086 => gk.false_origin_easting = get_double(),
            3087 => gk.false_origin_northing = get_double(),
            3088 => gk.center_long = get_double(),
            3089 => gk.center_lat = get_double(),
            3092 => gk.scale_at_nat_origin = get_double(),
            3095 => gk.straight_vert_pole_long = get_double(),
            _ => {}
        }
    }
    gk
}

fn parse_ifd_entries(
    r: &Reader<'_>,
    ifd_offset: u64,
    is_bigtiff: bool,
) -> Result<(Vec<TagEntry>, u64), TiffError> {
    let (num_entries, entries_start, entry_size, inline_limit) = if is_bigtiff {
        let n = r.u64_at(ifd_offset)?;
        (n, ifd_offset + 8, 20u64, 8u64)
    } else {
        let n = r.u16_at(ifd_offset)? as u64;
        (n, ifd_offset + 2, 12u64, 4u64)
    };

    if num_entries > MAX_IFD_ENTRIES {
        return Err(TiffError::Corrupt("IFD entry count exceeds limit"));
    }
    let entries_bytes = num_entries
        .checked_mul(entry_size)
        .ok_or(TiffError::Corrupt("IFD entries byte size overflow"))?;
    let next_ifd_pos = entries_start
        .checked_add(entries_bytes)
        .ok_or(TiffError::Corrupt("IFD offset overflow"))?;
    let next_ifd_size = if is_bigtiff { 8u64 } else { 4u64 };
    let total_ifd_bytes = next_ifd_pos
        .checked_sub(ifd_offset)
        .and_then(|d| d.checked_add(next_ifd_size))
        .ok_or(TiffError::Corrupt("IFD size overflow"))?;
    let _ = r.need(ifd_offset, total_ifd_bytes)?;

    let mut entries = Vec::with_capacity(num_entries as usize);
    for i in 0..num_entries {
        let off = entries_start + i * entry_size;
        let tag = r.u16_at(off)?;
        let ty = r.u16_at(off + 2)?;
        let (count, val_field_off) = if is_bigtiff {
            (r.u64_at(off + 4)?, off + 12)
        } else {
            (r.u32_at(off + 4)? as u64, off + 8)
        };
        let total_bytes = type_size(ty).unwrap_or(0).saturating_mul(count);
        let value_offset = if total_bytes <= inline_limit {
            val_field_off
        } else if is_bigtiff {
            r.u64_at(val_field_off)?
        } else {
            r.u32_at(val_field_off)? as u64
        };
        entries.push(TagEntry {
            tag,
            ty,
            count,
            value_offset,
        });
    }

    let next_ifd = if is_bigtiff {
        r.u64_at(next_ifd_pos)?
    } else {
        r.u32_at(next_ifd_pos)? as u64
    };

    Ok((entries, next_ifd))
}

/// Parse a float32 COG's TIFF/BigTIFF header, all overview IFDs, GeoKeys, and tile offset tables
/// from a prefix slice (e.g., `mmap` slice or initial speculative HTTP Range bytes).
///
/// If an IFD or tag array lies beyond `bytes.len()`, returns `Err(TiffError::NeedMoreBytes { required, .. })`.
pub fn parse_cog(bytes: &[u8]) -> Result<CogIndex, TiffError> {
    if bytes.len() < 8 {
        return Err(TiffError::NeedMoreBytes {
            required: 16,
            available: bytes.len() as u64,
        });
    }
    let bo = match &bytes[0..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _ => return Err(TiffError::InvalidHeader),
    };
    let r = Reader { buf: bytes, bo };
    let magic = r.u16_at(2)?;
    let (is_bigtiff, mut next_ifd) = match magic {
        42 => (false, r.u32_at(4)? as u64),
        43 => {
            let offset_size = r.u16_at(4)?;
            let zero = r.u16_at(6)?;
            if offset_size != 8 || zero != 0 {
                return Err(TiffError::InvalidHeader);
            }
            (true, r.u64_at(8)?)
        }
        _ => return Err(TiffError::InvalidHeader),
    };

    let mut ifds: Vec<IfdMeta> = Vec::with_capacity(8);
    let mut mask_ifds: Vec<MaskIfdMeta> = Vec::new();
    let mut geo_keys = GeoKeys::default();
    let mut nodata: Option<f32> = None;
    let mut gdal_metadata_xml: Option<String> = None;
    let mut base_affine: Option<Affine2D> = None;
    let mut base_dims: Option<(u32, u32)> = None;

    // Guard against cyclic IFD chains
    let mut ifd_count = 0usize;
    while next_ifd != 0 && ifd_count < 64 {
        ifd_count += 1;
        let (entries, following) = parse_ifd_entries(&r, next_ifd, is_bigtiff)?;
        next_ifd = following;

        let mut subfile_type = 0u32;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut tile_width = 0u32;
        let mut tile_height = 0u32;
        let mut samples_per_pixel = 1u16;
        let mut planar_config = PlanarConfig::Chunky;
        let mut compression = Compression::None;
        let mut predictor = Predictor::None;
        let mut photometric = 1u16;
        let mut extra_samples: Vec<u16> = Vec::new();
        let mut jpeg_tables: Option<Box<[u8]>> = None;
        let mut bits_per_sample: Option<Vec<u16>> = None;
        let mut sample_format: Option<Vec<u16>> = None;
        let mut tile_offsets: Option<Box<[u64]>> = None;
        let mut tile_byte_counts: Option<Box<[u32]>> = None;
        let mut strip_offsets: Option<Box<[u64]>> = None;
        let mut strip_byte_counts: Option<Box<[u32]>> = None;
        let mut rows_per_strip: u32 = 0;

        // GeoTIFF tags (only expected on IFD 0)
        let mut pixel_scale: Option<Vec<f64>> = None;
        let mut tiepoint: Option<Vec<f64>> = None;
        let mut model_transform: Option<Vec<f64>> = None;
        let mut raw_geo_keys: Option<Vec<u16>> = None;
        let mut double_params: Vec<f64> = Vec::new();

        for entry in &entries {
            match entry.tag {
                TAG_NEW_SUBFILE_TYPE => subfile_type = entry.read_u64_scalar(&r)? as u32,
                TAG_IMAGE_WIDTH => width = entry.read_u64_scalar(&r)? as u32,
                TAG_IMAGE_LENGTH => height = entry.read_u64_scalar(&r)? as u32,
                TAG_BITS_PER_SAMPLE => bits_per_sample = Some(entry.read_u16_vec(&r)?),
                TAG_COMPRESSION => {
                    let code = entry.read_u64_scalar(&r)? as u16;
                    compression = match code {
                        1 => Compression::None,
                        5 => Compression::Lzw,
                        7 => Compression::Jpeg,
                        8 | 32946 => Compression::Deflate,
                        50000 => Compression::Zstd,
                        50001 => Compression::Webp,
                        other => return Err(TiffError::UnsupportedCompression(other)),
                    };
                }
                TAG_PHOTOMETRIC => photometric = entry.read_u64_scalar(&r)? as u16,
                TAG_STRIP_OFFSETS => strip_offsets = Some(entry.read_u64_vec(&r)?),
                TAG_SAMPLES_PER_PIXEL => samples_per_pixel = entry.read_u64_scalar(&r)? as u16,
                TAG_ROWS_PER_STRIP => rows_per_strip = entry.read_u64_scalar(&r)? as u32,
                TAG_STRIP_BYTE_COUNTS => strip_byte_counts = Some(entry.read_u32_vec(&r)?),
                TAG_PLANAR_CONFIG => {
                    planar_config = match entry.read_u64_scalar(&r)? {
                        1 => PlanarConfig::Chunky,
                        2 => PlanarConfig::Planar,
                        _ => return Err(TiffError::Corrupt("invalid PlanarConfiguration")),
                    };
                }
                TAG_PREDICTOR => {
                    let code = entry.read_u64_scalar(&r)? as u16;
                    predictor = match code {
                        1 => Predictor::None,
                        2 => Predictor::HorizontalInt,
                        3 => Predictor::FloatingPoint,
                        other => return Err(TiffError::UnsupportedPredictor(other)),
                    };
                }
                TAG_TILE_WIDTH => tile_width = entry.read_u64_scalar(&r)? as u32,
                TAG_TILE_LENGTH => tile_height = entry.read_u64_scalar(&r)? as u32,
                TAG_TILE_OFFSETS => tile_offsets = Some(entry.read_u64_vec(&r)?),
                TAG_TILE_BYTE_COUNTS => tile_byte_counts = Some(entry.read_u32_vec(&r)?),
                TAG_EXTRA_SAMPLES => extra_samples = entry.read_u16_vec(&r)?,
                TAG_SAMPLE_FORMAT => sample_format = Some(entry.read_u16_vec(&r)?),
                TAG_JPEG_TABLES => jpeg_tables = Some(entry.read_bytes(&r)?),
                TAG_MODEL_PIXEL_SCALE if ifds.is_empty() => {
                    pixel_scale = Some(entry.read_f64_vec(&r)?);
                }
                TAG_MODEL_TIEPOINT if ifds.is_empty() => {
                    tiepoint = Some(entry.read_f64_vec(&r)?);
                }
                TAG_MODEL_TRANSFORMATION if ifds.is_empty() => {
                    model_transform = Some(entry.read_f64_vec(&r)?);
                }
                TAG_GEO_KEY_DIRECTORY if ifds.is_empty() => {
                    raw_geo_keys = Some(entry.read_u16_vec(&r)?);
                }
                TAG_GEO_DOUBLE_PARAMS if ifds.is_empty() => {
                    double_params = entry.read_f64_vec(&r)?;
                }
                TAG_GDAL_METADATA if ifds.is_empty() => {
                    gdal_metadata_xml = Some(entry.read_ascii(&r)?);
                }
                TAG_GDAL_NODATA if ifds.is_empty() => {
                    let s = entry.read_ascii(&r)?;
                    if s.eq_ignore_ascii_case("nan") {
                        nodata = Some(f32::NAN);
                    } else if let Ok(v) = s.parse::<f32>() {
                        nodata = Some(v);
                    }
                }
                _ => {}
            }
        }

        // Capture internal transparency mask IFDs (bit 2 of NewSubfileType == 4)
        if (subfile_type & 4) != 0 {
            let mask_bps = bits_per_sample
                .as_deref()
                .and_then(|v| v.first().copied())
                .unwrap_or(1);
            if tile_width > 0 && tile_height > 0 && (mask_bps == 1 || mask_bps == 8) {
                if let (Some(offs), Some(cnts)) = (tile_offsets, tile_byte_counts) {
                    mask_ifds.push(MaskIfdMeta {
                        width,
                        height,
                        tile_width,
                        tile_height,
                        tiles_across: width.div_ceil(tile_width),
                        tiles_down: height.div_ceil(tile_height),
                        bits_per_sample: mask_bps,
                        compression,
                        predictor,
                        tile_offsets: offs,
                        tile_byte_counts: cnts,
                    });
                }
            }
            continue;
        }

        // Accept all 8 standard GDAL numeric sample types (widened to f32 at block decode)
        let bps = bits_per_sample
            .as_deref()
            .and_then(|v| v.first().copied())
            .unwrap_or(0);
        let fmt = sample_format
            .as_deref()
            .and_then(|v| v.first().copied())
            .unwrap_or(1);
        let sample_type = match (fmt, bps) {
            (1, 8) => SampleType::Uint8,
            (2, 8) => SampleType::Int8,
            (1, 16) => SampleType::Uint16,
            (2, 16) => SampleType::Int16,
            (1, 32) => SampleType::Uint32,
            (2, 32) => SampleType::Int32,
            (3, 32) => SampleType::Float32,
            (3, 64) => SampleType::Float64,
            _ => {
                return Err(TiffError::UnsupportedSampleFormat {
                    format: fmt,
                    bits: bps,
                });
            }
        };

        if width == 0 || height == 0 || samples_per_pixel == 0 {
            return Err(TiffError::Corrupt("IFD has zero width, height, or samples_per_pixel"));
        }

        let (tile_width, tile_height, tile_offsets, tile_byte_counts, is_stripped) =
            if tile_width > 0 && tile_height > 0 {
                let to = tile_offsets.ok_or(TiffError::NotTiled)?;
                let tbc = tile_byte_counts.ok_or(TiffError::NotTiled)?;
                (tile_width, tile_height, to, tbc, false)
            } else if let (Some(so), Some(sbc)) = (strip_offsets, strip_byte_counts) {
                let rps = if rows_per_strip == 0 || rows_per_strip > height {
                    height
                } else {
                    rows_per_strip
                };
                (width, rps, so, sbc, true)
            } else {
                return Err(TiffError::NotTiled);
            };

        let tiles_across = width.div_ceil(tile_width);
        let tiles_down = height.div_ceil(tile_height);
        let planes: u64 = match planar_config {
            PlanarConfig::Chunky => 1,
            PlanarConfig::Planar => samples_per_pixel as u64,
        };
        let expected_tiles = (tiles_across as u64)
            .checked_mul(tiles_down as u64)
            .and_then(|n| n.checked_mul(planes))
            .ok_or(TiffError::Corrupt("tile grid count overflow"))?;
        if (tile_offsets.len() as u64) < expected_tiles
            || (tile_byte_counts.len() as u64) < expected_tiles
        {
            return Err(TiffError::Corrupt("truncated tile/strip offset or byte-count table"));
        }

        let pixel_to_crs = if ifds.is_empty() {
            if let Some(rk) = raw_geo_keys {
                geo_keys = parse_geo_keys(&rk, &double_params);
            }
            let aff = if let Some(m) = model_transform {
                if m.len() < 16 {
                    return Err(TiffError::MissingGeoTransform);
                }
                let mut c = m[3];
                let mut f = m[7];
                // PixelIsPoint (GTRasterTypeGeoKey == 2) shifts origin by half pixel along both axes
                if geo_keys.raster_type == Some(2) {
                    c -= 0.5 * (m[0] + m[1]);
                    f -= 0.5 * (m[4] + m[5]);
                }
                Affine2D {
                    a: m[0],
                    b: m[1],
                    c,
                    d: m[4],
                    e: m[5],
                    f,
                }
            } else if let (Some(ps), Some(tp)) = (pixel_scale, tiepoint) {
                if ps.len() < 2 || tp.len() < 6 {
                    return Err(TiffError::MissingGeoTransform);
                }
                let sx = ps[0];
                let sy = -ps[1];
                let mut c = tp[3] - tp[0] * sx;
                let mut f = tp[4] - tp[1] * sy;
                // PixelIsPoint (GTRasterTypeGeoKey == 2) shifts origin by half pixel
                if geo_keys.raster_type == Some(2) {
                    c -= 0.5 * sx;
                    f -= 0.5 * sy;
                }
                Affine2D {
                    a: sx,
                    b: 0.0,
                    c,
                    d: 0.0,
                    e: sy,
                    f,
                }
            } else {
                return Err(TiffError::MissingGeoTransform);
            };
            base_affine = Some(aff);
            base_dims = Some((width, height));
            aff
        } else {
            let base = base_affine.ok_or(TiffError::MissingGeoTransform)?;
            let (bw, bh) = base_dims.unwrap();
            let scale_x = (bw as f64) / (width as f64);
            let scale_y = (bh as f64) / (height as f64);
            Affine2D {
                a: base.a * scale_x,
                b: base.b * scale_y,
                c: base.c,
                d: base.d * scale_x,
                e: base.e * scale_y,
                f: base.f,
            }
        };

        let crs_to_pixel = pixel_to_crs
            .invert()
            .ok_or(TiffError::Corrupt("singular GeoTransform affine"))?;

        ifds.push(IfdMeta {
            width,
            height,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
            sample_type,
            samples_per_pixel,
            planar_config,
            compression,
            predictor,
            photometric,
            extra_samples,
            jpeg_tables,
            mask_ifd: None,
            is_stripped,
            pixel_to_crs,
            crs_to_pixel,
            tile_offsets,
            tile_byte_counts,
        });
    }

    if ifds.is_empty() {
        return Err(TiffError::Corrupt("no valid float32 image IFDs found"));
    }

    // Sort IFDs from highest resolution (smallest pixel size) to coarsest overview
    ifds.sort_by(|a, b| {
        let ra = a.pixel_resolution().0;
        let rb = b.pixel_resolution().0;
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Attach internal transparency mask IFDs to matching image IFDs by (width, height)
    for m in mask_ifds {
        if let Some(target) = ifds
            .iter_mut()
            .find(|i| i.width == m.width && i.height == m.height && i.mask_ifd.is_none())
        {
            target.mask_ifd = Some(Box::new(m));
        }
    }

    let n_bands = ifds[0].samples_per_pixel as usize;
    let mut scales = vec![1.0f32; n_bands];
    let mut offsets = vec![0.0f32; n_bands];
    if let Some(xml) = &gdal_metadata_xml {
        parse_gdal_scale_offset(xml, &mut scales, &mut offsets);
    }

    Ok(CogIndex {
        byte_order: bo,
        is_bigtiff,
        nodata,
        scales,
        offsets,
        geo_keys,
        ifds,
    })
}

/// Lightweight zero-dep extractor for `<Item name="SCALE" sample="0">...</Item>` and `OFFSET`
/// inside TIFF tag 42112 (`GDAL_METADATA`).
fn parse_gdal_scale_offset(xml: &str, scales: &mut [f32], offsets: &mut [f32]) {
    let mut rest = xml;
    while let Some(pos) = rest.find("<Item ") {
        rest = &rest[pos + 6..];
        let Some(tag_end) = rest.find('>') else { break };
        let attrs = &rest[..tag_end];
        rest = &rest[tag_end + 1..];
        // Self-closing `<Item .../>` has no body; do not swallow up to a later `</Item>`!
        if attrs.trim_end().ends_with('/') {
            continue;
        }
        let Some(val_end) = rest.find("</Item>") else { break };
        let val_str = rest[..val_end].trim();
        rest = &rest[val_end + 7..];

        // Ignore items belonging to a non-default GDAL metadata domain (e.g. domain="xml:ESRI").
        if let Some(dp) = attrs.find("domain=\"") {
            let dval = &attrs[dp + 8..];
            if let Some(dom) = dval.split('"').next()
                && !dom.is_empty()
            {
                continue;
            }
        }

        let is_scale = attrs.contains("name=\"SCALE\"") || attrs.contains("name=\"scale\"");
        let is_offset = attrs.contains("name=\"OFFSET\"") || attrs.contains("name=\"offset\"");
        if !is_scale && !is_offset {
            continue;
        }
        let Ok(v) = val_str.parse::<f32>() else {
            continue;
        };
        if !v.is_finite() || (is_scale && v == 0.0) {
            continue;
        }
        let sample_idx = attrs.find("sample=\"").and_then(|sp| {
            let s = &attrs[sp + 8..];
            s.split('"').next()?.parse::<usize>().ok()
        });
        match sample_idx {
            Some(idx) if idx < scales.len() => {
                if is_scale { scales[idx] = v; } else { offsets[idx] = v; }
            }
            None => {
                if is_scale { scales.fill(v); } else { offsets.fill(v); }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdal_metadata_handles_self_closing_items_and_non_default_domains() {
        let xml = r#"<GDALMetadata>
            <Item name="DESCRIPTION" sample="0"/>
            <Item name="SCALE" sample="0">0.0001</Item>
            <Item name="OFFSET" sample="0">-0.2</Item>
            <Item name="SCALE" sample="0" domain="xml:ESRI">999.0</Item>
            <Item name="SCALE" sample="1">0.0</Item>
        </GDALMetadata>"#;
        let mut scales = [1.0f32; 2];
        let mut offsets = [0.0f32; 2];
        parse_gdal_scale_offset(xml, &mut scales, &mut offsets);
        assert!((scales[0] - 0.0001).abs() < 1e-7);
        assert!((offsets[0] + 0.2).abs() < 1e-6);
        // Zero scale on band 1 must be ignored (keep default 1.0)
        assert_eq!(scales[1], 1.0);
    }

    #[test]
    fn hostile_bigtiff_num_entries_overflow_is_rejected_cleanly() {
        // 32-byte BigTIFF header pointing to IFD at offset 16 with num_entries = 1 << 62
        // (formerly wrapped `num_entries * 20 == 0` in release and aborted in Vec::with_capacity).
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(b"II"); // little-endian
        buf.extend_from_slice(&43u16.to_le_bytes()); // BigTIFF
        buf.extend_from_slice(&8u16.to_le_bytes()); // offset byte size
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
        buf.extend_from_slice(&16u64.to_le_bytes()); // first IFD offset
        buf.extend_from_slice(&(1u64 << 62).to_le_bytes()); // overflowing num_entries
        buf.extend_from_slice(&0u64.to_le_bytes()); // next IFD
        let err = parse_cog(&buf).unwrap_err();
        assert!(matches!(err, TiffError::Corrupt(_)), "got {err:?}");
    }
}
