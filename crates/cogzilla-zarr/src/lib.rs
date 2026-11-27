//! Zarr v2 and v3 array metadata, for serving Zarr the way cogzilla already serves COGs.
//!
//! # Why this crate is small on purpose
//!
//! cogzilla's render core is already format-agnostic. Everything downstream of a decoded block —
//! resampling, statistics, colour maps, expressions, terrain — reaches it through
//! [`cogzilla_kernel::BandSource`], whose constructor takes pure geometry plus decoded blocks:
//!
//! ```text
//! BandSource::new(ifd_w, ifd_h, tile_w, tile_h, tx0, ty0, grid_w, grid_h,
//!                 blocks: &[Option<&DecodedBand>])
//! ```
//!
//! There is no TIFF type in that signature. A Zarr array is the same shape of thing: an array of
//! samples cut into a regular grid of chunks, where a chunk that was never written is simply
//! absent — which the kernel already represents as `None` and treats as all-nodata.
//!
//! So this crate's job is only to answer four questions and then get out of the way:
//!
//! 1. what shape is the array, in what dtype, cut into what chunks ([`ArrayMeta`]);
//! 2. which object holds chunk `(i, j, ..)` ([`ChunkKeyEncoding::key`]);
//! 3. how is that object's byte payload decoded (the codec chain, [`CodecSpec`]);
//! 4. where on the Earth does the array sit (CRS and affine, from CF or GeoZarr attributes).
//!
//! Points 1 and 2 are implemented here. Points 3 and 4 land in the follow-up PRs listed in
//! `docs/zarr-integration.md`; the types they will fill in are already named so that the chunk
//! reader, the codec chain and the georeferencing can be written against a settled surface.
//!
//! Fetching is **not** this crate's job either: `cogzilla-io` already does S3, GCS, Azure and HTTP
//! with the credential chain, concurrency limiting, caching and eviction. A Zarr chunk is one more
//! object to ask it for.

pub mod blosc;
pub mod codecs;
pub mod crs;
pub mod geo;
pub mod hierarchy;
pub mod reader;
pub mod sel;
pub mod select;
pub mod sharding;

use std::collections::BTreeMap;

use serde_json::Value;

pub use codecs::{MAX_DECODED_CHUNK_BYTES, SUPPORTED_CODECS, decode_chunk_bytes};
pub use crs::{ZarrCrs, resolve_crs};
pub use geo::{ZarrGeo, read_coord_1d, resolve_geo};
pub use hierarchy::{FsStore, GroupMeta, GroupView, MemoryStore, OpenMode, ZarrHierarchy, ZarrNode, ZarrStore};
pub use reader::{
    ChunkWindow, DEFAULT_MAX_CHUNKS_PER_REQUEST, DEFAULT_MAX_DECODED_WINDOW_BYTES, ZarrCaps,
    ZarrWindowBlocks, read_chunk_window, read_chunk_window_for_band,
    read_chunk_window_for_band_with_caps,
};
pub use sel::{
    CfCalendar, CfTimeUnits, SelDimSpec, SelMethod, parse_sel_params, resolve_sel_indices,
};
pub use select::{
    ArrangedDims, SelectedVariable, arrange_dims, select_variable, select_variables, valid_min_max,
};
pub use sharding::{IndexLocation, ShardIndex, ShardIndexCache, ShardingConfig};

/// Anything that can go wrong reading Zarr metadata, navigating a store, or decoding chunks.
#[derive(Debug, thiserror::Error)]
pub enum ZarrError {
    #[error("metadata is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("metadata field {field} is missing")]
    MissingField { field: &'static str },
    #[error("metadata field {field} has an unexpected shape: {detail}")]
    BadField { field: &'static str, detail: String },
    #[error("unsupported zarr format version {0} (this build reads v2 and v3)")]
    UnsupportedVersion(u64),
    #[error("unsupported data type {0}")]
    UnsupportedDataType(String),
    #[error("array has {0} dimensions; a 2D or 3D array is required")]
    UnsupportedRank(usize),
    #[error("prefix is not a Zarr store (looked for {searched})")]
    NotAZarrStore { searched: &'static str },
    #[error("group {path:?} not found in Zarr store")]
    GroupNotFound { path: String },
    #[error("store I/O error on key {key:?}: {message}")]
    Io { key: String, message: String },
    #[error("unsupported codec {codec:?} on chunk {chunk_key:?} (supported: {supported})")]
    UnsupportedCodec {
        codec: String,
        chunk_key: String,
        supported: &'static str,
    },
    #[error("codec {codec:?} failed on chunk {chunk_key:?}: {detail}")]
    Codec {
        codec: String,
        chunk_key: String,
        detail: String,
    },
    #[error("chunk {chunk_key:?} decoded size {bytes} exceeds cap {cap} bytes")]
    ChunkTooLarge {
        chunk_key: String,
        bytes: usize,
        cap: usize,
    },
    #[error("chunk {chunk_key:?} is missing and array has no fill value")]
    MissingChunk { chunk_key: String },
    #[error("invalid pixel or band selection: {detail}")]
    InvalidSelection { detail: String },
    #[error("unsupported CRS {crs:?}: {detail}")]
    UnsupportedCrs { crs: String, detail: String },
    #[error("irregular or curvilinear grid on dimension {dim:?} at index {index}: {detail}")]
    IrregularGrid {
        dim: String,
        index: usize,
        detail: String,
    },
    #[error("variable {requested:?} not found (available variables: {available:?})")]
    VariableNotFound {
        requested: String,
        available: Vec<String>,
    },
    #[error("variable parameter is required or ambiguous (available variables: {available:?})")]
    AmbiguousVariable { available: Vec<String> },
    #[error("invalid sel parameter: {detail}")]
    InvalidSel { detail: String },
    #[error("sel value {requested:?} for dimension {dim:?} is outside available range [{min}, {max}]")]
    SelOutOfRange {
        dim: String,
        requested: String,
        min: String,
        max: String,
    },
    #[error("unsupported CF calendar {calendar:?}: {detail}")]
    UnsupportedCalendar { calendar: String, detail: String },
    #[error("store or key {key:?} not found at origin")]
    NotFound { key: String },
    #[error("origin refused access to {key:?} (HTTP {status})")]
    UpstreamRefused { key: String, status: u16 },
    #[error("origin timed out reading {key:?}: {detail}")]
    UpstreamTimeout { key: String, detail: String },
    #[error("origin error or truncated response reading {key:?}: {detail}")]
    UpstreamError { key: String, detail: String },
    #[error("request {kind} ({computed}) exceeds configured cap ({cap})")]
    CapExceeded {
        kind: &'static str,
        computed: usize,
        cap: usize,
    },
}

/// Which revision of the Zarr specification a store was written to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrVersion {
    V2,
    V3,
}

/// Byte order of the stored samples. Zarr v3 carries this in the `bytes` codec rather than the
/// dtype, so it is kept beside [`DataType`] instead of inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

/// The sample types cogzilla can render. Zarr's complex and variable-length types are rejected
/// rather than silently approximated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
}

impl DataType {
    /// Bytes per sample as stored.
    pub fn size(self) -> usize {
        match self {
            DataType::Bool | DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
        }
    }

    /// True when the type cannot be carried exactly by `f32`, so the data paths have to keep it in
    /// its own type or in `f64`. This is the same rule the GeoTIFF side settled in OO1.
    pub fn needs_wide_path(self) -> bool {
        matches!(
            self,
            DataType::Int32
                | DataType::UInt32
                | DataType::Int64
                | DataType::UInt64
                | DataType::Float64
        )
    }

    /// Parse a v2 dtype string such as `"<f4"`, `">i2"` or `"|u1"`.
    pub fn from_v2_dtype(s: &str) -> Result<(Self, Endianness), ZarrError> {
        let mut chars = s.chars();
        let (endian, rest) = match chars.next() {
            Some('<') => (Endianness::Little, chars.as_str()),
            Some('>') => (Endianness::Big, chars.as_str()),
            // `|` means "not applicable" (single-byte types); anything else is an implicit
            // native-order spelling, and every store we care about is little-endian.
            Some('|') => (Endianness::Little, chars.as_str()),
            _ => (Endianness::Little, s),
        };
        let dt = match rest {
            "b1" => DataType::Bool,
            "i1" => DataType::Int8,
            "i2" => DataType::Int16,
            "i4" => DataType::Int32,
            "i8" => DataType::Int64,
            "u1" => DataType::UInt8,
            "u2" => DataType::UInt16,
            "u4" => DataType::UInt32,
            "u8" => DataType::UInt64,
            "f2" => DataType::Float16,
            "f4" => DataType::Float32,
            "f8" => DataType::Float64,
            other => return Err(ZarrError::UnsupportedDataType(other.to_string())),
        };
        Ok((dt, endian))
    }

    /// Parse a v3 `data_type` name such as `"float32"`.
    pub fn from_v3_name(s: &str) -> Result<Self, ZarrError> {
        Ok(match s {
            "bool" => DataType::Bool,
            "int8" => DataType::Int8,
            "int16" => DataType::Int16,
            "int32" => DataType::Int32,
            "int64" => DataType::Int64,
            "uint8" => DataType::UInt8,
            "uint16" => DataType::UInt16,
            "uint32" => DataType::UInt32,
            "uint64" => DataType::UInt64,
            "float16" => DataType::Float16,
            "float32" => DataType::Float32,
            "float64" => DataType::Float64,
            other => return Err(ZarrError::UnsupportedDataType(other.to_string())),
        })
    }
}

/// One entry of the codec chain, kept as its name plus its untouched configuration so that the
/// decoder PR can interpret `blosc`, `zstd`, `gzip`, `zlib`, `lz4`, `transpose`, `bytes` and
/// `sharding_indexed` without this crate having to know about them yet.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecSpec {
    pub name: String,
    pub configuration: Value,
}

impl CodecSpec {
    fn from_v3(value: &Value) -> Result<Self, ZarrError> {
        // v3 allows a bare string as shorthand for a codec with no configuration.
        if let Some(name) = value.as_str() {
            return Ok(CodecSpec {
                name: name.to_string(),
                configuration: Value::Null,
            });
        }
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .ok_or(ZarrError::MissingField {
                field: "codecs[].name",
            })?;
        Ok(CodecSpec {
            name: name.to_string(),
            configuration: value.get("configuration").cloned().unwrap_or(Value::Null),
        })
    }

    /// v2 keeps `compressor` and `filters` as numcodecs objects with an `id` rather than a `name`.
    fn from_v2(value: &Value) -> Result<Self, ZarrError> {
        let name = value
            .get("id")
            .and_then(Value::as_str)
            .ok_or(ZarrError::MissingField {
                field: "compressor.id",
            })?;
        Ok(CodecSpec {
            name: name.to_string(),
            configuration: value.clone(),
        })
    }
}

/// How a chunk's grid index becomes a key in the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkKeyEncoding {
    /// v2, and v3's `v2` encoding: the indices joined by a separator, `0.1.2`.
    V2 { separator: char },
    /// v3's `default` encoding: a `c` prefix then the indices, `c/0/1/2`.
    Default { separator: char },
}

impl ChunkKeyEncoding {
    /// The key for one chunk, relative to the array's own prefix in the store.
    ///
    /// A zero-dimensional array has the single key `0` under the v2 encoding and `c` under the
    /// default one, which is what both specifications require.
    pub fn key(&self, index: &[u64]) -> String {
        match self {
            ChunkKeyEncoding::V2 { separator } => {
                if index.is_empty() {
                    return "0".to_string();
                }
                join(index, *separator)
            }
            ChunkKeyEncoding::Default { separator } => {
                if index.is_empty() {
                    return "c".to_string();
                }
                format!("c{separator}{}", join(index, *separator))
            }
        }
    }
}

fn join(index: &[u64], separator: char) -> String {
    let mut out = String::with_capacity(index.len() * 4);
    for (i, v) in index.iter().enumerate() {
        if i > 0 {
            out.push(separator);
        }
        out.push_str(itoa(*v).as_str());
    }
    out
}

fn itoa(v: u64) -> String {
    v.to_string()
}

/// Everything needed to address and decode one Zarr array, from either specification version.
#[derive(Debug, Clone)]
pub struct ArrayMeta {
    pub version: ZarrVersion,
    /// Array shape, slowest-varying dimension first.
    pub shape: Vec<u64>,
    /// Chunk shape, same order as [`ArrayMeta::shape`].
    pub chunk_shape: Vec<u64>,
    pub dtype: DataType,
    pub endianness: Endianness,
    /// `None` where the store says `null`, which means "no fill value": a missing chunk is then an
    /// error rather than a hole.
    pub fill_value: Option<f64>,
    /// Decode order: the chain as written, outermost last, exactly as the store declares it.
    pub codecs: Vec<CodecSpec>,
    pub chunk_key: ChunkKeyEncoding,
    /// v3 `dimension_names`, or the `_ARRAY_DIMENSIONS` attribute that xarray writes in v2. This is
    /// what lets `?sel=time=...` name a dimension.
    pub dimension_names: Option<Vec<Option<String>>>,
    /// `.zattrs` in v2, the `attributes` object in v3. Carries the CF and GeoZarr georeferencing
    /// that the CRS work will read.
    pub attributes: BTreeMap<String, Value>,
    /// True when a v2 array declares `"order": "F"` (Fortran / column-major memory order).
    pub order_f: bool,
}

impl ArrayMeta {
    /// Parse a v2 array from its `.zarray`, with `.zattrs` if the store has one.
    pub fn from_v2(zarray: &[u8], zattrs: Option<&[u8]>) -> Result<Self, ZarrError> {
        let v: Value = serde_json::from_slice(zarray)?;
        let attrs_val = match zattrs {
            Some(bytes) => Some(serde_json::from_slice::<Value>(bytes)?),
            None => None,
        };
        Self::from_v2_value(&v, attrs_val.as_ref())
    }

    /// Parse a v2 array from already-parsed JSON values (used by `.zmetadata` consolidation).
    pub fn from_v2_value(v: &Value, zattrs: Option<&Value>) -> Result<Self, ZarrError> {
        match v.get("zarr_format").and_then(Value::as_u64) {
            Some(2) | None => {}
            Some(other) => return Err(ZarrError::UnsupportedVersion(other)),
        }
        let shape = u64_vec(v, "shape")?;
        let chunk_shape = u64_vec(v, "chunks")?;
        validate_shape_and_chunks(&shape, &chunk_shape)?;
        let dtype_str = v
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or(ZarrError::MissingField { field: "dtype" })?;
        let (dtype, endianness) = DataType::from_v2_dtype(dtype_str)?;

        let mut codecs = Vec::new();
        if let Some(filters) = v.get("filters").and_then(Value::as_array) {
            for f in filters {
                codecs.push(CodecSpec::from_v2(f)?);
            }
        }
        if let Some(compressor) = v.get("compressor")
            && !compressor.is_null()
        {
            codecs.push(CodecSpec::from_v2(compressor)?);
        }

        let separator = v
            .get("dimension_separator")
            .and_then(Value::as_str)
            .and_then(|s| s.chars().next())
            .unwrap_or('.');

        let attributes = zattrs.map(object_map).unwrap_or_default();
        // xarray records dimension names in v2 stores under this attribute.
        let dimension_names = attributes.get("_ARRAY_DIMENSIONS").and_then(|v| {
            v.as_array()
                .map(|a| a.iter().map(|n| n.as_str().map(str::to_string)).collect())
        });

        let order_f = v.get("order").and_then(Value::as_str) == Some("F");

        Ok(ArrayMeta {
            version: ZarrVersion::V2,
            shape,
            chunk_shape,
            dtype,
            endianness,
            fill_value: fill_value(v.get("fill_value")),
            codecs,
            chunk_key: ChunkKeyEncoding::V2 { separator },
            dimension_names,
            attributes,
            order_f,
        })
    }

    /// Parse a v3 array from its `zarr.json`.
    pub fn from_v3(zarr_json: &[u8]) -> Result<Self, ZarrError> {
        let v: Value = serde_json::from_slice(zarr_json)?;
        Self::from_v3_value(&v)
    }

    /// Parse a v3 array from an already-parsed JSON value (used by v3 consolidated metadata).
    pub fn from_v3_value(v: &Value) -> Result<Self, ZarrError> {
        match v.get("zarr_format").and_then(Value::as_u64) {
            Some(3) | None => {}
            Some(other) => return Err(ZarrError::UnsupportedVersion(other)),
        }
        let shape = u64_vec(v, "shape")?;
        let dtype = DataType::from_v3_name(
            v.get("data_type")
                .and_then(Value::as_str)
                .ok_or(ZarrError::MissingField { field: "data_type" })?,
        )?;

        let grid = v.get("chunk_grid").ok_or(ZarrError::MissingField {
            field: "chunk_grid",
        })?;
        let chunk_shape = grid
            .get("configuration")
            .map(|c| u64_vec(c, "chunk_shape"))
            .transpose()?
            .ok_or(ZarrError::MissingField {
                field: "chunk_grid.configuration.chunk_shape",
            })?;
        validate_shape_and_chunks(&shape, &chunk_shape)?;

        let mut codecs = Vec::new();
        if let Some(list) = v.get("codecs").and_then(Value::as_array) {
            for c in list {
                codecs.push(CodecSpec::from_v3(c)?);
            }
        }
        // v3 states byte order in the `bytes` codec rather than in the dtype.
        let endianness = codecs
            .iter()
            .find(|c| c.name == "bytes")
            .and_then(|c| c.configuration.get("endian").and_then(Value::as_str))
            .map(|e| {
                if e == "big" {
                    Endianness::Big
                } else {
                    Endianness::Little
                }
            })
            .unwrap_or(Endianness::Little);

        let chunk_key = match v.get("chunk_key_encoding") {
            None => ChunkKeyEncoding::Default { separator: '/' },
            Some(enc) => {
                let name = enc.get("name").and_then(Value::as_str).unwrap_or("default");
                let separator = enc
                    .get("configuration")
                    .and_then(|c| c.get("separator"))
                    .and_then(Value::as_str)
                    .and_then(|s| s.chars().next());
                match name {
                    "v2" => ChunkKeyEncoding::V2 {
                        separator: separator.unwrap_or('.'),
                    },
                    _ => ChunkKeyEncoding::Default {
                        separator: separator.unwrap_or('/'),
                    },
                }
            }
        };

        let dimension_names = v.get("dimension_names").and_then(Value::as_array).map(|a| {
            a.iter()
                .map(|n| n.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        });

        Ok(ArrayMeta {
            version: ZarrVersion::V3,
            shape,
            chunk_shape,
            dtype,
            endianness,
            fill_value: fill_value(v.get("fill_value")),
            codecs,
            chunk_key,
            dimension_names,
            attributes: v.get("attributes").map(object_map).unwrap_or_default(),
            order_f: false,
        })
    }

    /// Number of chunks along each dimension, rounding up: the last chunk of a dimension may hang
    /// off the end of the array and is truncated when read.
    pub fn chunks_per_dim(&self) -> Vec<u64> {
        self.shape
            .iter()
            .zip(&self.chunk_shape)
            .map(|(&n, &c)| if c == 0 { 0 } else { n.div_ceil(c) })
            .collect()
    }

    /// Total number of chunks the array would have if every one were written.
    pub fn chunk_count(&self) -> u64 {
        self.chunks_per_dim().iter().product()
    }

    /// The store key for one chunk, relative to the array prefix.
    pub fn chunk_key(&self, index: &[u64]) -> String {
        self.chunk_key.key(index)
    }

    /// Rank of the array. cogzilla renders 2D and 3D arrays, as TiTiler does; anything else is
    /// refused by name rather than squeezed into shape silently.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check the rank before any work is done, so the refusal is cheap and explicit.
    pub fn require_renderable_rank(&self) -> Result<(), ZarrError> {
        match self.rank() {
            2 | 3 => Ok(()),
            other => Err(ZarrError::UnsupportedRank(other)),
        }
    }

    /// Effective nodata value: CF `_FillValue`, then `missing_value`, then Zarr `fill_value`.
    pub fn effective_nodata(&self) -> Option<f64> {
        fill_value(self.attributes.get("_FillValue"))
            .or_else(|| fill_value(self.attributes.get("missing_value")))
            .or(self.fill_value)
    }

    /// CF `scale_factor` and `add_offset` if either is non-identity.
    pub fn scale_offset(&self) -> Option<(f32, f32)> {
        let scale = self
            .attributes
            .get("scale_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let offset = self
            .attributes
            .get("add_offset")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;
        if scale != 1.0 || offset != 0.0 {
            Some((scale, offset))
        } else {
            None
        }
    }
}

fn u64_vec(v: &Value, field: &'static str) -> Result<Vec<u64>, ZarrError> {
    let arr = v
        .get(field)
        .and_then(Value::as_array)
        .ok_or(ZarrError::MissingField { field })?;
    if arr.len() > 16 {
        return Err(ZarrError::BadField {
            field,
            detail: format!("dimension count {} exceeds sanity limit 16", arr.len()),
        });
    }
    arr.iter()
        .map(|x| {
            x.as_u64().ok_or(ZarrError::BadField {
                field,
                detail: format!("{x} is not a non-negative integer"),
            })
        })
        .collect()
}

fn validate_shape_and_chunks(shape: &[u64], chunk_shape: &[u64]) -> Result<(), ZarrError> {
    if shape.len() != chunk_shape.len() {
        return Err(ZarrError::BadField {
            field: "chunks",
            detail: format!(
                "chunk_shape rank {} != array shape rank {}",
                chunk_shape.len(),
                shape.len()
            ),
        });
    }
    let mut total_chunks: u64 = 1;
    for (ax, (&s, &c)) in shape.iter().zip(chunk_shape).enumerate() {
        if s > 0 && c == 0 {
            return Err(ZarrError::BadField {
                field: "chunks",
                detail: format!("chunk dimension {ax} is 0 for non-empty axis of size {s}"),
            });
        }
        if c > 0 {
            let n = s.div_ceil(c);
            total_chunks = total_chunks.checked_mul(n).ok_or_else(|| ZarrError::BadField {
                field: "chunks",
                detail: "total chunk count overflows u64".into(),
            })?;
        }
    }
    let _ = total_chunks;
    Ok(())
}

fn object_map(v: &Value) -> BTreeMap<String, Value> {
    v.as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

/// Both versions allow a number, `null`, or one of the three non-finite spellings. Raw base64 fill
/// values for byte-array dtypes are not rendered by cogzilla and read as "no fill value".
fn fill_value(v: Option<&Value>) -> Option<f64> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        Some(Value::String(s)) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V2: &[u8] = br#"{
        "zarr_format": 2,
        "shape": [12, 1800, 3600],
        "chunks": [1, 600, 600],
        "dtype": "<f4",
        "fill_value": "NaN",
        "order": "C",
        "dimension_separator": "/",
        "filters": [{"id": "shuffle", "elementsize": 4}],
        "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1}
    }"#;

    const V2_ATTRS: &[u8] = br#"{"_ARRAY_DIMENSIONS": ["time", "lat", "lon"], "units": "K"}"#;

    const V3: &[u8] = br#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1800, 3600],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [512, 512]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "big"}},
            {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}
        ],
        "dimension_names": ["y", "x"],
        "attributes": {"grid_mapping": "spatial_ref"}
    }"#;

    #[test]
    fn v2_array_is_read_including_xarray_dimension_names() {
        let m = ArrayMeta::from_v2(V2, Some(V2_ATTRS)).unwrap();
        assert_eq!(m.version, ZarrVersion::V2);
        assert_eq!(m.shape, vec![12, 1800, 3600]);
        assert_eq!(m.chunk_shape, vec![1, 600, 600]);
        assert_eq!(m.dtype, DataType::Float32);
        assert_eq!(m.endianness, Endianness::Little);
        assert!(m.fill_value.unwrap().is_nan());
        // filters come before the compressor, in declared order
        assert_eq!(m.codecs.len(), 2);
        assert_eq!(m.codecs[0].name, "shuffle");
        assert_eq!(m.codecs[1].name, "blosc");
        assert_eq!(
            m.dimension_names,
            Some(vec![
                Some("time".to_string()),
                Some("lat".to_string()),
                Some("lon".to_string())
            ])
        );
        assert_eq!(m.rank(), 3);
        m.require_renderable_rank().unwrap();
    }

    #[test]
    fn v2_chunk_grid_rounds_up_and_keys_use_the_declared_separator() {
        let m = ArrayMeta::from_v2(V2, None).unwrap();
        assert_eq!(m.chunks_per_dim(), vec![12, 3, 6]);
        assert_eq!(m.chunk_count(), 12 * 3 * 6);
        assert_eq!(m.chunk_key(&[0, 1, 2]), "0/1/2");
    }

    #[test]
    fn v2_default_separator_is_a_dot() {
        let src = br#"{"zarr_format":2,"shape":[4,4],"chunks":[2,2],"dtype":"|u1","fill_value":0}"#;
        let m = ArrayMeta::from_v2(src, None).unwrap();
        assert_eq!(m.chunk_key(&[1, 0]), "1.0");
        assert_eq!(m.dtype, DataType::UInt8);
        assert_eq!(m.fill_value, Some(0.0));
        assert!(m.codecs.is_empty());
    }

    #[test]
    fn v3_array_is_read_with_endianness_from_the_bytes_codec() {
        let m = ArrayMeta::from_v3(V3).unwrap();
        assert_eq!(m.version, ZarrVersion::V3);
        assert_eq!(m.shape, vec![1800, 3600]);
        assert_eq!(m.chunk_shape, vec![512, 512]);
        assert_eq!(m.dtype, DataType::UInt16);
        assert_eq!(m.endianness, Endianness::Big);
        assert_eq!(m.fill_value, Some(0.0));
        assert_eq!(m.chunks_per_dim(), vec![4, 8]);
        assert_eq!(m.chunk_key(&[3, 7]), "c/3/7");
        assert_eq!(
            m.attributes.get("grid_mapping").and_then(Value::as_str),
            Some("spatial_ref")
        );
    }

    #[test]
    fn v3_v2_chunk_key_encoding_is_honoured() {
        let src = br#"{
            "zarr_format": 3, "shape": [8, 8], "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4]}},
            "chunk_key_encoding": {"name": "v2"},
            "fill_value": "-Infinity", "codecs": ["bytes"]
        }"#;
        let m = ArrayMeta::from_v3(src).unwrap();
        assert_eq!(m.chunk_key(&[1, 1]), "1.1");
        assert_eq!(m.fill_value, Some(f64::NEG_INFINITY));
        assert_eq!(m.codecs[0].name, "bytes");
        assert!(m.dtype.needs_wide_path());
    }

    #[test]
    fn a_rank_that_cannot_be_rendered_is_named_not_squeezed() {
        let src = br#"{
            "zarr_format": 3, "shape": [2, 3, 4, 5], "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 2, 2]}},
            "fill_value": null, "codecs": ["bytes"]
        }"#;
        let m = ArrayMeta::from_v3(src).unwrap();
        assert_eq!(m.fill_value, None);
        let err = m.require_renderable_rank().unwrap_err();
        assert!(matches!(err, ZarrError::UnsupportedRank(4)), "{err}");
    }

    #[test]
    fn an_unreadable_dtype_is_refused_by_name() {
        let src = br#"{"zarr_format":2,"shape":[2,2],"chunks":[1,1],"dtype":"<c8"}"#;
        let err = ArrayMeta::from_v2(src, None).unwrap_err();
        assert!(
            matches!(&err, ZarrError::UnsupportedDataType(d) if d == "c8"),
            "{err}"
        );
    }

    #[test]
    fn zero_dimensional_arrays_use_the_specified_single_key() {
        assert_eq!(ChunkKeyEncoding::V2 { separator: '.' }.key(&[]), "0");
        assert_eq!(ChunkKeyEncoding::Default { separator: '/' }.key(&[]), "c");
    }

    #[test]
    fn wide_dtypes_are_flagged_so_the_data_paths_keep_them_exact() {
        for dt in [
            DataType::Int32,
            DataType::UInt32,
            DataType::Int64,
            DataType::UInt64,
            DataType::Float64,
        ] {
            assert!(dt.needs_wide_path(), "{dt:?}");
        }
        for dt in [DataType::UInt8, DataType::Int16, DataType::Float32] {
            assert!(!dt.needs_wide_path(), "{dt:?}");
        }
    }
}
