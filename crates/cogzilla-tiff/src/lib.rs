#![forbid(unsafe_code)]

pub mod gcp;
pub mod parser;
pub mod types;

pub use gcp::{GcpFit, GcpPoly, fit_gcps};
pub use parser::{parse_cog, parse_cog_sparse};
pub use types::{
    Affine2D, ByteOrder, CogIndex, Compression, GeoKeys, IfdMeta, MaskIfdMeta, PlanarConfig,
    Predictor, SampleType, TiffError,
};
