#![forbid(unsafe_code)]

pub mod aimd;
pub mod algorithms;
pub mod catalog;
pub mod config;
pub mod engine;
#[cfg(feature = "gpu")]
pub mod gpu_backend;
#[cfg(not(feature = "gpu"))]
#[path = "gpu_stub.rs"]
pub mod gpu_backend;
pub mod geotiff;
pub mod http;
pub mod openapi;
pub mod stac;
pub mod tile_cache;
pub mod multi_asset;
pub mod stats;
pub mod tms;
pub mod zonal;

pub use aimd::{AimdLimiter, AimdPermit, AimdSnapshot};
pub use config::{Config, ConfigWarning};

pub use catalog::{
    InMemoryCatalog, MosaicJsonCatalog, SceneCatalog, SceneEntry, load_catalog_json_file,
    load_catalog_json_str, tile_to_quadkey, wgs84_bbox_to_3857,
};
pub use engine::{
    BandRequest, EngineBootConfig, EngineError, EngineStats, TileEngine, TileRequest,
};
pub use http::{handle_bytes_request, handle_request, serve, serve_listener};
pub use tile_cache::{CachedTile, TileCache};
