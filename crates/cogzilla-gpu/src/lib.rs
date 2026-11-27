#![forbid(unsafe_code)]

pub mod batch;
pub mod cache;
pub mod coalesce;
pub mod colormap;
pub mod context;
pub mod expr;
pub mod jpeg;
pub mod mosaic;
pub mod policy;
pub mod resample;
pub(crate) mod shaders;
pub mod stats;
pub mod weighted_stats;
pub(crate) mod wgpu_util;

pub use batch::{GpuTileBatcher, TileCoordSlice};
pub use cache::{GpuCacheStats, GpuMultiBandCache, GpuSlabCache, MultiBandSlabKey, SlabKey};
pub use coalesce::{CoalescedJob, CoalescedOutput, CoalescerStats, GpuCoalescer, OwnedCoordPlan};
pub use colormap::{GpuColormapMode, GpuPixelOutput, pack_rgb_lut};
pub use context::{GpuAdapterInfo, GpuContext, GpuError, GpuOptions};
pub use expr::{GpuColorFormula, GpuInstr, GpuMultiBandSlab};
pub use jpeg::{GpuJpegEncoder, huffman_encode_coeffs};
pub use mosaic::{
    GpuMosaicCompositor, GpuPixelSelection, GpuSceneStack, MosaicCompositeResult, MosaicOutput,
    MosaicUpload,
};
pub use policy::{
    DispatchDecision, DispatchPolicy, DispatchReason, DispatchRequest, GpuOp, GpuProfile,
    PolicyOverride,
};
pub use resample::GpuSlab;
pub use stats::{GpuBandStats, GpuStatsReducer, decode_stats_readback};
pub use cogzilla_raster::DemAlgorithm;
pub use weighted_stats::{GpuWeightedBandStats, GpuWeightedStatsReducer};
pub use wgpu;
