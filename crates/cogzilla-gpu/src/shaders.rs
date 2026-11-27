//! Composed WGSL shader sources assembled at compile time from shared fragments.

pub(crate) const COLORMAP_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/colormap.wgsl"),
    "\n",
    include_str!("shaders/common_sample_1band.wgsl"),
    "\n",
    include_str!("shaders/common_eval_uv.wgsl"),
);

pub(crate) const BATCH_TILE_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/batch_tile.wgsl"),
    "\n",
    include_str!("shaders/common_sample_1band.wgsl"),
);

pub(crate) const TERRAIN_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/terrain.wgsl"),
    "\n",
    include_str!("shaders/common_sample_1band.wgsl"),
    "\n",
    include_str!("shaders/common_eval_uv.wgsl"),
);

pub(crate) const EXPR_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/common_expr_vm.wgsl"),
    "\n",
    include_str!("shaders/expr.wgsl"),
    "\n",
    include_str!("shaders/common_eval_uv.wgsl"),
);

pub(crate) const MOSAIC_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/mosaic.wgsl"),
);

pub(crate) const WEIGHTED_EXPR_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/common_expr_vm.wgsl"),
    "\n",
    include_str!("shaders/weighted_expr.wgsl"),
);

pub(crate) const JPEG_DCT_WGSL: &str = include_str!("shaders/jpeg_dct.wgsl");
pub(crate) const STATS_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/stats.wgsl"),
);
pub(crate) const WEIGHTED_STATS_WGSL: &str = concat!(
    include_str!("shaders/common_math.wgsl"),
    "\n",
    include_str!("shaders/weighted_stats.wgsl"),
);
