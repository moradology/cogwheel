//! Per-band statistics accumulators, histogram builders, and TiTiler `BandStats` serialization.

use cogzilla_kernel::ResampleMode;

use crate::engine::TileEngine;
use crate::stats::histogram::{AscendingBins, HistDtype, bin_of, sorted_bin_ranges};

/// Per-band statistics accumulator: uses exact O(1)-memory counting histograms for `U8`/`U16`
/// bands (zero sort, zero per-pixel enum dispatch) and O(N) radix sort for `F32` bands.
#[allow(clippy::large_enum_variant)] // Inline 2 KiB U8 histogram avoids a heap alloc per band
pub(super) enum BandStatsAcc {
    HistU8([u64; 256]),
    HistU16(Vec<u64>),
    Floats(Vec<f32>),
    Floats64(Vec<f64>),
}

impl BandStatsAcc {
    pub(super) fn for_sample_type(st: cogzilla_raster::SampleType, capacity_hint: usize) -> Self {
        match st {
            cogzilla_raster::SampleType::Uint1
            | cogzilla_raster::SampleType::Uint4
            | cogzilla_raster::SampleType::Uint8 => Self::HistU8([0u64; 256]),
            cogzilla_raster::SampleType::Uint16 => Self::HistU16(vec![0u64; 65_536]),
            s if s.needs_f64() => Self::Floats64(Vec::with_capacity(capacity_hint)),
            _ => Self::Floats(Vec::with_capacity(capacity_hint)),
        }
    }

    /// Accumulate an unmasked tile region `[0..x_max) x [0..y_max)` directly from `BandData`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_tile_unmasked(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        use cogzilla_codec::BandData;
        match (self, data) {
            (Self::HistU8(hist), BandData::U8 { data: slice, nodata }) => {
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            hist[v as usize] += 1;
                        }
                    }
                }
            }
            (
                Self::HistU8(hist),
                BandData::RgbU8Channel {
                    rgb,
                    channel,
                    nodata,
                },
            ) => {
                let ch = *channel as usize;
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let base = ly * tw * 3 + ch;
                        for lx in 0..x_max {
                            let v = rgb[base + lx * 3];
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let base = ly * tw * 3 + ch;
                        for lx in 0..x_max {
                            hist[rgb[base + lx * 3] as usize] += 1;
                        }
                    }
                }
            }
            (Self::HistU16(hist), BandData::U16 { data: slice, nodata }) => {
                if let Some(nd) = *nodata {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v != nd {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            hist[v as usize] += 1;
                        }
                    }
                }
            }
            (Self::Floats(vals), BandData::F32(slab)) => {
                let slice = slab.as_slice();
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                if unscale {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() && nd_ov != Some(v) {
                                vals.push(v * scale + offset);
                            }
                        }
                    }
                } else {
                    for ly in 0..y_max {
                        let row = &slice[ly * tw..ly * tw + x_max];
                        for &v in row {
                            if v.is_finite() && nd_ov != Some(v) {
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            (Self::Floats64(vals), other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan()).map(|v| v as f64);
                let sc = scale as f64;
                let off = offset as f64;
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let mut v = other.get_f64(base + lx);
                        if v.is_finite() && nd_ov != Some(v) {
                            if unscale {
                                v = v * sc + off;
                            }
                            vals.push(v);
                        }
                    }
                }
            }
            // Fallback if tile storage variant differs from IFD sample type
            (Self::Floats(vals), other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let mut v = other.get_f32(base + lx);
                        if v.is_finite() && nd_ov != Some(v) {
                            if unscale {
                                v = v * scale + offset;
                            }
                            vals.push(v);
                        }
                    }
                }
            }
            (Self::HistU8(hist), other) => {
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let v = other.get_f32(base + lx);
                        if v.is_finite() {
                            hist[(v.clamp(0.0, 255.0)) as usize] += 1;
                        }
                    }
                }
            }
            (Self::HistU16(hist), other) => {
                for ly in 0..y_max {
                    let base = ly * tw;
                    for lx in 0..x_max {
                        let v = other.get_f32(base + lx);
                        if v.is_finite() {
                            hist[(v.clamp(0.0, 65535.0)) as usize] += 1;
                        }
                    }
                }
            }
        }
    }

    /// Accumulate masked pixels from a tile where `mask_bits[ly * x_max + lx] != 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_tile_masked(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        mask_bits: &[u8],
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        use cogzilla_codec::BandData;
        match (self, data) {
            (Self::HistU8(hist), BandData::U8 { data: slice, nodata }) => {
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = row[lx];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (
                Self::HistU8(hist),
                BandData::RgbU8Channel {
                    rgb,
                    channel,
                    nodata,
                },
            ) => {
                let ch = *channel as usize;
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw * 3 + ch;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = rgb[base + lx * 3];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (Self::HistU16(hist), BandData::U16 { data: slice, nodata }) => {
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let v = row[lx];
                            if *nodata != Some(v) {
                                hist[v as usize] += 1;
                            }
                        }
                    }
                }
            }
            (Self::Floats(vals), BandData::F32(slab)) => {
                let slice = slab.as_slice();
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let row = &slice[ly * tw..ly * tw + x_max];
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = row[lx];
                            if v.is_finite() && nd_ov != Some(v) {
                                if unscale {
                                    v = v * scale + offset;
                                }
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            (Self::Floats64(vals), other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan()).map(|v| v as f64);
                let sc = scale as f64;
                let off = offset as f64;
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = other.get_f64(base + lx);
                            if v.is_finite() && nd_ov != Some(v) {
                                if unscale {
                                    v = v * sc + off;
                                }
                                vals.push(v);
                            }
                        }
                    }
                }
            }
            (acc, other) => {
                let nd_ov = nodata_override.filter(|v| !v.is_nan());
                for ly in 0..y_max {
                    let mrow = &mask_bits[ly * x_max..(ly + 1) * x_max];
                    let base = ly * tw;
                    for lx in 0..x_max {
                        if mrow[lx] != 0 {
                            let mut v = other.get_f32(base + lx);
                            if v.is_finite() && nd_ov != Some(v) {
                                match acc {
                                    Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                                    Self::HistU16(hist) => {
                                        hist[(v.clamp(0.0, 65535.0)) as usize] += 1
                                    }
                                    Self::Floats(vals) => {
                                        if unscale {
                                            v = v * scale + offset;
                                        }
                                        vals.push(v);
                                    }
                                    Self::Floats64(_) => unreachable!(),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Accumulate strided pixels (`lx0, lx0+step, ...` × `ly0, ly0+step, ...`) when an overview-less
    /// IFD exceeds `max_size`. `mask_bits`, when present, is compact (`nx * ny` bytes matching the
    /// strided grid).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push_tile_strided(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        x_max: usize,
        y_max: usize,
        lx0: usize,
        ly0: usize,
        step: usize,
        mask_bits: Option<&[u8]>,
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        let nd_ov = nodata_override.filter(|v| !v.is_nan());
        let mut mi = 0usize;
        let mut ly = ly0;
        while ly < y_max {
            let base = ly * tw;
            let mut lx = lx0;
            while lx < x_max {
                let keep = match mask_bits {
                    Some(bits) => {
                        let k = bits.get(mi).copied().unwrap_or(0) != 0;
                        mi += 1;
                        k
                    }
                    None => true,
                };
                if keep {
                    if let Self::Floats64(vals) = self {
                        let mut v = data.get_f64(base + lx);
                        if v.is_finite() && nd_ov.map(|n| n as f64) != Some(v) {
                            if unscale {
                                v = v * (scale as f64) + (offset as f64);
                            }
                            vals.push(v);
                        }
                    } else {
                        let mut v = data.get_f32(base + lx);
                        if v.is_finite() && nd_ov != Some(v) {
                            match self {
                                Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                                Self::HistU16(hist) => hist[(v.clamp(0.0, 65535.0)) as usize] += 1,
                                Self::Floats(vals) => {
                                    if unscale {
                                        v = v * scale + offset;
                                    }
                                    vals.push(v);
                                }
                                Self::Floats64(_) => unreachable!(),
                            }
                        }
                    }
                }
                lx += step;
            }
            ly += step;
        }
    }

    pub(super) fn push_tile_coords(
        &mut self,
        data: &cogzilla_codec::BandData,
        tw: usize,
        xs: &[usize],
        ys: &[usize],
        unscale: bool,
        scale: f32,
        offset: f32,
        nodata_override: Option<f32>,
    ) {
        let nd_ov = nodata_override.filter(|v| !v.is_nan());
        for &ly in ys {
            let base = ly * tw;
            for &lx in xs {
                if let Self::Floats64(vals) = self {
                    let mut v = data.get_f64(base + lx);
                    if v.is_finite() && nd_ov.map(|n| n as f64) != Some(v) {
                        if unscale {
                            v = v * (scale as f64) + (offset as f64);
                        }
                        vals.push(v);
                    }
                } else {
                    let mut v = data.get_f32(base + lx);
                    if v.is_finite() && nd_ov != Some(v) {
                        match self {
                            Self::HistU8(hist) => hist[(v.clamp(0.0, 255.0)) as usize] += 1,
                            Self::HistU16(hist) => hist[(v.clamp(0.0, 65535.0)) as usize] += 1,
                            Self::Floats(vals) => {
                                if unscale {
                                    v = v * scale + offset;
                                }
                                vals.push(v);
                            }
                            Self::Floats64(_) => unreachable!(),
                        }
                    }
                }
            }
        }
    }
}

/// Optional statistics customization matching TiTiler's `/statistics` query parameters.
#[derive(Debug, Clone)]
pub struct StatsOptions {
    pub percentiles: Vec<f64>,
    pub histogram_bins: usize,
    pub histogram_edges: Option<Vec<f64>>,
    pub histogram_range: Option<(f64, f64)>,
    pub categorical: bool,
    pub categories: Option<Vec<f64>>,
    pub bidx: Option<Vec<u16>>,
    pub expressions: Option<Vec<(String, cogzilla_kernel::ExprProgram)>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub max_size_explicit: bool,
    pub area_weighted: bool,
    /// The type numpy builds the histogram's bin edges in, for paths that only see `f32` values;
    /// `None` infers it from the values (see [`HistDtype::infer`]).
    pub hist_dtype: Option<HistDtype>,
    /// `resampling`: how a whole raster's statistics read it at a size between its levels
    /// (rio-tiler reads its preview with it). Nearest picks the pixel under each centre.
    pub resample: ResampleMode,
}

impl StatsOptions {
    /// numpy's histogram bin edges for this request over data spanning `min_max`.
    pub fn histogram_bin_edges(&self, min_max: Option<(f64, f64)>, dtype: HistDtype) -> Vec<f64> {
        crate::stats::histogram::bin_edges(
            self.histogram_bins.clamp(1, 1000),
            self.histogram_edges.as_deref(),
            self.histogram_range,
            min_max,
            dtype,
        )
    }
}

impl Default for StatsOptions {
    fn default() -> Self {
        Self {
            percentiles: vec![2.0, 98.0],
            histogram_bins: 10,
            histogram_edges: None,
            histogram_range: None,
            categorical: false,
            categories: None,
            bidx: None,
            expressions: None,
            width: None,
            height: None,
            max_size_explicit: false,
            area_weighted: true,
            hist_dtype: None,
            resample: ResampleMode::Nearest,
        }
    }
}

fn build_categorical_histogram(
    observed: &[(f64, u64)],
    req_cats: &Option<Vec<f64>>,
) -> serde_json::Value {
    let val_json = |v: f64| -> serde_json::Value {
        if v.fract() == 0.0 && v >= (i64::MIN as f64) && v <= (i64::MAX as f64) {
            serde_json::json!(v as i64)
        } else {
            serde_json::json!(v)
        }
    };
    if let Some(req) = req_cats {
        let mut counts = Vec::with_capacity(req.len());
        let mut vals = Vec::with_capacity(req.len());
        for &c in req {
            let cnt = observed
                .iter()
                .find(|(v, _)| (v - c).abs() <= 1e-6)
                .map(|(_, n)| *n)
                .unwrap_or(0);
            counts.push(cnt);
            vals.push(val_json(c));
        }
        serde_json::json!([counts, vals])
    } else {
        let counts: Vec<u64> = observed.iter().map(|(_, n)| *n).collect();
        let vals: Vec<serde_json::Value> = observed.iter().map(|(v, _)| val_json(*v)).collect();
        serde_json::json!([counts, vals])
    }
}

/// Compute TiTiler-compatible band statistics from a dense `f32` slice where `NaN` marks masked pixels.
pub fn compute_f32_band_stats(
    pixels: &[f32],
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    let denom = pixels.len();
    let valid: Vec<f32> = pixels.iter().copied().filter(|v| v.is_finite()).collect();
    BandStatsAcc::Floats(valid).finalize(&BandStatsFinalize {
        denom_pixels: denom,
        unscale: false,
        scale: 1.0,
        offset: 0.0,
        nodata_override: None,
        description,
        opts,
        dtype: opts.hist_dtype,
    })
}

/// GPU-accelerated variant of `compute_f32_band_stats`: when built with `--features gpu` and
/// above the adapter-calibrated pixel crossover (non-categorical, standard histogram bins),
/// runs the two-pass WGSL `GpuStatsReducer` (20×–51× faster than CPU `sort_unstable_by` on
/// 4096² rasters) and formats the identical TiTiler JSON object.
pub fn compute_f32_band_stats_with_engine(
    engine: &TileEngine,
    pixels: &[f32],
    description: &str,
    opts: &StatsOptions,
) -> serde_json::Value {
    {
        if !opts.categorical
            && opts.histogram_edges.is_none()
            && let Some(gpu) = engine.gpu.as_ref()
            && let Some(gs) = gpu.try_reduce_f32_stats(
                pixels,
                None,
                opts.histogram_bins.clamp(1, 1000),
                opts.histogram_range,
                &opts.percentiles,
            )
        {
            let dtype = opts
                .hist_dtype
                .unwrap_or_else(|| HistDtype::infer(pixels.iter().copied()));
            return gpu_band_stats_to_json(&gs, pixels.len(), description, opts, dtype, Some(pixels));
        }
    }
    let _ = engine;
    compute_f32_band_stats(pixels, description, opts)
}

/// Scalar band-statistic value preserving native `f32` vs `f64` JSON formatting.
#[derive(Debug, Clone, Copy)]
pub enum StatScalar {
    F32(f32),
    F64(f64),
}

impl From<f32> for StatScalar {
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}

impl From<f64> for StatScalar {
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}

impl StatScalar {
    fn to_json(self) -> serde_json::Value {
        match self {
            Self::F32(v) => serde_json::json!(v),
            Self::F64(v) => serde_json::json!(v),
        }
    }
}

/// Count or pixel-total field: exact integer on regular raster stats, coverage float on zonal stats.
#[derive(Debug, Clone, Copy)]
pub enum StatCount {
    Int(usize),
    Float(f64),
}

impl StatCount {
    fn to_json(self) -> serde_json::Value {
        match self {
            Self::Int(v) => serde_json::json!(v),
            Self::Float(v) => serde_json::json!(v),
        }
    }
}

/// Band histogram payload for [`BandStats`].
#[derive(Debug, Clone)]
pub enum BandHistogram {
    NumericInt(Vec<u64>, Vec<f64>),
    NumericFloat(Vec<f64>, Vec<f64>),
    Categorical(serde_json::Value),
}

impl BandHistogram {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::NumericInt(counts, edges) => serde_json::json!([counts, edges]),
            Self::NumericFloat(counts, edges) => serde_json::json!([counts, edges]),
            Self::Categorical(val) => val.clone(),
        }
    }
}

/// Unified typed representation of a single band's statistics response (#416).
///
/// Constructed from the CPU accumulator, the GPU reduction, or the coverage-weighted zonal pass,
/// and serialised through a single [`BandStats::to_json`] implementation.
#[derive(Debug, Clone)]
pub struct BandStats {
    pub min: Option<StatScalar>,
    pub max: Option<StatScalar>,
    pub mean: Option<f64>,
    pub count: StatCount,
    pub sum: f64,
    pub std: Option<f64>,
    pub median: Option<StatScalar>,
    pub majority: Option<StatScalar>,
    pub minority: Option<StatScalar>,
    pub unique: StatCount,
    pub valid_percent: f64,
    pub masked_pixels: StatCount,
    pub valid_pixels: StatCount,
    pub percentiles: Vec<(f64, Option<StatScalar>)>,
    pub histogram: BandHistogram,
    pub categories: Option<serde_json::Map<String, serde_json::Value>>,
    pub category_areas: Option<serde_json::Map<String, serde_json::Value>>,
    /// `None` omits the key (non-zonal); `Some(None)` emits `"area": null` (Point/Line);
    /// `Some(Some(m2))` emits `"area": m2` (Polygon).
    pub area: Option<Option<f64>>,
    pub length: Option<f64>,
    pub description: Option<String>,
}

impl BandStats {
    /// Construct an empty (all-masked / zero-valid-pixel) band statistics struct.
    pub fn empty(
        masked_pixels: StatCount,
        description: Option<String>,
        opts: &StatsOptions,
        dtype: HistDtype,
    ) -> Self {
        let histogram = if opts.categorical {
            BandHistogram::Categorical(serde_json::json!([[], []]))
        } else {
            let edges = opts.histogram_bin_edges(None, dtype);
            let counts = vec![0u64; edges.len().saturating_sub(1)];
            BandHistogram::NumericInt(counts, edges)
        };
        Self {
            min: None,
            max: None,
            mean: None,
            count: StatCount::Int(0),
            sum: 0.0,
            std: None,
            median: None,
            majority: None,
            minority: None,
            unique: StatCount::Int(0),
            valid_percent: 0.0,
            masked_pixels,
            valid_pixels: StatCount::Int(0),
            percentiles: opts.percentiles.iter().map(|&p| (p, None)).collect(),
            histogram,
            categories: opts.categorical.then(serde_json::Map::new),
            category_areas: None,
            area: None,
            length: None,
            description,
        }
    }

    /// Serialise into the TiTiler-compatible JSON object.
    pub fn to_json(&self) -> serde_json::Value {
        let opt_scalar = |v: Option<StatScalar>| match v {
            Some(s) => s.to_json(),
            None => serde_json::Value::Null,
        };
        let mut obj = serde_json::Map::new();
        obj.insert("min".into(), opt_scalar(self.min));
        obj.insert("max".into(), opt_scalar(self.max));
        obj.insert("mean".into(), serde_json::json!(self.mean));
        obj.insert("count".into(), self.count.to_json());
        obj.insert("sum".into(), serde_json::json!(self.sum));
        obj.insert("std".into(), serde_json::json!(self.std));
        obj.insert("median".into(), opt_scalar(self.median));
        obj.insert("majority".into(), opt_scalar(self.majority));
        obj.insert("minority".into(), opt_scalar(self.minority));
        obj.insert("unique".into(), self.unique.to_json());
        obj.insert("valid_percent".into(), serde_json::json!(self.valid_percent));
        obj.insert("masked_pixels".into(), self.masked_pixels.to_json());
        obj.insert("valid_pixels".into(), self.valid_pixels.to_json());
        for &(p, val) in &self.percentiles {
            obj.insert(percentile_key(p), opt_scalar(val));
        }
        obj.insert("histogram".into(), self.histogram.to_json());
        if let Some(ref cats) = self.categories {
            obj.insert("categories".into(), serde_json::Value::Object(cats.clone()));
        }
        if let Some(ref cat_areas) = self.category_areas {
            obj.insert(
                "category_areas".into(),
                serde_json::Value::Object(cat_areas.clone()),
            );
        }
        if let Some(area_opt) = self.area {
            obj.insert("area".into(), serde_json::json!(area_opt));
        }
        if let Some(len) = self.length {
            obj.insert("length".into(), serde_json::json!(len));
        }
        if let Some(ref desc) = self.description {
            obj.insert("description".into(), serde_json::json!(desc));
        }
        serde_json::Value::Object(obj)
    }
}

/// Parameters for [`BandStatsAcc::finalize`].
pub(super) struct BandStatsFinalize<'a> {
    pub(super) denom_pixels: usize,
    pub(super) unscale: bool,
    pub(super) scale: f32,
    pub(super) offset: f32,
    pub(super) nodata_override: Option<f32>,
    pub(super) description: &'a str,
    pub(super) opts: &'a StatsOptions,
    pub(super) dtype: Option<HistDtype>,
}

/// `values`, when the band's values are at hand, are binned here as numpy bins them; without
/// them the GPU's own counts are kept against numpy's edges.
pub(super) fn gpu_band_stats_to_json(
    gs: &crate::gpu_backend::GpuBandStats,
    denom_pixels: usize,
    description: &str,
    opts: &StatsOptions,
    dtype: HistDtype,
    values: Option<&[f32]>,
) -> serde_json::Value {
    let masked_pixels = denom_pixels.saturating_sub(gs.valid_pixels);
    if gs.valid_pixels == 0 {
        return empty_band_stats_json(masked_pixels, description, opts, dtype);
    }
    let edges = opts.histogram_bin_edges(gs.min.zip(gs.max), dtype);
    let counts = match values {
        Some(v) => crate::stats::histogram::unsorted_counts(v, &edges),
        None if gs.min.is_some() && gs.min == gs.max => {
            let mut c = vec![0u64; edges.len().saturating_sub(1)];
            if let Some(v) = gs.min
                && let Some(b) = crate::stats::histogram::bin_of(&edges, v)
            {
                c[b] = gs.valid_pixels as u64;
            }
            c
        }
        None => gs.hist_counts.clone(),
    };
    let valid_percent = if denom_pixels > 0 {
        ((gs.valid_pixels as f64 / denom_pixels as f64) * 100.0 * 100.0).round() / 100.0
    } else {
        0.0
    };
    BandStats {
        min: gs.min.map(StatScalar::from),
        max: gs.max.map(StatScalar::from),
        mean: gs.mean,
        count: StatCount::Int(gs.valid_pixels),
        sum: gs.sum,
        std: gs.std,
        median: gs.median.map(StatScalar::from),
        majority: gs.majority.map(StatScalar::from),
        minority: gs.minority.map(StatScalar::from),
        unique: StatCount::Int(gs.unique_bins),
        valid_percent,
        masked_pixels: StatCount::Int(masked_pixels),
        valid_pixels: StatCount::Int(gs.valid_pixels),
        percentiles: gs
            .percentiles
            .iter()
            .map(|&(p, val)| (p, val.map(StatScalar::from)))
            .collect(),
        histogram: BandHistogram::NumericInt(counts, edges),
        categories: None,
        category_areas: None,
        area: None,
        length: None,
        description: Some(description.to_string()),
    }
    .to_json()
}

/// The rank rio-tiler takes a quantile `q` (0 to 1) at among `n` sorted values: with every
/// pixel's weight 1, `numpy.searchsorted(cumsum, q * n)` is the first value whose count reaches
/// `q * n`, so an even count's median is the lower of its two middle values.
pub(crate) fn quantile_rank(n: usize, q: f64) -> usize {
    let target = q * n as f64;
    (target.ceil() as usize).saturating_sub(1).min(n.saturating_sub(1))
}

pub(crate) fn percentile_key(p: f64) -> String {
    if p.fract() == 0.0 {
        format!("percentile_{}", p as i64)
    } else {
        format!("percentile_{p}")
    }
}

pub(crate) fn format_category_key(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn matches_category_filter(v: f64, filter: &Option<Vec<f64>>) -> bool {
    match filter {
        None => true,
        Some(list) => list.iter().any(|&c| (c - v).abs() <= 1e-6),
    }
}

fn empty_band_stats_json(
    masked_px: usize,
    description: &str,
    opts: &StatsOptions,
    dtype: HistDtype,
) -> serde_json::Value {
    BandStats::empty(
        StatCount::Int(masked_px),
        Some(description.to_string()),
        opts,
        dtype,
    )
    .to_json()
}

impl BandStatsAcc {
    #[allow(clippy::too_many_arguments)]
    fn finalize_hist_with_opts(
        hist: &[u64],
        denom_pixels: usize,
        unscale: bool,
        scale: f32,
        offset: f32,
        description: &str,
        opts: &StatsOptions,
        dtype: HistDtype,
    ) -> serde_json::Value {
        let valid_px: usize = hist.iter().map(|&c| c as usize).sum();
        let masked_px = denom_pixels.saturating_sub(valid_px);
        let valid_pct = if denom_pixels > 0 {
            (valid_px as f64 / denom_pixels as f64) * 100.0
        } else {
            0.0
        };
        if valid_px == 0 {
            return empty_band_stats_json(masked_px, description, opts, dtype);
        }
        let reverse = unscale && scale < 0.0;
        let to_val = |k: usize| -> f32 {
            if unscale {
                (k as f32) * scale + offset
            } else {
                k as f32
            }
        };
        let rank_med = quantile_rank(valid_px, 0.5);
        let p_ranks: Vec<usize> = opts
            .percentiles
            .iter()
            .map(|&p| quantile_rank(valid_px, p.clamp(0.0, 100.0) / 100.0))
            .collect();
        let mut p_vals: Vec<f32> = vec![f32::NAN; p_ranks.len()];

        let mut min = f32::NAN;
        let mut max = f32::NAN;
        let mut median = f32::NAN;
        let mut sum = 0.0f64;
        let mut cum = 0usize;
        let mut unique = 0usize;
        let mut maj_val = f32::NAN;
        let mut maj_cnt = 0usize;
        let mut min_val = f32::NAN;
        let mut min_cnt = usize::MAX;
        let mut categories = if opts.categorical {
            Some(serde_json::Map::new())
        } else {
            None
        };
        let mut cat_observed: Vec<(f64, u64)> = Vec::new();

        let n = hist.len();
        for step in 0..n {
            let k = if reverse { n - 1 - step } else { step };
            let c = hist[k] as usize;
            if c == 0 {
                continue;
            }
            unique += 1;
            let v = to_val(k);
            if let Some(ref mut cats) = categories {
                let vf = v as f64;
                cat_observed.push((vf, c as u64));
                if matches_category_filter(vf, &opts.categories) {
                    cats.insert(format_category_key(vf), serde_json::json!(c));
                }
            }
            if c > maj_cnt {
                maj_cnt = c;
                maj_val = v;
            }
            if c < min_cnt {
                min_cnt = c;
                min_val = v;
            }
            if min.is_nan() {
                min = v;
            }
            max = v;
            let prev = cum;
            cum += c;
            if prev <= rank_med && rank_med < cum {
                median = v;
            }
            for (idx, &rk) in p_ranks.iter().enumerate() {
                if prev <= rk && rk < cum {
                    p_vals[idx] = v;
                }
            }
            sum += (v as f64) * (c as f64);
        }

        let mean = sum / (valid_px as f64);
        let mut var_sum = 0.0f64;
        // Each distinct value goes to numpy's bin with its whole count.
        let edges = opts.histogram_bin_edges(Some((min as f64, max as f64)), dtype);
        let mut counts = vec![0u64; edges.len() - 1];
        let mut ascending = AscendingBins::new(&edges);
        for (k, &c) in hist.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let v = to_val(k) as f64;
            let d = v - mean;
            var_sum += d * d * (c as f64);
            let bin = if reverse { bin_of(&edges, v) } else { ascending.bin_of(v) };
            if let Some(bin) = bin {
                counts[bin] += c;
            }
        }
        let std = (var_sum / (valid_px as f64)).sqrt();
        let histogram = if categories.is_some() {
            BandHistogram::Categorical(build_categorical_histogram(&cat_observed, &opts.categories))
        } else {
            BandHistogram::NumericInt(counts, edges)
        };
        BandStats {
            min: Some(StatScalar::from(min)),
            max: Some(StatScalar::from(max)),
            mean: Some(mean),
            count: StatCount::Int(valid_px),
            sum,
            std: Some(std),
            median: Some(StatScalar::from(median)),
            majority: Some(StatScalar::from(maj_val)),
            minority: Some(StatScalar::from(min_val)),
            unique: StatCount::Int(unique),
            valid_percent: (valid_pct * 100.0).round() / 100.0,
            masked_pixels: StatCount::Int(masked_px),
            valid_pixels: StatCount::Int(valid_px),
            percentiles: opts
                .percentiles
                .iter()
                .zip(p_vals.iter())
                .map(|(&p, &pv)| (p, Some(StatScalar::from(pv))))
                .collect(),
            histogram,
            categories,
            category_areas: None,
            area: None,
            length: None,
            description: Some(description.to_string()),
        }
        .to_json()
    }

    fn finalize_sorted<T>(
        mut vals: Vec<T>,
        denom_pixels: usize,
        description: &str,
        opts: &StatsOptions,
        dtype: HistDtype,
        sort_vals: impl FnOnce(&mut [T]),
    ) -> serde_json::Value
    where
        T: Copy + PartialEq + Into<f64> + Into<StatScalar>,
        f64: From<T>,
    {
        let valid_px = vals.len();
        let masked_px = denom_pixels.saturating_sub(valid_px);
        let valid_pct = if denom_pixels > 0 {
            (valid_px as f64 / denom_pixels as f64) * 100.0
        } else {
            0.0
        };
        if valid_px == 0 {
            return empty_band_stats_json(masked_px, description, opts, dtype);
        }
        sort_vals(&mut vals);
        let min = vals[0];
        let max = vals[valid_px - 1];
        let mut sum = 0.0f64;
        let mut unique = 0usize;
        let mut maj_val = min;
        let mut maj_cnt = 0usize;
        let mut min_val = min;
        let mut min_cnt = usize::MAX;
        let mut run_val = min;
        let mut run_cnt = 0usize;
        let mut categories = if opts.categorical {
            Some(serde_json::Map::new())
        } else {
            None
        };
        let mut cat_observed: Vec<(f64, u64)> = Vec::new();
        for &v in &vals {
            sum += f64::from(v);
            if v == run_val {
                run_cnt += 1;
            } else {
                unique += 1;
                if let Some(ref mut cats) = categories {
                    let vf: f64 = run_val.into();
                    cat_observed.push((vf, run_cnt as u64));
                    if matches_category_filter(vf, &opts.categories) {
                        cats.insert(format_category_key(vf), serde_json::json!(run_cnt));
                    }
                }
                if run_cnt > maj_cnt {
                    maj_cnt = run_cnt;
                    maj_val = run_val;
                }
                if run_cnt < min_cnt {
                    min_cnt = run_cnt;
                    min_val = run_val;
                }
                run_val = v;
                run_cnt = 1;
            }
        }
        unique += 1;
        if let Some(ref mut cats) = categories {
            let vf: f64 = run_val.into();
            cat_observed.push((vf, run_cnt as u64));
            if matches_category_filter(vf, &opts.categories) {
                cats.insert(format_category_key(vf), serde_json::json!(run_cnt));
            }
        }
        if run_cnt > maj_cnt {
            maj_val = run_val;
        }
        if run_cnt < min_cnt {
            min_val = run_val;
        }
        let n_f = valid_px as f64;
        let mean = if sum.is_finite() {
            sum / n_f
        } else {
            vals.iter().map(|&v| f64::from(v) / n_f).sum::<f64>()
        };
        let min_f: f64 = min.into();
        let max_f: f64 = max.into();
        // Scale variance accumulation when values exceed ~1e150 so `(v - mean)^2` never
        // overflows `f64::MAX` to `+inf` (which JSON serializes as `null`).
        let max_abs = min_f.abs().max(max_f.abs());
        let var_scale = if max_abs > 1e150 { max_abs } else { 1.0 };
        let inv_var_scale = 1.0 / var_scale;
        let mean_scaled = mean * inv_var_scale;
        let mut var_sum = 0.0f64;
        for &v in &vals {
            let d = f64::from(v) * inv_var_scale - mean_scaled;
            var_sum += d * d;
        }
        let edges = opts.histogram_bin_edges(Some((min_f, max_f)), dtype);
        let counts: Vec<u64> = sorted_bin_ranges(&vals, &edges, |&v| v.into())
            .iter()
            .map(|r| r.len() as u64)
            .collect();
        let std = (var_sum / n_f).sqrt() * var_scale;
        let median = vals[quantile_rank(valid_px, 0.5)];
        let p_vals: Vec<T> = opts
            .percentiles
            .iter()
            .map(|&p| vals[quantile_rank(valid_px, p.clamp(0.0, 100.0) / 100.0)])
            .collect();
        let histogram = if categories.is_some() {
            BandHistogram::Categorical(build_categorical_histogram(&cat_observed, &opts.categories))
        } else {
            BandHistogram::NumericInt(counts, edges)
        };
        BandStats {
            min: Some(min.into()),
            max: Some(max.into()),
            mean: Some(mean),
            count: StatCount::Int(valid_px),
            sum,
            std: Some(std),
            median: Some(median.into()),
            majority: Some(maj_val.into()),
            minority: Some(min_val.into()),
            unique: StatCount::Int(unique),
            valid_percent: (valid_pct * 100.0).round() / 100.0,
            masked_pixels: StatCount::Int(masked_px),
            valid_pixels: StatCount::Int(valid_px),
            percentiles: opts
                .percentiles
                .iter()
                .zip(p_vals.iter())
                .map(|(&p, &pv)| (p, Some(pv.into())))
                .collect(),
            histogram,
            categories,
            category_areas: None,
            area: None,
            length: None,
            description: Some(description.to_string()),
        }
        .to_json()
    }

    /// The band's statistics. `dtype` is the type numpy would build the histogram's edges in;
    /// `None` takes it from the accumulator (integer bands, or float32 when unscaled) or, for
    /// `f32` values of unknown source, infers it from the values.
    // Runs once per band: keep it out of the caller that holds the per-tile accumulation loops.
    #[inline(never)]
    pub(super) fn finalize(self, params: &BandStatsFinalize<'_>) -> serde_json::Value {
        let BandStatsFinalize {
            denom_pixels,
            unscale,
            scale,
            offset,
            nodata_override,
            description,
            opts,
            dtype,
        } = *params;
        match self {
            Self::HistU8(mut hist) => {
                if let Some(nd) = nodata_override
                    && nd.is_finite()
                    && (0.0..=255.0).contains(&nd)
                    && nd.fract() == 0.0
                {
                    hist[nd as usize] = 0;
                }
                Self::finalize_hist_with_opts(
                    &hist,
                    denom_pixels,
                    unscale,
                    scale,
                    offset,
                    description,
                    opts,
                    dtype.unwrap_or(HistDtype::for_sample_type(
                        cogzilla_raster::SampleType::Uint16,
                        unscale,
                    )),
                )
            }
            Self::HistU16(mut hist) => {
                if let Some(nd) = nodata_override
                    && nd.is_finite()
                    && (0.0..=65535.0).contains(&nd)
                    && nd.fract() == 0.0
                {
                    hist[nd as usize] = 0;
                }
                Self::finalize_hist_with_opts(
                    &hist,
                    denom_pixels,
                    unscale,
                    scale,
                    offset,
                    description,
                    opts,
                    dtype.unwrap_or(HistDtype::for_sample_type(
                        cogzilla_raster::SampleType::Uint16,
                        unscale,
                    )),
                )
            }
            Self::Floats(vals) => {
                let eff_dtype = if vals.is_empty() {
                    dtype.unwrap_or(HistDtype::Float64)
                } else {
                    dtype.unwrap_or_else(|| HistDtype::infer(vals.iter().copied()))
                };
                Self::finalize_sorted(
                    vals,
                    denom_pixels,
                    description,
                    opts,
                    eff_dtype,
                    crate::zonal::sort_f32_radix,
                )
            }
            Self::Floats64(vals) => Self::finalize_sorted(
                vals,
                denom_pixels,
                description,
                opts,
                dtype.unwrap_or(HistDtype::Float64),
                |s| s.sort_unstable_by(f64::total_cmp),
            ),
        }
    }
}

#[cfg(test)]
mod quantile_rank_tests {
    use super::quantile_rank;

    #[test]
    fn ranks_are_rio_tilers() {
        // An even count's median is its lower middle value; an odd count's, its middle one.
        assert_eq!(quantile_rank(4, 0.5), 1);
        assert_eq!(quantile_rank(5, 0.5), 2);
        assert_eq!(quantile_rank(65536, 0.5), 32767);
        // A whole `q * n` is that many values in: the last of them.
        assert_eq!(quantile_rank(100, 0.02), 1);
        assert_eq!(quantile_rank(65536, 0.25), 16383);
        // Otherwise the value whose count first passes it.
        assert_eq!(quantile_rank(50577, 0.02), 1011);
        assert_eq!(quantile_rank(50577, 0.98), 49565);
        assert_eq!((quantile_rank(1, 0.5), quantile_rank(0, 0.5), quantile_rank(7, 0.0), quantile_rank(7, 1.0)), (0, 0, 0, 6));
    }
}
