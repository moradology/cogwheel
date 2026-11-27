//! `numpy.histogram` as rio-tiler's `get_array_statistics` calls it, shared by every
//! statistics route so their bin edges and bin counts agree with TiTiler's.
//!
//! - Equal bins: the edges are `numpy.linspace(first, last, n + 1)`, i.e. `k * ((last - first) / n)
//!   + first` with the last edge set to `last` exactly. `first, last` are the request's `range`, or
//!   else the data's least and greatest values, widened to `first - 0.5 .. last + 0.5` when they are
//!   equal (and `0 .. 1` when there is no data).
//! - numpy works the edges in the data's type when that is float32 (`linspace(..., dtype=float32)`)
//!   and in float64 otherwise. Between the data's own float32 least and greatest values the step and
//!   every edge are float32 arithmetic; between a range given in the request (Python floats) the
//!   edges are worked in float64 and cast to float32.
//! - A value lands in bin `i` when `edges[i] <= v < edges[i + 1]`, the last bin also holding a value
//!   equal to its upper edge; values outside `edges[0] ..= edges[n]` are dropped. This is what
//!   numpy's index estimate settles to after its correction against the edges, and it is the rule
//!   for custom edges (`bins` given as a list), where `range` is ignored.

use std::ops::Range;

/// The type numpy builds equal-width bin edges in: float32 for float32 data, float64 for any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistDtype {
    Float32,
    Float64,
}

impl HistDtype {
    /// The edge type numpy uses for a band read with this sample type. rio-tiler casts a band to
    /// float32 when it applies scales and offsets, so an unscaled band of any type is float32.
    pub fn for_sample_type(st: cogzilla_tiff::SampleType, unscale: bool) -> Self {
        if unscale || st == cogzilla_tiff::SampleType::Float32 {
            Self::Float32
        } else {
            Self::Float64
        }
    }

    /// The edge type for an expression's values: numexpr keeps float32 bands in float32 unless a
    /// float literal (a float64 constant) promotes the result; any other bands give float64.
    pub fn for_expression(expr: &str, bands: Self) -> Self {
        if bands == Self::Float32 && !has_float_literal(expr) {
            Self::Float32
        } else {
            Self::Float64
        }
    }

    /// The edge type for values whose source type is not known: whole numbers are taken to be an
    /// integer band (float64 edges), anything else a float32 band.
    pub fn infer(values: impl IntoIterator<Item = f32>) -> Self {
        if values.into_iter().all(|v| !v.is_finite() || v.fract() == 0.0) {
            Self::Float64
        } else {
            Self::Float32
        }
    }
}

/// Whether `expr` holds a number written as a float (`0.5`, `.5`, `2.`, `1e3`), not counting the
/// digits of names such as `b1`.
fn has_float_literal(expr: &str) -> bool {
    let b = expr.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            // A name: skip it with its digits.
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            continue;
        }
        if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' {
                return true;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    return true;
                }
            }
            continue;
        }
        i += 1;
    }
    false
}

/// The bin edges `numpy.histogram` makes. `custom` edges (two or more) are returned as given;
/// otherwise `bins` equal bins span `range`, or the data's `min ..= max` (`None` when there is no
/// data). `min`/`max` of a float32 band must be its float32 values.
pub fn bin_edges(
    bins: usize,
    custom: Option<&[f64]>,
    range: Option<(f64, f64)>,
    data_min_max: Option<(f64, f64)>,
    dtype: HistDtype,
) -> Vec<f64> {
    if let Some(edges) = custom
        && edges.len() >= 2
    {
        return edges.to_vec();
    }
    let n = bins.max(1);
    match (range, data_min_max, dtype) {
        // The data's own least and greatest float32 values: numpy's `linspace` runs in float32.
        (None, Some((lo, hi)), HistDtype::Float32) => {
            let (mut first, mut last) = (lo as f32, hi as f32);
            if first == last {
                first -= 0.5;
                last += 0.5;
            }
            let step = (last - first) / n as f32;
            let mut edges: Vec<f64> = (0..=n).map(|k| f64::from(k as f32 * step + first)).collect();
            edges[n] = f64::from(last);
            edges
        }
        _ => {
            // A range from the request (Python floats), integer data, or float64 data: `linspace`
            // runs in float64 and, for float32 data, each edge is cast to float32.
            let (mut first, mut last) = range.or(data_min_max).unwrap_or((0.0, 1.0));
            if first == last {
                first -= 0.5;
                last += 0.5;
            }
            let step = (last - first) / n as f64;
            let mut edges: Vec<f64> = if step.is_finite() {
                (0..=n).map(|k| k as f64 * step + first).collect()
            } else {
                // A span beyond `f64::MAX` (numpy gives up here): interpolate instead.
                (0..=n)
                    .map(|k| {
                        let t = k as f64 / n as f64;
                        first * (1.0 - t) + last * t
                    })
                    .collect()
            };
            edges[n] = last;
            if dtype == HistDtype::Float32 {
                for e in &mut edges {
                    *e = f64::from(*e as f32);
                }
            }
            edges
        }
    }
}

/// The bin of `v` among `edges` (two or more, increasing), or `None` when `v` is outside them.
#[inline]
pub fn bin_of(edges: &[f64], v: f64) -> Option<usize> {
    let n = edges.len().checked_sub(1).filter(|&n| n > 0)?;
    if !(v >= edges[0] && v <= edges[n]) {
        return None;
    }
    Some(edges[1..n].partition_point(|&e| e <= v))
}

/// [`bin_of`] for values that arrive in ascending order: a pointer walks up the edges instead of
/// searching them for each value.
pub struct AscendingBins<'a> {
    edges: &'a [f64],
    bin: usize,
}

impl<'a> AscendingBins<'a> {
    pub fn new(edges: &'a [f64]) -> Self {
        Self { edges, bin: 0 }
    }

    /// The bin of `v`, which is not below any value given before.
    #[inline]
    pub fn bin_of(&mut self, v: f64) -> Option<usize> {
        let n = self.edges.len().checked_sub(1).filter(|&n| n > 0)?;
        if !(v >= self.edges[0] && v <= self.edges[n]) {
            return None;
        }
        while self.bin + 1 < n && v >= self.edges[self.bin + 1] {
            self.bin += 1;
        }
        Some(self.bin)
    }
}

/// For values sorted ascending, the index range of the values in each bin of `edges`.
pub fn sorted_bin_ranges<T>(sorted: &[T], edges: &[f64], value: impl Fn(&T) -> f64) -> Vec<Range<usize>> {
    let n = edges.len().saturating_sub(1);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut start = sorted.partition_point(|t| value(t) < edges[0]);
    for bin in 0..n {
        let hi = edges[bin + 1];
        let rest = &sorted[start..];
        let end = start
            + if bin + 1 < n {
                rest.partition_point(|t| value(t) < hi)
            } else {
                rest.partition_point(|t| value(t) <= hi)
            };
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn np_counts(values: &[f64], edges: &[f64]) -> Vec<u64> {
        let mut counts = vec![0u64; edges.len() - 1];
        for &v in values {
            if let Some(b) = bin_of(edges, v) {
                counts[b] += 1;
            }
        }
        counts
    }

    #[test]
    fn float64_edges_follow_linspace() {
        // numpy.histogram([7, 255], bins=10)[1]
        let e = bin_edges(10, None, None, Some((7.0, 255.0)), HistDtype::Float64);
        assert_eq!(e[3], 81.4);
        assert_eq!(e[6], 155.8);
        assert_eq!(e[9], 230.20000000000002);
        assert_eq!(e[10], 255.0);
        // numpy.histogram(np.array([-8399, 8399], np.int16), bins=10)[1]
        let e = bin_edges(10, None, None, Some((-8399.0, 8399.0)), HistDtype::Float64);
        assert_eq!(e[4], -1679.8000000000002);
        assert_eq!(e[7], 3359.6000000000004);
        assert_eq!(e[9], 6719.199999999999);
    }

    #[test]
    fn float32_edges_between_data_bounds() {
        // numpy.histogram(np.array([80.00247, 919.9995], np.float32), bins=10)[1]
        let lo = f64::from(80.002_47_f32);
        let hi = f64::from(919.999_5_f32);
        let e = bin_edges(10, None, None, Some((lo, hi)), HistDtype::Float32);
        assert_eq!(e[0], 80.00247192382812);
        assert_eq!(e[1], 164.00218200683594);
        assert_eq!(e[2], 248.00189208984375);
        assert_eq!(e[10], 919.99951171875);
    }

    #[test]
    fn float32_edges_from_a_requested_range_are_cast() {
        // numpy.histogram(np.array([1, 2, 3], np.float32), bins=3, range=(0.1, 3.0))[1]
        let e = bin_edges(3, None, Some((0.1, 3.0)), Some((1.0, 3.0)), HistDtype::Float32);
        assert_eq!(e, vec![f64::from(0.1_f32), f64::from(1.066_666_7_f32), f64::from(2.033_333_3_f32), 3.0]);
    }

    #[test]
    fn one_value_widens_by_half() {
        let e = bin_edges(10, None, None, Some((7.0, 7.0)), HistDtype::Float64);
        assert_eq!(e[0], 6.5);
        assert_eq!(e[5], 7.0);
        assert_eq!(e[10], 7.5);
        assert_eq!(np_counts(&[7.0, 7.0], &e), vec![0, 0, 0, 0, 0, 2, 0, 0, 0, 0]);
        // numpy.histogram(np.full(4, 1037.8, np.float32))[1]
        let v = f64::from(1037.8_f32);
        let e = bin_edges(10, None, None, Some((v, v)), HistDtype::Float32);
        assert_eq!(e[0], 1037.300048828125);
        assert_eq!(e[1], 1037.4000244140625);
        assert_eq!(e[5], 1037.800048828125);
        assert_eq!(e[10], 1038.300048828125);
        assert_eq!(np_counts(&[v], &e)[5], 1);
        // No data: numpy's range is 0 .. 1.
        let e = bin_edges(10, None, None, None, HistDtype::Float64);
        assert_eq!(e[3], 0.30000000000000004);
        assert_eq!(e[10], 1.0);
    }

    #[test]
    fn values_on_edges_and_outside() {
        let e = [0.0, 10.0, 20.0];
        assert_eq!(bin_of(&e, -0.1), None);
        assert_eq!(bin_of(&e, 0.0), Some(0));
        assert_eq!(bin_of(&e, 10.0), Some(1));
        assert_eq!(bin_of(&e, 20.0), Some(1));
        assert_eq!(bin_of(&e, 20.000001), None);
        assert_eq!(bin_of(&e, f64::NAN), None);
        // numpy.histogram(np.float32(1037.8), bins=[0, 1037.8]): float32 1037.8 is above the edge.
        assert_eq!(bin_of(&[0.0, 1037.8], f64::from(1037.8_f32)), None);
    }

    #[test]
    fn sorted_ranges_match_bin_of() {
        let vals: Vec<f64> = (0..=40).map(|i| i as f64 * 0.5).collect();
        for edges in [vec![0.0, 10.0, 20.0], vec![2.0, 2.5, 7.0, 7.5], vec![-1.0, 25.0]] {
            let ranges = sorted_bin_ranges(&vals, &edges, |v| *v);
            let counts: Vec<u64> = ranges.iter().map(|r| r.len() as u64).collect();
            assert_eq!(counts, np_counts(&vals, &edges), "{edges:?}");
            let mut walk = AscendingBins::new(&edges);
            for &v in &vals {
                assert_eq!(walk.bin_of(v), bin_of(&edges, v), "{v} in {edges:?}");
            }
        }
    }

    #[test]
    fn expression_types() {
        let f32b = HistDtype::Float32;
        assert_eq!(HistDtype::for_expression("b1*2", f32b), HistDtype::Float32);
        assert_eq!(HistDtype::for_expression("(b1-b2)/(b1+b2)", f32b), HistDtype::Float32);
        assert_eq!(HistDtype::for_expression("b1/3", f32b), HistDtype::Float32);
        assert_eq!(HistDtype::for_expression("b1+0.5", f32b), HistDtype::Float64);
        assert_eq!(HistDtype::for_expression("b1*.5", f32b), HistDtype::Float64);
        assert_eq!(HistDtype::for_expression("b1*2.", f32b), HistDtype::Float64);
        assert_eq!(HistDtype::for_expression("b1*1e3", f32b), HistDtype::Float64);
        assert_eq!(HistDtype::for_expression("b12*b1e", f32b), HistDtype::Float32);
        assert_eq!(HistDtype::for_expression("b1*2", HistDtype::Float64), HistDtype::Float64);
    }

    #[test]
    fn dtype_inference() {
        assert_eq!(HistDtype::infer([1.0, 2.0, f32::NAN]), HistDtype::Float64);
        assert_eq!(HistDtype::infer([1.0, 2.5]), HistDtype::Float32);
        assert_eq!(
            HistDtype::for_sample_type(cogzilla_tiff::SampleType::Uint16, true),
            HistDtype::Float32
        );
        assert_eq!(
            HistDtype::for_sample_type(cogzilla_tiff::SampleType::Int16, false),
            HistDtype::Float64
        );
    }
}
