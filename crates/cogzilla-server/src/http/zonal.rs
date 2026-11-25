//! Statistics of the pixels under a polygon, the way rio-tiler computes them: a polygon
//! burned onto the pixel grid by GDAL's all-touched rule, a coverage fraction per pixel
//! from the same polygon burned on a finer grid, and coverage-weighted statistics.

use serde_json::{Value, json};

/// A ring in pixel coordinates: `x` to the right, `y` down, one unit per pixel.
pub(super) type Ring = Vec<(f64, f64)>;

/// For every pixel of a `width` x `height` grid, how many of its `scale` x `scale`
/// sub-cells the polygon touches. With `scale` 1 this is rasterio's
/// `rasterize(all_touched=True)`; a sub-cell is touched when its centre is inside the
/// polygon (even-odd over its rings) or a ring passes through it. Each polygon is burned
/// on its own, as GDAL burns them, so where two overlap a sub-cell is counted once.
///
/// The polygon and the scale come from a request, so the work they ask for is counted
/// first and refused when it is more than a statistics request has any use for.
pub(super) fn cover_counts(polygons: &[Vec<Ring>], width: usize, height: usize, scale: usize) -> Result<Vec<u32>, String> {
    let scale = scale.max(1);
    let (fw, fh) = (width * scale, height * scale);
    let mut counts = vec![0u32; width * height];
    if fw == 0 || fh == 0 {
        return Ok(counts);
    }
    // Every row of the fine grid looks at every segment, and a segment burns a cell for
    // every row and column it crosses inside the grid.
    let too_much = || "the polygon has too many vertices for this size and cover_scale; simplify it, or lower cover_scale".to_string();
    let segments: u64 = polygons.iter().flatten().map(|ring| ring.len() as u64).sum();
    if segments.saturating_mul(fh as u64) > MAX_ROW_CHECKS {
        return Err(too_much());
    }
    let s = scale as f64;
    let mut burned = 0u64;
    for ring in polygons.iter().flatten() {
        let closing = ring.first().zip(ring.last()).filter(|(a, b)| a != b).map(|(a, b)| [*b, *a]);
        for pair in ring.windows(2).map(|p| [p[0], p[1]]).chain(closing) {
            let across = (pair[0].0 * s).clamp(0.0, fw as f64) - (pair[1].0 * s).clamp(0.0, fw as f64);
            let down = (pair[0].1 * s).clamp(0.0, fh as f64) - (pair[1].1 * s).clamp(0.0, fh as f64);
            // Not a number clamps to itself and counts as nothing; such a segment burns nothing.
            let cells = across.abs() + down.abs() + 2.0;
            burned = burned.saturating_add(if cells.is_finite() { cells as u64 } else { 0 });
        }
    }
    if burned > MAX_EDGE_CELLS {
        return Err(too_much());
    }
    let fine: Vec<Vec<Ring>> = polygons
        .iter()
        .map(|rings| {
            rings
                .iter()
                .map(|ring| {
                    let mut ring: Ring = ring.iter().map(|&(x, y)| (x * s, y * s)).collect();
                    // Rings are closed for the edge walk whether or not the input repeats the start.
                    if ring.len() > 1 && ring.first() != ring.last() {
                        ring.push(ring[0]);
                    }
                    ring
                })
                .filter(|ring| ring.len() >= 2)
                .collect()
        })
        .collect();

    // Cells the rings pass through, as (row, column).
    let mut edge_cells: Vec<(u32, u32)> = Vec::new();
    for ring in fine.iter().flatten() {
        for pair in ring.windows(2) {
            burn_line(pair[0], pair[1], fw as i64, fh as i64, &mut edge_cells);
        }
    }
    edge_cells.sort_unstable();
    edge_cells.dedup();

    let mut next_edge = 0usize;
    let mut crossings: Vec<i64> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for row in 0..fh {
        // Cells whose centre is inside: GDAL's scanline fill at the row's centre line.
        let dy = row as f64 + 0.5;
        spans.clear();
        for rings in &fine {
            crossings.clear();
            for ring in rings {
                for pair in ring.windows(2) {
                    let ((mut x1, mut y1), (mut x2, mut y2)) = (pair[0], pair[1]);
                    if (y1 < dy && y2 < dy) || (y1 > dy && y2 > dy) || y1 == y2 {
                        continue;
                    }
                    if y1 > y2 {
                        std::mem::swap(&mut x1, &mut x2);
                        std::mem::swap(&mut y1, &mut y2);
                    }
                    if dy < y2 && dy >= y1 {
                        let x = (dy - y1) * (x2 - x1) / (y2 - y1) + x1;
                        crossings.push((x + 0.5).floor() as i64);
                    }
                }
            }
            crossings.sort_unstable();
            for pair in crossings.chunks_exact(2) {
                let (a, b) = (pair[0].max(0), (pair[1] - 1).min(fw as i64 - 1));
                if a <= b {
                    spans.push((a as usize, b as usize));
                }
            }
        }
        // In order and apart: the spans of polygons that overlap are run together.
        if fine.len() > 1 {
            spans.sort_unstable();
            let mut kept = 0;
            for i in 1..spans.len() {
                if spans[i].0 <= spans[kept].1 {
                    spans[kept].1 = spans[kept].1.max(spans[i].1);
                } else {
                    kept += 1;
                    spans[kept] = spans[i];
                }
            }
            spans.truncate((kept + 1).min(spans.len()));
        }
        let out_row = (row / scale) * width;
        for &(a, b) in &spans {
            for cx in a / scale..=b / scale {
                let lo = a.max(cx * scale);
                let hi = b.min(cx * scale + scale - 1);
                counts[out_row + cx] += (hi - lo + 1) as u32;
            }
        }
        // Cells a ring passes through that the fill did not already take.
        while next_edge < edge_cells.len() && edge_cells[next_edge].0 as usize == row {
            let x = edge_cells[next_edge].1 as usize;
            // The spans are in order and apart, so the one that could hold `x` is found, not looked for.
            let at = spans.partition_point(|&(_, b)| b < x);
            if spans.get(at).is_none_or(|&(a, _)| x < a) {
                counts[out_row + x / scale] += 1;
            }
            next_edge += 1;
        }
    }
    Ok(counts)
}

/// How many cells the rings of one polygon may burn: eight bytes each before they are sorted.
const MAX_EDGE_CELLS: u64 = 16_000_000;

/// How many times a row of the fine grid may look at a segment. The largest polygon a
/// request may carry, at the default size and a `cover_scale` of ten, is within it.
const MAX_ROW_CHECKS: u64 = 1_000_000_000;

/// Every cell a segment passes through, as GDAL's all-touched line burner walks it.
fn burn_line(from: (f64, f64), to: (f64, f64), width: i64, height: i64, out: &mut Vec<(u32, u32)>) {
    let ((mut x, mut y), (mut x_end, mut y_end)) = (from, to);
    let (w, h) = (width as f64, height as f64);
    if (y < 0.0 && y_end < 0.0) || (y > h && y_end > h) || (x < 0.0 && x_end < 0.0) || (x > w && x_end > w)
    {
        return;
    }
    if x > x_end {
        std::mem::swap(&mut x, &mut x_end);
        std::mem::swap(&mut y, &mut y_end);
    }
    // Vertical, or close to it.
    if x.floor() == x_end.floor() || (x - x_end).abs() < 0.01 {
        if y_end < y {
            std::mem::swap(&mut y, &mut y_end);
        }
        let col = x_end.floor() as i64;
        if col < 0 || col >= width {
            return;
        }
        let (first, last) = ((y.floor() as i64).max(0), (y_end.floor() as i64).min(height - 1));
        out.extend((first..=last).map(|row| (row as u32, col as u32)));
        return;
    }
    // Horizontal, or close to it.
    if y.floor() == y_end.floor() || (y - y_end).abs() < 0.01 {
        let row = y.floor() as i64;
        if row < 0 || row >= height {
            return;
        }
        let (first, last) = ((x.floor() as i64).max(0), (x_end.floor() as i64).min(width - 1));
        out.extend((first..=last).map(|col| (row as u32, col as u32)));
        return;
    }
    let slope = (y_end - y) / (x_end - x);
    // Clip to the grid.
    if x_end > w {
        y_end -= (x_end - w) * slope;
        x_end = w;
    }
    if x < 0.0 {
        y += (0.0 - x) * slope;
        x = 0.0;
    }
    if y_end > y {
        if y < 0.0 {
            x += (0.0 - y) / slope;
            y = 0.0;
        }
        if y_end >= h {
            x_end += (y_end - h) / slope;
        }
    } else {
        if y >= h {
            x += (h - y) / slope;
            y = h;
        }
        if y_end < 0.0 {
            x_end -= y_end / slope;
        }
    }
    // Step from cell to cell. Every step enters a new column or a new row, so the walk is
    // bounded by the size of the grid; the budget only stops a walk that rounding stalls.
    let mut budget = 4 * (width + height) + 64;
    while x >= 0.0 && x < x_end && budget > 0 {
        budget -= 1;
        let (col, row) = (x.floor() as i64, y.floor() as i64);
        if row >= 0 && row < height && col < width {
            out.push((row as u32, col as u32));
        }
        let mut step_x = (x + 1.0).floor() - x;
        let mut step_y = step_x * slope;
        if (y + step_y).floor() as i64 == row {
            // The next column, same row.
        } else if slope < 0.0 {
            step_y = (row as f64 - y).min(-0.000_000_001);
            step_x = step_y / slope;
        } else {
            step_y = ((row + 1) as f64 - y).max(0.000_000_001);
            step_x = step_y / slope;
        }
        x += step_x;
        y += step_y;
    }
}

/// The bins of a histogram: numpy's `bins` argument.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Bins {
    Count(usize),
    Edges(Vec<f64>),
}

/// TiTiler's statistics and histogram options.
#[derive(Debug, Clone)]
pub(super) struct StatsOptions {
    pub categorical: bool,
    pub categories: Vec<f64>,
    pub percentiles: Vec<i64>,
    pub bins: Bins,
    pub range: Option<(f64, f64)>,
}

impl Default for StatsOptions {
    fn default() -> Self {
        Self {
            categorical: false,
            categories: Vec::new(),
            percentiles: vec![2, 98],
            bins: Bins::Count(10),
            range: None,
        }
    }
}

/// numpy's `histogram(values, bins, range)` for sorted `values`: counts and edges.
fn histogram(sorted: &[f64], bins: &Bins, range: Option<(f64, f64)>) -> (Vec<u64>, Vec<f64>) {
    let edges: Vec<f64> = match bins {
        Bins::Edges(edges) => edges.clone(),
        Bins::Count(n) => {
            let n = (*n).max(1);
            let (mut first, mut last) = match (range, sorted.first(), sorted.last()) {
                (Some(r), _, _) => r,
                (None, Some(a), Some(b)) => (*a, *b),
                _ => (0.0, 1.0),
            };
            if first == last {
                first -= 0.5;
                last += 0.5;
            }
            // numpy.linspace: a multiple of the step from the start, and the end exactly.
            let step = (last - first) / n as f64;
            let mut edges: Vec<f64> = (0..=n).map(|i| i as f64 * step + first).collect();
            edges[n] = last;
            edges
        }
    };
    let n = edges.len().saturating_sub(1);
    let mut counts = vec![0u64; n];
    if n == 0 {
        return (counts, edges);
    }
    let (first, last) = (edges[0], edges[n]);
    for &v in sorted {
        if v < first || v > last {
            continue;
        }
        // The bin whose edges hold the value; the last bin includes its upper edge.
        let bin = edges[1..n].partition_point(|e| *e <= v);
        counts[bin] += 1;
    }
    (counts, edges)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// One band's statistics in TiTiler's shape. `values` are the band's pixels, a pixel that
/// is not finite being masked; `coverage` is the fraction of each pixel under the polygon.
pub(super) fn band_statistics(
    values: &[f64],
    coverage: &[f32],
    description: &str,
    opts: &StatsOptions,
) -> Value {
    let mut pairs: Vec<(f64, f32)> = values
        .iter()
        .zip(coverage)
        .filter(|(v, _)| v.is_finite())
        .map(|(v, w)| (*v, *w))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let sorted: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let valid = pairs.len();
    let covered = coverage.iter().filter(|c| **c != 0.0).count();
    let valid_percent = if covered == 0 {
        0.0
    } else {
        round2((valid as f64 / covered as f64).min(1.0) * 100.0)
    };

    // Distinct values and how often each occurs.
    let mut keys: Vec<f64> = Vec::new();
    let mut key_counts: Vec<u64> = Vec::new();
    for &v in &sorted {
        if keys.last() == Some(&v) {
            *key_counts.last_mut().unwrap_or(&mut 0) += 1;
        } else {
            keys.push(v);
            key_counts.push(1);
        }
    }

    let integral = sorted.iter().all(|v| v.fract() == 0.0);
    // A whole number only where it is one and an `i64` holds it: a category of 2.5 asked of
    // whole-numbered data keeps its name.
    let number = |v: f64| if integral && v.fract() == 0.0 && v.abs() < 9.0e15 { json!(v as i64) } else { json!(v) };
    let histogram = if opts.categorical {
        let wanted = if opts.categories.is_empty() { keys.clone() } else { opts.categories.clone() };
        let counts: Vec<u64> = wanted
            .iter()
            .map(|c| keys.binary_search_by(|k| k.total_cmp(c)).map_or(0, |i| key_counts[i]))
            .collect();
        json!([counts, wanted.iter().map(|c| number(*c)).collect::<Vec<_>>()])
    } else {
        let (counts, edges) = histogram(&sorted, &opts.bins, opts.range);
        json!([counts, edges])
    };

    let mut out = serde_json::Map::new();
    let mut put = |key: &str, value: Value| {
        out.insert(key.to_string(), value);
    };
    let percentile_names: Vec<String> =
        opts.percentiles.iter().map(|p| format!("percentile_{p}")).collect();
    if valid == 0 {
        for key in ["min", "max", "mean"] {
            put(key, Value::Null);
        }
        put("count", json!(0.0));
        put("sum", json!(0.0));
        for key in ["std", "median", "majority", "minority"] {
            put(key, Value::Null);
        }
        put("unique", json!(0.0));
        put("histogram", histogram);
        put("valid_percent", json!(valid_percent));
        put("masked_pixels", json!(values.len() as f64));
        put("valid_pixels", json!(0.0));
        put("description", json!(description));
        for name in &percentile_names {
            put(name, Value::Null);
        }
        return Value::Object(out);
    }

    let count: f64 = pairs.iter().map(|p| f64::from(p.1)).sum();
    let sum: f64 = pairs.iter().map(|p| p.0 * f64::from(p.1)).sum();
    let mean = sum / count;
    let variance = pairs.iter().map(|p| f64::from(p.1) * (p.0 - mean).powi(2)).sum::<f64>() / count;

    // Weighted quantiles: the first value at which the running weight reaches the share.
    // The running weight is kept in single precision, as rio-tiler's coverage array is.
    let mut running = Vec::with_capacity(valid);
    let mut acc = 0f32;
    for p in &pairs {
        acc += p.1;
        running.push(acc);
    }
    let total = f64::from(acc);
    let quantile = |q: f64| -> f64 {
        let target = q * total;
        let at = running.partition_point(|c| f64::from(*c) < target).min(valid - 1);
        sorted[at]
    };

    let most = key_counts.iter().copied().max().unwrap_or(0);
    let least = key_counts.iter().copied().min().unwrap_or(0);
    let first_with = |n: u64| keys[key_counts.iter().position(|c| *c == n).unwrap_or(0)];

    put("min", json!(sorted[0]));
    put("max", json!(sorted[valid - 1]));
    put("mean", json!(mean));
    put("count", json!(count));
    put("sum", json!(sum));
    put("std", json!(variance.sqrt()));
    put("median", json!(quantile(0.5)));
    put("majority", json!(first_with(most)));
    put("minority", json!(first_with(least)));
    put("unique", json!(keys.len() as f64));
    put("histogram", histogram);
    put("valid_percent", json!(valid_percent));
    put("masked_pixels", json!((values.len() - valid) as f64));
    put("valid_pixels", json!(valid as f64));
    put("description", json!(description));
    for (name, p) in percentile_names.iter().zip(&opts.percentiles) {
        put(name, json!(quantile(*p as f64 / 100.0)));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(counts: &[u32], width: usize) -> Vec<String> {
        counts
            .chunks(width)
            .map(|row| row.iter().map(|c| if *c > 0 { '#' } else { '.' }).collect())
            .collect()
    }

    #[test]
    fn all_touched_takes_centres_and_edges() {
        // A square from (1.2, 1.2) to (3.6, 3.6): centres of cells 2 and 3 are inside
        // (1.5 is too), and the outline also touches row and column 1.
        let square = vec![vec![(1.2, 1.2), (3.6, 1.2), (3.6, 3.6), (1.2, 3.6), (1.2, 1.2)]];
        let mask = cover_counts(std::slice::from_ref(&square), 5, 5, 1).unwrap();
        assert_eq!(grid(&mask, 5), [".....", ".###.", ".###.", ".###.", "....."]);

        // A thin sliver holds no cell centre and is still burned along its length.
        let sliver = vec![vec![(0.1, 0.1), (3.9, 2.2), (3.9, 2.3), (0.1, 0.2), (0.1, 0.1)]];
        let mask = cover_counts(std::slice::from_ref(&sliver), 4, 3, 1).unwrap();
        assert_eq!(grid(&mask, 4), ["##..", ".###", "...#"]);

        // An unclosed ring is closed, and a hole is left out (even-odd).
        let holed = vec![
            vec![(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)],
            vec![(1.6, 1.6), (4.4, 1.6), (4.4, 4.4), (1.6, 4.4), (1.6, 1.6)],
        ];
        let mask = cover_counts(std::slice::from_ref(&holed), 6, 6, 1).unwrap();
        assert_eq!(grid(&mask, 6), ["######", "######", "##..##", "##..##", "######", "######"]);
    }

    #[test]
    fn coverage_is_the_share_of_sub_cells() {
        // The left half of a 2x1 grid, plus a tenth of the right pixel.
        let ring = vec![vec![(0.0, 0.0), (1.1, 0.0), (1.1, 1.0), (0.0, 1.0), (0.0, 0.0)]];
        let counts = cover_counts(std::slice::from_ref(&ring), 2, 1, 10).unwrap();
        assert_eq!(counts[0], 100);
        // Sub-columns 10 (centre inside) and 11 (the edge at 11.0 runs along its left side).
        assert_eq!(counts[1], 20);
        // Vertices far outside the grid: the walk is clipped to it and still ends.
        let huge = vec![vec![(-1.0e9, -1.0e9), (2.0e9, -1.0e9), (0.0, 2.0e9), (-1.0e9, -1.0e9)]];
        assert_eq!(cover_counts(std::slice::from_ref(&huge), 2, 1, 10).unwrap(), [100, 100]);
        // A polygon away from the grid touches nothing.
        let far = vec![vec![(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 10.0)]];
        assert!(cover_counts(std::slice::from_ref(&far), 2, 1, 10).unwrap().iter().all(|c| *c == 0));
        // A polygon that asks for more work than statistics have a use for is refused.
        let comb: Vec<Ring> = (0..4096).map(|i| vec![(0.0, i as f64), (64.0, i as f64), (64.0, i as f64 + 0.5), (0.0, i as f64)]).collect();
        assert!(cover_counts(std::slice::from_ref(&comb), 64, 64, 1).is_ok());
        assert!(cover_counts(std::slice::from_ref(&comb), 64, 64, 1000).is_err());
    }

    #[test]
    fn statistics_follow_rio_tiler() {
        // numpy: values 1..=10 with two masked, weights all one.
        let values: Vec<f64> =
            vec![1.0, 2.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, f64::NAN, 10.0];
        let coverage = vec![1f32; 10];
        let s = band_statistics(&values, &coverage, "red_b1", &StatsOptions::default());
        assert_eq!(s["min"], 1.0);
        assert_eq!(s["max"], 10.0);
        assert_eq!(s["mean"], 4.5);
        assert_eq!(s["count"], 8.0);
        assert_eq!(s["sum"], 36.0);
        assert_eq!(s["median"], 3.0);
        assert_eq!(s["majority"], 2.0);
        assert_eq!(s["minority"], 1.0);
        assert_eq!(s["unique"], 7.0);
        assert_eq!(s["valid_pixels"], 8.0);
        assert_eq!(s["masked_pixels"], 2.0);
        assert_eq!(s["valid_percent"], 80.0);
        assert_eq!(s["description"], "red_b1");
        assert_eq!(s["percentile_2"], 1.0);
        assert_eq!(s["percentile_98"], 10.0);
        // numpy.histogram(bins=10): edges 1, 1.9, ..., 10 and the last bin closed.
        assert_eq!(s["histogram"][0], json!([1, 2, 1, 0, 1, 1, 1, 0, 0, 1]));
        assert_eq!(s["histogram"][1][1], 1.9);
        assert_eq!(s["histogram"][1][10], 10.0);
        let std = s["std"].as_f64().unwrap();
        assert!((std - 8.25f64.sqrt()).abs() < 1e-12, "{std}");

        // Weights move the mean and the median.
        let s = band_statistics(&[1.0, 3.0], &[0.25, 0.75], "x", &StatsOptions::default());
        assert_eq!(s["mean"], 2.5);
        assert_eq!(s["count"], 1.0);
        assert_eq!(s["median"], 3.0);

        // Categories, bin edges and a range.
        let opts = StatsOptions { categorical: true, categories: vec![2.0, 4.0], ..Default::default() };
        let s = band_statistics(&values, &coverage, "x", &opts);
        assert_eq!(s["histogram"], json!([[2, 0], [2, 4]]));
        let opts = StatsOptions { bins: Bins::Edges(vec![0.0, 2.0, 10.0]), ..Default::default() };
        assert_eq!(band_statistics(&values, &coverage, "x", &opts)["histogram"][0], json!([1, 7]));
        let opts = StatsOptions { bins: Bins::Count(2), range: Some((0.0, 4.0)), percentiles: vec![50], ..Default::default() };
        let s = band_statistics(&values, &coverage, "x", &opts);
        assert_eq!(s["histogram"], json!([[1, 3], [0.0, 2.0, 4.0]]));
        assert_eq!(s["percentile_50"], 3.0);
        assert!(s.get("percentile_2").is_none());

        // Nothing valid.
        let s = band_statistics(&[f64::NAN], &[1.0], "x", &StatsOptions::default());
        assert_eq!(s["min"], Value::Null);
        assert_eq!(s["valid_pixels"], 0.0);
        assert_eq!(s["histogram"][0], json!([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    }
    #[test]
    fn polygons_that_overlap_are_counted_once() {
        let square = |x: f64| vec![(x, 0.0), (x + 2.0, 0.0), (x + 2.0, 2.0), (x, 2.0), (x, 0.0)];
        // Half of column 0, then columns 1 to 3, the last by the edge that runs along it:
        // the overlap is in both squares and counted once.
        let both = [vec![square(0.5)], vec![square(1.5)]];
        let counts = cover_counts(&both, 5, 2, 2).unwrap();
        assert_eq!(counts, [2, 4, 4, 4, 0, 2, 4, 4, 4, 0]);
        // As one polygon of two rings the overlap is a hole.
        let one = [vec![square(0.5), square(1.5)]];
        assert_ne!(cover_counts(&one, 5, 2, 2).unwrap(), counts);
    }

}
