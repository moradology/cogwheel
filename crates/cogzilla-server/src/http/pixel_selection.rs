//! rio-tiler's pixel selection methods, other than `first`.
//!
//! A mosaic is made from every scene's values for the tile. `first` needs no arithmetic
//! and stops when the tile is full, so it stays on the renderer's fast path; the methods
//! here read all the scenes and decide each value of each band on its own, as rio-tiler
//! does on masked arrays (`rio_tiler/mosaic/methods/defaults.py`):
//!
//! - `highest` and `lowest` keep the largest or smallest valid value;
//! - `mean`, `median` and `stdev` are numpy's over the valid values, the first two cast
//!   back to the source's type, which for integers drops the fraction;
//! - `count` is how many scenes had a value, as one band that is valid everywhere.
//!
//! A value is not valid when the scene has no data there, which the renderer writes as NaN.

/// A pixel selection method that is not `first`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Selection {
    Highest,
    Lowest,
    Mean,
    Median,
    Stdev,
    Count,
}

impl Selection {
    /// The methods by TiTiler's names. `lastbandlow` and `lastbandhigh` are not here yet.
    pub(super) fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "highest" => Self::Highest,
            "lowest" => Self::Lowest,
            "mean" => Self::Mean,
            "median" => Self::Median,
            "stdev" => Self::Stdev,
            "count" => Self::Count,
            _ => return None,
        })
    }
}

/// A value a selection is worked out over: `f32` for pictures, `f64` for statistics.
pub(super) trait Sample: Copy + PartialOrd {
    const NAN: Self;
    const ZERO: Self;
    const ONE: Self;
    fn is_nan(self) -> bool;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn total_order(a: &Self, b: &Self) -> std::cmp::Ordering;
    fn add(self, other: Self) -> Self;
}

macro_rules! sample {
    ($t:ty) => {
        impl Sample for $t {
            const NAN: Self = <$t>::NAN;
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn total_order(a: &Self, b: &Self) -> std::cmp::Ordering {
                a.total_cmp(b)
            }
            fn add(self, other: Self) -> Self {
                self + other
            }
        }
    };
}
sample!(f32);
sample!(f64);

/// Band-major values, NaN where no scene had one.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Selected<T = f32> {
    pub bands: usize,
    pub values: Vec<T>,
}

/// A pixel selection being worked out, one scene at a time. Only `median` has to keep the
/// scenes; the others keep a running figure for each value.
pub(super) struct Fold<T = f32> {
    method: Selection,
    bands: usize,
    pixels: usize,
    integer: bool,
    scenes: usize,
    state: State<T>,
}

enum State<T> {
    /// `highest` and `lowest`: the value kept so far.
    Extreme(Vec<T>),
    /// `mean` and `stdev`: the sum, the sum of squares and how many values went into them.
    Moments { sum: Vec<f64>, squares: Vec<f64>, count: Vec<u32> },
    /// `median`: every scene's values, scene after scene.
    All(Vec<T>),
    /// `count`: how many scenes had a value in the first band.
    Count(Vec<T>),
}

impl Fold<f32> {
    /// Take one scene as the renderer writes it: little-endian `f32`, band after band.
    pub(super) fn add_le_bytes(&mut self, scene: &[u8]) {
        debug_assert_eq!(scene.len(), self.bands * self.pixels * 4);
        self.add(scene.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])));
    }
}

impl<T: Sample> Fold<T> {
    /// `integer` says the source holds whole numbers, which `mean` and `median` are cast
    /// back to.
    pub(super) fn new(method: Selection, bands: usize, pixels: usize, integer: bool) -> Self {
        let len = bands * pixels;
        let state = match method {
            Selection::Highest | Selection::Lowest => State::Extreme(vec![T::NAN; len]),
            Selection::Mean | Selection::Stdev => {
                State::Moments { sum: vec![0.0; len], squares: vec![0.0; len], count: vec![0; len] }
            }
            Selection::Median => State::All(Vec::new()),
            Selection::Count => State::Count(vec![T::ZERO; pixels]),
        };
        Self { method, bands, pixels, integer, scenes: 0, state }
    }

    /// The bytes `scenes` scenes would be held in, for the method that holds them.
    pub(super) fn held_bytes(method: Selection, scenes: usize, bands: usize, pixels: usize) -> usize {
        match method {
            Selection::Median => scenes.saturating_mul(bands * pixels * std::mem::size_of::<T>()),
            _ => 0,
        }
    }

    /// Take one scene: `bands * pixels` values, band after band, NaN where it has no data.
    pub(super) fn add(&mut self, scene: impl Iterator<Item = T>) {
        self.scenes += 1;
        match &mut self.state {
            State::Extreme(kept) => {
                let highest = self.method == Selection::Highest;
                for (kept, v) in kept.iter_mut().zip(scene) {
                    // Take the scene's value where nothing is kept yet, or where it is a
                    // valid value beyond the one kept.
                    let beyond = if highest { v > *kept } else { v < *kept };
                    if kept.is_nan() || beyond {
                        *kept = v;
                    }
                }
            }
            State::Moments { sum, squares, count } => {
                for (i, v) in scene.enumerate().take(sum.len()) {
                    if !v.is_nan() {
                        let v = v.to_f64();
                        sum[i] += v;
                        squares[i] += v * v;
                        count[i] += 1;
                    }
                }
            }
            State::All(all) => {
                let len = self.bands * self.pixels;
                let before = all.len();
                all.extend(scene.take(len));
                all.resize(before + len, T::NAN);
            }
            State::Count(counts) => {
                // rio-tiler keeps the first band's counts.
                for (n, v) in counts.iter_mut().zip(scene) {
                    if !v.is_nan() {
                        *n = n.add(T::ONE);
                    }
                }
            }
        }
    }

    /// The values the method leaves. `None` when no scene was taken.
    pub(super) fn finish(self) -> Option<Selected<T>> {
        if self.scenes == 0 {
            return None;
        }
        let len = self.bands * self.pixels;
        let integer = self.integer;
        // As `astype` does for an integer type: toward zero.
        let cast = |v: f64| T::from_f64(if integer { v.trunc() } else { v });
        let values = match self.state {
            State::Extreme(kept) => kept,
            State::Moments { sum, squares, count } => {
                let mut values = vec![T::NAN; len];
                for i in 0..len {
                    if count[i] == 0 {
                        continue;
                    }
                    let n = f64::from(count[i]);
                    let mean = sum[i] / n;
                    values[i] = if self.method == Selection::Mean {
                        cast(mean)
                    } else {
                        // The population figure, as `numpy.ma.std` gives; never cast.
                        T::from_f64((squares[i] / n - mean * mean).max(0.0).sqrt())
                    };
                }
                values
            }
            State::All(all) => {
                let mut values = vec![T::NAN; len];
                let mut column: Vec<T> = Vec::with_capacity(self.scenes);
                for (i, out) in values.iter_mut().enumerate() {
                    column.clear();
                    column.extend(all.iter().skip(i).step_by(len).copied().filter(|v| !v.is_nan()));
                    if column.is_empty() {
                        continue;
                    }
                    column.sort_unstable_by(T::total_order);
                    let mid = column.len() / 2;
                    // An even number of values has no middle one: numpy takes the mean of the two.
                    let median = if column.len() % 2 == 1 {
                        column[mid].to_f64()
                    } else {
                        (column[mid - 1].to_f64() + column[mid].to_f64()) / 2.0
                    };
                    *out = cast(median);
                }
                values
            }
            // One band, and no pixel of it is masked.
            State::Count(counts) => return Some(Selected { bands: 1, values: counts }),
        };
        Some(Selected { bands: self.bands, values })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: f32 = f32::NAN;

    /// `method` over `scenes`, taken in order.
    fn select(method: Selection, scenes: &[Vec<u8>], bands: usize, pixels: usize, integer: bool) -> Option<Selected> {
        let mut fold = Fold::new(method, bands, pixels, integer);
        scenes.iter().for_each(|s| fold.add_le_bytes(s));
        fold.finish()
    }

    fn scene(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Compare allowing for NaN, which is equal to nothing.
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.is_nan() && y.is_nan()) || x == y)
    }

    /// One band, four pixels, three scenes. The expected values are numpy's for
    /// `numpy.ma.masked_invalid([[3, nan, 7, nan], [5, 2, nan, nan], [4, 9, 8, nan]])`.
    fn three() -> Vec<Vec<u8>> {
        vec![scene(&[3.0, N, 7.0, N]), scene(&[5.0, 2.0, N, N]), scene(&[4.0, 9.0, 8.0, N])]
    }

    #[test]
    fn names_are_titilers() {
        for (name, method) in [
            ("highest", Selection::Highest),
            ("lowest", Selection::Lowest),
            ("mean", Selection::Mean),
            ("median", Selection::Median),
            ("stdev", Selection::Stdev),
            ("count", Selection::Count),
        ] {
            assert_eq!(Selection::parse(name), Some(method));
        }
        for name in ["first", "lastbandlow", "lastbandhigh", "sum", "Highest", ""] {
            assert_eq!(Selection::parse(name), None, "{name}");
        }
    }

    #[test]
    fn highest_and_lowest_ignore_missing_values() {
        let got = select(Selection::Highest, &three(), 1, 4, true).unwrap();
        assert!(same(&got.values, &[5.0, 9.0, 8.0, N]), "{got:?}");
        let got = select(Selection::Lowest, &three(), 1, 4, true).unwrap();
        assert!(same(&got.values, &[3.0, 2.0, 7.0, N]), "{got:?}");
    }

    #[test]
    fn mean_and_median_are_cast_back_for_whole_numbers() {
        // Means 4, 5.5, 7.5; medians 4, 5.5, 7.5.
        let got = select(Selection::Mean, &three(), 1, 4, true).unwrap();
        assert!(same(&got.values, &[4.0, 5.0, 7.0, N]), "{got:?}");
        let got = select(Selection::Mean, &three(), 1, 4, false).unwrap();
        assert!(same(&got.values, &[4.0, 5.5, 7.5, N]), "{got:?}");
        let got = select(Selection::Median, &three(), 1, 4, true).unwrap();
        assert!(same(&got.values, &[4.0, 5.0, 7.0, N]), "{got:?}");
        let got = select(Selection::Median, &three(), 1, 4, false).unwrap();
        assert!(same(&got.values, &[4.0, 5.5, 7.5, N]), "{got:?}");
        // Toward zero, not down: the mean of -3 and -4 is -3.5 and casts to -3.
        let got = select(Selection::Mean, &[scene(&[-3.0]), scene(&[-4.0])], 1, 1, true).unwrap();
        assert_eq!(got.values, [-3.0]);
    }

    #[test]
    fn stdev_is_the_population_figure_and_is_never_cast() {
        let got = select(Selection::Stdev, &three(), 1, 4, true).unwrap();
        // std([3, 5, 4]) = 0.8165, std([2, 9]) = 3.5, std([7, 8]) = 0.5.
        assert!((got.values[0] - 0.816_496_6).abs() < 1e-6, "{got:?}");
        assert_eq!(&got.values[1..3], [3.5, 0.5]);
        assert!(got.values[3].is_nan());
        // One value has no spread.
        let got = select(Selection::Stdev, &[scene(&[41.0])], 1, 1, true).unwrap();
        assert_eq!(got.values, [0.0]);
    }

    #[test]
    fn count_is_one_band_valid_everywhere() {
        let got = select(Selection::Count, &three(), 1, 4, true).unwrap();
        assert_eq!(got, Selected { bands: 1, values: vec![3.0, 2.0, 2.0, 0.0] });
        // Of several bands, the first's counts.
        let two_bands = [scene(&[1.0, N, N, N]), scene(&[1.0, 1.0, N, 1.0])];
        let got = select(Selection::Count, &two_bands, 2, 2, true).unwrap();
        assert_eq!(got, Selected { bands: 1, values: vec![2.0, 1.0] });
    }

    #[test]
    fn each_band_is_decided_on_its_own() {
        // Two bands of two pixels: band one is higher in the first scene, band two in the second.
        let scenes = [scene(&[9.0, 9.0, 1.0, 1.0]), scene(&[2.0, N, 8.0, 8.0])];
        let got = select(Selection::Highest, &scenes, 2, 2, true).unwrap();
        assert_eq!(got, Selected { bands: 2, values: vec![9.0, 9.0, 8.0, 8.0] });
    }

    #[test]
    fn no_scenes_is_no_picture() {
        assert_eq!(select(Selection::Mean, &[], 1, 4, true), None);
        assert_eq!(select(Selection::Count, &[], 1, 4, true), None);
    }

    #[test]
    fn statistics_are_worked_out_in_double_precision() {
        let scenes = [vec![0.1f64, f64::NAN, 3.0], vec![0.2, 5.0, 4.0], vec![0.4, 7.0, f64::NAN]];
        let run = |method, integer| {
            let mut fold = Fold::<f64>::new(method, 1, 3, integer);
            scenes.iter().for_each(|s| fold.add(s.iter().copied()));
            fold.finish().unwrap().values
        };
        assert_eq!(run(Selection::Mean, false), [(0.1 + 0.2 + 0.4) / 3.0, 6.0, 3.5]);
        assert_eq!(run(Selection::Median, false), [0.2, 6.0, 3.5]);
        assert_eq!(run(Selection::Median, true), [0.0, 6.0, 3.0]);
        assert_eq!(run(Selection::Highest, false), [0.4, 7.0, 4.0]);
        assert_eq!(run(Selection::Count, false), [3.0, 2.0, 2.0]);
    }

    #[test]
    fn only_median_holds_its_scenes() {
        assert_eq!(Fold::<f32>::held_bytes(Selection::Median, 10, 3, 256 * 256), 10 * 3 * 65536 * 4);
        assert_eq!(Fold::<f64>::held_bytes(Selection::Median, 10, 3, 256 * 256), 10 * 3 * 65536 * 8);
        for method in [Selection::Highest, Selection::Lowest, Selection::Mean, Selection::Stdev, Selection::Count] {
            assert_eq!(Fold::<f32>::held_bytes(method, 10_000, 3, 512 * 512), 0);
        }
    }
}
