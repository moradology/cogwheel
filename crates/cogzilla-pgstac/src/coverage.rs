//! Which items add to a picture.
//!
//! pgSTAC decides this with exact geometry: an item is kept when the union of the
//! footprints so far, clipped to the request, grows by any amount. Here the request is a
//! grid of sample points, one or more per output pixel, and an item is kept when its
//! footprint holds a sample no earlier footprint held. An item that pgSTAC keeps for a
//! sliver between samples is passed over; no item pgSTAC passes over is kept.
//!
//! Footprints are planar in longitude and latitude, as they are in PostGIS.

use serde_json::Value;

pub(crate) type Ring = Vec<(f64, f64)>;
/// An outer ring and its holes; a point is inside when it is inside an odd number of rings.
pub(crate) type Polygon = Vec<Ring>;

/// The sample points of a request: `cols` by `rows` over the bounding box of its geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleGrid {
    pub cols: u32,
    pub rows: u32,
    /// Rows are evenly spaced in Web Mercator `y`, as a map tile's are, not in latitude.
    pub mercator_rows: bool,
}

impl SampleGrid {
    /// The largest side a grid may have.
    pub const MAX_SIDE: u32 = 4096;

    /// The grid for a picture of `width` by `height` pixels: five samples a side per pixel
    /// where that fits, else three, else one. An odd count puts a sample on each pixel
    /// centre. `None` for a picture too large to sample once per pixel.
    pub fn for_image(width: u32, height: u32, mercator_rows: bool) -> Option<Self> {
        let side = width.max(height);
        if width == 0 || height == 0 {
            return None;
        }
        let factor = [5u32, 3, 1].into_iter().find(|f| side.saturating_mul(*f) <= Self::MAX_SIDE)?;
        Some(Self { cols: width * factor, rows: height * factor, mercator_rows })
    }
}

/// The polygons of a WKB geometry (ISO or EWKB). Points and lines have none. `None` for
/// bytes this does not read: a curve, a surface, a truncated value.
pub(crate) fn polygons_from_wkb(wkb: &[u8]) -> Option<Vec<Polygon>> {
    let mut reader = Wkb { bytes: wkb, at: 0 };
    let mut out = Vec::new();
    reader.geometry(&mut out, 0)?;
    Some(out)
}

struct Wkb<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Wkb<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(n)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn u32(&mut self, little: bool) -> Option<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().ok()?;
        Some(if little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }

    fn f64(&mut self, little: bool) -> Option<f64> {
        let b: [u8; 8] = self.take(8)?.try_into().ok()?;
        Some(if little { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
    }

    /// Skip `count` points of `dims` ordinates each.
    fn skip_points(&mut self, count: u32, dims: usize) -> Option<()> {
        self.take((count as usize).checked_mul(dims * 8)?).map(|_| ())
    }

    fn geometry(&mut self, out: &mut Vec<Polygon>, depth: u8) -> Option<()> {
        if depth > 8 {
            return None;
        }
        let little = match self.take(1)?[0] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let code = self.u32(little)?;
        // EWKB carries Z, M and SRID as flag bits; ISO adds 1000, 2000 or 3000 to the type.
        let (has_z, has_m, has_srid) = (code & 0x8000_0000 != 0, code & 0x4000_0000 != 0, code & 0x2000_0000 != 0);
        let plain = code & 0x0FFF_FFFF;
        let (kind, iso_dims) = match plain / 1000 {
            0 => (plain, 0),
            1 | 2 => (plain % 1000, 1),
            3 => (plain % 1000, 2),
            _ => return None,
        };
        let dims = 2 + iso_dims + usize::from(has_z) + usize::from(has_m);
        if has_srid {
            self.u32(little)?;
        }
        match kind {
            1 => self.skip_points(1, dims),
            2 => {
                let n = self.u32(little)?;
                self.skip_points(n, dims)
            }
            3 => {
                let rings = self.u32(little)?;
                let mut polygon = Polygon::new();
                for _ in 0..rings {
                    let n = self.u32(little)? as usize;
                    // Each point takes at least sixteen bytes, so a count the value cannot hold is refused before allocating.
                    if n > (self.bytes.len() - self.at) / 16 {
                        return None;
                    }
                    let mut ring = Ring::with_capacity(n);
                    for _ in 0..n {
                        let x = self.f64(little)?;
                        let y = self.f64(little)?;
                        self.take((dims - 2) * 8)?;
                        ring.push((x, y));
                    }
                    polygon.push(ring);
                }
                out.push(polygon);
                Some(())
            }
            4..=7 => {
                let n = self.u32(little)?;
                for _ in 0..n {
                    self.geometry(out, depth + 1)?;
                }
                Some(())
            }
            _ => None,
        }
    }
}

/// The polygons of a GeoJSON `Polygon` or `MultiPolygon`. `None` for any other geometry:
/// pgSTAC does no coverage arithmetic for those either.
pub(crate) fn polygons_from_geojson(geometry: &Value) -> Option<Vec<Polygon>> {
    let ring = |v: &Value| -> Option<Ring> {
        v.as_array()?
            .iter()
            .map(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?)))
            .collect()
    };
    let polygon = |v: &Value| -> Option<Polygon> { v.as_array()?.iter().map(ring).collect() };
    let coordinates = geometry.get("coordinates")?;
    match geometry.get("type")?.as_str()? {
        "Polygon" => Some(vec![polygon(coordinates)?]),
        "MultiPolygon" => coordinates.as_array()?.iter().map(polygon).collect(),
        _ => None,
    }
}

/// The samples of a request and which of them a footprint has covered so far.
pub(crate) struct Coverage {
    cols: usize,
    words_per_row: usize,
    west: f64,
    /// Width of a column in degrees.
    dx: f64,
    /// Latitude of each row's samples, north to south.
    row_lat: Vec<f64>,
    /// Samples of the request no footprint holds yet.
    open: Vec<u64>,
    scratch: Vec<u64>,
    crossings: Vec<f64>,
    remaining: usize,
}

impl Coverage {
    /// `None` when the area holds no sample: it is empty, or thinner than the grid.
    pub(crate) fn new(area: &[Polygon], grid: SampleGrid) -> Option<Self> {
        let points = || area.iter().flatten().flatten();
        let fold = |f: fn(f64, f64) -> f64, pick: fn(&(f64, f64)) -> f64, start: f64| points().map(pick).fold(start, f);
        let west = fold(f64::min, |p| p.0, f64::INFINITY);
        let east = fold(f64::max, |p| p.0, f64::NEG_INFINITY);
        let south = fold(f64::min, |p| p.1, f64::INFINITY);
        let north = fold(f64::max, |p| p.1, f64::NEG_INFINITY);
        let (cols, rows) = (grid.cols as usize, grid.rows as usize);
        let sized = cols > 0 && rows > 0 && grid.cols <= SampleGrid::MAX_SIDE && grid.rows <= SampleGrid::MAX_SIDE;
        // An empty area leaves the bounds infinite, and the comparisons false.
        let bounded = east > west && north > south && [west, east, south, north].iter().all(|v| v.is_finite());
        if !(sized && bounded) {
            return None;
        }
        let mercator_y = |lat: f64| (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln();
        let (top, bottom) = (mercator_y(north), mercator_y(south));
        let row_lat: Vec<f64> = if grid.mercator_rows && top.is_finite() && bottom.is_finite() {
            (0..rows)
                .map(|r| {
                    let y = top - (r as f64 + 0.5) * (top - bottom) / rows as f64;
                    (2.0 * y.exp().atan() - std::f64::consts::FRAC_PI_2).to_degrees()
                })
                .collect()
        } else {
            (0..rows).map(|r| north - (r as f64 + 0.5) * (north - south) / rows as f64).collect()
        };
        let words_per_row = cols.div_ceil(64);
        let mut this = Self {
            cols,
            words_per_row,
            west,
            dx: (east - west) / cols as f64,
            row_lat,
            open: vec![0; words_per_row * rows],
            scratch: vec![0; words_per_row * rows],
            crossings: Vec::new(),
            remaining: 0,
        };
        // The request's own samples: all of them for a box, else those inside its outline.
        if is_box(area, [west, south, east, north]) {
            for row in this.scratch.chunks_exact_mut(words_per_row) {
                set_bits(row, 0, cols);
            }
        } else {
            this.rasterize(area, false);
        }
        std::mem::swap(&mut this.open, &mut this.scratch);
        this.remaining = this.open.iter().map(|w| w.count_ones() as usize).sum();
        (this.remaining > 0).then_some(this)
    }

    /// Take a footprint. Returns how many samples it is the first to hold.
    pub(crate) fn add(&mut self, footprint: &[Polygon]) -> usize {
        let Some((first, last)) = self.rasterize(footprint, true) else {
            return 0;
        };
        let mut gained = 0;
        for i in first * self.words_per_row..(last + 1) * self.words_per_row {
            let fresh = self.scratch[i] & self.open[i];
            gained += fresh.count_ones() as usize;
            self.open[i] &= !fresh;
            self.scratch[i] = 0;
        }
        self.remaining -= gained;
        gained
    }

    /// Every sample of the request is held by some footprint.
    pub(crate) fn is_full(&self) -> bool {
        self.remaining == 0
    }

    /// Set, in `scratch`, the samples inside any of `polygons`. Returns the first and last
    /// row written. With `only_open`, rows where the polygon's extent has no open sample
    /// are passed over: nothing it set there could count. A search walks far more
    /// footprints over ground already covered than over ground that is not, and this is
    /// what makes those cheap.
    fn rasterize(&mut self, polygons: &[Polygon], only_open: bool) -> Option<(usize, usize)> {
        let rows = self.row_lat.len();
        let mut touched: Option<(usize, usize)> = None;
        for polygon in polygons {
            let lats = || polygon.iter().flatten().map(|p| p.1);
            let lons = || polygon.iter().flatten().map(|p| p.0);
            let top = lats().fold(f64::NEG_INFINITY, f64::max);
            let bottom = lats().fold(f64::INFINITY, f64::min);
            let left = lons().fold(f64::INFINITY, f64::min);
            let right = lons().fold(f64::NEG_INFINITY, f64::max);
            if top.is_nan() || bottom.is_nan() || top < bottom || left.is_nan() || right.is_nan() {
                continue;
            }
            // Rows run north to south. A sample is inside from the bottom edge up to, but not
            // on, the top one, the same half-open rule the crossings below follow: a row on
            // the edge two footprints share belongs to the upper one and to no other.
            let first = self.row_lat.partition_point(|lat| *lat >= top);
            let end = self.row_lat.partition_point(|lat| *lat >= bottom);
            // The words the polygon's extent reaches into.
            let column = |x: f64| ((x - self.west) / self.dx - 0.5).ceil().clamp(0.0, self.cols as f64) as usize;
            let (from, to) = (column(left), column(right));
            if from >= to {
                continue;
            }
            let words = from / 64..(to - 1) / 64 + 1;
            for row in first..end.min(rows) {
                let base = row * self.words_per_row;
                if only_open && self.open[base + words.start..base + words.end].iter().all(|w| *w == 0) {
                    continue;
                }
                let lat = self.row_lat[row];
                self.crossings.clear();
                for ring in polygon {
                    for (a, b) in ring.iter().zip(ring.iter().skip(1)) {
                        // Half-open in latitude, so a sample on a shared edge belongs to one side only.
                        if (a.1 <= lat) != (b.1 <= lat) {
                            self.crossings.push(a.0 + (lat - a.1) * (b.0 - a.0) / (b.1 - a.1));
                        }
                    }
                }
                if self.crossings.len() < 2 {
                    continue;
                }
                self.crossings.sort_unstable_by(f64::total_cmp);
                let mut wrote = false;
                for pair in self.crossings.chunks_exact(2) {
                    // The columns whose centre lies in [pair[0], pair[1]).
                    let (from, to) = (column(pair[0]), column(pair[1]));
                    if from < to {
                        set_bits(&mut self.scratch[base..base + self.words_per_row], from, to);
                        wrote = true;
                    }
                }
                if wrote {
                    touched = Some(match touched {
                        Some((lo, hi)) => (lo.min(row), hi.max(row)),
                        None => (row, row),
                    });
                }
            }
        }
        touched
    }
}

/// Set bits `from..to` of a row.
fn set_bits(row: &mut [u64], from: usize, to: usize) {
    let (first, last) = (from / 64, (to - 1) / 64);
    let head = !0u64 << (from % 64);
    let tail = !0u64 >> (63 - (to - 1) % 64);
    if first == last {
        row[first] |= head & tail;
    } else {
        row[first] |= head;
        row[first + 1..last].fill(!0);
        row[last] |= tail;
    }
}

/// One ring that is the box itself: four corners, sides along the axes.
fn is_box(area: &[Polygon], [west, south, east, north]: [f64; 4]) -> bool {
    let [polygon] = area else { return false };
    let [ring] = polygon.as_slice() else { return false };
    let closed = ring.len() == 5 && ring.first() == ring.last();
    let corners = if closed { &ring[..4] } else { ring.as_slice() };
    corners.len() == 4
        && corners.iter().all(|p| (p.0 == west || p.0 == east) && (p.1 == south || p.1 == north))
        && (0..4).all(|i| (0..i).all(|j| corners[i] != corners[j]))
        // Neighbours share exactly one coordinate, which rules out a bow-tie through the corners.
        && (0..4).all(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            (a.0 == b.0) != (a.1 == b.1)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        vec![vec![(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]]
    }

    fn grid(side: u32) -> SampleGrid {
        SampleGrid { cols: side, rows: side, mercator_rows: false }
    }

    #[test]
    fn grid_for_image_puts_a_sample_on_each_pixel_centre() {
        assert_eq!(SampleGrid::for_image(256, 256, true), Some(SampleGrid { cols: 1280, rows: 1280, mercator_rows: true }));
        assert_eq!(SampleGrid::for_image(1024, 512, false), Some(SampleGrid { cols: 3072, rows: 1536, mercator_rows: false }));
        assert_eq!(SampleGrid::for_image(4096, 10, false), Some(SampleGrid { cols: 4096, rows: 10, mercator_rows: false }));
        assert_eq!(SampleGrid::for_image(4097, 10, false), None);
        assert_eq!(SampleGrid::for_image(0, 10, false), None);
    }

    #[test]
    fn two_halves_fill_a_box_and_a_third_adds_nothing() {
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(10)).unwrap();
        assert_eq!(cover.add(&[square(-5.0, -5.0, 5.0, 15.0)]), 50);
        assert!(!cover.is_full());
        // The same half again, and a part of it, hold nothing new.
        assert_eq!(cover.add(&[square(-5.0, -5.0, 5.0, 15.0)]), 0);
        assert_eq!(cover.add(&[square(1.0, 1.0, 4.0, 4.0)]), 0);
        // The other half shares the edge at x = 5: no sample is counted twice or missed.
        assert_eq!(cover.add(&[square(5.0, -5.0, 15.0, 15.0)]), 50);
        assert!(cover.is_full());
    }

    #[test]
    fn a_row_of_samples_on_a_shared_edge_belongs_to_one_side() {
        // Ten rows put samples at 9.5, 8.5, ...: two footprints that meet at 4.5 meet on a row.
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(10)).unwrap();
        assert_eq!(cover.add(&[square(0.0, 4.5, 10.0, 15.0)]), 60);
        assert_eq!(cover.add(&[square(0.0, -5.0, 10.0, 4.5)]), 40);
        assert!(cover.is_full());
        // An odd number of rows puts one on the middle of the box.
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(3)).unwrap();
        assert_eq!(cover.add(&[square(0.0, -5.0, 10.0, 5.0)]), 3);
        assert_eq!(cover.add(&[square(0.0, 5.0, 10.0, 15.0)]), 6);
        assert!(cover.is_full());
        // And the same for columns.
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(3)).unwrap();
        assert_eq!(cover.add(&[square(-5.0, 0.0, 5.0, 10.0)]) + cover.add(&[square(5.0, 0.0, 15.0, 10.0)]), 9);
        assert!(cover.is_full());
    }

    #[test]
    fn a_ring_through_the_corners_is_not_always_the_box() {
        let bounds = [0.0, 0.0, 10.0, 10.0];
        assert!(is_box(&[square(0.0, 0.0, 10.0, 10.0)], bounds));
        // Out along one side and back: the same extent, no area.
        let folded: Polygon = vec![vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)]];
        assert!(!is_box(&[folded], bounds));
        let bow_tie: Polygon = vec![vec![(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)]];
        assert!(!is_box(&[bow_tie], bounds));
    }

    #[test]
    fn a_sliver_between_samples_adds_nothing() {
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(10)).unwrap();
        // Samples sit at 0.5, 1.5, ...: a strip from 0.6 to 1.4 holds none of them.
        assert_eq!(cover.add(&[square(0.6, 0.0, 1.4, 10.0)]), 0);
        // One that reaches 1.5 holds a column of ten.
        assert_eq!(cover.add(&[square(0.6, 0.0, 1.6, 10.0)]), 10);
    }

    #[test]
    fn holes_and_multipolygons() {
        let mut cover = Coverage::new(&[square(0.0, 0.0, 10.0, 10.0)], grid(10)).unwrap();
        let mut ring_with_hole = square(0.0, 0.0, 10.0, 10.0);
        ring_with_hole.push(square(2.0, 2.0, 8.0, 8.0).remove(0));
        assert_eq!(cover.add(&[ring_with_hole]), 100 - 36);
        // Two overlapping parts of one footprint are a union, not an exclusive-or.
        assert_eq!(cover.add(&[square(2.0, 2.0, 6.0, 8.0), square(4.0, 2.0, 8.0, 8.0)]), 36);
        assert!(cover.is_full());
    }

    #[test]
    fn a_request_that_is_not_a_box_counts_only_its_own_samples() {
        // A right triangle: half the box.
        let triangle: Polygon = vec![vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)]];
        let mut cover = Coverage::new(&[triangle], grid(10)).unwrap();
        let wanted = cover.remaining;
        assert!((40..=55).contains(&wanted), "{wanted}");
        // A footprint over the other half of the box holds none of the triangle's samples.
        let other: Polygon = vec![vec![(10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)]];
        assert_eq!(cover.add(&[other]), 0);
        assert_eq!(cover.add(&[square(-1.0, -1.0, 11.0, 11.0)]), wanted);
        assert!(cover.is_full());
    }

    #[test]
    fn areas_with_no_sample_are_refused() {
        assert!(Coverage::new(&[], grid(10)).is_none());
        assert!(Coverage::new(&[vec![vec![(0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]]], grid(10)).is_none());
        // An L along two sides, thinner than the first sample is from the edge (1.25).
        let thin: Polygon = vec![vec![(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (1.0, 1.0), (1.0, 10.0), (0.0, 10.0), (0.0, 0.0)]];
        assert!(Coverage::new(std::slice::from_ref(&thin), grid(4)).is_none());
        assert!(Coverage::new(&[thin], grid(8)).is_some());
        assert!(Coverage::new(&[square(0.0, 0.0, 1.0, 1.0)], SampleGrid { cols: 5000, rows: 10, mercator_rows: false }).is_none());
    }

    #[test]
    fn mercator_rows_follow_the_tile_not_the_latitude() {
        // The top half of a tall box in latitude is less than half of it in Mercator rows.
        let area = [square(0.0, 0.0, 10.0, 80.0)];
        let mut linear = Coverage::new(&area, grid(100)).unwrap();
        let mut mercator = Coverage::new(&area, SampleGrid { cols: 100, rows: 100, mercator_rows: true }).unwrap();
        let top_half = [square(0.0, 40.0, 10.0, 80.0)];
        assert_eq!(linear.add(&top_half), 5000);
        let gained = mercator.add(&top_half);
        // y(40°) / y(80°) = 0.7629 / 2.4362, so the top half holds about 69% of the rows.
        assert!((6800..=7000).contains(&gained), "{gained}");
    }

    #[test]
    fn wide_rows_cross_word_boundaries() {
        let mut cover = Coverage::new(&[square(0.0, 0.0, 200.0, 2.0)], SampleGrid { cols: 200, rows: 2, mercator_rows: false }).unwrap();
        assert_eq!(cover.add(&[square(10.0, 0.0, 190.0, 2.0)]), 360);
        assert_eq!(cover.add(&[square(63.0, 0.0, 65.0, 2.0)]), 0);
        assert_eq!(cover.add(&[square(0.0, 0.0, 200.0, 1.0)]), 20);
        assert_eq!(cover.add(&[square(0.0, 0.0, 200.0, 2.0)]), 20);
        assert!(cover.is_full());
    }

    fn wkb_polygon(code: u32, extra: usize, rings: &[&[(f64, f64)]]) -> Vec<u8> {
        let mut out = vec![1u8];
        out.extend(code.to_le_bytes());
        out.extend((rings.len() as u32).to_le_bytes());
        for ring in rings {
            out.extend((ring.len() as u32).to_le_bytes());
            for (x, y) in *ring {
                out.extend(x.to_le_bytes());
                out.extend(y.to_le_bytes());
                out.extend(std::iter::repeat_n(0u8, extra * 8));
            }
        }
        out
    }

    #[test]
    fn wkb_polygons_in_every_dimension() {
        let ring: &[(f64, f64)] = &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)];
        let expect = vec![vec![ring.to_vec()]];
        assert_eq!(polygons_from_wkb(&wkb_polygon(3, 0, &[ring])), Some(expect.clone()));
        assert_eq!(polygons_from_wkb(&wkb_polygon(1003, 1, &[ring])), Some(expect.clone()));
        assert_eq!(polygons_from_wkb(&wkb_polygon(3003, 2, &[ring])), Some(expect.clone()));
        assert_eq!(polygons_from_wkb(&wkb_polygon(0x8000_0003, 1, &[ring])), Some(expect.clone()));
        // A multipolygon of two, big-endian header on the collection.
        let mut multi = vec![0u8];
        multi.extend(6u32.to_be_bytes());
        multi.extend(2u32.to_be_bytes());
        multi.extend(wkb_polygon(3, 0, &[ring]));
        multi.extend(wkb_polygon(3, 0, &[ring]));
        assert_eq!(polygons_from_wkb(&multi).map(|p| p.len()), Some(2));
        // A point has no polygons; a truncated polygon and an unknown type are refused.
        let mut point = vec![1u8];
        point.extend(1u32.to_le_bytes());
        point.extend([0u8; 16]);
        assert_eq!(polygons_from_wkb(&point), Some(vec![]));
        let whole = wkb_polygon(3, 0, &[ring]);
        assert_eq!(polygons_from_wkb(&whole[..whole.len() - 1]), None);
        assert_eq!(polygons_from_wkb(&wkb_polygon(10, 0, &[ring])), None);
        // A point count far beyond the bytes present is refused without allocating for it.
        let mut lying = vec![1u8];
        lying.extend(3u32.to_le_bytes());
        lying.extend(1u32.to_le_bytes());
        lying.extend(u32::MAX.to_le_bytes());
        assert_eq!(polygons_from_wkb(&lying), None);
    }

    #[test]
    fn geojson_polygons() {
        let polygon = json!({"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]});
        assert_eq!(polygons_from_geojson(&polygon).map(|p| p.len()), Some(1));
        let multi = json!({"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 1], [1, 1], [0, 0]]], [[[5, 5], [5, 6], [6, 6], [5, 5]]]]});
        assert_eq!(polygons_from_geojson(&multi).map(|p| p.len()), Some(2));
        assert_eq!(polygons_from_geojson(&json!({"type": "Point", "coordinates": [1, 2]})), None);
        assert_eq!(polygons_from_geojson(&json!({"type": "Polygon", "coordinates": [[["a", 0]]]})), None);
    }
}
