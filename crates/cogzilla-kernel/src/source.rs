use cogzilla_codec::DecodedBand;

/// Lightweight view over the small rectangular grid of decoded COG blocks (`tx0..=tx1`, `ty0..=ty1`)
/// needed to render a single output XYZ tile for one band.
pub struct BandSource<'a> {
    pub ifd_w: i32,
    pub ifd_h: i32,
    pub tile_w: u32,
    pub tile_h: u32,
    pub tx0: u32,
    pub ty0: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    /// `grid_w * grid_h` entries; `None` represents a sparse unwritten all-nodata block.
    pub blocks: &'a [Option<&'a DecodedBand>],
    pub all_nodata: bool,
    pub no_nodata: bool,
    /// Optional runtime `?nodata=` override; when `Some(nd)`, values equal to `nd` become `NaN`.
    pub nodata_override: Option<f32>,
    /// Fast path when tile_w/tile_h are powers of two (e.g. 256 or 512).
    shift_x: Option<u32>,
    mask_x: u32,
    shift_y: Option<u32>,
    mask_y: u32,
}

impl<'a> BandSource<'a> {
    pub fn new(
        ifd_w: u32,
        ifd_h: u32,
        tile_w: u32,
        tile_h: u32,
        tx0: u32,
        ty0: u32,
        grid_w: u32,
        grid_h: u32,
        blocks: &'a [Option<&'a DecodedBand>],
    ) -> Self {
        let mut all_nodata = true;
        let mut no_nodata = true;
        for blk in blocks {
            match blk {
                Some(b) => {
                    if !b.stats.all_nodata {
                        all_nodata = false;
                    }
                    if !b.stats.no_nodata {
                        no_nodata = false;
                    }
                }
                None => {
                    no_nodata = false;
                }
            }
        }
        Self {
            ifd_w: ifd_w as i32,
            ifd_h: ifd_h as i32,
            tile_w,
            tile_h,
            tx0,
            ty0,
            grid_w,
            grid_h,
            blocks,
            all_nodata,
            no_nodata,
            nodata_override: None,
            shift_x: tile_w.is_power_of_two().then(|| tile_w.trailing_zeros()),
            mask_x: tile_w.wrapping_sub(1),
            shift_y: tile_h.is_power_of_two().then(|| tile_h.trailing_zeros()),
            mask_y: tile_h.wrapping_sub(1),
        }
    }

    #[inline(always)]
    pub fn with_nodata_override(mut self, nodata: Option<f32>) -> Self {
        if let Some(nd) = nodata {
            if !nd.is_nan() {
                self.nodata_override = Some(nd);
                self.no_nodata = false;
            }
        }
        self
    }

    /// Sample source pixel `(col, row)` with bounds check. Out-of-bounds returns `f32::NAN`.
    #[inline(always)]
    pub fn sample_checked(&self, col: i32, row: i32) -> f32 {
        if col < 0 || row < 0 || col >= self.ifd_w || row >= self.ifd_h {
            return f32::NAN;
        }
        let v = self.sample_inbounds(col as u32, row as u32);
        if let Some(nd) = self.nodata_override {
            if v == nd {
                return f32::NAN;
            }
        }
        v
    }

    /// Sample source pixel `(col, row)` guaranteed in `[0..ifd_w, 0..ifd_h]`.
    #[inline(always)]
    pub fn sample_inbounds(&self, col: u32, row: u32) -> f32 {
        let (tx, lx) = match self.shift_x {
            Some(s) => (col >> s, col & self.mask_x),
            None => (col / self.tile_w, col % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (row >> s, row & self.mask_y),
            None => (row / self.tile_h, row % self.tile_h),
        };
        let gx = tx.wrapping_sub(self.tx0);
        let gy = ty.wrapping_sub(self.ty0);
        if gx >= self.grid_w || gy >= self.grid_h {
            return f32::NAN;
        }
        let bidx = (gy * self.grid_w + gx) as usize;
        match unsafe { *self.blocks.get_unchecked(bidx) } {
            Some(blk) => {
                let pidx = (ly as usize) * (self.tile_w as usize) + (lx as usize);
                unsafe { blk.data.get_f32_unchecked(pidx) }
            }
            None => f32::NAN,
        }
    }

    /// Fast-path check: true if all pixel bounds `(c0..=c1, r0..=r1)` lie strictly inside `[0..ifd_w, 0..ifd_h]`
    /// and every intersecting block has `no_nodata == true`.
    #[inline(always)]
    pub fn is_clean_inbounds(&self, c0: i32, c1: i32, r0: i32, r1: i32) -> bool {
        self.no_nodata && c0 >= 0 && r0 >= 0 && c1 < self.ifd_w && r1 < self.ifd_h
    }

    /// If the tile intersects a single decoded `f32` block (`grid_w == 1 && grid_h == 1`), return
    /// `(flat_slice, base_col, base_row, row_stride)` so hot scanline loops can hoist pointers
    /// and bypass per-pixel tile-grid indexing completely!
    #[inline(always)]
    pub fn single_block_flat(&self) -> Option<(&'a [f32], u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 {
            let blk = self.blocks[0]?;
            let slice = blk.data.as_f32_slice()?;
            Some((
                slice,
                self.tx0 * self.tile_w,
                self.ty0 * self.tile_h,
                self.tile_w as usize,
            ))
        } else {
            None
        }
    }
}
