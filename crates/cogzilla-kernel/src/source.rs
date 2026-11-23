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

    /// Fast 2×2 bilinear quad fetch when all 4 taps `(ux0..=ux0+1, uy0..=uy0+1)` are in-bounds.
    /// For the 99.6% interior case where all 4 taps lie in the same internal COG block, performs
    /// only 1 block lookup and 1 `BandData` dispatch instead of 4!
    #[inline(always)]
    pub fn sample_quad_inbounds(&self, ux0: u32, uy0: u32) -> (f32, f32, f32, f32) {
        let (tx, lx) = match self.shift_x {
            Some(s) => (ux0 >> s, ux0 & self.mask_x),
            None => (ux0 / self.tile_w, ux0 % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy0 >> s, uy0 & self.mask_y),
            None => (uy0 / self.tile_h, uy0 % self.tile_h),
        };
        if lx + 1 < self.tile_w && ly + 1 < self.tile_h {
            let gx = tx.wrapping_sub(self.tx0);
            let gy = ty.wrapping_sub(self.ty0);
            if gx < self.grid_w && gy < self.grid_h {
                let bidx = (gy * self.grid_w + gx) as usize;
                return match unsafe { *self.blocks.get_unchecked(bidx) } {
                    Some(blk) => {
                        let tw = self.tile_w as usize;
                        let p00 = (ly as usize) * tw + (lx as usize);
                        let p01 = p00 + tw;
                        match &blk.data {
                            cogzilla_codec::BandData::F32(s) => {
                                let sl = s.as_slice();
                                unsafe {
                                    (
                                        *sl.get_unchecked(p00),
                                        *sl.get_unchecked(p00 + 1),
                                        *sl.get_unchecked(p01),
                                        *sl.get_unchecked(p01 + 1),
                                    )
                                }
                            }
                            cogzilla_codec::BandData::U16 { data, nodata } => unsafe {
                                let a = *data.get_unchecked(p00);
                                let b = *data.get_unchecked(p00 + 1);
                                let c = *data.get_unchecked(p01);
                                let d = *data.get_unchecked(p01 + 1);
                                match *nodata {
                                    None => (a as f32, b as f32, c as f32, d as f32),
                                    Some(nd) => (
                                        if a == nd { f32::NAN } else { a as f32 },
                                        if b == nd { f32::NAN } else { b as f32 },
                                        if c == nd { f32::NAN } else { c as f32 },
                                        if d == nd { f32::NAN } else { d as f32 },
                                    ),
                                }
                            },
                            cogzilla_codec::BandData::U8 { data, nodata } => unsafe {
                                let a = *data.get_unchecked(p00);
                                let b = *data.get_unchecked(p00 + 1);
                                let c = *data.get_unchecked(p01);
                                let d = *data.get_unchecked(p01 + 1);
                                match *nodata {
                                    None => (a as f32, b as f32, c as f32, d as f32),
                                    Some(nd) => (
                                        if a == nd { f32::NAN } else { a as f32 },
                                        if b == nd { f32::NAN } else { b as f32 },
                                        if c == nd { f32::NAN } else { c as f32 },
                                        if d == nd { f32::NAN } else { d as f32 },
                                    ),
                                }
                            },
                        }
                    }
                    None => (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
                };
            }
            return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        }
        (
            self.sample_inbounds(ux0, uy0),
            self.sample_inbounds(ux0 + 1, uy0),
            self.sample_inbounds(ux0, uy0 + 1),
            self.sample_inbounds(ux0 + 1, uy0 + 1),
        )
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

    /// Single-block raw typed slice view for clean (`no_nodata`) `U8`/`U16`/`F32` bands.
    #[inline(always)]
    pub fn single_block_u16(&self) -> Option<(&'a [u16], u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 && self.no_nodata {
            let blk = self.blocks[0]?;
            if let cogzilla_codec::BandData::U16 { data, nodata: None } = &blk.data {
                return Some((
                    data.as_ref(),
                    self.tx0 * self.tile_w,
                    self.ty0 * self.tile_h,
                    self.tile_w as usize,
                ));
            }
        }
        None
    }

    /// Single-block raw `u8` slice view for clean (`no_nodata`) `U8` bands.
    #[inline(always)]
    pub fn single_block_u8(&self) -> Option<(&'a [u8], u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 && self.no_nodata {
            let blk = self.blocks[0]?;
            if let cogzilla_codec::BandData::U8 { data, nodata: None } = &blk.data {
                return Some((
                    data.as_ref(),
                    self.tx0 * self.tile_w,
                    self.ty0 * self.tile_h,
                    self.tile_w as usize,
                ));
            }
        }
        None
    }

    /// True if `other` shares the exact same tile grid geometry as `self`, allowing multi-band
    /// RGB kernels to compute tile/pixel indices once per pixel instead of 3×.
    #[inline(always)]
    pub fn same_grid_as(&self, other: &Self) -> bool {
        self.tile_w == other.tile_w
            && self.tile_h == other.tile_h
            && self.tx0 == other.tx0
            && self.ty0 == other.ty0
            && self.grid_w == other.grid_w
            && self.grid_h == other.grid_h
    }

    /// Compute `(bidx, p00, p01)` for a 2×2 quad whose top-left `(ux0, uy0)` stays strictly
    /// inside a single internal block (`lx + 1 < tile_w && ly + 1 < tile_h`).
    #[inline(always)]
    pub fn interior_quad_loc(&self, ux0: u32, uy0: u32) -> Option<(usize, usize, usize)> {
        let (tx, lx) = match self.shift_x {
            Some(s) => (ux0 >> s, ux0 & self.mask_x),
            None => (ux0 / self.tile_w, ux0 % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy0 >> s, uy0 & self.mask_y),
            None => (uy0 / self.tile_h, uy0 % self.tile_h),
        };
        if lx + 1 < self.tile_w && ly + 1 < self.tile_h {
            let gx = tx.wrapping_sub(self.tx0);
            let gy = ty.wrapping_sub(self.ty0);
            if gx < self.grid_w && gy < self.grid_h {
                let bidx = (gy * self.grid_w + gx) as usize;
                let tw = self.tile_w as usize;
                let p00 = (ly as usize) * tw + (lx as usize);
                return Some((bidx, p00, p00 + tw));
            }
        }
        None
    }
}
