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
    /// Optional GDAL `scale`/`offset` applied when `?unscale=true` (`v * scale + offset`).
    pub scale_offset: Option<(f32, f32)>,
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
            scale_offset: None,
            shift_x: tile_w.is_power_of_two().then(|| tile_w.trailing_zeros()),
            mask_x: tile_w.wrapping_sub(1),
            shift_y: tile_h.is_power_of_two().then(|| tile_h.trailing_zeros()),
            mask_y: tile_h.wrapping_sub(1),
        }
    }

    #[inline(always)]
    pub fn with_nodata_override(mut self, nodata: Option<f32>) -> Self {
        if let Some(nd) = nodata
            && !nd.is_nan() {
                self.nodata_override = Some(nd);
                self.no_nodata = false;
            }
        self
    }

    #[inline(always)]
    pub fn with_scale_offset(mut self, scale: f32, offset: f32) -> Self {
        if scale != 1.0 || offset != 0.0 {
            self.scale_offset = Some((scale, offset));
            self.no_nodata = false;
        }
        self
    }

    /// Whether 1D sample coordinate `c` lies inside the pixel-center extent
    /// `[-0.5, dim - 0.5]` (inclusive on both edges, finite only).
    ///
    /// Source pixels are indexed by pixel center (`0.0` = center of pixel 0,
    /// `dim - 1.0` = center of the last pixel), so the raster covers
    /// `[-0.5, dim - 0.5]` along each axis.
    #[inline(always)]
    pub fn axis_contains_sample(c: f32, dim: i32) -> bool {
        c.is_finite() && c >= -0.5 && c <= dim as f32 - 0.5
    }

    /// `f64` twin of [`Self::axis_contains_sample`].
    #[inline(always)]
    pub fn axis_contains_sample_f64(c: f64, dim: i32) -> bool {
        c.is_finite() && c >= -0.5 && c <= dim as f64 - 0.5
    }

    /// Whether 2D sample coordinate `(u, v)` lies inside the raster's
    /// `[-0.5, w - 0.5] × [-0.5, h - 0.5]` sample extent.
    #[inline(always)]
    pub fn dims_contain_sample(w: i32, h: i32, u: f32, v: f32) -> bool {
        Self::axis_contains_sample(u, w) && Self::axis_contains_sample(v, h)
    }

    /// Whether 2D sample coordinate `(u, v)` lies inside this band's
    /// `[-0.5, ifd_w - 0.5] × [-0.5, ifd_h - 0.5]` sample extent.
    #[inline(always)]
    pub fn contains_sample(&self, u: f32, v: f32) -> bool {
        Self::dims_contain_sample(self.ifd_w, self.ifd_h, u, v)
    }

    /// `f64` twin of [`Self::contains_sample`].
    #[inline(always)]
    pub fn contains_sample_f64(&self, u: f64, v: f64) -> bool {
        Self::axis_contains_sample_f64(u, self.ifd_w) && Self::axis_contains_sample_f64(v, self.ifd_h)
    }

    /// Sample source pixel `(col, row)` with bounds check. Out-of-bounds returns `f32::NAN`.
    #[inline(always)]
    pub fn sample_checked(&self, col: i32, row: i32) -> f32 {
        if col < 0 || row < 0 || col >= self.ifd_w || row >= self.ifd_h {
            return f32::NAN;
        }
        self.sample_inbounds(col as u32, row as u32)
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
        let v = match unsafe { *self.blocks.get_unchecked(bidx) } {
            Some(blk) if ly < blk.height && lx < blk.width => {
                let pidx = (ly as usize) * (self.tile_w as usize) + (lx as usize);
                unsafe { blk.data.get_f32_unchecked(pidx) }
            }
            _ => f32::NAN,
        };
        if let Some(nd) = self.nodata_override
            && v == nd {
                return f32::NAN;
            }
        if let Some((sc, off)) = self.scale_offset
            && !v.is_nan() {
                return v * sc + off;
            }
        v
    }

    /// Exact `f64` bounds-checked pixel read (for wide dtypes `Uint32`/`Int32`/`Uint64`/`Int64`/`Float64`).
    #[inline]
    pub fn sample_f64_checked(&self, col: i32, row: i32) -> f64 {
        if col < 0 || row < 0 || col >= self.ifd_w || row >= self.ifd_h {
            return f64::NAN;
        }
        let col = col as u32;
        let row = row as u32;
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
            return f64::NAN;
        }
        let bidx = (gy * self.grid_w + gx) as usize;
        let v = match unsafe { *self.blocks.get_unchecked(bidx) } {
            Some(blk) if ly < blk.height && lx < blk.width => {
                let pidx = (ly as usize) * (self.tile_w as usize) + (lx as usize);
                unsafe { blk.data.get_f64_unchecked(pidx) }
            }
            _ => f64::NAN,
        };
        if let Some(nd) = self.nodata_override
            && !nd.is_nan()
            && v == nd as f64
        {
            return f64::NAN;
        }
        if let Some((sc, off)) = self.scale_offset
            && !v.is_nan()
        {
            return v * (sc as f64) + (off as f64);
        }
        v
    }

    /// Fast 2×2 bilinear quad fetch when all 4 taps `(ux0..=ux0+1, uy0..=uy0+1)` are in-bounds.
    /// For the 99.6% interior case where all 4 taps lie in the same internal COG block, performs
    /// only 1 block lookup and 1 `BandData` dispatch instead of 4!
    #[inline(always)]
    pub fn sample_quad_inbounds(&self, ux0: u32, uy0: u32) -> (f32, f32, f32, f32) {
        if self.scale_offset.is_some() || self.nodata_override.is_some() {
            return (
                self.sample_inbounds(ux0, uy0),
                self.sample_inbounds(ux0 + 1, uy0),
                self.sample_inbounds(ux0, uy0 + 1),
                self.sample_inbounds(ux0 + 1, uy0 + 1),
            );
        }
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
                    Some(blk) if ly + 1 < blk.height && lx + 1 < blk.width => {
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
                            cogzilla_codec::BandData::F64(s) => unsafe {
                                (
                                    *s.get_unchecked(p00) as f32,
                                    *s.get_unchecked(p00 + 1) as f32,
                                    *s.get_unchecked(p01) as f32,
                                    *s.get_unchecked(p01 + 1) as f32,
                                )
                            },
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
                            cogzilla_codec::BandData::RgbU8Channel {
                                rgb,
                                channel,
                                nodata,
                            } => unsafe {
                                let ch = *channel as usize;
                                let a = *rgb.get_unchecked(p00 * 3 + ch);
                                let b = *rgb.get_unchecked((p00 + 1) * 3 + ch);
                                let c = *rgb.get_unchecked(p01 * 3 + ch);
                                let d = *rgb.get_unchecked((p01 + 1) * 3 + ch);
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
                    Some(_) => (
                        self.sample_inbounds(ux0, uy0),
                        self.sample_inbounds(ux0 + 1, uy0),
                        self.sample_inbounds(ux0, uy0 + 1),
                        self.sample_inbounds(ux0 + 1, uy0 + 1),
                    ),
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

    #[inline(always)]
    fn single_block_base_exact(&self) -> Option<(u32, u32)> {
        let base_col = self.tx0.checked_mul(self.tile_w)?;
        let base_row = self.ty0.checked_mul(self.tile_h)?;
        // Reject bases > 2^24 that cannot be represented exactly in f32 so
        // `u_coords[px] - (base_col as f32)` can never round down and over-read past row_stride.
        if ((base_col as f32) as u32) != base_col || ((base_row as f32) as u32) != base_row {
            return None;
        }
        Some((base_col, base_row))
    }

    /// If the tile intersects a single decoded `f32` block (`grid_w == 1 && grid_h == 1`), return
    /// `(flat_slice, base_col, base_row, row_stride)` so hot scanline loops can hoist pointers
    /// and bypass per-pixel tile-grid indexing completely!
    #[inline(always)]
    pub fn single_block_flat(&self) -> Option<(&'a [f32], u32, u32, usize)> {
        if self.grid_w == 1
            && self.grid_h == 1
            && self.scale_offset.is_none()
            && self.nodata_override.is_none()
        {
            let (base_col, base_row) = self.single_block_base_exact()?;
            let blk = self.blocks[0]?;
            if blk.width < 2 || blk.height < 2 {
                return None;
            }
            let slice = blk.data.as_f32_slice()?;
            Some((slice, base_col, base_row, self.tile_w as usize))
        } else {
            None
        }
    }

    /// Single-block raw typed slice view for clean (`no_nodata`) `U8`/`U16`/`F32` bands.
    #[inline(always)]
    pub fn single_block_u16(&self) -> Option<(&'a [u16], u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 && self.no_nodata {
            let (base_col, base_row) = self.single_block_base_exact()?;
            let blk = self.blocks[0]?;
            if blk.width < 2 || blk.height < 2 {
                return None;
            }
            if let cogzilla_codec::BandData::U16 { data, .. } = &blk.data {
                return Some((data.as_ref(), base_col, base_row, self.tile_w as usize));
            }
        }
        None
    }

    /// Single-block raw `u8` slice view for clean (`no_nodata`) `U8` bands.
    #[inline(always)]
    pub fn single_block_u8(&self) -> Option<(&'a [u8], u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 && self.no_nodata {
            let (base_col, base_row) = self.single_block_base_exact()?;
            let blk = self.blocks[0]?;
            if blk.width < 2 || blk.height < 2 {
                return None;
            }
            if let cogzilla_codec::BandData::U8 { data, .. } = &blk.data {
                return Some((data.as_ref(), base_col, base_row, self.tile_w as usize));
            }
        }
        None
    }

    /// Single-block interleaved RGB8 view `(rgb_slice, channel, base_col, base_row, stride)`
    /// for clean chunky 3-band `uint8` COGs.
    #[inline(always)]
    pub fn single_block_rgb_u8(&self) -> Option<(&'a [u8], u8, u32, u32, usize)> {
        if self.grid_w == 1 && self.grid_h == 1 && self.no_nodata {
            let (base_col, base_row) = self.single_block_base_exact()?;
            let blk = self.blocks[0]?;
            if blk.width < 2 || blk.height < 2 {
                return None;
            }
            if let cogzilla_codec::BandData::RgbU8Channel { rgb, channel, .. } = &blk.data {
                return Some((
                    rgb.as_ref(),
                    *channel,
                    base_col,
                    base_row,
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
                if let Some(blk) = unsafe { *self.blocks.get_unchecked(bidx) }
                    && ly + 1 < blk.height && lx + 1 < blk.width {
                        let tw = self.tile_w as usize;
                        let p00 = (ly as usize) * tw + (lx as usize);
                        return Some((bidx, p00, p00 + tw));
                    }
            }
        }
        None
    }

    /// Locate a single in-bounds pixel `(col, row)` in the block grid, returning `(bidx, pidx)`.
    /// Used by aligned multi-band nearest-neighbor fast paths to compute the tile lookup once for
    /// all 3 RGB bands.
    #[inline(always)]
    pub fn pixel_loc(&self, col: u32, row: u32) -> Option<(usize, usize)> {
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
        if gx < self.grid_w && gy < self.grid_h {
            let bidx = (gy * self.grid_w + gx) as usize;
            if let Some(blk) = unsafe { *self.blocks.get_unchecked(bidx) }
                && ly < blk.height && lx < blk.width {
                    let pidx = (ly as usize) * (self.tile_w as usize) + (lx as usize);
                    return Some((bidx, pidx));
                }
        }
        None
    }

    /// The block holding the whole `w x h` pixel rectangle at `(x0, y0)` when it is in bounds and
    /// inside one decoded block: `(band, index_of_(x0, y0), row_stride)`. Raw block values: the
    /// caller applies `nodata_override` / `scale_offset` (or checks `no_nodata`).
    #[inline(always)]
    pub fn rect_in_block(&self, x0: i32, y0: i32, w: u32, h: u32) -> Option<(&'a DecodedBand, usize, usize)> {
        if w == 0
            || h == 0
            || x0 < 0
            || y0 < 0
            || x0 as i64 + w as i64 > self.ifd_w as i64
            || y0 as i64 + h as i64 > self.ifd_h as i64
        {
            return None;
        }
        let (ux, uy) = (x0 as u32, y0 as u32);
        let (tx, lx) = match self.shift_x {
            Some(s) => (ux >> s, ux & self.mask_x),
            None => (ux / self.tile_w, ux % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy >> s, uy & self.mask_y),
            None => (uy / self.tile_h, uy % self.tile_h),
        };
        if lx + w > self.tile_w || ly + h > self.tile_h {
            return None;
        }
        let gx = tx.wrapping_sub(self.tx0);
        let gy = ty.wrapping_sub(self.ty0);
        if gx >= self.grid_w || gy >= self.grid_h {
            return None;
        }
        let blk = unsafe { *self.blocks.get_unchecked((gy * self.grid_w + gx) as usize) }?;
        if lx + w > blk.width || ly + h > blk.height {
            return None;
        }
        let tw = self.tile_w as usize;
        Some((blk, ly as usize * tw + lx as usize, tw))
    }

    /// Zero-copy slice of `len` contiguous `f32` pixels starting at `(col0, row)` when the entire
    /// span lies within a single decoded `F32` block and has no nodata override or scale/offset.
    #[inline(always)]
    pub fn row_segment_f32_ref(&self, row: i32, col0: i32, len: usize) -> Option<&[f32]> {
        if self.nodata_override.is_some()
            || self.scale_offset.is_some()
            || row < 0
            || row >= self.ifd_h
            || col0 < 0
            || len == 0
        {
            return None;
        }
        let ux = col0 as u32;
        let uy = row as u32;
        let (tx, lx) = match self.shift_x {
            Some(s) => (ux >> s, ux & self.mask_x),
            None => (ux / self.tile_w, ux % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy >> s, uy & self.mask_y),
            None => (uy / self.tile_h, uy % self.tile_h),
        };
        let gx = tx.wrapping_sub(self.tx0);
        let gy = ty.wrapping_sub(self.ty0);
        if gx >= self.grid_w || gy >= self.grid_h {
            return None;
        }
        let b = unsafe { *self.blocks.get_unchecked((gy * self.grid_w + gx) as usize) }?;
        if ly >= b.height || (lx as usize) + len > b.width as usize {
            return None;
        }
        let sl = b.data.as_f32_slice()?;
        let base = ly as usize * self.tile_w as usize + lx as usize;
        sl.get(base..base + len)
    }

    /// Fill `out[i]` with pixel `(col0 + i, row)`, `NaN` outside the raster or the fetched blocks
    /// and for nodata (same values as [`Self::sample_checked`], read a block run at a time).
    pub fn read_row_segment(&self, row: i32, col0: i32, out: &mut [f32]) {
        if row < 0 || row >= self.ifd_h {
            out.fill(f32::NAN);
            return;
        }
        let uy = row as u32;
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy >> s, uy & self.mask_y),
            None => (uy / self.tile_h, uy % self.tile_h),
        };
        let gy = ty.wrapping_sub(self.ty0);
        let mut i = 0usize;
        while i < out.len() {
            let col = col0 + i as i32;
            if col < 0 || col >= self.ifd_w {
                out[i] = f32::NAN;
                i += 1;
                continue;
            }
            let ux = col as u32;
            let (tx, lx) = match self.shift_x {
                Some(s) => (ux >> s, ux & self.mask_x),
                None => (ux / self.tile_w, ux % self.tile_w),
            };
            let run = (out.len() - i)
                .min((self.tile_w - lx) as usize)
                .min((self.ifd_w - col) as usize);
            let dst = &mut out[i..i + run];
            let gx = tx.wrapping_sub(self.tx0);
            let blk = if gx < self.grid_w && gy < self.grid_h {
                unsafe { *self.blocks.get_unchecked((gy * self.grid_w + gx) as usize) }
            } else {
                None
            };
            match blk {
                Some(b) if ly < b.height => {
                    let base = ly as usize * self.tile_w as usize + lx as usize;
                    let valid = (b.width.saturating_sub(lx) as usize).min(run);
                    match b.data.as_f32_slice() {
                        Some(sl) if self.nodata_override.is_none() && self.scale_offset.is_none() => {
                            dst[..valid].copy_from_slice(&sl[base..base + valid]);
                        }
                        _ => {
                            for (k, d) in dst[..valid].iter_mut().enumerate() {
                                let mut v = unsafe { b.data.get_f32_unchecked(base + k) };
                                if let Some(nd) = self.nodata_override
                                    && v == nd
                                {
                                    v = f32::NAN;
                                }
                                if let Some((sc, off)) = self.scale_offset
                                    && !v.is_nan()
                                {
                                    v = v * sc + off;
                                }
                                *d = v;
                            }
                        }
                    }
                    dst[valid..].fill(f32::NAN);
                }
                _ => dst.fill(f32::NAN),
            }
            i += run;
        }
    }

    /// Fast pointer to the top-left `(x0 - 1, y0 - 1)` of a 4×4 Catmull-Rom cubic neighborhood
    /// when all 16 pixels live strictly inside a single clean (`no_nodata`) `f32` block.
    #[inline(always)]
    pub fn interior_4x4_f32(&self, x0: i32, y0: i32) -> Option<(*const f32, usize)> {
        if !self.no_nodata || x0 < 1 || y0 < 1 || x0 + 2 >= self.ifd_w || y0 + 2 >= self.ifd_h {
            return None;
        }
        let ux = x0 as u32;
        let uy = y0 as u32;
        let (tx, lx) = match self.shift_x {
            Some(s) => (ux >> s, ux & self.mask_x),
            None => (ux / self.tile_w, ux % self.tile_w),
        };
        let (ty, ly) = match self.shift_y {
            Some(s) => (uy >> s, uy & self.mask_y),
            None => (uy / self.tile_h, uy % self.tile_h),
        };
        if lx >= 1 && lx + 2 < self.tile_w && ly >= 1 && ly + 2 < self.tile_h {
            let gx = tx.wrapping_sub(self.tx0);
            let gy = ty.wrapping_sub(self.ty0);
            if gx < self.grid_w && gy < self.grid_h {
                let bidx = (gy * self.grid_w + gx) as usize;
                let blk = unsafe { *self.blocks.get_unchecked(bidx) }?;
                if ly + 2 >= blk.height || lx + 2 >= blk.width {
                    return None;
                }
                let sl = blk.data.as_f32_slice()?;
                let tw = self.tile_w as usize;
                let off = (ly as usize - 1) * tw + (lx as usize - 1);
                return Some((unsafe { sl.as_ptr().add(off) }, tw));
            }
        }
        None
    }
}

/// Copy a rectangular sub-region `(lx0..lx0+copy_w, ly0..ly0+copy_h)` from a decoded `BandData`
/// tile into a destination `f32` buffer at `(dst_x0, dst_y0)` with `dst_stride`, applying both
/// internal `GDAL_NODATA` and optional `nodata_override` + `(scale, offset)` using hoisted
/// dtype-specialized loops.
///
/// When `reject_non_finite` is `true`, `±Inf` values are masked to `f32::NAN` (used by zonal
/// statistics); otherwise `F32` tiles without overrides copy via fast `copy_from_slice`.
#[allow(clippy::too_many_arguments)]
pub fn copy_band_data_rect_to_f32(
    data: &cogzilla_codec::BandData,
    src_stride: usize,
    lx0: usize,
    ly0: usize,
    copy_w: usize,
    copy_h: usize,
    dst: &mut [f32],
    dst_stride: usize,
    dst_x0: usize,
    dst_y0: usize,
    nodata_override: Option<f32>,
    scale_offset: Option<(f32, f32)>,
    reject_non_finite: bool,
) {
    use cogzilla_codec::BandData;

    if copy_w == 0 || copy_h == 0 {
        return;
    }
    let nd_ovr = nodata_override.filter(|nd| !nd.is_nan());

    match data {
        BandData::F32(slab) => {
            let sdata = slab.as_slice();
            if nd_ovr.is_none() && scale_offset.is_none() && !reject_non_finite {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    dst[d0..d0 + copy_w].copy_from_slice(&sdata[s0..s0 + copy_w]);
                }
            } else {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let srow = &sdata[s0..s0 + copy_w];
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (d, &v) in drow.iter_mut().zip(srow) {
                        if reject_non_finite && !v.is_finite() {
                            *d = f32::NAN;
                            continue;
                        }
                        if let Some(nd) = nd_ovr
                            && v == nd
                        {
                            *d = f32::NAN;
                            continue;
                        }
                        *d = match scale_offset {
                            Some((sc, off)) if !v.is_nan() => v * sc + off,
                            _ => v,
                        };
                    }
                }
            }
        }
        BandData::F64(sdata) => {
            for dy in 0..copy_h {
                let s0 = (ly0 + dy) * src_stride + lx0;
                let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                let srow = &sdata[s0..s0 + copy_w];
                let drow = &mut dst[d0..d0 + copy_w];
                for (d, &v64) in drow.iter_mut().zip(srow) {
                    let v = v64 as f32;
                    if reject_non_finite && !v.is_finite() {
                        *d = f32::NAN;
                        continue;
                    }
                    if let Some(nd) = nd_ovr
                        && v == nd
                    {
                        *d = f32::NAN;
                        continue;
                    }
                    *d = match scale_offset {
                        Some((sc, off)) if !v.is_nan() => v * sc + off,
                        _ => v,
                    };
                }
            }
        }
        BandData::U16 { data: sdata, nodata } => {
            let nd_u16 = *nodata;
            if nd_u16.is_none() && nd_ovr.is_none() && scale_offset.is_none() {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let srow = &sdata[s0..s0 + copy_w];
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (d, &s) in drow.iter_mut().zip(srow) {
                        *d = s as f32;
                    }
                }
            } else {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let srow = &sdata[s0..s0 + copy_w];
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (d, &s) in drow.iter_mut().zip(srow) {
                        if nd_u16 == Some(s) {
                            *d = f32::NAN;
                            continue;
                        }
                        let v = s as f32;
                        if let Some(nd) = nd_ovr
                            && v == nd
                        {
                            *d = f32::NAN;
                            continue;
                        }
                        *d = match scale_offset {
                            Some((sc, off)) => v * sc + off,
                            None => v,
                        };
                    }
                }
            }
        }
        BandData::U8 { data: sdata, nodata } => {
            let nd_u8 = *nodata;
            if nd_u8.is_none() && nd_ovr.is_none() && scale_offset.is_none() {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let srow = &sdata[s0..s0 + copy_w];
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (d, &s) in drow.iter_mut().zip(srow) {
                        *d = s as f32;
                    }
                }
            } else {
                for dy in 0..copy_h {
                    let s0 = (ly0 + dy) * src_stride + lx0;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let srow = &sdata[s0..s0 + copy_w];
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (d, &s) in drow.iter_mut().zip(srow) {
                        if nd_u8 == Some(s) {
                            *d = f32::NAN;
                            continue;
                        }
                        let v = s as f32;
                        if let Some(nd) = nd_ovr
                            && v == nd
                        {
                            *d = f32::NAN;
                            continue;
                        }
                        *d = match scale_offset {
                            Some((sc, off)) => v * sc + off,
                            None => v,
                        };
                    }
                }
            }
        }
        BandData::RgbU8Channel {
            rgb,
            channel,
            nodata,
        } => {
            let ch = *channel as usize;
            let nd_u8 = *nodata;
            let raw: &[u8] = rgb;
            if nd_u8.is_none() && nd_ovr.is_none() && scale_offset.is_none() {
                for dy in 0..copy_h {
                    let s_base = ((ly0 + dy) * src_stride + lx0) * 3 + ch;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (i, d) in drow.iter_mut().enumerate() {
                        *d = raw[s_base + i * 3] as f32;
                    }
                }
            } else {
                for dy in 0..copy_h {
                    let s_base = ((ly0 + dy) * src_stride + lx0) * 3 + ch;
                    let d0 = (dst_y0 + dy) * dst_stride + dst_x0;
                    let drow = &mut dst[d0..d0 + copy_w];
                    for (i, d) in drow.iter_mut().enumerate() {
                        let s = raw[s_base + i * 3];
                        if nd_u8 == Some(s) {
                            *d = f32::NAN;
                            continue;
                        }
                        let v = s as f32;
                        if let Some(nd) = nd_ovr
                            && v == nd
                        {
                            *d = f32::NAN;
                            continue;
                        }
                        *d = match scale_offset {
                            Some((sc, off)) => v * sc + off,
                            None => v,
                        };
                    }
                }
            }
        }
    }
}

/// A band math result where every band it reads has a value: a number all the same, as
/// rio-tiler passes it through numpy's `nan_to_num` (zero for `0/0`, the largest number of
/// either sign for a division by zero), so only a missing input leaves a pixel missing.
pub fn band_math_nan_to_num(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else if v.is_nan() {
        0.0
    } else if v > 0.0 {
        f32::MAX
    } else {
        f32::MIN
    }
}

/// Double-precision twin of [`band_math_nan_to_num`] for `f64` band-math evaluators.
pub fn band_math_nan_to_num_f64(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else if v.is_nan() {
        0.0
    } else if v > 0.0 {
        f64::MAX
    } else {
        f64::MIN
    }
}

/// One band of band math, block by block: for a terrain algorithm (hillshade, slope, ...)
/// over an expression's first output, read like any one band. `inputs` are the blocks of
/// each band `prog` reads, in its slot order (`prog.used_bands`), each with the request's
/// `nodata` and the band's scale and offset; a pixel missing in any band is missing in the
/// result (`NaN`), as rio-tiler masks band math. Blocks missing in every band stay missing.
pub fn band_math_blocks(
    prog: &crate::expr_vm::ExprProgram,
    inputs: &[(&[Option<&DecodedBand>], Option<f32>, Option<(f32, f32)>)],
) -> Vec<Option<DecodedBand>> {
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats};
    let blocks = inputs.iter().map(|(b, _, _)| b.len()).max().unwrap_or(0);
    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); inputs.len()];
    let mut vals = vec![0.0f32; inputs.len()];
    (0..blocks)
        .map(|i| {
            let shape = inputs.iter().find_map(|(b, _, _)| b.get(i).copied().flatten()).map(|band| (band.width, band.height))?;
            let (w, h) = (shape.0 as usize, shape.1 as usize);
            let n = w * h;
            for (plane, (b, nodata, scale_offset)) in planes.iter_mut().zip(inputs) {
                plane.clear();
                plane.resize(n, f32::NAN);
                if let Some(band) = b.get(i).copied().flatten() {
                    let (bw, bh) = (band.width as usize, band.height as usize);
                    copy_band_data_rect_to_f32(&band.data, bw, 0, 0, bw.min(w), bh.min(h), plane, w, 0, 0, *nodata, *scale_offset, true);
                }
            }
            let mut out = AlignedF32Slab::new_uninit(n);
            let (mut min, mut max, mut any, mut all) = (f32::INFINITY, f32::NEG_INFINITY, false, true);
            for (px, o) in out.as_mut_slice().iter_mut().enumerate() {
                let mut missing = false;
                for (v, plane) in vals.iter_mut().zip(&planes) {
                    *v = plane[px];
                    missing |= v.is_nan();
                }
                *o = if missing { f32::NAN } else { band_math_nan_to_num(prog.eval(&vals)) };
                if o.is_nan() {
                    all = false;
                } else {
                    any = true;
                    min = min.min(*o);
                    max = max.max(*o);
                }
            }
            Some(DecodedBand {
                width: shape.0,
                height: shape.1,
                data: BandData::F32(out),
                stats: if any { BlockStats { all_nodata: false, no_nodata: all, min, max } } else { BlockStats::ALL_NODATA },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

    #[test]
    fn band_math_blocks_read_as_one_band() {
        let block = |vals: &[f32]| {
            let mut slab = AlignedF32Slab::new_uninit(vals.len());
            slab.as_mut_slice().copy_from_slice(vals);
            DecodedBand {
                width: 2,
                height: 2,
                data: BandData::F32(slab),
                stats: BlockStats { all_nodata: false, no_nodata: false, min: 0.0, max: 0.0 },
            }
        };
        let (a, b) = (block(&[1.0, 2.0, f32::NAN, 4.0]), block(&[10.0, 20.0, 30.0, 40.0]));
        let prog = crate::expr_vm::ExprProgram::compile("b1*2+b2").unwrap();
        let (ar, br) = ([Some(&a), None], [Some(&b), None]);
        let out = band_math_blocks(&prog, &[(&ar[..], None, None), (&br[..], None, Some((0.5, 1.0)))]);
        // b2 is unscaled first; a pixel missing in b1 is missing; a block missing in both stays so.
        let first = out[0].as_ref().unwrap();
        let BandData::F32(slab) = &first.data else { panic!("f32") };
        let v = slab.as_slice();
        assert_eq!((v[0], v[1], v[3]), (8.0, 15.0, 29.0));
        assert!(v[2].is_nan());
        assert!(!first.stats.all_nodata && !first.stats.no_nodata);
        assert!(out[1].is_none());
        // A request nodata masks the input's matching pixels.
        let out = band_math_blocks(&prog, &[(&ar[..], Some(2.0), None), (&br[..], None, None)]);
        let BandData::F32(slab) = &out[0].as_ref().unwrap().data else { panic!("f32") };
        assert!(slab.as_slice()[1].is_nan());
        // Arithmetic with no value to give over bands that have one is a number, as numpy's
        // `nan_to_num` makes it (0/0 and the root of a negative are 0); a missing band is not.
        for expr in ["(b1-1)/(b1-1)", "sqrt(1-b1)+1"] {
            let prog = crate::expr_vm::ExprProgram::compile(expr).unwrap();
            let out = band_math_blocks(&prog, &[(&ar[..], None, None)]);
            let BandData::F32(slab) = &out[0].as_ref().unwrap().data else { panic!("f32") };
            let v = slab.as_slice();
            assert_eq!((v[0], v[1]), if expr.starts_with("sqrt") { (1.0, 0.0) } else { (0.0, 1.0) }, "{expr}");
            assert!(v[2].is_nan(), "{expr}");
        }
    }

    #[test]
    fn band_source_scale_offset_applies_consistently_and_disables_raw_fastpaths() {
        let mut slab = AlignedF32Slab::new_uninit(4);
        slab.as_mut_slice().copy_from_slice(&[100.0, 200.0, f32::NAN, 400.0]);
        let band = DecodedBand {
            width: 2,
            height: 2,
            data: BandData::F32(slab),
            stats: BlockStats {
                all_nodata: false,
                no_nodata: false,
                min: 100.0,
                max: 400.0,
            },
        };
        let blocks = [Some(&band)];
        let src = BandSource::new(2, 2, 2, 2, 0, 0, 1, 1, &blocks).with_scale_offset(0.01, -0.5);
        assert!(!src.no_nodata);
        assert!(src.single_block_flat().is_none());
        assert!((src.sample_checked(0, 0) - 0.5).abs() < 1e-6);
        assert!((src.sample_inbounds(1, 0) - 1.5).abs() < 1e-6);
        assert!(src.sample_checked(0, 1).is_nan());
        let (a, b, c, d) = src.sample_quad_inbounds(0, 0);
        assert!((a - 0.5).abs() < 1e-6);
        assert!((b - 1.5).abs() < 1e-6);
        assert!(c.is_nan());
        assert!((d - 3.5).abs() < 1e-6);

        // Identity scale/offset (1.0, 0.0) must remain a no-op so fast paths stay active.
        let src_id = BandSource::new(2, 2, 2, 2, 0, 0, 1, 1, &blocks).with_scale_offset(1.0, 0.0);
        assert!(src_id.scale_offset.is_none());
        assert!(src_id.single_block_flat().is_some());

        // Clean (`no_nodata == true`) U8/U16/RgbU8 blocks with GDAL_NODATA metadata (e.g. Some(0))
        // must still unlock single_block_* SIMD fast paths!
        let u16_band = DecodedBand {
            width: 2,
            height: 2,
            data: BandData::U16 {
                data: vec![10u16, 20, 30, 40].into_boxed_slice(),
                nodata: Some(0),
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 10.0,
                max: 40.0,
            },
        };
        let u16_blocks = [Some(&u16_band)];
        let src_u16 = BandSource::new(2, 2, 2, 2, 0, 0, 1, 1, &u16_blocks);
        assert!(src_u16.single_block_u16().is_some());

        // nodata_override must disable single_block_flat and apply in sample_quad_inbounds.
        let src_nd = BandSource::new(2, 2, 2, 2, 0, 0, 1, 1, &blocks).with_nodata_override(Some(200.0));
        assert!(src_nd.single_block_flat().is_none());
        let (a, b, c, d) = src_nd.sample_quad_inbounds(0, 0);
        assert_eq!(a, 100.0);
        assert!(b.is_nan());
        assert!(c.is_nan());
        assert_eq!(d, 400.0);
    }

    #[test]
    fn single_block_fast_paths_reject_f32_inexact_base_coords() {
        let mut slab = AlignedF32Slab::new_uninit(4);
        slab.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let band = DecodedBand {
            width: 2,
            height: 2,
            data: BandData::F32(slab),
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 1.0,
                max: 4.0,
            },
        };
        let blocks = [Some(&band)];
        // base_col = 263_173 * 255 = 67_109_115 > 2^24, not exactly representable in f32.
        let src_inexact = BandSource::new(
            70_000_000,
            255,
            255,
            255,
            263_173,
            0,
            1,
            1,
            &blocks,
        );
        assert!(src_inexact.single_block_flat().is_none());

        // Power-of-two aligned base_col (262_144 * 256 = 67_108_864 = 2^26) IS exact in f32.
        let src_exact = BandSource::new(
            70_000_000,
            256,
            256,
            256,
            262_144,
            0,
            1,
            1,
            &blocks,
        );
        assert!(src_exact.single_block_flat().is_some());
    }

    #[test]
    fn sample_extent_helpers_pin_inclusive_half_pixel_bounds() {
        let blocks: [Option<&DecodedBand>; 0] = [];
        let src = BandSource::new(10, 20, 16, 16, 0, 0, 0, 0, &blocks);

        // Inclusive edges: [-0.5, dim - 0.5]
        assert!(src.contains_sample(-0.5, -0.5));
        assert!(src.contains_sample(9.5, 19.5));
        assert!(src.contains_sample(0.0, 0.0));
        assert!(src.contains_sample_f64(-0.5, -0.5));
        assert!(src.contains_sample_f64(9.5, 19.5));

        // Strictly outside by 1 ulp / small epsilon
        assert!(!src.contains_sample(-0.5001, 0.0));
        assert!(!src.contains_sample(0.0, -0.5001));
        assert!(!src.contains_sample(9.5001, 0.0));
        assert!(!src.contains_sample(0.0, 19.5001));
        assert!(!src.contains_sample(10.0, 0.0));
        assert!(!src.contains_sample_f64(-0.5000001, 0.0));
        assert!(!src.contains_sample_f64(9.5000001, 0.0));
        assert!(!src.contains_sample_f64(0.0, 19.5000001));

        // Non-finite coordinates are always outside
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!src.contains_sample(bad, 0.0));
            assert!(!src.contains_sample(0.0, bad));
            assert!(!BandSource::axis_contains_sample(bad, 10));
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!src.contains_sample_f64(bad, 0.0));
            assert!(!src.contains_sample_f64(0.0, bad));
            assert!(!BandSource::axis_contains_sample_f64(bad, 10));
        }

        // Static dim helpers match BandSource methods
        assert!(BandSource::dims_contain_sample(10, 20, 9.5, 19.5));
        assert!(!BandSource::dims_contain_sample(10, 20, 9.6, 19.5));
        assert!(!BandSource::axis_contains_sample(0.0, 0));
    }
}
