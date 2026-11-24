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
        if let Some(nd) = nodata {
            if !nd.is_nan() {
                self.nodata_override = Some(nd);
                self.no_nodata = false;
            }
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
        if let Some(nd) = self.nodata_override {
            if v == nd {
                return f32::NAN;
            }
        }
        if let Some((sc, off)) = self.scale_offset {
            if !v.is_nan() {
                return v * sc + off;
            }
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
                if let Some(blk) = unsafe { *self.blocks.get_unchecked(bidx) } {
                    if ly + 1 < blk.height && lx + 1 < blk.width {
                        let tw = self.tile_w as usize;
                        let p00 = (ly as usize) * tw + (lx as usize);
                        return Some((bidx, p00, p00 + tw));
                    }
                }
            }
        }
        None
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

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

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
}
