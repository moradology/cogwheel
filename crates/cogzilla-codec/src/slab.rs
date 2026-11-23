use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// 64-byte-aligned heap-allocated contiguous `f32` slab for SIMD-friendly tile storage.
pub struct AlignedF32Slab {
    ptr: NonNull<f32>,
    len: usize,
}

unsafe impl Send for AlignedF32Slab {}
unsafe impl Sync for AlignedF32Slab {}

impl AlignedF32Slab {
    pub fn new_uninit(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), 64)
            .expect("valid aligned f32 layout");
        // SAFETY: layout size > 0 and align is power of two.
        let raw = unsafe { alloc(layout) as *mut f32 };
        let ptr = NonNull::new(raw).expect("OOM allocating AlignedF32Slab");
        Self { ptr, len }
    }

    pub fn filled(len: usize, val: f32) -> Self {
        let mut s = Self::new_uninit(len);
        s.as_mut_slice().fill(val);
        s
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl Deref for AlignedF32Slab {
    type Target = [f32];
    #[inline(always)]
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl DerefMut for AlignedF32Slab {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl Drop for AlignedF32Slab {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = Layout::from_size_align(self.len * std::mem::size_of::<f32>(), 64)
                .expect("valid aligned f32 layout");
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

/// Summary statistics computed in registers during block decode / unshuffling.
/// Enables `<200ns` empty-tile short-circuiting and branchless `CleanFastPath` kernels.
#[derive(Debug, Clone, Copy)]
pub struct BlockStats {
    pub all_nodata: bool,
    pub no_nodata: bool,
    pub min: f32,
    pub max: f32,
}

impl BlockStats {
    pub const ALL_NODATA: Self = Self {
        all_nodata: true,
        no_nodata: false,
        min: f32::NAN,
        max: f32::NAN,
    };
}

/// Single decoded planar `f32` tile band, with nodata values normalized to canonical `f32::NAN`.
pub struct DecodedBand {
    pub data: AlignedF32Slab,
    pub width: u32,
    pub height: u32,
    pub stats: BlockStats,
}

/// Decoded COG internal tile (1 band for single-band/planar COGs, N bands for chunky multi-band COGs).
pub struct DecodedTile {
    pub bands: Vec<DecodedBand>,
}
