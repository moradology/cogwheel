use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

struct SendF32Ptr(*mut f32);
unsafe impl Send for SendF32Ptr {}

struct SlabPools {
    /// 256×256 f32 (256 KiB each, max 16 = 4 MiB)
    p256: Mutex<Vec<SendF32Ptr>>,
    /// 512×512 f32 (1 MiB each, max 16 = 16 MiB)
    p512: Mutex<Vec<SendF32Ptr>>,
    /// 1024×1024 f32 (4 MiB each, max 4 = 16 MiB)
    p1024: Mutex<Vec<SendF32Ptr>>,
}

fn slab_pools() -> &'static SlabPools {
    static POOLS: OnceLock<SlabPools> = OnceLock::new();
    POOLS.get_or_init(|| SlabPools {
        p256: Mutex::new(Vec::with_capacity(16)),
        p512: Mutex::new(Vec::with_capacity(16)),
        p1024: Mutex::new(Vec::with_capacity(4)),
    })
}

#[inline]
fn pool_for_len(len: usize) -> Option<(&'static Mutex<Vec<SendF32Ptr>>, usize)> {
    match len {
        65_536 => Some((&slab_pools().p256, 16)),
        262_144 => Some((&slab_pools().p512, 16)),
        1_048_576 => Some((&slab_pools().p1024, 4)),
        _ => None,
    }
}

/// 64-byte-aligned heap-allocated contiguous `f32` slab for SIMD-friendly tile storage.
/// Recycles standard COG block sizes (`256×256`, `512×512`, `1024×1024`) through a bounded
/// free-list pool to avoid repeated `mmap`/`munmap` zero-page faults under cold-cache churn.
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
        if let Some((pool, _)) = pool_for_len(len) {
            if let Ok(mut guard) = pool.lock() {
                if let Some(SendF32Ptr(raw)) = guard.pop() {
                    return Self {
                        ptr: unsafe { NonNull::new_unchecked(raw) },
                        len,
                    };
                }
            }
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
            if let Some((pool, cap)) = pool_for_len(self.len) {
                if let Ok(mut guard) = pool.lock() {
                    if guard.len() < cap {
                        guard.push(SendF32Ptr(self.ptr.as_ptr()));
                        return;
                    }
                }
            }
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

/// Compact planar tile storage: keeps `uint8` (Sentinel-2 TCI, visual RGB) at 1 byte/px
/// and `uint16` (Landsat C2, Sentinel-2 L2A) at 2 bytes/px, while storing `f32` and other
/// numeric types in 64-byte-aligned `AlignedF32Slab`.
pub enum BandData {
    F32(AlignedF32Slab),
    U16 {
        data: Box<[u16]>,
        nodata: Option<u16>,
    },
    U8 {
        data: Box<[u8]>,
        nodata: Option<u8>,
    },
}

impl From<AlignedF32Slab> for BandData {
    #[inline(always)]
    fn from(slab: AlignedF32Slab) -> Self {
        Self::F32(slab)
    }
}

impl BandData {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::U16 { data, .. } => data.len(),
            Self::U8 { data, .. } => data.len(),
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resident heap byte size of this band (1 B/px for `U8`, 2 B/px for `U16`, 4 B/px for `F32`).
    #[inline(always)]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::F32(s) => s.len() * 4,
            Self::U16 { data, .. } => data.len() * 2,
            Self::U8 { data, .. } => data.len(),
        }
    }

    /// Borrow the underlying `f32` slice if this band is stored as `F32`.
    #[inline(always)]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::F32(s) => Some(s.as_slice()),
            _ => None,
        }
    }

    /// Read pixel `idx` as `f32` without bounds checking (`nodata` normalized to `f32::NAN`).
    ///
    /// # Safety
    /// `idx` must be `< self.len()`.
    #[inline(always)]
    pub unsafe fn get_f32_unchecked(&self, idx: usize) -> f32 {
        match self {
            Self::F32(s) => unsafe { *s.as_slice().get_unchecked(idx) },
            Self::U16 { data, nodata } => {
                let v = unsafe { *data.get_unchecked(idx) };
                if *nodata == Some(v) {
                    f32::NAN
                } else {
                    v as f32
                }
            }
            Self::U8 { data, nodata } => {
                let v = unsafe { *data.get_unchecked(idx) };
                if *nodata == Some(v) {
                    f32::NAN
                } else {
                    v as f32
                }
            }
        }
    }

    /// Bounds-checked read of pixel `idx` as `f32` (`nodata` normalized to `f32::NAN`).
    #[inline(always)]
    pub fn get_f32(&self, idx: usize) -> f32 {
        match self {
            Self::F32(s) => s.as_slice()[idx],
            Self::U16 { data, nodata } => {
                let v = data[idx];
                if *nodata == Some(v) {
                    f32::NAN
                } else {
                    v as f32
                }
            }
            Self::U8 { data, nodata } => {
                let v = data[idx];
                if *nodata == Some(v) {
                    f32::NAN
                } else {
                    v as f32
                }
            }
        }
    }

    /// Materialize this band into a mutable `AlignedF32Slab` (in-place if already `F32`,
    /// promoting `U8`/`U16` when arbitrary per-pixel `NaN` masking is needed).
    pub fn make_f32_mut(&mut self) -> &mut [f32] {
        if !matches!(self, Self::F32(_)) {
            let n = self.len();
            let mut slab = AlignedF32Slab::new_uninit(n);
            let dst = slab.as_mut_slice();
            for (i, d) in dst.iter_mut().enumerate() {
                *d = unsafe { self.get_f32_unchecked(i) };
            }
            *self = Self::F32(slab);
        }
        match self {
            Self::F32(s) => s.as_mut_slice(),
            _ => unreachable!(),
        }
    }
}

/// Single decoded planar tile band, with nodata values normalized to canonical `f32::NAN` on read.
pub struct DecodedBand {
    pub data: BandData,
    pub width: u32,
    pub height: u32,
    pub stats: BlockStats,
}

/// Decoded COG internal tile (1 band for single-band/planar COGs, N bands for chunky multi-band COGs).
pub struct DecodedTile {
    pub bands: Vec<DecodedBand>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::{reconstruct_u16, reconstruct_u8};
    use cogzilla_tiff::ByteOrder;

    #[test]
    fn u16_and_u8_slabs_store_compact_bytes_with_exact_nodata() {
        // 16x2 u16 tile with Predictor 2 (horizontal diff): row 0 = [0, 10, 20, ...], nodata = 0
        let w = 16u32;
        let h = 2u32;
        let mut raw_u16 = Vec::new();
        for y in 0..h {
            let mut prev = 0u16;
            for x in 0..w {
                let val = if x == 0 && y == 0 { 0u16 } else { (y * 100 + x * 10) as u16 };
                let diff = val.wrapping_sub(prev);
                raw_u16.extend_from_slice(&diff.to_le_bytes());
                prev = val;
            }
        }
        let bands = reconstruct_u16(&raw_u16, w, h, 1, ByteOrder::Little, true, Some(0.0));
        assert_eq!(bands.len(), 1);
        assert!(matches!(bands[0].data, BandData::U16 { .. }));
        assert_eq!(bands[0].data.byte_len(), (w * h * 2) as usize); // 2 B/px (2x less than f32)
        assert!(bands[0].data.get_f32(0).is_nan());
        assert_eq!(bands[0].data.get_f32(1), 10.0);
        assert_eq!(bands[0].data.get_f32(16), 100.0);
        assert!(!bands[0].stats.no_nodata);
        assert_eq!(bands[0].stats.min, 10.0);
        assert_eq!(bands[0].stats.max, 250.0);

        // 16x2 u8 tile with nodata = 255
        let mut raw_u8 = vec![42u8; (w * h) as usize];
        raw_u8[0] = 255;
        let bands8 = reconstruct_u8(&raw_u8, w, h, 1, false, Some(255.0));
        assert!(matches!(bands8[0].data, BandData::U8 { .. }));
        assert_eq!(bands8[0].data.byte_len(), (w * h) as usize); // 1 B/px (4x less than f32)
        assert!(bands8[0].data.get_f32(0).is_nan());
        assert_eq!(bands8[0].data.get_f32(1), 42.0);
        assert_eq!(bands8[0].stats.min, 42.0);
        assert_eq!(bands8[0].stats.max, 42.0);
    }

    #[test]
    fn slab_arena_recycles_standard_block_allocations() {
        let s1 = AlignedF32Slab::new_uninit(256 * 256);
        let p1 = s1.as_slice().as_ptr();
        drop(s1);
        let s2 = AlignedF32Slab::new_uninit(256 * 256);
        let p2 = s2.as_slice().as_ptr();
        assert_eq!(p1, p2, "expected 256x256 slab pointer to be recycled from free-list");
    }
}
