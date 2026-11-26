use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

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

/// Minimum slab size (2 MiB) for requesting Linux Transparent Huge Pages (`MADV_HUGEPAGE`).
const HUGE_SLAB_BYTES: usize = 2 * 1024 * 1024;

#[inline]
fn slab_layout(len: usize) -> Layout {
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .expect("AlignedF32Slab size overflow");
    // Page-align large slabs (>= 2 MiB) so `madvise(MADV_HUGEPAGE)` on Linux is guaranteed
    // page-aligned, reducing TLB pressure in multi-GiB block caches.
    let align = if bytes >= HUGE_SLAB_BYTES { 4096 } else { 64 };
    Layout::from_size_align(bytes, align).expect("valid aligned f32 layout")
}

#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
unsafe fn advise_hugepage(ptr: *mut f32, bytes: usize) {
    // MADV_HUGEPAGE = 14 on Linux x86_64 and aarch64.
    const MADV_HUGEPAGE: std::ffi::c_int = 14;
    unsafe extern "C" {
        fn madvise(
            addr: *mut std::ffi::c_void,
            length: usize,
            advice: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
    if bytes >= HUGE_SLAB_BYTES {
        let _ = unsafe { madvise(ptr as *mut std::ffi::c_void, bytes, MADV_HUGEPAGE) };
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
        let layout = slab_layout(len);
        // SAFETY: layout size > 0 and align is power of two; zero-initializing ensures
        // subsequent `&[f32]`/`&mut [f32]` slice views are always valid initialized memory.
        // On Linux (x86_64/aarch64) for >= 2 MiB slabs, allocate uninitialized first and call
        // `madvise(MADV_HUGEPAGE)` *before* zeroing so the first-touch page faults allocate
        // 2 MiB transparent huge pages under `madvise` mode instead of 4 KiB pages.
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        let ptr = if layout.size() >= HUGE_SLAB_BYTES {
            let raw = unsafe { std::alloc::alloc(layout) as *mut f32 };
            let ptr = NonNull::new(raw).expect("OOM allocating AlignedF32Slab");
            unsafe {
                advise_hugepage(raw, layout.size());
                std::ptr::write_bytes(raw, 0, len);
            }
            ptr
        } else {
            let raw = unsafe { alloc_zeroed(layout) as *mut f32 };
            NonNull::new(raw).expect("OOM allocating AlignedF32Slab")
        };
        #[cfg(not(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        let ptr = {
            let raw = unsafe { alloc_zeroed(layout) as *mut f32 };
            NonNull::new(raw).expect("OOM allocating AlignedF32Slab")
        };
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
            let layout = slab_layout(self.len);
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
    /// Zero-deinterleave view into a shared interleaved RGB8 (`[R, G, B, ...]`) block
    /// for chunky 3-band `uint8` COGs (Sentinel-2 TCI, NAIP, aerial JPEG/WebP RGB).
    RgbU8Channel {
        rgb: Arc<[u8]>,
        channel: u8,
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
            Self::RgbU8Channel { rgb, .. } => rgb.len() / 3,
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resident heap byte size of this band (1 B/px for `U8`/`RgbU8Channel`, 2 B/px for `U16`, 4 B/px for `F32`).
    #[inline(always)]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::F32(s) => s.len() * 4,
            Self::U16 { data, .. } => data.len() * 2,
            Self::U8 { data, .. } => data.len(),
            Self::RgbU8Channel { rgb, .. } => rgb.len() / 3,
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
            Self::RgbU8Channel {
                rgb,
                channel,
                nodata,
            } => {
                let v = unsafe { *rgb.get_unchecked(idx * 3 + (*channel as usize)) };
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
            Self::RgbU8Channel {
                rgb,
                channel,
                nodata,
            } => {
                let v = rgb[idx * 3 + (*channel as usize)];
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

    #[test]
    fn chunky_3band_u8_uses_shared_interleaved_rgb_slab() {
        let raw: Vec<u8> = (0..24u8).collect();
        let bands = crate::predictor::reconstruct_u8(&raw, 4, 2, 3, false, None);
        assert_eq!(bands.len(), 3);
        let ptrs: Vec<*const u8> = bands
            .iter()
            .enumerate()
            .map(|(ch, b)| match &b.data {
                BandData::RgbU8Channel { rgb, channel, .. } => {
                    assert_eq!(*channel as usize, ch);
                    assert_eq!(b.data.byte_len(), 8);
                    rgb.as_ptr()
                }
                _ => panic!("expected BandData::RgbU8Channel"),
            })
            .collect();
        assert_eq!(ptrs[0], ptrs[1]);
        assert_eq!(ptrs[1], ptrs[2]);
        assert_eq!(bands[0].data.get_f32(3), 9.0);
        assert_eq!(bands[1].data.get_f32(3), 10.0);
        assert_eq!(bands[2].data.get_f32(3), 11.0);

        let p2 = vec![
            10, 20, 30, 1, 2, 3, 1, 2, 3, 1, 2, 3,
            5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7,
        ];
        let b_p2 = crate::predictor::reconstruct_u8(&p2, 4, 2, 3, true, None);
        assert_eq!(b_p2[0].data.get_f32(3), 13.0);
        assert_eq!(b_p2[1].data.get_f32(3), 26.0);
        assert_eq!(b_p2[2].data.get_f32(3), 39.0);
        assert_eq!(b_p2[0].data.get_f32(5), 10.0);
    }

    #[test]
    #[should_panic(expected = "AlignedF32Slab size overflow")]
    fn slab_new_uninit_panics_on_size_overflow_wrap_to_zero() {
        let _ = AlignedF32Slab::new_uninit(usize::MAX / std::mem::size_of::<f32>() + 1);
    }

    #[test]
    #[should_panic(expected = "AlignedF32Slab size overflow")]
    fn slab_new_uninit_panics_on_size_overflow_small_wrap() {
        let _ = AlignedF32Slab::new_uninit(usize::MAX / std::mem::size_of::<f32>() + 2);
    }

    #[test]
    fn slab_hugepage_size_is_zero_initialized_and_page_aligned() {
        let len = HUGE_SLAB_BYTES / std::mem::size_of::<f32>();
        let slab = AlignedF32Slab::new_uninit(len);
        assert_eq!(slab.len(), len);
        assert_eq!((slab.as_slice().as_ptr() as usize) % 4096, 0);
        assert!(slab.as_slice().iter().all(|&v| v == 0.0));
    }
}
