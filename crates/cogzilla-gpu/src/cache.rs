use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use crate::context::GpuError;
use crate::expr::GpuMultiBandSlab;
use crate::resample::GpuSlab;

/// Cache key identifying a decoded raster block/slab resident on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabKey {
    pub dataset_id: u64,
    pub ifd_idx: u16,
    pub band: u16,
    pub tx: u32,
    pub ty: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuCacheStats {
    pub entries: usize,
    pub used_bytes: usize,
    pub max_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct LruEntry<V> {
    value: Arc<V>,
    bytes: usize,
    last_used: u64,
}

struct LruInner<K, V> {
    map: HashMap<K, LruEntry<V>>,
    used_bytes: usize,
    max_bytes: usize,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

struct LruByteCache<K, V> {
    inner: Mutex<LruInner<K, V>>,
}

impl<K: Eq + Hash + Copy, V> LruByteCache<K, V> {
    fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(LruInner {
                map: HashMap::new(),
                used_bytes: 0,
                max_bytes: max_bytes.max(1024 * 1024),
                tick: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }

    fn stats(&self) -> GpuCacheStats {
        let g = self.inner.lock().expect("gpu LRU cache mutex poisoned");
        GpuCacheStats {
            entries: g.map.len(),
            used_bytes: g.used_bytes,
            max_bytes: g.max_bytes,
            hits: g.hits,
            misses: g.misses,
            evictions: g.evictions,
        }
    }

    fn get(&self, key: &K) -> Option<Arc<V>> {
        let mut g = self.inner.lock().expect("gpu LRU cache mutex poisoned");
        let next_tick = g.tick.wrapping_add(1);
        g.tick = next_tick;
        if let Some(entry) = g.map.get_mut(key) {
            entry.last_used = next_tick;
            let value = Arc::clone(&entry.value);
            g.hits += 1;
            Some(value)
        } else {
            g.misses += 1;
            None
        }
    }

    fn insert(&self, key: K, value: Arc<V>, bytes: usize) {
        let mut g = self.inner.lock().expect("gpu LRU cache mutex poisoned");
        if bytes > g.max_bytes {
            return;
        }
        if let Some(old) = g.map.remove(&key) {
            g.used_bytes = g.used_bytes.saturating_sub(old.bytes);
        }
        while g.used_bytes.saturating_add(bytes) > g.max_bytes && !g.map.is_empty() {
            let lru_key = g
                .map
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            if let Some(ek) = lru_key
                && let Some(removed) = g.map.remove(&ek)
            {
                g.used_bytes = g.used_bytes.saturating_sub(removed.bytes);
                g.evictions += 1;
            } else {
                break;
            }
        }
        let next_tick = g.tick.wrapping_add(1);
        g.tick = next_tick;
        g.used_bytes = g.used_bytes.saturating_add(bytes);
        g.map.insert(
            key,
            LruEntry {
                value,
                bytes,
                last_used: next_tick,
            },
        );
    }

    fn clear(&self) {
        let mut g = self.inner.lock().expect("gpu LRU cache mutex poisoned");
        g.map.clear();
        g.used_bytes = 0;
    }
}

/// Thread-safe byte-bounded LRU cache of GPU-resident `GpuSlab` buffers.
///
/// Uploading a decoded COG block once and keeping it resident in VRAM / unified memory
/// eliminates host->device copy overhead across overlapping tiles, zoom levels, and styling changes.
pub struct GpuSlabCache {
    inner: LruByteCache<SlabKey, GpuSlab>,
}

impl GpuSlabCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: LruByteCache::new(max_bytes),
        }
    }

    pub fn stats(&self) -> GpuCacheStats {
        self.inner.stats()
    }

    pub fn get(&self, key: &SlabKey) -> Option<Arc<GpuSlab>> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: SlabKey, slab: Arc<GpuSlab>) {
        let bytes = (slab.height as usize)
            .saturating_mul(slab.stride as usize)
            .saturating_mul(std::mem::size_of::<f32>());
        self.inner.insert(key, slab, bytes);
    }

    /// Look up `key`, or upload `data` on miss and cache the resulting `GpuSlab`.
    pub fn get_or_upload(
        &self,
        key: SlabKey,
        data: &[f32],
        width: u32,
        height: u32,
        stride: u32,
        device: &wgpu::Device,
    ) -> Result<Arc<GpuSlab>, GpuError> {
        if let Some(hit) = self.get(&key) {
            return Ok(hit);
        }
        let slab = Arc::new(GpuSlab::upload(device, data, width, height, stride)?);
        self.insert(key, Arc::clone(&slab));
        Ok(slab)
    }

    pub fn clear(&self) {
        self.inner.clear();
    }
}

/// Cache key identifying a stacked multi-band `GpuMultiBandSlab` resident on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiBandSlabKey {
    pub dataset_id: u64,
    pub ifd_idx: u16,
    pub num_bands: u8,
    pub bands: [u16; 8],
    pub tx: u32,
    pub ty: u32,
}

/// Thread-safe byte-bounded LRU cache of GPU-resident `GpuMultiBandSlab` buffers.
///
/// Eliminates repeated CPU multi-band stitching and host->device PCIe uploads across warm
/// `CustomExpr` (NDVI/EVI/SAR) and 3-band `RGB` tile and `/cog/{preview,bbox}` requests.
pub struct GpuMultiBandCache {
    inner: LruByteCache<MultiBandSlabKey, GpuMultiBandSlab>,
}

impl GpuMultiBandCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: LruByteCache::new(max_bytes),
        }
    }

    pub fn stats(&self) -> GpuCacheStats {
        self.inner.stats()
    }

    pub fn get(&self, key: &MultiBandSlabKey) -> Option<Arc<GpuMultiBandSlab>> {
        self.inner.get(key)
    }

    pub fn insert(&self, key: MultiBandSlabKey, slab: Arc<GpuMultiBandSlab>) {
        let bytes = (slab.num_bands as usize)
            .saturating_mul(slab.band_stride as usize)
            .saturating_mul(std::mem::size_of::<f32>());
        self.inner.insert(key, slab, bytes);
    }

    pub fn clear(&self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuContext;

    #[test]
    fn gpu_slab_cache_lru_eviction_and_stats() {
        let ctx = match GpuContext::new_default() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping GPU cache test (no adapter): {e}");
                return;
            }
        };
        // Each 512x512 f32 slab is 1 MiB; cap at 2 MiB so 3rd insert evicts LRU.
        let cache = GpuSlabCache::new(2 * 1024 * 1024);
        let data = vec![1.0f32; 512 * 512];

        let k0 = SlabKey { dataset_id: 1, ifd_idx: 0, band: 0, tx: 0, ty: 0 };
        let k1 = SlabKey { dataset_id: 1, ifd_idx: 0, band: 0, tx: 1, ty: 0 };
        let k2 = SlabKey { dataset_id: 1, ifd_idx: 0, band: 0, tx: 2, ty: 0 };

        let _ = cache.get_or_upload(k0, &data, 512, 512, 512, &ctx.device).unwrap();
        let _ = cache.get_or_upload(k1, &data, 512, 512, 512, &ctx.device).unwrap();
        // Touch k0 so k1 becomes LRU
        assert!(cache.get(&k0).is_some());
        let _ = cache.get_or_upload(k2, &data, 512, 512, 512, &ctx.device).unwrap();

        let st = cache.stats();
        assert_eq!(st.entries, 2);
        assert_eq!(st.evictions, 1);
        assert_eq!(st.hits, 1);
        assert_eq!(st.misses, 3);
        assert!(cache.get(&k0).is_some());
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
    }
}
