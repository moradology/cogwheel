use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use cogwheel_codec::DecodedTile;
use dashmap::DashMap;
use tokio::sync::Notify;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub dataset_id: u64,
    pub ifd_idx: u16,
    pub tile_idx: u32,
}

enum CacheEntry {
    Ready(Arc<DecodedTile>),
    InFlight(Arc<Notify>),
}

/// Sharded lock-free singleflight cache for decoded `AlignedF32Slab` COG blocks.
/// Deduplicates concurrent misses across parallel HTTP tile requests (`BM-07: PanBurst32`).
pub struct SingleflightBlockCache {
    map: DashMap<BlockKey, CacheEntry, ahash::RandomState>,
    current_bytes: AtomicUsize,
    max_bytes: usize,
    evicting: AtomicBool,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub coalesced_waiters: AtomicU64,
}

pub enum LookupResult {
    Hit(Arc<DecodedTile>),
    /// Caller owns the miss and MUST call `complete(key, tile)` or `abort(key)` when done!
    Owner,
    /// Another worker is currently fetching/decoding this block; await this Notify and retry.
    Wait(Arc<Notify>),
}

impl SingleflightBlockCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            map: DashMap::with_capacity_and_hasher(2048, ahash::RandomState::new()),
            current_bytes: AtomicUsize::new(0),
            max_bytes,
            evicting: AtomicBool::new(false),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            coalesced_waiters: AtomicU64::new(0),
        }
    }

    /// Check current state of `key`: `Some(Ok(tile))` if ready, `Some(Err(current_notify))` if
    /// in-flight (returning the live owner's `Notify`), or `None` if removed/aborted.
    #[inline(always)]
    pub fn peek_state(&self, key: &BlockKey) -> Option<Result<Arc<DecodedTile>, Arc<Notify>>> {
        self.map.get(key).map(|entry| match entry.value() {
            CacheEntry::Ready(tile) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Ok(tile.clone())
            }
            CacheEntry::InFlight(n) => Err(n.clone()),
        })
    }

    #[inline(always)]
    pub fn try_get_fast(&self, key: &BlockKey) -> Option<Arc<DecodedTile>> {
        if let Some(entry) = self.map.get(key) {
            if let CacheEntry::Ready(tile) = entry.value() {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(tile.clone());
            }
        }
        None
    }

    /// Acquire either a cached tile, singleflight ownership to fetch it, or a waiter handle.
    pub fn acquire(&self, key: BlockKey) -> LookupResult {
        if let Some(tile) = self.try_get_fast(&key) {
            return LookupResult::Hit(tile);
        }

        use dashmap::mapref::entry::Entry;
        match self.map.entry(key) {
            Entry::Occupied(occ) => match occ.get() {
                CacheEntry::Ready(tile) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    LookupResult::Hit(tile.clone())
                }
                CacheEntry::InFlight(notify) => {
                    self.coalesced_waiters.fetch_add(1, Ordering::Relaxed);
                    LookupResult::Wait(notify.clone())
                }
            },
            Entry::Vacant(vac) => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                vac.insert(CacheEntry::InFlight(Arc::new(Notify::new())));
                LookupResult::Owner
            }
        }
    }

    /// Complete an in-flight fetch, storing the decoded tile and waking all singleflight waiters.
    pub fn complete(&self, key: BlockKey, tile: Arc<DecodedTile>) {
        let tile_bytes: usize = tile.bands.iter().map(|b| b.data.len() * 4).sum();
        let notify = match self.map.insert(key, CacheEntry::Ready(tile)) {
            Some(CacheEntry::InFlight(n)) => Some(n),
            _ => None,
        };
        if let Some(n) = notify {
            n.notify_waiters();
        }
        let prev = self.current_bytes.fetch_add(tile_bytes, Ordering::Relaxed);
        if prev + tile_bytes > self.max_bytes {
            self.evict_some();
        }
    }

    /// Abort an in-flight fetch on error or cancellation so waiters can retry cleanly.
    pub fn abort(&self, key: &BlockKey) {
        if let Some((_, CacheEntry::InFlight(n))) = self.map.remove(key) {
            n.notify_waiters();
        }
    }

    /// Create an RAII guard that automatically calls `abort(&key)` on `Drop` (e.g. HTTP client
    /// disconnect or panic) unless `complete(tile)` is called.
    pub fn guard(self: &Arc<Self>, key: BlockKey) -> InFlightGuard {
        InFlightGuard {
            cache: self.clone(),
            key,
            done: false,
        }
    }

    /// Clear all cached blocks (used for cold-cache benchmarks).
    pub fn clear(&self) {
        self.map.clear();
        self.current_bytes.store(0, Ordering::Relaxed);
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    fn evict_some(&self) {
        // Non-blocking guard: at most 1 worker evicts while all other workers stay lock-free
        if self
            .evicting
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        struct Reset<'a>(&'a AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _reset = Reset(&self.evicting);

        let target_drop = self.max_bytes / 4;
        let mut dropped = 0usize;
        let mut keys_to_remove = Vec::with_capacity(64);
        for item in self.map.iter() {
            if let CacheEntry::Ready(tile) = item.value() {
                let sz: usize = tile.bands.iter().map(|b| b.data.len() * 4).sum();
                keys_to_remove.push((*item.key(), sz));
                dropped += sz;
                if dropped >= target_drop {
                    break;
                }
            }
        }
        for (k, sz) in keys_to_remove {
            if self.map.remove(&k).is_some() {
                let _ = self
                    .current_bytes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                        Some(cur.saturating_sub(sz))
                    });
            }
        }
    }
}

/// RAII guard guaranteeing an `InFlight` singleflight slot is always completed or aborted,
/// even if the owning HTTP request future is cancelled/dropped mid-S3 fetch.
pub struct InFlightGuard {
    cache: Arc<SingleflightBlockCache>,
    key: BlockKey,
    done: bool,
}

impl InFlightGuard {
    pub fn complete(mut self, tile: Arc<DecodedTile>) {
        self.done = true;
        self.cache.complete(self.key, tile);
    }

    pub fn key(&self) -> BlockKey {
        self.key
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if !self.done {
            self.cache.abort(&self.key);
        }
    }
}
