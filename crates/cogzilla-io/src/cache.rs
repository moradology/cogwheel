use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use cogzilla_codec::DecodedTile;
use dashmap::DashMap;
use tokio::sync::Notify;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub dataset_id: u64,
    pub ifd_idx: u16,
    pub tile_idx: u32,
}

enum CacheEntry {
    Ready {
        tile: Arc<DecodedTile>,
        /// Value of the cache clock when this block was last inserted or read.
        last_used: AtomicU64,
    },
    InFlight(Arc<Notify>),
}

#[inline(always)]
fn tile_bytes(tile: &DecodedTile) -> usize {
    tile.bands.iter().map(|b| b.data.byte_len()).sum()
}

/// Sharded lock-free singleflight cache for decoded `AlignedF32Slab` COG blocks.
/// Deduplicates concurrent misses across parallel HTTP tile requests (`BM-07: PanBurst32`).
///
/// Eviction is least-recently-used, approximated with a per-entry timestamp: every insert and hit
/// stamps the entry from a shared counter, and eviction removes the entries with the oldest stamps.
/// Hits stay lock-free (one relaxed store into the entry).
pub struct SingleflightBlockCache {
    map: DashMap<BlockKey, CacheEntry, ahash::RandomState>,
    current_bytes: AtomicUsize,
    max_bytes: usize,
    evicting: AtomicBool,
    /// Logical clock for recency stamps; ticks on every insert and hit.
    clock: AtomicU64,
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
            clock: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            coalesced_waiters: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Check current state of `key`: `Some(Ok(tile))` if ready, `Some(Err(current_notify))` if
    /// in-flight (returning the live owner's `Notify`), or `None` if removed/aborted.
    #[inline(always)]
    pub fn peek_state(&self, key: &BlockKey) -> Option<Result<Arc<DecodedTile>, Arc<Notify>>> {
        self.map.get(key).map(|entry| match entry.value() {
            CacheEntry::Ready { tile, last_used } => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                last_used.store(self.tick(), Ordering::Relaxed);
                Ok(tile.clone())
            }
            CacheEntry::InFlight(n) => Err(n.clone()),
        })
    }

    #[inline(always)]
    pub fn try_get_fast(&self, key: &BlockKey) -> Option<Arc<DecodedTile>> {
        if let Some(entry) = self.map.get(key) {
            if let CacheEntry::Ready { tile, last_used } = entry.value() {
                self.hits.fetch_add(1, Ordering::Relaxed);
                last_used.store(self.tick(), Ordering::Relaxed);
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
                CacheEntry::Ready { tile, last_used } => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    last_used.store(self.tick(), Ordering::Relaxed);
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
        let new_bytes = tile_bytes(&tile);
        let entry = CacheEntry::Ready {
            tile,
            last_used: AtomicU64::new(self.tick()),
        };
        let (notify, old_bytes) = match self.map.insert(key, entry) {
            Some(CacheEntry::InFlight(n)) => (Some(n), 0),
            Some(CacheEntry::Ready { tile: old, .. }) => (None, tile_bytes(&old)),
            None => (None, 0),
        };
        if let Some(n) = notify {
            n.notify_waiters();
        }
        let now_bytes = if new_bytes >= old_bytes {
            let delta = new_bytes - old_bytes;
            self.current_bytes.fetch_add(delta, Ordering::AcqRel) + delta
        } else {
            let delta = old_bytes - new_bytes;
            self.current_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                    Some(cur.saturating_sub(delta))
                })
                .unwrap_or(0)
                .saturating_sub(delta)
        };
        if now_bytes > self.max_bytes {
            self.evict_some();
        }
    }

    /// Abort an in-flight fetch on error or cancellation so waiters can retry cleanly.
    pub fn abort(&self, key: &BlockKey) {
        if let Some((_, CacheEntry::InFlight(n))) =
            self.map.remove_if(key, |_, v| matches!(v, CacheEntry::InFlight(_)))
        {
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
        self.current_bytes.store(0, Ordering::Release);
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Acquire)
    }

    /// Recompute actual Ready bytes in the map (for diagnostics and invariant tests).
    pub fn exact_ready_bytes(&self) -> usize {
        self.map
            .iter()
            .map(|item| match item.value() {
                CacheEntry::Ready { tile, .. } => tile_bytes(tile),
                CacheEntry::InFlight(_) => 0,
            })
            .sum()
    }

    /// Number of currently in-flight entries (should be 0 when idle).
    pub fn in_flight_count(&self) -> usize {
        self.map
            .iter()
            .filter(|item| matches!(item.value(), CacheEntry::InFlight(_)))
            .count()
    }

    fn evict_some(&self) {
        loop {
            if self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                return;
            }
            // Non-blocking guard: at most 1 worker evicts while all other workers stay lock-free
            if self
                .evicting
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return;
            }

            let mut removed_any = false;
            {
                struct Reset<'a>(&'a AtomicBool);
                impl Drop for Reset<'_> {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _reset = Reset(&self.evicting);

                // Drain until within cap (aiming to free excess + 25% of max_bytes so we don't
                // re-trigger on every subsequent insert).
                loop {
                    let cur = self.current_bytes.load(Ordering::Acquire);
                    if cur <= self.max_bytes {
                        break;
                    }
                    let excess = cur - self.max_bytes;
                    let target_drop = excess.max(self.max_bytes / 4);

                    let mut candidates: Vec<(u64, BlockKey)> = self
                        .map
                        .iter()
                        .filter_map(|item| match item.value() {
                            CacheEntry::Ready { last_used, .. } => {
                                Some((last_used.load(Ordering::Relaxed), *item.key()))
                            }
                            CacheEntry::InFlight(_) => None,
                        })
                        .collect();
                    if candidates.is_empty() {
                        break;
                    }
                    candidates.sort_unstable_by_key(|&(stamp, _)| stamp);

                    let mut dropped = 0usize;
                    let mut skipped: Vec<BlockKey> = Vec::new();

                    // Pass 1: drop genuinely cold blocks whose recency stamp hasn't advanced.
                    for (stamp, key) in candidates {
                        if dropped >= target_drop {
                            break;
                        }
                        let removed = self.map.remove_if(&key, |_, v| {
                            matches!(v, CacheEntry::Ready { last_used, .. }
                                if last_used.load(Ordering::Relaxed) == stamp)
                        });
                        if let Some((_, CacheEntry::Ready { tile, .. })) = removed {
                            let sz = tile_bytes(&tile);
                            dropped += sz;
                            removed_any = true;
                            let _ = self.current_bytes.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |c| Some(c.saturating_sub(sz)),
                            );
                        } else {
                            skipped.push(key);
                        }
                    }

                    // Pass 2: if concurrent reads touched every candidate and we are still over cap,
                    // evict the oldest Ready entries from the snapshot regardless of recent hits so
                    // memory stays strictly bounded under thundering herds.
                    if self.current_bytes.load(Ordering::Acquire) > self.max_bytes {
                        for key in skipped {
                            if self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                                break;
                            }
                            let removed = self
                                .map
                                .remove_if(&key, |_, v| matches!(v, CacheEntry::Ready { .. }));
                            if let Some((_, CacheEntry::Ready { tile, .. })) = removed {
                                let sz = tile_bytes(&tile);
                                dropped += sz;
                                removed_any = true;
                                let _ = self.current_bytes.fetch_update(
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                    |c| Some(c.saturating_sub(sz)),
                                );
                            }
                        }
                    }

                    if dropped == 0 {
                        break;
                    }
                }
            }

            // Re-check after releasing `evicting` to close the lost-wakeup race where another
            // thread inserted while we were exiting `evict_some`.
            if !removed_any || self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                return;
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

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_codec::{AlignedF32Slab, BlockStats, DecodedBand};

    const BLOCK_FLOATS: usize = 1024;
    const BLOCK_BYTES: usize = BLOCK_FLOATS * 4;

    fn key(i: u32) -> BlockKey {
        BlockKey {
            dataset_id: 1,
            ifd_idx: 0,
            tile_idx: i,
        }
    }

    fn block() -> Arc<DecodedTile> {
        Arc::new(DecodedTile {
            bands: vec![DecodedBand {
                data: AlignedF32Slab::filled(BLOCK_FLOATS, 1.0).into(),
                width: 32,
                height: 32,
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 1.0,
                    max: 1.0,
                },
            }],
        })
    }

    fn insert(cache: &SingleflightBlockCache, i: u32) {
        assert!(matches!(cache.acquire(key(i)), LookupResult::Owner));
        cache.complete(key(i), block());
    }

    fn contains(cache: &SingleflightBlockCache, i: u32) -> bool {
        cache.map.contains_key(&key(i))
    }

    #[test]
    fn evicts_oldest_inserted_blocks_first() {
        let cache = SingleflightBlockCache::new(16 * BLOCK_BYTES);
        for i in 0..16 {
            insert(&cache, i);
        }
        assert_eq!(cache.current_bytes(), 16 * BLOCK_BYTES);

        // The 17th block pushes the cache over its cap: the oldest quarter (4 blocks) goes.
        insert(&cache, 16);
        for i in 0..4 {
            assert!(!contains(&cache, i), "block {i} should have been evicted");
        }
        for i in 4..=16 {
            assert!(contains(&cache, i), "block {i} should have been kept");
        }
        assert_eq!(cache.current_bytes(), 13 * BLOCK_BYTES);
    }

    #[test]
    fn recently_read_blocks_survive_eviction() {
        let cache = SingleflightBlockCache::new(16 * BLOCK_BYTES);
        for i in 0..16 {
            insert(&cache, i);
        }
        // Blocks 0 and 2 are the oldest by insertion but the most recently read, through each
        // of the hit paths.
        assert!(cache.try_get_fast(&key(0)).is_some());
        assert!(matches!(cache.acquire(key(2)), LookupResult::Hit(_)));
        assert!(matches!(cache.peek_state(&key(3)), Some(Ok(_))));

        insert(&cache, 16);
        for i in [0, 2, 3] {
            assert!(contains(&cache, i), "recently read block {i} was evicted");
        }
        // The four coldest blocks are now 1, 4, 5 and 6.
        for i in [1, 4, 5, 6] {
            assert!(!contains(&cache, i), "cold block {i} should have been evicted");
        }
        assert_eq!(cache.current_bytes(), 13 * BLOCK_BYTES);
    }

    #[test]
    fn a_hot_block_outlives_many_rounds_of_churn() {
        let cache = SingleflightBlockCache::new(16 * BLOCK_BYTES);
        insert(&cache, 0);
        for i in 1..2000 {
            insert(&cache, i);
            assert!(
                cache.try_get_fast(&key(0)).is_some(),
                "hot block evicted after {i} inserts"
            );
        }
        assert!(cache.current_bytes() <= 16 * BLOCK_BYTES);
    }

    #[test]
    fn in_flight_entries_are_never_evicted() {
        let cache = SingleflightBlockCache::new(4 * BLOCK_BYTES);
        assert!(matches!(cache.acquire(key(999)), LookupResult::Owner));
        for i in 0..64 {
            insert(&cache, i);
        }
        assert!(matches!(cache.acquire(key(999)), LookupResult::Wait(_)));
    }
}
