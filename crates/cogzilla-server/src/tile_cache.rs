use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::Notify;

use crate::engine::EngineError;

#[derive(Clone)]
pub struct CachedTile {
    pub bytes: Bytes,
    pub content_type: &'static str,
}

struct ReadyEntry {
    tile: CachedTile,
    byte_len: usize,
    last_used: AtomicU64,
}

struct InFlightSlot {
    notify: Notify,
    result: std::sync::Mutex<Option<Result<CachedTile, Arc<EngineError>>>>,
}

struct InFlightGuard<'a> {
    inflight: &'a DashMap<String, Arc<InFlightSlot>, ahash::RandomState>,
    key: String,
    slot: Arc<InFlightSlot>,
    completed: bool,
}

impl<'a> InFlightGuard<'a> {
    fn complete(mut self, res: Result<CachedTile, Arc<EngineError>>) {
        self.completed = true;
        {
            let mut lock = self.slot.result.lock().unwrap();
            *lock = Some(res);
        }
        self.inflight
            .remove_if(&self.key, |_, v| Arc::ptr_eq(v, &self.slot));
        self.slot.notify.notify_waiters();
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            // Owner task was cancelled/dropped: remove slot and wake waiters so they can retry.
            self.inflight
                .remove_if(&self.key, |_, v| Arc::ptr_eq(v, &self.slot));
            self.slot.notify.notify_waiters();
        }
    }
}

/// Request-level tile singleflight deduplicator + optional bounded LRU encoded-tile micro-cache.
///
/// - **Always on (singleflight):** When N concurrent requests arrive for the exact same rendered
///   tile key `(path, query)`, only 1 request runs resampling + encoding; the other N-1 waiters
///   receive a zero-copy `Bytes::clone()` of the result.
/// - **Opt-in (`max_bytes > 0`):** Also retains recently encoded tiles in a bounded LRU map so
///   repeated identical tile requests return in ~1 µs without resampling or re-encoding.
pub struct TileCache {
    max_bytes: usize,
    ready: DashMap<String, ReadyEntry, ahash::RandomState>,
    inflight: DashMap<String, Arc<InFlightSlot>, ahash::RandomState>,
    current_bytes: AtomicUsize,
    clock: AtomicU64,
    evicting: AtomicBool,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub coalesced: AtomicU64,
}

impl TileCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ready: DashMap::with_hasher(ahash::RandomState::new()),
            inflight: DashMap::with_hasher(ahash::RandomState::new()),
            current_bytes: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            evicting: AtomicBool::new(false),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            coalesced: AtomicU64::new(0),
        }
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn clear(&self) {
        self.ready.clear();
        self.current_bytes.store(0, Ordering::Release);
    }

    #[inline]
    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn insert_ready(&self, key: String, tile: CachedTile) {
        if self.max_bytes == 0 {
            return;
        }
        let byte_len = tile.bytes.len() + key.len() + 64;
        if byte_len > self.max_bytes {
            return;
        }
        let stamp = self.tick();
        let old_len = self
            .ready
            .insert(
                key,
                ReadyEntry {
                    tile,
                    byte_len,
                    last_used: AtomicU64::new(stamp),
                },
            )
            .map(|old| old.byte_len)
            .unwrap_or(0);

        let now = if byte_len >= old_len {
            let delta = byte_len - old_len;
            self.current_bytes.fetch_add(delta, Ordering::AcqRel) + delta
        } else {
            let delta = old_len - byte_len;
            self.current_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| {
                    Some(c.saturating_sub(delta))
                })
                .unwrap_or(0)
                .saturating_sub(delta)
        };
        if now > self.max_bytes {
            self.evict_some();
        }
    }

    fn evict_some(&self) {
        loop {
            if self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                return;
            }
            if self
                .evicting
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return;
            }

            let mut dropped_any = false;
            {
                struct Reset<'a>(&'a AtomicBool);
                impl Drop for Reset<'_> {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _reset = Reset(&self.evicting);

                let cur = self.current_bytes.load(Ordering::Acquire);
                if cur > self.max_bytes {
                    let excess = cur - self.max_bytes;
                    let target_drop = excess.max(self.max_bytes / 4);
                    let mut candidates: Vec<(u64, String, usize)> = self
                        .ready
                        .iter()
                        .map(|r| {
                            (
                                r.last_used.load(Ordering::Relaxed),
                                r.key().clone(),
                                r.byte_len,
                            )
                        })
                        .collect();
                    candidates.sort_unstable_by_key(|(ts, _, _)| *ts);
                    let mut dropped = 0usize;
                    for (_, k, _) in candidates {
                        if let Some((_, entry)) = self.ready.remove(&k) {
                            let _ = self.current_bytes.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |c| Some(c.saturating_sub(entry.byte_len)),
                            );
                            dropped += entry.byte_len;
                            dropped_any = true;
                            if dropped >= target_drop {
                                break;
                            }
                        }
                    }
                }
            }
            if !dropped_any {
                return;
            }
        }
    }

    /// Execute `render_fn` with request-level singleflight deduplication and optional LRU caching.
    pub async fn get_or_render<F, Fut>(
        &self,
        key: String,
        render_fn: F,
    ) -> Result<CachedTile, EngineError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<(Bytes, &'static str), EngineError>>,
    {
        // 1. Fast path: encoded tile micro-cache hit
        if self.max_bytes > 0 {
            if let Some(entry) = self.ready.get(&key) {
                entry.last_used.store(self.tick(), Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(entry.tile.clone());
            }
        }

        // 2. Singleflight loop: either join an in-flight render or become the owner
        let mut render_opt = Some(render_fn);
        loop {
            // Re-check ready cache in case a previous owner just completed
            if self.max_bytes > 0 {
                if let Some(entry) = self.ready.get(&key) {
                    entry.last_used.store(self.tick(), Ordering::Relaxed);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(entry.tile.clone());
                }
            }

            let existing = {
                use dashmap::mapref::entry::Entry;
                match self.inflight.entry(key.clone()) {
                    Entry::Occupied(occ) => Some(occ.get().clone()),
                    Entry::Vacant(vac) => {
                        let slot = Arc::new(InFlightSlot {
                            notify: Notify::new(),
                            result: std::sync::Mutex::new(None),
                        });
                        let _ = vac.insert(slot.clone());

                        // We own this render!
                        self.misses.fetch_add(1, Ordering::Relaxed);
                        let guard = InFlightGuard {
                            inflight: &self.inflight,
                            key: key.clone(),
                            slot,
                            completed: false,
                        };
                        let f = render_opt.take().expect("render_fn called once");
                        return match f().await {
                            Ok((bytes, content_type)) => {
                                let tile = CachedTile {
                                    bytes,
                                    content_type,
                                };
                                // Wake all singleflight waiters immediately BEFORE any LRU eviction scan.
                                guard.complete(Ok(tile.clone()));
                                self.insert_ready(key, tile.clone());
                                Ok(tile)
                            }
                            Err(e) => {
                                let shared = Arc::new(e);
                                guard.complete(Err(shared.clone()));
                                Err(EngineError::Shared(shared))
                            }
                        };
                    }
                }
            };

            if let Some(slot) = existing {
                self.coalesced.fetch_add(1, Ordering::Relaxed);
                let notified = slot.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();

                // Re-check result after enabling notification to prevent lost wakeups.
                if let Some(res) = slot.result.lock().unwrap().clone() {
                    return res.map_err(EngineError::Shared);
                }
                // Also check if slot was already removed (cancelled owner).
                let still_present = self
                    .inflight
                    .get(&key)
                    .map(|v| Arc::ptr_eq(v.value(), &slot))
                    .unwrap_or(false);
                if still_present {
                    notified.await;
                }
                if let Some(res) = slot.result.lock().unwrap().clone() {
                    return res.map_err(EngineError::Shared);
                }
                // Owner cancelled without a result: loop to retry and become the new owner.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn singleflight_collapses_32_concurrent_renders_into_one() {
        let cache = Arc::new(TileCache::new(0)); // singleflight only, no retention
        let calls = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..32 {
            let c = cache.clone();
            let n = calls.clone();
            handles.push(tokio::spawn(async move {
                c.get_or_render("tile_k".into(), || async move {
                    n.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(25)).await;
                    Ok((Bytes::from_static(b"PNG_DATA"), "image/png"))
                })
                .await
                .unwrap()
            }));
        }
        for h in handles {
            let res = h.await.unwrap();
            assert_eq!(res.bytes.as_ref(), b"PNG_DATA");
            assert_eq!(res.content_type, "image/png");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.coalesced.load(Ordering::Relaxed), 31);
    }

    #[tokio::test]
    async fn cancelled_owner_wakes_waiter_and_never_deadlocks() {
        let cache = Arc::new(TileCache::new(0));
        let c1 = cache.clone();
        let owner = tokio::spawn(async move {
            let _ = c1
                .get_or_render("cancel_k".into(), || async move {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    Ok((Bytes::from_static(b"NEVER"), "image/png"))
                })
                .await;
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let c2 = cache.clone();
        let waiter = tokio::spawn(async move {
            c2.get_or_render("cancel_k".into(), || async move {
                Ok((Bytes::from_static(b"RECOVERED"), "image/png"))
            })
            .await
            .unwrap()
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        owner.abort();
        let res = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("waiter should not deadlock")
            .unwrap();
        assert_eq!(res.bytes.as_ref(), b"RECOVERED");
    }

    #[tokio::test]
    async fn lru_micro_cache_hits_and_enforces_byte_cap() {
        let cache = TileCache::new(1024);
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..5 {
            let n = calls.clone();
            let res = cache
                .get_or_render("k1".into(), || async move {
                    n.fetch_add(1, Ordering::SeqCst);
                    Ok((Bytes::from(vec![7u8; 256]), "image/png"))
                })
                .await
                .unwrap();
            assert_eq!(res.bytes.len(), 256);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 4);

        // Insert 10 more entries of 256 bytes each -> cap 1024 must be enforced
        for i in 0..10 {
            let _ = cache
                .get_or_render(format!("overflow_{i}"), || async move {
                    Ok((Bytes::from(vec![1u8; 256]), "image/png"))
                })
                .await
                .unwrap();
        }
        assert!(cache.current_bytes() <= 1024);
    }
}
