//! Shared lock-free byte-bounded LRU eviction policy used by `SingleflightBlockCache`,
//! `DiskBlockCache`, and `TileCache`.
//!
//! All three caches share:
//! - atomic byte accounting (`add` / saturating `sub`),
//! - a shared logical recency clock (`touch`),
//! - a single-evictor CAS guard with an RAII reset so concurrent writers stay lock-free,
//! - 25% hysteresis (`excess.max(max_bytes / 4)`) so an insert that trips the cap does not
//!   re-trigger eviction on every subsequent insert, and
//! - a two-pass eviction loop: Pass 1 drops cold entries whose recency stamp has not advanced;
//!   Pass 2 drops the oldest skipped entries unconditionally if concurrent reads touched every
//!   candidate and the cache is still over budget (keeping memory strictly bounded under
//!   thundering herds).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Operations a cache exposes to [`ByteBudget::evict_to_budget`].
pub trait EvictTarget {
    type Key;

    /// Snapshot `(last_used_stamp, key)` for every currently evictable entry.
    fn candidates(&self) -> Vec<(u64, Self::Key)>;

    /// Pass 1: remove `key` only if its recency stamp is still `expected_stamp`,
    /// returning the freed byte count if removed.
    fn remove_if_cold(&self, key: &Self::Key, expected_stamp: u64) -> Option<u64>;

    /// Pass 2: unconditionally remove `key` if still present and evictable, returning the
    /// freed byte count. Invoked only when Pass 1 could not bring the cache under budget
    /// because concurrent readers advanced every candidate's stamp.
    fn remove_now(&self, key: &Self::Key) -> Option<u64>;
}

/// Lock-free byte budget, logical recency clock, and single-evictor controller.
pub struct ByteBudget {
    current: AtomicU64,
    max: u64,
    evicting: AtomicBool,
    clock: AtomicU64,
}

impl ByteBudget {
    pub const fn new(max_bytes: u64) -> Self {
        Self {
            current: AtomicU64::new(0),
            max: max_bytes,
            evicting: AtomicBool::new(false),
            clock: AtomicU64::new(1),
        }
    }

    #[inline(always)]
    pub const fn max_bytes(&self) -> u64 {
        self.max
    }

    #[inline(always)]
    pub fn current_bytes(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn current_bytes_acquire(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }

    /// Advance and return a monotonically increasing recency stamp.
    #[inline(always)]
    pub fn touch(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Reserve `bytes` in the budget and return the updated total.
    #[inline(always)]
    pub fn add(&self, bytes: u64) -> u64 {
        self.current.fetch_add(bytes, Ordering::AcqRel) + bytes
    }

    /// Saturating atomic subtract of `bytes`, returning the updated total.
    #[inline]
    pub fn sub(&self, bytes: u64) -> u64 {
        let mut cur = self.current.load(Ordering::Acquire);
        loop {
            let next = cur.saturating_sub(bytes);
            match self
                .current
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return next,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Run the single-evictor loop with 25% hysteresis and two-pass cold-first eviction until
    /// `current_bytes <= max_bytes` or no evictable entries remain.
    pub fn evict_to_budget<T: EvictTarget>(&self, target: &T) {
        loop {
            if self.current.load(Ordering::Acquire) <= self.max {
                return;
            }
            // Non-blocking guard: at most 1 worker evicts while all other workers stay lock-free.
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

                // Drain until within cap (aiming to free excess + 25% of max so we don't
                // re-trigger on every subsequent insert).
                loop {
                    let cur = self.current.load(Ordering::Acquire);
                    if cur <= self.max {
                        break;
                    }
                    let excess = cur - self.max;
                    let target_drop = excess.max(self.max / 4);

                    let mut candidates = target.candidates();
                    if candidates.is_empty() {
                        break;
                    }
                    candidates.sort_unstable_by_key(|(stamp, _)| *stamp);

                    let mut dropped = 0u64;
                    let mut skipped: Vec<T::Key> = Vec::new();

                    // Pass 1: drop genuinely cold entries whose recency stamp hasn't advanced.
                    for (stamp, key) in candidates {
                        if dropped >= target_drop {
                            break;
                        }
                        if let Some(freed) = target.remove_if_cold(&key, stamp) {
                            dropped = dropped.saturating_add(freed);
                            removed_any = true;
                            self.sub(freed);
                        } else {
                            skipped.push(key);
                        }
                    }

                    // Pass 2: if concurrent reads touched every candidate and we are still over cap,
                    // evict the oldest entries from the snapshot regardless of recent hits so
                    // memory/disk stays strictly bounded under thundering herds.
                    if self.current.load(Ordering::Acquire) > self.max {
                        for key in skipped {
                            if self.current.load(Ordering::Acquire) <= self.max {
                                break;
                            }
                            if let Some(freed) = target.remove_now(&key) {
                                dropped = dropped.saturating_add(freed);
                                removed_any = true;
                                self.sub(freed);
                            }
                        }
                    }

                    if dropped == 0 {
                        break;
                    }
                }
            }

            // Re-check after releasing `evicting` to close the lost-wakeup race where another
            // thread inserted while we were exiting `evict_to_budget`.
            if !removed_any || self.current.load(Ordering::Acquire) <= self.max {
                return;
            }
        }
    }
}
