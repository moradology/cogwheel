//! Lock-free AIMD (Additive-Increase / Multiplicative-Decrease) concurrency limiter for
//! `cogzilla-server` HTTP admission control.
//!
//! Automatically adapts concurrency limits to the underlying hardware (CPU-only 4-core containers,
//! Apple Silicon unified-memory GPUs, or discrete Vulkan GPUs) and active workload mix without
//! static `--max-concurrent` tuning:
//! - When admitted RTT `<= target_ms` under load (`in_flight >= limit / 2`), additively increases
//!   `limit` up to `max_limit`.
//! - When admitted RTT `> target_ms`, multiplicatively decreases `limit` (`× 0.85`, rate-limited
//!   by a 100 ms cooldown) down to `min_limit`.
//! - Excess requests beyond `limit` are immediately shed with HTTP `429 Too Many Requests`
//!   (`Retry-After: 1`) instead of queueing into multi-second latency collapse or Tokio worker
//!   starvation.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Snapshot of AIMD limiter telemetry (exported via `/metrics`).
#[derive(Debug, Clone, Copy)]
pub struct AimdSnapshot {
    pub enabled: bool,
    pub in_flight: usize,
    pub limit: usize,
    pub admitted: u64,
    pub rejected_429: u64,
}

/// Lock-free adaptive AIMD concurrency limiter.
pub struct AimdLimiter {
    enabled: bool,
    in_flight: AtomicUsize,
    limit: AtomicUsize,
    min_limit: usize,
    max_limit: usize,
    target_us: u64,
    last_adjust_ms: AtomicI64,
    epoch: Instant,
    admitted: AtomicU64,
    rejected_429: AtomicU64,
}

/// RAII permit acquired from [`AimdLimiter::try_acquire`]. Releases the in-flight slot and
/// updates the AIMD window on drop using the observed request RTT.
pub struct AimdPermit<'a> {
    limiter: &'a AimdLimiter,
    started: Instant,
    record_rtt: bool,
}

impl<'a> AimdPermit<'a> {
    /// Skip RTT-based AIMD feedback on drop (e.g. for `/healthz`, `/metrics`, or `4xx` bad
    /// requests whose sub-millisecond RTT does not reflect rendering capacity).
    pub fn ignore_rtt(&mut self) {
        self.record_rtt = false;
    }
}

impl Drop for AimdPermit<'_> {
    fn drop(&mut self) {
        let rtt = self.started.elapsed();
        self.limiter.release(rtt, self.record_rtt);
    }
}

impl AimdLimiter {
    /// Construct an `AimdLimiter` calibrated for either GPU-backed or CPU-only execution,
    /// honoring environment overrides:
    /// - `COGZILLA_AIMD=off|0|false` disables admission control.
    /// - `COGZILLA_AIMD_TARGET_MS` sets target RTT in milliseconds (default `150`).
    /// - `COGZILLA_AIMD_MIN_LIMIT` sets the minimum concurrency floor.
    /// - `COGZILLA_AIMD_MAX_LIMIT` sets the maximum concurrency ceiling (default `4096`).
    pub fn from_env(has_gpu: bool) -> Self {
        let enabled = std::env::var("COGZILLA_AIMD")
            .ok()
            .map(|v| {
                let s = v.trim().to_ascii_lowercase();
                !matches!(s.as_str(), "0" | "off" | "false" | "disabled" | "none")
            })
            .unwrap_or(true);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let default_min = if has_gpu { 32 } else { cores.clamp(4, 32) };
        let default_init = if has_gpu {
            256
        } else {
            (cores * 4).clamp(16, 128)
        };
        let target_ms = std::env::var("COGZILLA_AIMD_TARGET_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(150)
            .max(10);
        let min_limit = std::env::var("COGZILLA_AIMD_MIN_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_min)
            .max(1);
        let max_limit = std::env::var("COGZILLA_AIMD_MAX_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            .max(min_limit);
        Self::new(enabled, default_init, min_limit, max_limit, target_ms)
    }

    pub fn new(
        enabled: bool,
        initial_limit: usize,
        min_limit: usize,
        max_limit: usize,
        target_ms: u64,
    ) -> Self {
        let min_limit = min_limit.max(1);
        let max_limit = max_limit.max(min_limit);
        let init = initial_limit.clamp(min_limit, max_limit);
        Self {
            enabled,
            in_flight: AtomicUsize::new(0),
            limit: AtomicUsize::new(init),
            min_limit,
            max_limit,
            target_us: target_ms.max(1) * 1000,
            last_adjust_ms: AtomicI64::new(-1000),
            epoch: Instant::now(),
            admitted: AtomicU64::new(0),
            rejected_429: AtomicU64::new(0),
        }
    }

    /// Attempt to admit a request. Returns `Some(AimdPermit)` if admitted, or `None` if the
    /// server is currently at its adaptive concurrency limit (caller should return HTTP `429`).
    #[inline]
    pub fn try_acquire(&self) -> Option<AimdPermit<'_>> {
        if !self.enabled {
            return Some(AimdPermit {
                limiter: self,
                started: Instant::now(),
                record_rtt: false,
            });
        }
        let cur = self.in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        if cur > self.limit.load(Ordering::Relaxed) {
            self.in_flight.fetch_sub(1, Ordering::AcqRel);
            self.rejected_429.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.admitted.fetch_add(1, Ordering::Relaxed);
        Some(AimdPermit {
            limiter: self,
            started: Instant::now(),
            record_rtt: true,
        })
    }

    /// Manually record an overload signal (e.g. GPU coalescer queue full) to trigger an
    /// immediate multiplicative decrease.
    pub fn record_overload(&self) {
        if !self.enabled {
            return;
        }
        self.decrease_limit();
    }

    fn release(&self, rtt: Duration, record_rtt: bool) {
        if !self.enabled {
            return;
        }
        let prev_in_flight = self.in_flight.fetch_sub(1, Ordering::AcqRel);
        if !record_rtt {
            return;
        }
        let rtt_us = rtt.as_micros() as u64;
        if rtt_us > self.target_us {
            self.decrease_limit();
        } else {
            let lim = self.limit.load(Ordering::Relaxed);
            if prev_in_flight >= lim / 2 && lim < self.max_limit {
                let now_ms = self.epoch.elapsed().as_millis() as i64;
                let prev = self.last_adjust_ms.load(Ordering::Relaxed);
                if now_ms - prev >= 20
                    && self
                        .last_adjust_ms
                        .compare_exchange(prev, now_ms, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                {
                    let step = (lim / 16).max(2);
                    let next = (lim + step).min(self.max_limit);
                    self.limit.store(next, Ordering::Relaxed);
                }
            }
        }
    }

    fn decrease_limit(&self) {
        let now_ms = self.epoch.elapsed().as_millis() as i64;
        let prev = self.last_adjust_ms.load(Ordering::Relaxed);
        if now_ms - prev >= 100
            && self
                .last_adjust_ms
                .compare_exchange(prev, now_ms, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            let lim = self.limit.load(Ordering::Relaxed);
            let next = ((lim as f64 * 0.85) as usize).max(self.min_limit);
            self.limit.store(next, Ordering::Relaxed);
        }
    }

    pub fn current_limit(&self) -> usize {
        self.limit.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> AimdSnapshot {
        AimdSnapshot {
            enabled: self.enabled,
            in_flight: self.in_flight.load(Ordering::Relaxed),
            limit: self.limit.load(Ordering::Relaxed),
            admitted: self.admitted.load(Ordering::Relaxed),
            rejected_429: self.rejected_429.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aimd_enforces_limit_and_returns_429_on_overflow() {
        let lim = AimdLimiter::new(true, 4, 2, 16, 100);
        let p1 = lim.try_acquire().unwrap();
        let p2 = lim.try_acquire().unwrap();
        let p3 = lim.try_acquire().unwrap();
        let p4 = lim.try_acquire().unwrap();
        assert!(lim.try_acquire().is_none(), "5th request must be shed");
        assert_eq!(lim.snapshot().rejected_429, 1);
        assert_eq!(lim.snapshot().admitted, 4);
        drop((p1, p2, p3, p4));
        assert_eq!(lim.snapshot().in_flight, 0);
        assert!(lim.try_acquire().is_some());
    }

    #[test]
    fn aimd_additive_increase_and_multiplicative_decrease() {
        let lim = AimdLimiter::new(true, 20, 4, 100, 50);
        // Simulate fast completion under >= 50% load -> additive increase
        let mut guards = Vec::new();
        for _ in 0..12 {
            guards.push(lim.try_acquire().unwrap());
        }
        drop(guards);
        let after_inc = lim.current_limit();
        assert!(after_inc > 20, "expected additive increase, got {after_inc}");

        // Simulate overload -> multiplicative decrease (after 100ms cooldown)
        std::thread::sleep(Duration::from_millis(105));
        lim.record_overload();
        let after_dec = lim.current_limit();
        assert!(
            after_dec < after_inc,
            "expected multiplicative decrease from {after_inc}, got {after_dec}"
        );
        assert!(after_dec >= 4);
    }

    #[test]
    fn aimd_disabled_never_rejects() {
        let lim = AimdLimiter::new(false, 1, 1, 1, 10);
        let permits: Vec<_> = (0..100).filter_map(|_| lim.try_acquire()).collect();
        assert_eq!(permits.len(), 100);
        assert_eq!(lim.snapshot().rejected_429, 0);
    }
}
