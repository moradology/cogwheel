use parking_lot::Mutex;
use tokio::sync::Notify;

struct LimiterState {
    limit: f64,
    in_flight: usize,
    throttles: u64,
}

/// Throttle-only AIMD (Additive Increase, Multiplicative Decrease) concurrency limiter for
/// upstream S3 / HTTP byte-range fetches.
///
/// - Starts at `max_limit` (no cold-start penalty on healthy upstreams).
/// - **Multiplicative decrease (`× 0.7`, floor `min_limit`)** only on explicit upstream distress
///   (`429`, `503 SlowDown`, or retryable transport/timeout errors) — ignores RTT so variable
///   byte-range sizes (8 KiB..4 MiB) and multi-region buckets never cause false throttling.
/// - **Additive increase (`+0.25` per clean response, i.e. `+1` every 4 successes)** back up to
///   `max_limit`.
/// - RAII [`AimdPermit`] guarantees `in_flight` is always decremented on task cancellation.
pub struct AimdLimiter {
    min_limit: usize,
    max_limit: usize,
    state: Mutex<LimiterState>,
    notify: Notify,
}

impl AimdLimiter {
    pub fn new(min_limit: usize, max_limit: usize) -> Self {
        let max_limit = max_limit.max(1);
        let min_limit = min_limit.max(1).min(max_limit);
        Self {
            min_limit,
            max_limit,
            state: Mutex::new(LimiterState {
                limit: max_limit as f64,
                in_flight: 0,
                throttles: 0,
            }),
            notify: Notify::new(),
        }
    }

    /// Construct from `COGZILLA_MAX_UPSTREAM_FETCHES` (default `64`, floor `4`).
    pub fn from_env() -> Self {
        let max_limit = std::env::var("COGZILLA_MAX_UPSTREAM_FETCHES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64);
        Self::new(4.min(max_limit), max_limit)
    }

    #[inline]
    fn effective_limit(&self, raw_limit: f64) -> usize {
        (raw_limit.floor() as usize).clamp(self.min_limit, self.max_limit)
    }

    /// Acquire a concurrency permit, waiting asynchronously if `in_flight >= current_limit`.
    /// Cancel-safe: dropping the returned future before completion never leaks a permit.
    pub async fn acquire(&self) -> AimdPermit<'_> {
        loop {
            {
                let mut st = self.state.lock();
                if st.in_flight < self.effective_limit(st.limit) {
                    st.in_flight += 1;
                    return AimdPermit {
                        limiter: self,
                        finished: false,
                    };
                }
            }
            let fut = self.notify.notified();
            tokio::pin!(fut);
            fut.as_mut().enable();
            {
                let mut st = self.state.lock();
                if st.in_flight < self.effective_limit(st.limit) {
                    st.in_flight += 1;
                    return AimdPermit {
                        limiter: self,
                        finished: false,
                    };
                }
            }
            fut.await;
        }
    }

    pub fn current_limit(&self) -> usize {
        let st = self.state.lock();
        self.effective_limit(st.limit)
    }

    pub fn in_flight(&self) -> usize {
        self.state.lock().in_flight
    }

    pub fn throttle_events(&self) -> u64 {
        self.state.lock().throttles
    }

    fn release_success(&self) {
        let wake_all = {
            let mut st = self.state.lock();
            st.in_flight = st.in_flight.saturating_sub(1);
            if st.limit < self.max_limit as f64 {
                let prev = self.effective_limit(st.limit);
                st.limit = (st.limit + 0.25).min(self.max_limit as f64);
                self.effective_limit(st.limit) > prev
            } else {
                false
            }
        };
        if wake_all {
            self.notify.notify_waiters();
        } else {
            self.notify.notify_one();
        }
    }

    fn release_throttle(&self) {
        let should_wake = {
            let mut st = self.state.lock();
            st.in_flight = st.in_flight.saturating_sub(1);
            st.throttles += 1;
            st.limit = (st.limit * 0.7).max(self.min_limit as f64);
            st.in_flight < self.effective_limit(st.limit)
        };
        if should_wake {
            self.notify.notify_one();
        }
    }

    fn release_neutral(&self) {
        let should_wake = {
            let mut st = self.state.lock();
            st.in_flight = st.in_flight.saturating_sub(1);
            st.in_flight < self.effective_limit(st.limit)
        };
        if should_wake {
            self.notify.notify_one();
        }
    }
}

/// RAII concurrency permit for a single upstream HTTP/S3 fetch attempt.
pub struct AimdPermit<'a> {
    limiter: &'a AimdLimiter,
    finished: bool,
}

impl AimdPermit<'_> {
    /// Mark this attempt as clean (`200`/`206`), gently recovering the concurrency window.
    pub fn on_success(mut self) {
        self.finished = true;
        self.limiter.release_success();
    }

    /// Mark this attempt as throttled/timed out (`429`, `503`, transport reset), backing off ×0.7.
    pub fn on_throttle(mut self) {
        self.finished = true;
        self.limiter.release_throttle();
    }
}

impl Drop for AimdPermit<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.limiter.release_neutral();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn backs_off_on_throttle_and_recovers_additively() {
        let lim = AimdLimiter::new(4, 20);
        assert_eq!(lim.current_limit(), 20);

        // Throttle cuts 20 -> 14 -> 9 -> 6 -> 4 (floor)
        lim.acquire().await.on_throttle();
        assert_eq!(lim.current_limit(), 14);
        lim.acquire().await.on_throttle();
        assert_eq!(lim.current_limit(), 9);
        for _ in 0..10 {
            lim.acquire().await.on_throttle();
        }
        assert_eq!(lim.current_limit(), 4);
        assert_eq!(lim.in_flight(), 0);

        // 4 clean successes bump limit by +1 (4 -> 5)
        for _ in 0..4 {
            lim.acquire().await.on_success();
        }
        assert_eq!(lim.current_limit(), 5);

        // Many clean successes recover back to max_limit (20), never above
        for _ in 0..200 {
            lim.acquire().await.on_success();
        }
        assert_eq!(lim.current_limit(), 20);
        assert_eq!(lim.in_flight(), 0);
    }

    #[tokio::test]
    async fn dropped_permit_never_leaks_and_wakes_waiter() {
        let lim = std::sync::Arc::new(AimdLimiter::new(1, 1));
        let p = lim.acquire().await;
        assert_eq!(lim.in_flight(), 1);

        let lim2 = lim.clone();
        let waiter = tokio::spawn(async move {
            let p2 = lim2.acquire().await;
            p2.on_success();
        });
        tokio::task::yield_now().await;
        drop(p);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("waiter deadlocked after permit drop")
            .unwrap();
        assert_eq!(lim.in_flight(), 0);
    }
}
