use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use cogzilla_io::CogStore;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::header::{ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, HeaderValue};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

async fn start_mock_range_server(file_bytes: Arc<Vec<u8>>, delay_ms: u64) -> (String, Arc<AtomicU64>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let reqs = Arc::new(AtomicU64::new(0));
    let reqs_srv = reqs.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let io = TokioIo::new(stream);
            let data = file_bytes.clone();
            let reqs_c = reqs_srv.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let data = data.clone();
                    let reqs_c = reqs_c.clone();
                    async move {
                        reqs_c.fetch_add(1, Ordering::Relaxed);
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        let total = data.len();
                        let range = req
                            .headers()
                            .get("range")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.strip_prefix("bytes="));
                        if let Some(r) = range {
                            if let Some((a, b)) = r.split_once('-') {
                                let start: usize = a.parse().unwrap_or(0);
                                let end_incl: usize = b.parse().unwrap_or(total - 1).min(total - 1);
                                let slice = &data[start..=end_incl];
                                let mut resp = Response::new(Full::new(Bytes::copy_from_slice(slice)));
                                *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
                                resp.headers_mut()
                                    .insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
                                resp.headers_mut().insert(
                                    CONTENT_RANGE,
                                    HeaderValue::from_str(&format!("bytes {start}-{end_incl}/{total}")).unwrap(),
                                );
                                resp.headers_mut().insert(
                                    CONTENT_LENGTH,
                                    HeaderValue::from_str(&slice.len().to_string()).unwrap(),
                                );
                                return Ok::<_, Infallible>(resp);
                            }
                        }
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::copy_from_slice(&data))))
                    }
                });
                let _ = ServerBuilder::new(TokioExecutor::new())
                    .serve_connection(io, svc)
                    .await;
            });
        }
    });
    (format!("http://{addr}/test.tif"), reqs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn singleflight_open_uri_and_midflight_cancellation_never_deadlock() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let (url, reqs) = start_mock_range_server(bytes, 25).await;

    let store = Arc::new(CogStore::new(64 * 1024 * 1024));

    // 1. 64 concurrent cold open_uri calls -> must issue EXACTLY 1 HTTP header GET!
    let mut open_tasks = Vec::new();
    for _ in 0..64 {
        let s = store.clone();
        let u = url.clone();
        open_tasks.push(tokio::spawn(async move { s.open_uri(&u).await.unwrap() }));
    }
    let mut cogs = Vec::new();
    for t in open_tasks {
        cogs.push(t.await.unwrap());
    }
    let open_reqs = reqs.load(Ordering::Relaxed);
    assert_eq!(
        open_reqs, 1,
        "expected exactly 1 singleflight header GET for 64 concurrent opens, got {open_reqs}"
    );
    let cog = cogs[0].clone();

    // 2. 64 concurrent get_blocks calls where the FIRST 16 (including the likely owner!) are
    //    aborted mid-flight after 5ms -> all 48 surviving tasks MUST recover and succeed with 0 deadlocks!
    let mut survivors = Vec::new();
    let mut doomed = Vec::new();
    for i in 0..64 {
        let s = store.clone();
        let c = cog.clone();
        let handle = tokio::spawn(async move { s.get_blocks(&c, 0, &[Some(0)]).await.unwrap() });
        if i < 16 {
            doomed.push(handle);
        } else {
            survivors.push(handle);
        }
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    for h in doomed {
        h.abort();
    }
    // Every surviving task must finish within 2 seconds (no orphaned Notify deadlock!)
    for h in survivors {
        let blocks = tokio::time::timeout(Duration::from_secs(2), h)
            .await
            .expect("waiter deadlocked after owner cancellation!")
            .unwrap();
        assert!(blocks[0].is_some());
        assert_eq!(blocks[0].as_ref().unwrap().bands[0].data.len(), 512 * 512);
    }

    // 3. Aggressive concurrent eviction stress under tiny 4 MiB cache
    let tiny_store = Arc::new(CogStore::new(4 * 1024 * 1024));
    let local_cog = tiny_store.register_local("merc", &path).unwrap();
    let mut evict_tasks = Vec::new();
    for t in 0..32u32 {
        let s = tiny_store.clone();
        let c = local_cog.clone();
        evict_tasks.push(tokio::spawn(async move {
            for k in 0..12u32 {
                let tidx = (t * 7 + k) % 64;
                let res = s.get_blocks(&c, 0, &[Some(tidx)]).await.unwrap();
                assert!(res[0].is_some());
            }
        }));
    }
    for h in evict_tasks {
        h.await.unwrap();
    }
    assert!(
        tiny_store.cache.current_bytes() <= 4 * 1024 * 1024,
        "cache exceeded 4 MiB cap: {}",
        tiny_store.cache.current_bytes()
    );
    assert_eq!(
        tiny_store.cache.current_bytes(),
        tiny_store.cache.exact_ready_bytes(),
        "current_bytes drifted from actual Ready bytes in map"
    );
    assert_eq!(tiny_store.cache.in_flight_count(), 0, "leaked InFlight entries");
}

fn make_test_tile(floats: usize, val: f32) -> Arc<cogzilla_codec::DecodedTile> {
    use cogzilla_codec::{AlignedF32Slab, BlockStats, DecodedBand, DecodedTile};
    Arc::new(DecodedTile {
        bands: vec![DecodedBand {
            data: AlignedF32Slab::filled(floats, val).into(),
            width: 32,
            height: 32,
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: val,
                max: val,
            },
        }],
    })
}

/// 64 tasks hammering a tiny 4-block cache across 64 keys with singleflight races,
/// mid-flight aborts, and continuous eviction. Verifies zero deadlocks, strict cap
/// enforcement, zero InFlight leaks, and exact byte accounting.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cache_thundering_herd_stays_bounded_and_exact() {
    use cogzilla_io::cache::{BlockKey, LookupResult, SingleflightBlockCache};

    const BLOCK_FLOATS: usize = 1024;
    const BLOCK_BYTES: usize = BLOCK_FLOATS * 4;
    const CAP_BLOCKS: usize = 4;
    const CAP_BYTES: usize = CAP_BLOCKS * BLOCK_BYTES;

    let cache = Arc::new(SingleflightBlockCache::new(CAP_BYTES));
    let mut tasks = Vec::new();

    for t in 0..64u32 {
        let c = cache.clone();
        tasks.push(tokio::spawn(async move {
            let mut rng = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            for step in 0..250u32 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let tidx = (rng % 48) as u32;
                let key = BlockKey {
                    dataset_id: 1,
                    ifd_idx: 0,
                    tile_idx: tidx,
                };
                match c.acquire(key) {
                    LookupResult::Hit(_) => {}
                    LookupResult::Owner => {
                        let guard = c.guard(key);
                        // Yield occasionally to widen the race window with waiters & evictors
                        if step % 5 == 0 {
                            tokio::task::yield_now().await;
                        }
                        // Abort 20% of owned fetches to hammer abort + waiter recovery
                        if rng % 5 == 0 {
                            drop(guard);
                        } else {
                            guard.complete(make_test_tile(BLOCK_FLOATS, tidx as f32));
                        }
                    }
                    LookupResult::Wait(notify) => {
                        let fut = notify.notified();
                        tokio::pin!(fut);
                        fut.as_mut().enable();
                        if let Some(Err(live)) = c.peek_state(&key) {
                            if Arc::ptr_eq(&notify, &live) {
                                let _ = tokio::time::timeout(Duration::from_secs(2), fut)
                                    .await
                                    .expect("singleflight waiter deadlocked!");
                            }
                        }
                    }
                }
            }
        }));
    }

    for h in tasks {
        tokio::time::timeout(Duration::from_secs(10), h)
            .await
            .expect("thundering herd task deadlocked!")
            .unwrap();
    }

    assert!(
        cache.current_bytes() <= CAP_BYTES,
        "cache exceeded cap after thundering herd: {} > {}",
        cache.current_bytes(),
        CAP_BYTES
    );
    assert_eq!(
        cache.current_bytes(),
        cache.exact_ready_bytes(),
        "byte counter drifted from exact map contents"
    );
    assert_eq!(cache.in_flight_count(), 0, "leaked InFlight entries");
}

/// 16 reader tasks continuously hot-stamping every cached key while 16 writer tasks insert
/// new keys into a small cap. Ensures Pass-2 eviction still enforces the memory cap even
/// when every entry's `last_used` stamp changes mid-snapshot.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cache_all_hot_read_storm_still_enforces_cap() {
    use cogzilla_io::cache::{BlockKey, LookupResult, SingleflightBlockCache};
    use std::sync::atomic::AtomicBool;

    const BLOCK_FLOATS: usize = 1024;
    const BLOCK_BYTES: usize = BLOCK_FLOATS * 4;
    const CAP_BYTES: usize = 8 * BLOCK_BYTES;

    let cache = Arc::new(SingleflightBlockCache::new(CAP_BYTES));
    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for r in 0..16u32 {
        let c = cache.clone();
        let s = stop.clone();
        readers.push(tokio::spawn(async move {
            let mut k = r;
            while !s.load(Ordering::Relaxed) {
                let key = BlockKey {
                    dataset_id: 1,
                    ifd_idx: 0,
                    tile_idx: k % 64,
                };
                let _ = c.try_get_fast(&key);
                k = k.wrapping_add(1);
                if k % 32 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }));
    }

    let mut writers = Vec::new();
    for w in 0..16u32 {
        let c = cache.clone();
        writers.push(tokio::spawn(async move {
            for i in 0..120u32 {
                let key = BlockKey {
                    dataset_id: 1,
                    ifd_idx: 0,
                    tile_idx: (w * 13 + i) % 64,
                };
                if matches!(c.acquire(key), LookupResult::Owner) {
                    let guard = c.guard(key);
                    guard.complete(make_test_tile(BLOCK_FLOATS, i as f32));
                }
                if i % 8 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }));
    }

    for h in writers {
        h.await.unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for h in readers {
        h.await.unwrap();
    }

    assert!(
        cache.current_bytes() <= CAP_BYTES,
        "all-hot storm broke cache cap: {} > {}",
        cache.current_bytes(),
        CAP_BYTES
    );
    assert_eq!(cache.current_bytes(), cache.exact_ready_bytes());
    assert_eq!(cache.in_flight_count(), 0);
}

/// Repeatedly cancel owners of `open_uri` at staggered intervals to exercise the
/// `OpenAbortGuard` + `Notify::enable()` race window across many rounds.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn open_uri_cancellation_storm_never_deadlocks() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let (base_url, _reqs) = start_mock_range_server(bytes, 15).await;
    let store = Arc::new(CogStore::new(16 * 1024 * 1024));

    for round in 0..8u32 {
        let url = format!("{base_url}?round={round}");
        let mut doomed = Vec::new();
        let mut survivors = Vec::new();
        for i in 0..48u32 {
            let s = store.clone();
            let u = url.clone();
            let h = tokio::spawn(async move { s.open_uri(&u).await.unwrap() });
            if i < 20 {
                doomed.push(h);
            } else {
                survivors.push(h);
            }
        }
        // Staggered aborts across 0..10 ms to hit both pre-enable and post-enable windows
        for (idx, h) in doomed.into_iter().enumerate() {
            if idx % 5 == 0 {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            h.abort();
        }
        for h in survivors {
            let cog = tokio::time::timeout(Duration::from_secs(3), h)
                .await
                .expect("open_uri waiter deadlocked after owner cancellation!")
                .unwrap();
            assert_eq!(cog.uri, url);
        }
    }
}

/// Flaky remote server (returns HTTP 500 on first 12 range requests, then succeeds) +
/// 48 concurrent multi-block `get_blocks` calls with duplicate tile indices and mid-flight aborts.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn flaky_remote_singleflight_and_aborts_never_deadlock() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let req_counter = Arc::new(AtomicU64::new(0));
    let rc_srv = req_counter.clone();
    let data_srv = bytes.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { break };
            let io = TokioIo::new(stream);
            let data = data_srv.clone();
            let rc = rc_srv.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let data = data.clone();
                    let rc = rc.clone();
                    async move {
                        let n = rc.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(8)).await;
                        // Succeed on header (n == 0), then fail requests 1..=6 with 500s
                        if (1..=6).contains(&n) {
                            let mut resp = Response::new(Full::new(Bytes::from_static(b"boom")));
                            *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                            return Ok::<_, Infallible>(resp);
                        }
                        let total = data.len();
                        let range = req
                            .headers()
                            .get("range")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.strip_prefix("bytes="));
                        if let Some(r) = range {
                            if let Some((a, b)) = r.split_once('-') {
                                let start: usize = a.parse().unwrap_or(0);
                                let end_incl: usize = b.parse().unwrap_or(total - 1).min(total - 1);
                                let slice = &data[start..=end_incl];
                                let mut resp = Response::new(Full::new(Bytes::copy_from_slice(slice)));
                                *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
                                resp.headers_mut()
                                    .insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
                                resp.headers_mut().insert(
                                    CONTENT_RANGE,
                                    HeaderValue::from_str(&format!("bytes {start}-{end_incl}/{total}")).unwrap(),
                                );
                                resp.headers_mut().insert(
                                    CONTENT_LENGTH,
                                    HeaderValue::from_str(&slice.len().to_string()).unwrap(),
                                );
                                return Ok::<_, Infallible>(resp);
                            }
                        }
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::copy_from_slice(&data))))
                    }
                });
                let _ = ServerBuilder::new(TokioExecutor::new())
                    .serve_connection(io, svc)
                    .await;
            });
        }
    });

    let url = format!("http://{addr}/flaky.tif");
    let store = Arc::new(CogStore::new(4 * 1024 * 1024));
    let cog = store.open_uri(&url).await.unwrap();

    let mut doomed = Vec::new();
    let mut survivors = Vec::new();
    for i in 0..48u32 {
        let s = store.clone();
        let c = cog.clone();
        let h = tokio::spawn(async move {
            // Overlapping batches with duplicate indices in the same request
            let b0 = i % 8;
            let b1 = (i + 1) % 8;
            s.get_blocks(&c, 0, &[Some(b0), Some(b1), Some(b0)]).await
        });
        if i < 12 {
            doomed.push(h);
        } else {
            survivors.push(h);
        }
    }
    tokio::time::sleep(Duration::from_millis(5)).await;
    for h in doomed {
        h.abort();
    }
    for h in survivors {
        let res = tokio::time::timeout(Duration::from_secs(5), h)
            .await
            .expect("flaky singleflight waiter deadlocked!")
            .unwrap();
        if let Ok(blocks) = res {
            assert_eq!(blocks.len(), 3);
            assert!(blocks[0].is_some() && blocks[1].is_some() && blocks[2].is_some());
        }
    }
    // After transient failures pass, a clean fetch must succeed and leave 0 leaked InFlight slots
    let clean = store.get_blocks(&cog, 0, &[Some(0), Some(1)]).await.unwrap();
    assert!(clean[0].is_some() && clean[1].is_some());
    assert!(store.cache.current_bytes() <= 4 * 1024 * 1024);
    assert_eq!(store.cache.current_bytes(), store.cache.exact_ready_bytes());
    assert_eq!(store.cache.in_flight_count(), 0);
}

/// Verifies that `AimdLimiter` strictly caps concurrent upstream range requests,
/// backs off on HTTP 503 SlowDown, recovers on clean 206s, and leaks 0 permits.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn aimd_limiter_bounds_upstream_concurrency_and_adapts_on_503() {
    use cogzilla_io::AimdLimiter;
    use std::sync::atomic::AtomicUsize;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let req_num = Arc::new(AtomicU64::new(0));

    let active_srv = active.clone();
    let max_srv = max_active.clone();
    let req_srv = req_num.clone();
    let data_srv = bytes.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { break };
            let io = TokioIo::new(stream);
            let data = data_srv.clone();
            let active = active_srv.clone();
            let max_active = max_srv.clone();
            let req_num = req_srv.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let data = data.clone();
                    let active = active.clone();
                    let max_active = max_active.clone();
                    let req_num = req_num.clone();
                    async move {
                        let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(cur, Ordering::SeqCst);
                        struct ActiveGuard(Arc<AtomicUsize>);
                        impl Drop for ActiveGuard {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        let _ag = ActiveGuard(active);

                        let n = req_num.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(8)).await;

                        // Header (n == 0) succeeds; requests 1..=4 return 503 SlowDown
                        if (1..=4).contains(&n) {
                            let mut resp = Response::new(Full::new(Bytes::from_static(b"SlowDown")));
                            *resp.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
                            return Ok::<_, Infallible>(resp);
                        }

                        let total = data.len();
                        let range = req
                            .headers()
                            .get("range")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.strip_prefix("bytes="));
                        if let Some(r) = range {
                            if let Some((a, b)) = r.split_once('-') {
                                let start: usize = a.parse().unwrap_or(0);
                                let end_incl: usize = b.parse().unwrap_or(total - 1).min(total - 1);
                                let slice = &data[start..=end_incl];
                                let mut resp = Response::new(Full::new(Bytes::copy_from_slice(slice)));
                                *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
                                resp.headers_mut()
                                    .insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
                                resp.headers_mut().insert(
                                    CONTENT_RANGE,
                                    HeaderValue::from_str(&format!("bytes {start}-{end_incl}/{total}")).unwrap(),
                                );
                                resp.headers_mut().insert(
                                    CONTENT_LENGTH,
                                    HeaderValue::from_str(&slice.len().to_string()).unwrap(),
                                );
                                return Ok::<_, Infallible>(resp);
                            }
                        }
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::copy_from_slice(&data))))
                    }
                });
                let _ = ServerBuilder::new(TokioExecutor::new())
                    .serve_connection(io, svc)
                    .await;
            });
        }
    });

    let max_cap = 6usize;
    let limiter = Arc::new(AimdLimiter::new(2, max_cap));
    let store = Arc::new(CogStore::with_limits(64 * 1024 * 1024, 64, limiter.clone()));
    let url = format!("http://{addr}/aimd.tif");
    let cog = store.open_uri(&url).await.unwrap();

    // Launch 24 concurrent distinct tile fetches against cap = 6
    let mut handles = Vec::new();
    for tidx in 0..24u32 {
        let s = store.clone();
        let c = cog.clone();
        handles.push(tokio::spawn(async move {
            s.get_blocks(&c, 0, &[Some(tidx)]).await.unwrap()
        }));
    }
    for h in handles {
        let blocks = tokio::time::timeout(Duration::from_secs(10), h)
            .await
            .expect("AIMD-limited fetch deadlocked!")
            .unwrap();
        assert!(blocks[0].is_some());
    }

    let observed_max = max_active.load(Ordering::SeqCst);
    assert!(
        observed_max <= max_cap,
        "upstream concurrency exceeded AIMD cap: {observed_max} > {max_cap}"
    );
    assert!(
        limiter.throttle_events() >= 4,
        "expected AIMD to record 503 throttle events, got {}",
        limiter.throttle_events()
    );
    assert_eq!(limiter.in_flight(), 0, "leaked AIMD permits");
    // After 24 clean completions, AIMD should have recovered back to max_cap
    assert_eq!(limiter.current_limit(), max_cap);
}

/// Verifies `datasets_by_uri` stays strictly bounded by `max_open_uris` under concurrent opens.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn datasets_by_uri_cap_enforced_under_concurrent_opens() {
    use cogzilla_io::AimdLimiter;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let (base_url, _reqs) = start_mock_range_server(bytes, 2).await;

    let max_uris = 8usize;
    let store = Arc::new(CogStore::with_limits(
        16 * 1024 * 1024,
        max_uris,
        Arc::new(AimdLimiter::new(4, 32)),
    ));

    let mut handles = Vec::new();
    for i in 0..40u32 {
        let s = store.clone();
        let u = format!("{base_url}?scene={i}");
        handles.push(tokio::spawn(async move { s.open_uri(&u).await.unwrap() }));
    }
    for h in handles {
        let cog = tokio::time::timeout(Duration::from_secs(5), h)
            .await
            .expect("open_uri deadlocked under URI eviction!")
            .unwrap();
        assert!(!cog.index.ifds.is_empty());
    }

    assert!(
        store.open_uri_count() <= max_uris,
        "datasets_by_uri exceeded cap: {} > {max_uris}",
        store.open_uri_count()
    );
}

/// End-to-end tier-2 disk cache test: fetch remote blocks once, clear RAM cache,
/// re-fetch and verify 0 additional HTTP requests + identical decoded pixel data.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disk_cache_serves_ram_misses_with_zero_remote_requests() {
    use cogzilla_io::DiskBlockCache;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    if !path.exists() {
        return;
    }
    let bytes = Arc::new(std::fs::read(&path).unwrap());
    let (url, reqs) = start_mock_range_server(bytes, 5).await;

    let dir = std::env::temp_dir().join(format!(
        "cogzilla_e2e_disk_cache_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let dc = DiskBlockCache::open(&dir, 32 * 1024 * 1024).unwrap();
    let store = Arc::new(CogStore::new(16 * 1024 * 1024).with_disk_cache(dc.clone()));

    let cog = store.open_uri(&url).await.unwrap();
    let first = store.get_blocks(&cog, 0, &[Some(0), Some(1), Some(2)]).await.unwrap();
    assert!(first.iter().all(|b| b.is_some()));

    // Wait briefly for the non-blocking background writer to flush the 3 blocks
    let mut flushed = false;
    for _ in 0..50 {
        if dc.stats().bytes_used > 0 {
            flushed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(flushed, "background disk cache writer never flushed");
    // Give any remaining queued block a moment to rename
    tokio::time::sleep(Duration::from_millis(30)).await;

    let reqs_after_first = reqs.load(Ordering::Relaxed);
    // Clear RAM completely!
    store.cache.clear();

    // Re-fetch the exact same 3 blocks -> must be served 100% from disk with 0 HTTP requests!
    let second = store.get_blocks(&cog, 0, &[Some(0), Some(1), Some(2)]).await.unwrap();
    assert_eq!(
        reqs.load(Ordering::Relaxed),
        reqs_after_first,
        "expected 0 remote HTTP requests on disk cache hit"
    );
    assert_eq!(dc.stats().hits, 3);
    for i in 0..3 {
        let a = first[i].as_ref().unwrap().bands[0].data.as_f32_slice().unwrap();
        let b = second[i].as_ref().unwrap().bands[0].data.as_f32_slice().unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[a.len() - 1].to_bits(), b[b.len() - 1].to_bits());
    }

    let _ = std::fs::remove_dir_all(&dir);
}
