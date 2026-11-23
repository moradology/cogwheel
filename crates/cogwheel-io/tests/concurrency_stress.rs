use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use cogwheel_io::CogStore;
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
    assert!(tiny_store.cache.current_bytes() <= 12 * 1024 * 1024);
}
