//! The scenes of a mosaic are read together, and composited in order.
//!
//! Storage is a local HTTP server that takes a fixed time over every request, serves one
//! fixture under any path, and records which paths were asked for and how many of them
//! had a request in flight at once. Needs the generated fixtures; skips itself without them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cogzilla_kernel::{ColormapName, OutputFormat, ResampleMode};
use cogzilla_proj::TileCoord;
use cogzilla_server::engine::MOSAIC_SCENES_AHEAD;
use cogzilla_server::{BandRequest, TileEngine, TileRequest};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const DELAY: Duration = Duration::from_millis(60);

#[derive(Default)]
struct Seen {
    /// Requests in flight, by path. A scene may have several; it is scenes that are counted.
    in_flight: Mutex<BTreeMap<String, usize>>,
    /// The most paths with a request in flight at one moment.
    peak: AtomicUsize,
    paths: Mutex<BTreeSet<String>>,
}

impl Seen {
    /// Forget what was seen, and return it: the peak and the paths.
    fn take(&self) -> (usize, BTreeSet<String>) {
        (self.peak.swap(0, Ordering::SeqCst), std::mem::take(&mut *self.paths.lock().unwrap()))
    }
}

/// Serve `body` under every path, with byte ranges, `DELAY` after each request arrives.
async fn slow_storage(body: Arc<Vec<u8>>, seen: Arc<Seen>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let (body, seen) = (body.clone(), seen.clone());
            tokio::spawn(async move {
                let mut pending = Vec::new();
                loop {
                    // One request head at a time; the client sends no bodies.
                    let end = loop {
                        if let Some(at) = pending.windows(4).position(|w| w == b"\r\n\r\n") {
                            break at + 4;
                        }
                        let mut chunk = [0u8; 4096];
                        match stream.read(&mut chunk).await {
                            Ok(n) if n > 0 => pending.extend_from_slice(&chunk[..n]),
                            _ => return,
                        }
                    };
                    let head = String::from_utf8_lossy(&pending[..end]).into_owned();
                    pending.drain(..end);
                    let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let range = head
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("range: bytes=").map(str::to_string))
                        .and_then(|r| {
                            let (a, b) = r.trim().split_once('-')?;
                            Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()))
                        });
                    seen.paths.lock().unwrap().insert(path.clone());
                    {
                        let mut in_flight = seen.in_flight.lock().unwrap();
                        *in_flight.entry(path.clone()).or_default() += 1;
                        seen.peak.fetch_max(in_flight.len(), Ordering::SeqCst);
                    }
                    tokio::time::sleep(DELAY).await;
                    {
                        let mut in_flight = seen.in_flight.lock().unwrap();
                        let left = in_flight.get_mut(&path).map(|n| {
                            *n -= 1;
                            *n
                        });
                        if left == Some(0) {
                            in_flight.remove(&path);
                        }
                    }
                    let total = body.len();
                    let (status, from, to) = match range {
                        Some((a, b)) if a < total => ("206 Partial Content", a, b.map_or(total, |b| (b + 1).min(total))),
                        Some(_) => ("416 Range Not Satisfiable", 0, 0),
                        None => ("200 OK", 0, total),
                    };
                    let mut reply = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: {}\r\naccept-ranges: bytes\r\n",
                        to - from
                    );
                    if status.starts_with("206") {
                        reply.push_str(&format!("content-range: bytes {from}-{}/{total}\r\n", to - 1));
                    }
                    reply.push_str("\r\n");
                    if stream.write_all(reply.as_bytes()).await.is_err()
                        || stream.write_all(&body[from..to]).await.is_err()
                    {
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mosaic_scenes_are_read_together_and_composited_in_order() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benches/fixtures/mercator_4k_zstd_p3.tif");
    let Ok(body) = std::fs::read(&fixture) else {
        eprintln!("fixtures not generated; skipping mosaic read-ahead test");
        return;
    };
    let seen = Arc::new(Seen::default());
    let storage = slow_storage(Arc::new(body), seen.clone()).await;
    let engine = TileEngine::new(64 * 1024 * 1024);
    // The fixture covers 100 km in Web Mercator. At zoom 12 it reaches 14% of the way into
    // column 652, so that tile is never full; tile 655/1583 lies wholly inside it.
    let request = |x, y| TileRequest {
        tile: TileCoord { z: 12, x, y },
        tile_size: 256,
        bands: BandRequest::Single { band_1based: 1, rescale: (0.0, 2500.0), colormap: ColormapName::Viridis },
        resample: ResampleMode::Nearest,
        format: OutputFormat::Png,
        unscale: false,
    };
    let (edge, inside) = (request(652, 1580), request(655, 1583));
    // Six scenes that are the same picture under six names, so nothing is shared in a cache.
    let scenes = |set: &str| -> Vec<Vec<String>> { (0..6).map(|i| vec![format!("{storage}/{set}{i}.tif")]).collect() };

    // What one scene costs here.
    let started = Instant::now();
    let single = engine.render_mosaic_scenes_ahead(&scenes("one")[..1], &edge, 1).await.unwrap();
    let one_scene = started.elapsed();
    let (peak, paths) = seen.take();
    assert_eq!(paths.len(), 1);
    assert_eq!(peak, 1);

    // All six at once: about what one costs, and the server sees them in flight together.
    let started = Instant::now();
    let together = engine.render_mosaic_scenes_ahead(&scenes("all"), &edge, MOSAIC_SCENES_AHEAD).await.unwrap();
    let all_at_once = started.elapsed();
    let (peak_together, paths) = seen.take();
    assert_eq!(paths.len(), 6, "a tile no scene fills reads every scene");
    assert_eq!(peak_together, 6, "six scenes should be in flight together");

    // One, then two, then the rest: more waiting, the same picture.
    let started = Instant::now();
    let ramped = engine.render_mosaic_scenes(&scenes("ramp"), &edge).await.unwrap();
    let ramp = started.elapsed();
    let (peak_ramped, paths) = seen.take();
    assert_eq!(paths.len(), 6);
    assert_eq!(peak_ramped, 3, "the ramp reads one scene, then two, then the three that are left");
    assert_eq!(together, ramped, "how many scenes are read at once must not change the tile");
    // Every scene is the same picture, so six of them composite to what one gives.
    assert_eq!(together, single);

    eprintln!("one scene {one_scene:?}; six together {all_at_once:?}; six ramped {ramp:?}");
    // Generous bounds: read one after another, six scenes would take six times one.
    assert!(all_at_once < one_scene * 3, "six together took {all_at_once:?}, one took {one_scene:?}");
    assert!(ramp < one_scene * 5, "six ramped took {ramp:?}, one took {one_scene:?}");

    // A tile the first scene fills: read alone, it is the only scene asked for.
    let full = engine.render_mosaic_scenes(&scenes("first"), &inside).await.unwrap();
    let (_, paths) = seen.take();
    assert_eq!(paths.into_iter().collect::<Vec<_>>(), [format!("/first0.tif")]);
    // Read together, later scenes may be asked for, and the tile is the same.
    let eager = engine.render_mosaic_scenes_ahead(&scenes("eager"), &inside, MOSAIC_SCENES_AHEAD).await.unwrap();
    assert_eq!(full, eager);
}
