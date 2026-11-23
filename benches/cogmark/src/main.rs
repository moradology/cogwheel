use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use clap::Parser;
use cogwheel_codec::decode_tile;
use cogwheel_kernel::{ColormapName, OutputFormat, ResampleMode, SpectralExpr};
use cogwheel_proj::TileCoord;
use cogwheel_server::{BandRequest, TileEngine, TileRequest, serve_listener};
use cogwheel_tiff::parse_cog;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::header::{ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, HeaderValue};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use serde::Serialize;
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "benches/fixtures")]
    fixtures: PathBuf,
    #[arg(long, default_value = "benches/results")]
    results_dir: PathBuf,
    /// Optional directory to dump raw f32 + PNG tiles for GDAL pixel parity verification
    #[arg(long)]
    dump_parity: Option<PathBuf>,
}

#[derive(Serialize)]
struct BenchResult {
    id: String,
    name: String,
    median_us: f64,
    p99_us: f64,
    throughput_note: String,
    extra: serde_json::Value,
}

fn summarize_us(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let median = samples[n / 2];
    let p99 = samples[((n as f64) * 0.99).min((n - 1) as f64) as usize];
    (median, p99)
}

async fn bench_tile_sync(
    engine: &TileEngine,
    cog: &cogwheel_io::OpenedCog,
    req: &TileRequest,
    iters: usize,
    clear_cache_each: bool,
) -> (f64, f64, usize) {
    // Warmup
    let mut last_len = 0usize;
    for _ in 0..5 {
        if clear_cache_each {
            engine.store.cache.clear();
        }
        let b = engine.render_tile(cog, req).await.unwrap();
        last_len = b.len();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        if clear_cache_each {
            engine.store.cache.clear();
        }
        let t0 = Instant::now();
        let b = engine.render_tile(cog, req).await.unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6;
        last_len = b.len();
        samples.push(us);
    }
    let (med, p99) = summarize_us(samples);
    (med, p99, last_len)
}

/// Mock HTTP Range server with injected per-request RTT for BM-08.
async fn start_delayed_range_server(
    file_bytes: Arc<Vec<u8>>,
    rtt_ms: u64,
) -> (String, Arc<AtomicU64>, Arc<AtomicU64>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let req_count = Arc::new(AtomicU64::new(0));
    let bytes_sent = Arc::new(AtomicU64::new(0));
    let rc = req_count.clone();
    let bs = bytes_sent.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let _ = stream.set_nodelay(true);
            let io = TokioIo::new(stream);
            let data = file_bytes.clone();
            let rc = rc.clone();
            let bs = bs.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let data = data.clone();
                    let rc = rc.clone();
                    let bs = bs.clone();
                    async move {
                        rc.fetch_add(1, Ordering::Relaxed);
                        if rtt_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(rtt_ms)).await;
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
                                let end_incl: usize =
                                    b.parse().unwrap_or(total - 1).min(total - 1);
                                let slice = if start <= end_incl && start < total {
                                    &data[start..=end_incl]
                                } else {
                                    &[]
                                };
                                bs.fetch_add(slice.len() as u64, Ordering::Relaxed);
                                let mut resp = Response::new(Full::new(Bytes::copy_from_slice(slice)));
                                *resp.status_mut() = StatusCode::PARTIAL_CONTENT;
                                resp.headers_mut()
                                    .insert(ACCEPT_RANGES, HeaderValue::from_static("bytes"));
                                resp.headers_mut().insert(
                                    CONTENT_RANGE,
                                    HeaderValue::from_str(&format!(
                                        "bytes {start}-{end_incl}/{total}"
                                    ))
                                    .unwrap(),
                                );
                                resp.headers_mut().insert(
                                    CONTENT_LENGTH,
                                    HeaderValue::from_str(&slice.len().to_string()).unwrap(),
                                );
                                return Ok::<_, Infallible>(resp);
                            }
                        }
                        bs.fetch_add(total as u64, Ordering::Relaxed);
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::copy_from_slice(&data))))
                    }
                });
                let _ = ServerBuilder::new(TokioExecutor::new())
                    .serve_connection(io, svc)
                    .await;
            });
        }
    });

    (format!("http://{addr}/mercator.tif"), req_count, bytes_sent)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.results_dir).unwrap();
    let mut results: Vec<BenchResult> = Vec::new();

    println!("=== COGMark Benchmark Suite (cogwheel Rust+SIMD) ===");

    // ------------------------------------------------------------------------
    // BM-01: ColdHeader
    // ------------------------------------------------------------------------
    {
        let path = args.fixtures.join("mercator_4k_zstd_p3.tif");
        let bytes = std::fs::read(&path).unwrap();
        for _ in 0..20 {
            let _ = std::hint::black_box(parse_cog(&bytes).unwrap());
        }
        let mut samples = Vec::with_capacity(500);
        for _ in 0..500 {
            let t0 = Instant::now();
            let cog = parse_cog(&bytes).unwrap();
            std::hint::black_box(&cog);
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let (med, p99) = summarize_us(samples);
        println!("BM-01 ColdHeader (5 IFDs, mmap parse): median={med:.2} us  p99={p99:.2} us");
        results.push(BenchResult {
            id: "BM-01".into(),
            name: "ColdHeader".into(),
            median_us: med,
            p99_us: p99,
            throughput_note: format!("{:.0} opens/s", 1e6 / med),
            extra: serde_json::json!({"ifds": 5}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-02: UnshufflePred3 (all 8 codec x predictor combinations, 512x512 f32 = 1 MiB)
    // ------------------------------------------------------------------------
    {
        let codecs = [
            ("NONE_P1", "codec_none_p1.tif"),
            ("DEFLATE_P1", "codec_deflate_p1.tif"),
            ("DEFLATE_P2", "codec_deflate_p2.tif"),
            ("DEFLATE_P3", "codec_deflate_p3.tif"),
            ("ZSTD_P1", "codec_zstd_p1.tif"),
            ("ZSTD_P3", "codec_zstd_p3.tif"),
            ("LZW_P1", "codec_lzw_p1.tif"),
            ("LZW_P3", "codec_lzw_p3.tif"),
        ];
        for (label, fname) in codecs {
            let bytes = std::fs::read(args.fixtures.join(fname)).unwrap();
            let cog = parse_cog(&bytes).unwrap();
            let ifd = cog.full_res();
            let (off, len) = ifd.tile_byte_range(0, 0, 0).unwrap();
            let slice = &bytes[off as usize..(off as usize + len as usize)];
            for _ in 0..10 {
                let _ = std::hint::black_box(
                    decode_tile(slice, ifd, cog.byte_order, cog.nodata).unwrap(),
                );
            }
            let mut samples = Vec::with_capacity(200);
            for _ in 0..200 {
                let t0 = Instant::now();
                let d = decode_tile(slice, ifd, cog.byte_order, cog.nodata).unwrap();
                std::hint::black_box(&d);
                samples.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            let (med, p99) = summarize_us(samples);
            let gb_s = (1048576.0 / (med * 1e-6)) / 1e9;
            println!(
                "BM-02 UnshufflePred3 [{label:10}]: median={med:7.1} us  p99={p99:7.1} us  ({gb_s:.2} GB/s)"
            );
            results.push(BenchResult {
                id: format!("BM-02-{label}"),
                name: format!("UnshufflePred3 ({label})"),
                median_us: med,
                p99_us: p99,
                throughput_note: format!("{gb_s:.2} GB/s"),
                extra: serde_json::json!({"compressed_bytes": len, "raw_bytes": 1048576}),
            });
        }
    }

    // Shared TileEngine for BM-03..BM-06
    let engine = Arc::new(TileEngine::new(512 * 1024 * 1024));
    let cog_merc = engine
        .store
        .register_local("mercator", args.fixtures.join("mercator_4k_zstd_p3.tif"))
        .unwrap();
    let cog_4326 = engine
        .store
        .register_local("geo4326", args.fixtures.join("geo4326_4k_deflate_p3.tif"))
        .unwrap();
    let cog_sat = engine
        .store
        .register_local("sat_rgb", args.fixtures.join("satellite_utm10n_rgb_zstd_p3.tif"))
        .unwrap();
    let cog_utm_lzw = engine
        .store
        .register_local("utm_lzw", args.fixtures.join("utm10n_2k_lzw_p3.tif"))
        .unwrap();
    let cog_albers = engine
        .store
        .register_local("albers", args.fixtures.join("albers5070_2k_zstd_p3.tif"))
        .unwrap();
    let cog_coast = engine
        .store
        .register_local("coastal", args.fixtures.join("coastal_sparse_4k_zstd_p3.tif"))
        .unwrap();

    // Optional parity dump for GDAL oracle comparison
    if let Some(dump_dir) = &args.dump_parity {
        std::fs::create_dir_all(dump_dir).unwrap();
        let dumps: Vec<(&str, &Arc<cogwheel_io::OpenedCog>, TileRequest)> = vec![
            (
                "mercator_z12_f32.bin",
                &cog_merc,
                TileRequest {
                    tile: TileCoord { z: 12, x: 657, y: 1582 },
                    tile_size: 256,
                    bands: BandRequest::Single {
                        band_1based: 1,
                        rescale: (0.0, 1000.0),
                        colormap: ColormapName::Viridis,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::RawF32,
                },
            ),
            (
                "mercator_z12.png",
                &cog_merc,
                TileRequest {
                    tile: TileCoord { z: 12, x: 657, y: 1582 },
                    tile_size: 256,
                    bands: BandRequest::Single {
                        band_1based: 1,
                        rescale: (0.0, 1000.0),
                        colormap: ColormapName::Viridis,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::Png,
                },
            ),
            (
                "geo4326_z12_f32.bin",
                &cog_4326,
                TileRequest {
                    tile: TileCoord { z: 12, x: 657, y: 1583 },
                    tile_size: 256,
                    bands: BandRequest::Single {
                        band_1based: 1,
                        rescale: (0.0, 1000.0),
                        colormap: ColormapName::Viridis,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::RawF32,
                },
            ),
            (
                "sat_utm10n_rgb_z12_f32.bin",
                &cog_sat,
                TileRequest {
                    tile: TileCoord { z: 12, x: 652, y: 1583 },
                    tile_size: 256,
                    bands: BandRequest::Rgb {
                        bands_1based: [1, 2, 3],
                        rescale_min: [0.0; 3],
                        rescale_max: [0.35; 3],
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::RawF32,
                },
            ),
            (
                "sat_utm10n_ndvi_z12_f32.bin",
                &cog_sat,
                TileRequest {
                    tile: TileCoord { z: 12, x: 652, y: 1583 },
                    tile_size: 256,
                    bands: BandRequest::Spectral2 {
                        expr: SpectralExpr::NormalizedDiff,
                        bands_1based: [1, 2],
                        rescale: (-0.5, 0.5),
                        colormap: ColormapName::Ndvi,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::RawF32,
                },
            ),
            (
                "albers5070_z12_f32.bin",
                &cog_albers,
                TileRequest {
                    tile: TileCoord { z: 12, x: 662, y: 1595 },
                    tile_size: 256,
                    bands: BandRequest::Single {
                        band_1based: 1,
                        rescale: (0.0, 1000.0),
                        colormap: ColormapName::Viridis,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::RawF32,
                },
            ),
        ];
        for (fname, cog, treq) in dumps {
            let data = engine.render_tile(cog, &treq).await.unwrap();
            std::fs::write(dump_dir.join(fname), &data).unwrap();
        }
        println!("Dumped parity tiles to {}", dump_dir.display());
    }

    // ------------------------------------------------------------------------
    // BM-03: NativeMercator (PNG, WebP lossy, WebP lossless, Raw f32 - warm & cold)
    // ------------------------------------------------------------------------
    let merc_tile = TileCoord { z: 12, x: 657, y: 1582 };
    for (label, fmt) in [
        ("PNG", OutputFormat::Png),
        ("WebP_Lossy82", OutputFormat::Webp { quality: Some(82.0) }),
        ("WebP_Lossless", OutputFormat::Webp { quality: None }),
        ("RawF32", OutputFormat::RawF32),
    ] {
        let req = TileRequest {
            tile: merc_tile,
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 1000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Bilinear,
            format: fmt,
        };
        let (med_cold, p99_cold, bytes_len) = bench_tile_sync(&engine, &cog_merc, &req, 80, true).await;
        let (med_warm, p99_warm, _) = bench_tile_sync(&engine, &cog_merc, &req, 200, false).await;
        println!(
            "BM-03 NativeMercator [{label:13}]: cold={med_cold:7.1} us  warm={med_warm:6.1} us  (size={bytes_len} B)"
        );
        results.push(BenchResult {
            id: format!("BM-03-{label}-cold"),
            name: format!("NativeMercator {label} (cold)"),
            median_us: med_cold,
            p99_us: p99_cold,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / med_cold),
            extra: serde_json::json!({"bytes": bytes_len}),
        });
        results.push(BenchResult {
            id: format!("BM-03-{label}-warm"),
            name: format!("NativeMercator {label} (warm)"),
            median_us: med_warm,
            p99_us: p99_warm,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / med_warm),
            extra: serde_json::json!({"bytes": bytes_len}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-04: Reproj4326 (EPSG:4326 DEFLATE+Pred3 -> Web Mercator PNG & RawF32)
    // ------------------------------------------------------------------------
    {
        let req = TileRequest {
            tile: TileCoord { z: 12, x: 657, y: 1583 },
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 1000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let (med_cold, p99_cold, sz) = bench_tile_sync(&engine, &cog_4326, &req, 60, true).await;
        let (med_warm, p99_warm, _) = bench_tile_sync(&engine, &cog_4326, &req, 200, false).await;
        println!("BM-04 Reproj4326 [PNG]: cold={med_cold:7.1} us  warm={med_warm:6.1} us  (size={sz} B)");
        results.push(BenchResult {
            id: "BM-04-cold".into(),
            name: "Reproj4326 PNG (cold)".into(),
            median_us: med_cold,
            p99_us: p99_cold,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / med_cold),
            extra: serde_json::json!({"bytes": sz}),
        });
        results.push(BenchResult {
            id: "BM-04-warm".into(),
            name: "Reproj4326 PNG (warm)".into(),
            median_us: med_warm,
            p99_us: p99_warm,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / med_warm),
            extra: serde_json::json!({"bytes": sz}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-05: ReprojProjected (Satellite 3-band UTM 10N RGB, UTM LZW, Albers 5070)
    // ------------------------------------------------------------------------
    {
        let req_sat_png = TileRequest {
            tile: TileCoord { z: 12, x: 652, y: 1583 },
            tile_size: 256,
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0; 3],
                rescale_max: [0.35; 3],
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let (mc, pc, sz) = bench_tile_sync(&engine, &cog_sat, &req_sat_png, 50, true).await;
        let (mw, pw, _) = bench_tile_sync(&engine, &cog_sat, &req_sat_png, 150, false).await;
        println!("BM-05a Satellite UTM10N 3-band RGB [PNG]: cold={mc:7.1} us  warm={mw:6.1} us  (size={sz} B)");
        results.push(BenchResult {
            id: "BM-05a-SatRGB-PNG-cold".into(),
            name: "Satellite UTM10N RGB PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": sz}),
        });
        results.push(BenchResult {
            id: "BM-05a-SatRGB-PNG-warm".into(),
            name: "Satellite UTM10N RGB PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": sz}),
        });

        let req_utm_lzw = TileRequest {
            tile: TileCoord { z: 12, x: 652, y: 1583 },
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 0.35),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let (mc, pc, sz) = bench_tile_sync(&engine, &cog_utm_lzw, &req_utm_lzw, 50, true).await;
        let (mw, pw, _) = bench_tile_sync(&engine, &cog_utm_lzw, &req_utm_lzw, 150, false).await;
        println!("BM-05b UTM10N LZW+Pred3 [PNG]:            cold={mc:7.1} us  warm={mw:6.1} us  (size={sz} B)");
        results.push(BenchResult {
            id: "BM-05b-UTM-LZW-cold".into(),
            name: "UTM10N LZW+Pred3 PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": sz}),
        });
        results.push(BenchResult {
            id: "BM-05b-UTM-LZW-warm".into(),
            name: "UTM10N LZW+Pred3 PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": sz}),
        });

        let req_alb = TileRequest {
            tile: TileCoord { z: 12, x: 662, y: 1595 },
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 1000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let (mc, pc, sz) = bench_tile_sync(&engine, &cog_albers, &req_alb, 50, true).await;
        let (mw, pw, _) = bench_tile_sync(&engine, &cog_albers, &req_alb, 150, false).await;
        println!("BM-05c Albers EPSG:5070 [PNG]:            cold={mc:7.1} us  warm={mw:6.1} us  (size={sz} B)");
        results.push(BenchResult {
            id: "BM-05c-Albers-cold".into(),
            name: "Albers 5070 PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": sz}),
        });
        results.push(BenchResult {
            id: "BM-05c-Albers-warm".into(),
            name: "Albers 5070 PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": sz}),
        });

        let req_ndvi = TileRequest {
            tile: TileCoord { z: 12, x: 652, y: 1583 },
            tile_size: 256,
            bands: BandRequest::Spectral2 {
                expr: SpectralExpr::NormalizedDiff,
                bands_1based: [1, 2],
                rescale: (-0.5, 0.5),
                colormap: ColormapName::Ndvi,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let (mc, pc, sz) = bench_tile_sync(&engine, &cog_sat, &req_ndvi, 50, true).await;
        let (mw, pw, _) = bench_tile_sync(&engine, &cog_sat, &req_ndvi, 150, false).await;
        println!("BM-05d Satellite Fused NDVI [PNG]:        cold={mc:7.1} us  warm={mw:6.1} us  (size={sz} B)");
        results.push(BenchResult {
            id: "BM-05d-FusedNDVI-cold".into(),
            name: "Satellite Fused NDVI PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": sz}),
        });
        results.push(BenchResult {
            id: "BM-05d-FusedNDVI-warm".into(),
            name: "Satellite Fused NDVI PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": sz}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-06: SparseCoastline (32-tile mixed 50% NaN ocean + 50% land viewport)
    // ------------------------------------------------------------------------
    {
        engine.store.cache.clear();
        let mut tiles = Vec::with_capacity(32);
        for y in 1581..1585 {
            for x in 654..662 {
                tiles.push(TileCoord { z: 12, x, y });
            }
        }
        // Warmup once
        for &t in &tiles {
            let req = TileRequest {
                tile: t,
                tile_size: 256,
                bands: BandRequest::Single {
                    band_1based: 1,
                    rescale: (0.0, 1000.0),
                    colormap: ColormapName::Viridis,
                },
                resample: ResampleMode::Bilinear,
                format: OutputFormat::Png,
            };
            let _ = engine.render_tile(&cog_coast, &req).await.unwrap();
        }
        let mut samples = Vec::with_capacity(60);
        for _ in 0..60 {
            let t0 = Instant::now();
            for &t in &tiles {
                let req = TileRequest {
                    tile: t,
                    tile_size: 256,
                    bands: BandRequest::Single {
                        band_1based: 1,
                        rescale: (0.0, 1000.0),
                        colormap: ColormapName::Viridis,
                    },
                    resample: ResampleMode::Bilinear,
                    format: OutputFormat::Png,
                };
                let b = engine.render_tile(&cog_coast, &req).await.unwrap();
                std::hint::black_box(&b);
            }
            samples.push(t0.elapsed().as_secs_f64() * 1e6 / 32.0);
        }
        let (med, p99) = summarize_us(samples);
        println!("BM-06 SparseCoastline (50% empty, avg/tile): median={med:6.1} us  p99={p99:6.1} us");
        results.push(BenchResult {
            id: "BM-06".into(),
            name: "SparseCoastline (avg/tile)".into(),
            median_us: med,
            p99_us: p99,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / med),
            extra: serde_json::json!({"tiles": 32}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-07: PanBurst32 (End-to-End HTTP server, 32 concurrent tiles across all cores)
    // ------------------------------------------------------------------------
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let srv_eng = engine.clone();
        tokio::spawn(async move {
            let _ = serve_listener(srv_eng, listener).await;
        });

        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .build()
            .unwrap();

        let mut urls_merc = Vec::with_capacity(32);
        for y in 1581..1585 {
            for x in 654..662 {
                urls_merc.push(format!(
                    "http://{addr}/tiles/mercator/12/{x}/{y}.png?rescale=0,1000&colormap=viridis"
                ));
            }
        }

        // Cold burst (tests singleflight deduplication + parallel decode!)
        engine.store.cache.clear();
        let t0 = Instant::now();
        let mut js = tokio::task::JoinSet::new();
        for u in &urls_merc {
            let c = client.clone();
            let u = u.clone();
            js.spawn(async move {
                let r = c.get(&u).send().await.unwrap();
                assert_eq!(r.status().as_u16(), 200);
                r.bytes().await.unwrap().len()
            });
        }
        while js.join_next().await.is_some() {}
        let cold_burst_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Warm sustained HTTP burst: 50 rounds of 32 concurrent requests = 1,600 tiles
        let rounds = 50usize;
        let t_start = Instant::now();
        for _ in 0..rounds {
            let mut js = tokio::task::JoinSet::new();
            for u in &urls_merc {
                let c = client.clone();
                let u = u.clone();
                js.spawn(async move {
                    let r = c.get(&u).send().await.unwrap();
                    r.bytes().await.unwrap().len()
                });
            }
            while js.join_next().await.is_some() {}
        }
        let elapsed_s = t_start.elapsed().as_secs_f64();
        let qps = ((rounds * 32) as f64) / elapsed_s;
        let avg_burst_ms = (elapsed_s * 1e3) / (rounds as f64);
        println!(
            "BM-07 PanBurst32 HTTP [Mercator PNG]: cold_32={cold_burst_ms:.2} ms  warm_32={avg_burst_ms:.2} ms  ({qps:.0} tiles/sec)"
        );
        results.push(BenchResult {
            id: "BM-07-HTTP-Mercator".into(),
            name: "PanBurst32 HTTP Mercator PNG".into(),
            median_us: avg_burst_ms * 1000.0,
            p99_us: cold_burst_ms * 1000.0,
            throughput_note: format!("{qps:.0} tiles/sec"),
            extra: serde_json::json!({"cold_32_ms": cold_burst_ms, "warm_32_ms": avg_burst_ms, "qps": qps}),
        });

        // Satellite 3-band UTM10N RGB HTTP burst
        let mut urls_sat = Vec::with_capacity(32);
        for y in 3164..3168 {
            for x in 1301..1309 {
                urls_sat.push(format!(
                    "http://{addr}/tiles/sat_rgb/13/{x}/{y}.png?bidx=1,2,3&rescale=0,0.35"
                ));
            }
        }
        engine.store.cache.clear();
        let t0 = Instant::now();
        let mut js = tokio::task::JoinSet::new();
        for u in &urls_sat {
            let c = client.clone();
            let u = u.clone();
            js.spawn(async move {
                let r = c.get(&u).send().await.unwrap();
                assert_eq!(r.status().as_u16(), 200);
                r.bytes().await.unwrap().len()
            });
        }
        while js.join_next().await.is_some() {}
        let sat_cold_ms = t0.elapsed().as_secs_f64() * 1e3;

        let rounds_sat = 30usize;
        let t_start = Instant::now();
        for _ in 0..rounds_sat {
            let mut js = tokio::task::JoinSet::new();
            for u in &urls_sat {
                let c = client.clone();
                let u = u.clone();
                js.spawn(async move {
                    let r = c.get(&u).send().await.unwrap();
                    r.bytes().await.unwrap().len()
                });
            }
            while js.join_next().await.is_some() {}
        }
        let elapsed_s = t_start.elapsed().as_secs_f64();
        let sat_qps = ((rounds_sat * 32) as f64) / elapsed_s;
        let sat_warm_ms = (elapsed_s * 1e3) / (rounds_sat as f64);
        println!(
            "BM-07 PanBurst32 HTTP [Sat RGB PNG]:  cold_32={sat_cold_ms:.2} ms  warm_32={sat_warm_ms:.2} ms  ({sat_qps:.0} tiles/sec)"
        );
        results.push(BenchResult {
            id: "BM-07-HTTP-SatRGB".into(),
            name: "PanBurst32 HTTP Satellite UTM10N RGB PNG".into(),
            median_us: sat_warm_ms * 1000.0,
            p99_us: sat_cold_ms * 1000.0,
            throughput_note: format!("{sat_qps:.0} tiles/sec"),
            extra: serde_json::json!({"cold_32_ms": sat_cold_ms, "warm_32_ms": sat_warm_ms, "qps": sat_qps}),
        });
    }

    // ------------------------------------------------------------------------
    // BM-08: RemoteRangeRTT (Injected 20ms per HTTP Range RTT)
    // ------------------------------------------------------------------------
    {
        let file_bytes = Arc::new(std::fs::read(args.fixtures.join("mercator_4k_zstd_p3.tif")).unwrap());
        let (url, req_count, bytes_sent) = start_delayed_range_server(file_bytes, 20).await;
        let remote_eng = TileEngine::new(128 * 1024 * 1024);

        let t0 = Instant::now();
        let cog = remote_eng.store.open_uri(&url).await.unwrap();
        let open_ms = t0.elapsed().as_secs_f64() * 1e3;
        let open_reqs = req_count.load(Ordering::Relaxed);

        let req = TileRequest {
            tile: TileCoord { z: 12, x: 657, y: 1582 },
            tile_size: 256,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 1000.0),
                colormap: ColormapName::Viridis,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let t1 = Instant::now();
        let _ = remote_eng.render_tile(&cog, &req).await.unwrap();
        let first_tile_ms = t1.elapsed().as_secs_f64() * 1e3;
        let total_reqs = req_count.load(Ordering::Relaxed);
        let total_bytes = bytes_sent.load(Ordering::Relaxed);

        let t2 = Instant::now();
        let _ = remote_eng.render_tile(&cog, &req).await.unwrap();
        let warm_tile_us = t2.elapsed().as_secs_f64() * 1e6;

        println!(
            "BM-08 RemoteRangeRTT (20ms RTT): open={open_ms:.1}ms ({open_reqs} RTT)  cold_tile={first_tile_ms:.1}ms ({} tile RTTs, total_reqs={total_reqs}, {:.1} KB)  warm={warm_tile_us:.1} us",
            total_reqs - open_reqs,
            total_bytes as f64 / 1024.0
        );
        results.push(BenchResult {
            id: "BM-08".into(),
            name: "RemoteRangeRTT (20ms RTT)".into(),
            median_us: (open_ms + first_tile_ms) * 1000.0,
            p99_us: warm_tile_us,
            throughput_note: format!("{total_reqs} HTTP Range requests total"),
            extra: serde_json::json!({
                "open_ms": open_ms,
                "open_reqs": open_reqs,
                "cold_tile_ms": first_tile_ms,
                "tile_reqs": total_reqs - open_reqs,
                "total_reqs": total_reqs,
                "total_bytes": total_bytes,
                "warm_tile_us": warm_tile_us,
            }),
        });
    }

    // ------------------------------------------------------------------------
    // BM-09: Landsat C2 Level-2 (3 separate uint16 DEFLATE+Pred2 COGs -> RGB PNG)
    // BM-10: OPERA RTC-S1 Dual-Pol SAR (2 separate f32 linear COGs -> fused SAR-RGB PNG & dB PNG)
    // BM-11: Multi-Zoom Pyramid (z=5, 8, 10, 12, 14, 16 across overviews & deep over-zoom)
    // ------------------------------------------------------------------------
    let lsat_b4 = args.fixtures.join("landsat_c2_B4.tif");
    if lsat_b4.exists() {
        let landsat_scene = engine
            .register_local_scene(
                "landsat_c2",
                &[
                    args.fixtures.join("landsat_c2_B4.tif"),
                    args.fixtures.join("landsat_c2_B3.tif"),
                    args.fixtures.join("landsat_c2_B2.tif"),
                ],
            )
            .unwrap();
        let sar_scene = engine
            .register_local_scene(
                "opera_sar",
                &[
                    args.fixtures.join("opera_rtc_s1_VV.tif"),
                    args.fixtures.join("opera_rtc_s1_VH.tif"),
                ],
            )
            .unwrap();

        let lsat_req = TileRequest {
            tile: TileCoord { z: 12, x: 652, y: 1583 },
            tile_size: 256,
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [7500.0; 3],
                rescale_max: [22500.0; 3],
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        // Cold + Warm BM-09
        let mut cold_s = Vec::new();
        for _ in 0..30 {
            engine.store.cache.clear();
            let t0 = Instant::now();
            let b = engine.render_multi_cog_tile(&landsat_scene, &lsat_req).await.unwrap();
            std::hint::black_box(&b);
            cold_s.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let mut warm_s = Vec::new();
        let mut lsat_sz = 0;
        for _ in 0..100 {
            let t0 = Instant::now();
            let b = engine.render_multi_cog_tile(&landsat_scene, &lsat_req).await.unwrap();
            lsat_sz = b.len();
            warm_s.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let (mc, pc) = summarize_us(cold_s);
        let (mw, pw) = summarize_us(warm_s);
        println!(
            "BM-09 Landsat C2 3-File uint16 RGB [PNG]: cold={mc:7.1} us  warm={mw:6.1} us  (size={lsat_sz} B)"
        );
        results.push(BenchResult {
            id: "BM-09-LandsatC2-cold".into(),
            name: "Landsat C2 3-File uint16 RGB PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": lsat_sz}),
        });
        results.push(BenchResult {
            id: "BM-09-LandsatC2-warm".into(),
            name: "Landsat C2 3-File uint16 RGB PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": lsat_sz}),
        });

        // BM-10: OPERA RTC-S1 Dual-Pol SAR False-Color RGB ([VV_dB, VH_dB, VV_dB - VH_dB])
        let sar_req = TileRequest {
            tile: TileCoord { z: 12, x: 652, y: 1583 },
            tile_size: 256,
            bands: BandRequest::SarRgb {
                bands_1based: [1, 2],
                rescale_min: [-20.0, -25.0, 0.0],
                rescale_max: [0.0, -5.0, 15.0],
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Png,
        };
        let mut cold_s = Vec::new();
        for _ in 0..30 {
            engine.store.cache.clear();
            let t0 = Instant::now();
            let b = engine.render_multi_cog_tile(&sar_scene, &sar_req).await.unwrap();
            std::hint::black_box(&b);
            cold_s.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let mut warm_s = Vec::new();
        let mut sar_sz = 0;
        for _ in 0..100 {
            let t0 = Instant::now();
            let b = engine.render_multi_cog_tile(&sar_scene, &sar_req).await.unwrap();
            sar_sz = b.len();
            warm_s.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let (mc, pc) = summarize_us(cold_s);
        let (mw, pw) = summarize_us(warm_s);
        println!(
            "BM-10 OPERA RTC-S1 Dual-Pol SAR-RGB [PNG]: cold={mc:7.1} us  warm={mw:6.1} us  (size={sar_sz} B)"
        );
        results.push(BenchResult {
            id: "BM-10-OperaSAR-RGB-cold".into(),
            name: "OPERA RTC-S1 Dual-Pol SAR-RGB PNG (cold)".into(),
            median_us: mc,
            p99_us: pc,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mc),
            extra: serde_json::json!({"bytes": sar_sz}),
        });
        results.push(BenchResult {
            id: "BM-10-OperaSAR-RGB-warm".into(),
            name: "OPERA RTC-S1 Dual-Pol SAR-RGB PNG (warm)".into(),
            median_us: mw,
            p99_us: pw,
            throughput_note: format!("{:.0} tiles/s/core", 1e6 / mw),
            extra: serde_json::json!({"bytes": sar_sz}),
        });

        // BM-11: Multi-Zoom Pyramid across z = 5, 8, 10, 12, 14, 16 on Landsat C2 3-file RGB!
        // Center lon=-122.650, lat=37.773 mapped down/up the XYZ pyramid:
        let zoom_tiles = [
            (5u8, 5u32, 12u32),
            (8, 40, 98),
            (10, 163, 395),
            (12, 652, 1583),
            (14, 2608, 6332),
            (16, 10432, 25328),
        ];
        println!("BM-11 Multi-Zoom Pyramid (Landsat C2 uint16 3-File RGB PNG):");
        for (z, x, y) in zoom_tiles {
            let zreq = TileRequest {
                tile: TileCoord { z, x, y },
                tile_size: 256,
                bands: BandRequest::Rgb {
                    bands_1based: [1, 2, 3],
                    rescale_min: [7500.0; 3],
                    rescale_max: [22500.0; 3],
                },
                resample: ResampleMode::Bilinear,
                format: OutputFormat::Png,
            };
            // Warmup + measure warm
            let b0 = engine.render_multi_cog_tile(&landsat_scene, &zreq).await.unwrap();
            let mut zs = Vec::with_capacity(50);
            for _ in 0..50 {
                let t0 = Instant::now();
                let b = engine.render_multi_cog_tile(&landsat_scene, &zreq).await.unwrap();
                std::hint::black_box(&b);
                zs.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            let (mz, pz) = summarize_us(zs);
            println!("  z={z:2} ({x:5},{y:5}): warm={mz:7.1} us  p99={pz:7.1} us  (size={} B)", b0.len());
            results.push(BenchResult {
                id: format!("BM-11-z{z}"),
                name: format!("MultiZoom Landsat C2 RGB z={z}"),
                median_us: mz,
                p99_us: pz,
                throughput_note: format!("{:.0} tiles/s/core", 1e6 / mz),
                extra: serde_json::json!({"z": z, "x": x, "y": y, "bytes": b0.len()}),
            });
        }

        // Dump parity tiles for Landsat uint16 RGB, OPERA SAR-RGB, and multi-zoom z=8 & z=14!
        if let Some(dump_dir) = &args.dump_parity {
            let lsat_f32 = engine
                .render_multi_cog_tile(
                    &landsat_scene,
                    &TileRequest {
                        format: OutputFormat::RawF32,
                        ..lsat_req.clone()
                    },
                )
                .await
                .unwrap();
            std::fs::write(dump_dir.join("landsat_c2_rgb_z12_f32.bin"), &lsat_f32).unwrap();

            let sar_f32 = engine
                .render_multi_cog_tile(
                    &sar_scene,
                    &TileRequest {
                        format: OutputFormat::RawF32,
                        ..sar_req.clone()
                    },
                )
                .await
                .unwrap();
            std::fs::write(dump_dir.join("opera_sar_rgb_z12_f32.bin"), &sar_f32).unwrap();

            for (z, x, y) in [(8u8, 40u32, 98u32), (10, 163, 395), (14, 2608, 6332)] {
                let z_f32 = engine
                    .render_multi_cog_tile(
                        &landsat_scene,
                        &TileRequest {
                            tile: TileCoord { z, x, y },
                            tile_size: 256,
                            bands: BandRequest::Rgb {
                                bands_1based: [1, 2, 3],
                                rescale_min: [7500.0; 3],
                                rescale_max: [22500.0; 3],
                            },
                            resample: ResampleMode::Bilinear,
                            format: OutputFormat::RawF32,
                        },
                    )
                    .await
                    .unwrap();
                std::fs::write(dump_dir.join(format!("landsat_c2_z{z}_f32.bin")), &z_f32).unwrap();
            }
        }
    }

    let out_json = args.results_dir.join("cogwheel_results.json");
    std::fs::write(&out_json, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    println!("Saved results to {}", out_json.display());
}
