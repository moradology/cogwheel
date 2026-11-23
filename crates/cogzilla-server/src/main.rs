use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use cogzilla_io::{CogStore, DiskBlockCache};
use cogzilla_server::{TileEngine, load_catalog_json_file, serve};

#[derive(Parser, Debug)]
#[command(name = "cogzilla-server", about = "Ultra-fast COG & STAC tile server")]
struct Args {
    /// Address to bind HTTP server
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Decoded block cache size in MiB (tier 1, RAM)
    #[arg(long, default_value_t = 512)]
    cache_mb: usize,

    /// Optional encoded-tile micro-cache size in MiB (0 = singleflight dedup only)
    #[arg(long, default_value_t = 0)]
    tile_cache_mb: usize,

    /// Optional directory for persistent compressed-block cache (tier 2, local disk/NVMe)
    #[arg(long)]
    disk_cache_dir: Option<PathBuf>,

    /// Maximum disk cache size in MiB (0 disables tier 2 even if --disk-cache-dir is set)
    #[arg(long, default_value_t = 0)]
    disk_cache_mb: u64,

    /// Pre-registered local COG datasets as `name=/path/to/file.tif` (repeatable)
    #[arg(long = "dataset")]
    datasets: Vec<String>,

    /// Pre-registered multi-COG scenes as `name=band1.tif,band2.tif[,band3.tif]` (Landsat RGB / SAR VV+VH)
    #[arg(long = "scene")]
    scenes: Vec<String>,

    /// Pre-registered spatial mosaic catalogs as `name=/path/to/catalog.json` (STAC FeatureCollection or JSON array)
    #[arg(long = "mosaic")]
    mosaics: Vec<String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut store = CogStore::new(args.cache_mb * 1024 * 1024);
    if args.disk_cache_mb > 0 {
        let dir = args
            .disk_cache_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("cogzilla-disk-cache"));
        match DiskBlockCache::open(&dir, args.disk_cache_mb * 1024 * 1024) {
            Ok(dc) => {
                tracing::info!(
                    dir = %dir.display(),
                    disk_cache_mb = args.disk_cache_mb,
                    "enabled tier-2 compressed-block disk cache"
                );
                store = store.with_disk_cache(dc);
            }
            Err(e) => {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "failed to open disk cache directory; continuing with tier 2 disabled"
                );
            }
        }
    }
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(
        Arc::new(store),
        args.tile_cache_mb * 1024 * 1024,
    ));

    for spec in &args.datasets {
        let Some((name, path)) = spec.split_once('=') else {
            eprintln!("Invalid --dataset '{spec}', expected name=/path/to/file.tif");
            std::process::exit(1);
        };
        match engine.store.register_local(name, path) {
            Ok(cog) => {
                let ifd = cog.index.full_res();
                tracing::info!(
                    dataset = name,
                    width = ifd.width,
                    height = ifd.height,
                    bands = ifd.samples_per_pixel,
                    overviews = cog.index.ifds.len().saturating_sub(1),
                    "registered local COG"
                );
            }
            Err(e) => {
                eprintln!("Failed to open dataset '{name}' ({path}): {e}");
                std::process::exit(1);
            }
        }
    }

    for spec in &args.scenes {
        let Some((name, paths_str)) = spec.split_once('=') else {
            eprintln!("Invalid --scene '{spec}', expected name=b1.tif,b2.tif[,b3.tif]");
            std::process::exit(1);
        };
        let paths: Vec<&str> = paths_str.split(',').collect();
        match engine.register_local_scene(name, &paths) {
            Ok(cogs) => {
                tracing::info!(scene = name, assets = cogs.len(), "registered multi-COG scene");
            }
            Err(e) => {
                eprintln!("Failed to open scene '{name}': {e}");
                std::process::exit(1);
            }
        }
    }

    for spec in &args.mosaics {
        let Some((name, path)) = spec.split_once('=') else {
            eprintln!("Invalid --mosaic '{spec}', expected name=/path/to/catalog.json");
            std::process::exit(1);
        };
        match load_catalog_json_file(path) {
            Ok(cat) => {
                let count = cat.len();
                engine.register_catalog(name, cat);
                tracing::info!(mosaic = name, scenes = count, "registered spatial mosaic catalog");
            }
            Err(e) => {
                eprintln!("Failed to load mosaic catalog '{name}' ({path}): {e}");
                std::process::exit(1);
            }
        }
    }

    tracing::info!(bind = %args.bind, cache_mb = args.cache_mb, "cogzilla-server listening");
    if let Err(e) = serve(engine, args.bind).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
