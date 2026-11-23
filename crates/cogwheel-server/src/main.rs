use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use cogwheel_server::{TileEngine, serve};

#[derive(Parser, Debug)]
#[command(name = "cogwheel-server", about = "Ultra-fast float32 COG tile server")]
struct Args {
    /// Address to bind HTTP server
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Decoded block cache size in MiB
    #[arg(long, default_value_t = 512)]
    cache_mb: usize,

    /// Pre-registered local COG datasets as `name=/path/to/file.tif` (repeatable)
    #[arg(long = "dataset")]
    datasets: Vec<String>,

    /// Pre-registered multi-COG scenes as `name=band1.tif,band2.tif[,band3.tif]` (Landsat RGB / SAR VV+VH)
    #[arg(long = "scene")]
    scenes: Vec<String>,
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
    let engine = Arc::new(TileEngine::new(args.cache_mb * 1024 * 1024));

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

    tracing::info!(bind = %args.bind, cache_mb = args.cache_mb, "cogwheel-server listening");
    if let Err(e) = serve(engine, args.bind).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
