//! JSON `/stats` and Prometheus `/metrics` exposition for [`TileEngine`].

use std::fmt::Write;
use std::sync::atomic::Ordering;

use crate::engine::TileEngine;

/// Serialize runtime counters and cache/pool telemetry as JSON for `GET /stats`.
pub(super) fn render_stats_json(engine: &TileEngine) -> String {
    let disk = engine.store.disk_cache.as_ref().map(|dc| {
        let s = dc.stats();
        serde_json::json!({
            "hits": s.hits,
            "misses": s.misses,
            "bytes_used": s.bytes_used,
            "max_bytes": s.max_bytes,
            "evictions": s.evictions,
            "write_errors": s.write_errors,
            "read_errors": s.read_errors,
        })
    });
    #[cfg(feature = "gpu")]
    let gpu_stats = engine.gpu.as_ref().map(|gpu| {
        let s = gpu.slab_cache.stats();
        let ms = gpu.multi_slab_cache.stats();
        serde_json::json!({
            "dispatches": engine.stats.gpu_dispatches.load(Ordering::Relaxed),
            "in_flight": gpu.in_flight(),
            "slab_cache": {
                "hits": s.hits,
                "misses": s.misses,
                "bytes_used": s.used_bytes,
                "max_bytes": s.max_bytes,
            },
            "multi_slab_cache": {
                "hits": ms.hits,
                "misses": ms.misses,
                "bytes_used": ms.used_bytes,
                "max_bytes": ms.max_bytes,
            },
        })
    });
    #[cfg(not(feature = "gpu"))]
    let gpu_stats: Option<serde_json::Value> = None;
    let pgstac_stats = engine.pgstac().map(|pg| {
        let s = pg.stats();
        serde_json::json!({
            "pool_size": s.pool_size,
            "pool_available": s.pool_available,
            "pool_waiting": s.pool_waiting,
            "cache_items": s.cache_items,
            "cache_searches": s.cache_searches,
            "cache_collections": s.cache_collections,
            "cache_item_docs": s.cache_item_docs,
            "cache_plans": s.cache_plans,
        })
    });
    serde_json::json!({
        "tiles_served": engine.stats.tiles_served.load(Ordering::Relaxed),
        "empty_short_circuits": engine.stats.empty_short_circuits.load(Ordering::Relaxed),
        "passthrough_hits": engine.stats.passthrough_hits.load(Ordering::Relaxed),
        "http_304_not_modified": engine.stats.http_304_not_modified.load(Ordering::Relaxed),
        "cache_hits": engine.store.cache.hits.load(Ordering::Relaxed),
        "cache_misses": engine.store.cache.misses.load(Ordering::Relaxed),
        "singleflight_coalesced": engine.store.cache.coalesced_waiters.load(Ordering::Relaxed),
        "cache_bytes": engine.store.cache.current_bytes(),
        "tile_cache_hits": engine.tile_cache.hits.load(Ordering::Relaxed),
        "tile_cache_misses": engine.tile_cache.misses.load(Ordering::Relaxed),
        "tile_cache_coalesced": engine.tile_cache.coalesced.load(Ordering::Relaxed),
        "tile_cache_bytes": engine.tile_cache.current_bytes(),
        "upstream_limit": engine.store.upstream_limiter.current_limit(),
        "upstream_in_flight": engine.store.upstream_limiter.in_flight(),
        "upstream_throttles": engine.store.upstream_limiter.throttle_events(),
        "remote_range_requests": engine.store.metrics.remote_range_requests.load(Ordering::Relaxed),
        "remote_bytes_fetched": engine.store.metrics.remote_bytes_fetched.load(Ordering::Relaxed),
        "disk_cache": disk,
        "gpu": gpu_stats,
        "pgstac": pgstac_stats,
    })
    .to_string()
}

/// Format Prometheus text exposition (`text/plain; version=0.0.4`) for `GET /metrics`.
pub(super) fn render_prometheus_metrics(engine: &TileEngine) -> String {
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "# HELP cogzilla_tiles_served_total Total tiles rendered or served.");
    let _ = writeln!(out, "# TYPE cogzilla_tiles_served_total counter");
    let _ = writeln!(out, "cogzilla_tiles_served_total {}", engine.stats.tiles_served.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_empty_short_circuits_total Empty/nodata tiles short-circuited.");
    let _ = writeln!(out, "# TYPE cogzilla_empty_short_circuits_total counter");
    let _ = writeln!(out, "cogzilla_empty_short_circuits_total {}", engine.stats.empty_short_circuits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_passthrough_hits_total Zero-decode JPEG/WebP pass-through tiles.");
    let _ = writeln!(out, "# TYPE cogzilla_passthrough_hits_total counter");
    let _ = writeln!(out, "cogzilla_passthrough_hits_total {}", engine.stats.passthrough_hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_http_304_not_modified_total Conditional GET 304 Not Modified responses.");
    let _ = writeln!(out, "# TYPE cogzilla_http_304_not_modified_total counter");
    let _ = writeln!(out, "cogzilla_http_304_not_modified_total {}", engine.stats.http_304_not_modified.load(Ordering::Relaxed));

    let _ = writeln!(out, "# TYPE cogzilla_block_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_hits_total {}", engine.store.cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_misses_total {}", engine.store.cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_coalesced_total {}", engine.store.cache.coalesced_waiters.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_block_cache_bytes {}", engine.store.cache.current_bytes());

    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_hits_total {}", engine.tile_cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_misses_total {}", engine.tile_cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_coalesced_total {}", engine.tile_cache.coalesced.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_tile_cache_bytes {}", engine.tile_cache.current_bytes());

    let _ = writeln!(out, "# TYPE cogzilla_upstream_limit gauge");
    let _ = writeln!(out, "cogzilla_upstream_limit {}", engine.store.upstream_limiter.current_limit());
    let _ = writeln!(out, "# TYPE cogzilla_upstream_in_flight gauge");
    let _ = writeln!(out, "cogzilla_upstream_in_flight {}", engine.store.upstream_limiter.in_flight());
    let _ = writeln!(out, "# TYPE cogzilla_upstream_throttles_total counter");
    let _ = writeln!(out, "cogzilla_upstream_throttles_total {}", engine.store.upstream_limiter.throttle_events());
    let _ = writeln!(out, "# TYPE cogzilla_remote_range_requests_total counter");
    let _ = writeln!(out, "cogzilla_remote_range_requests_total {}", engine.store.metrics.remote_range_requests.load(Ordering::Relaxed));
    let _ = writeln!(out, "# TYPE cogzilla_remote_bytes_fetched_total counter");
    let _ = writeln!(out, "cogzilla_remote_bytes_fetched_total {}", engine.store.metrics.remote_bytes_fetched.load(Ordering::Relaxed));

    let aimd = engine.aimd.snapshot();
    let _ = writeln!(out, "# HELP cogzilla_aimd_limit Current adaptive AIMD concurrency limit.");
    let _ = writeln!(out, "# TYPE cogzilla_aimd_limit gauge");
    let _ = writeln!(out, "cogzilla_aimd_limit {}", aimd.limit);
    let _ = writeln!(out, "# HELP cogzilla_aimd_in_flight Current in-flight HTTP requests admitted by AIMD.");
    let _ = writeln!(out, "# TYPE cogzilla_aimd_in_flight gauge");
    let _ = writeln!(out, "cogzilla_aimd_in_flight {}", aimd.in_flight);
    let _ = writeln!(out, "# HELP cogzilla_aimd_rejected_429_total Total HTTP requests shed with 429 Too Many Requests by AIMD.");
    let _ = writeln!(out, "# TYPE cogzilla_aimd_rejected_429_total counter");
    let _ = writeln!(out, "cogzilla_aimd_rejected_429_total {}", aimd.rejected_429);

    if let Some(dc) = &engine.store.disk_cache {
        let s = dc.stats();
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_hits_total {}", s.hits);
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_misses_total {}", s.misses);
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_disk_cache_bytes {}", s.bytes_used);
    }

    #[cfg(feature = "gpu")]
    if let Some(gpu) = &engine.gpu {
        let s = gpu.slab_cache.stats();
        let ms = gpu.multi_slab_cache.stats();
        let _ = writeln!(out, "# HELP cogzilla_gpu_dispatches_total Total GPU compute dispatches.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_dispatches_total counter");
        let _ = writeln!(
            out,
            "cogzilla_gpu_dispatches_total {}",
            engine.stats.gpu_dispatches.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# HELP cogzilla_gpu_in_flight Current in-flight GPU operations.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_in_flight gauge");
        let _ = writeln!(out, "cogzilla_gpu_in_flight {}", gpu.in_flight());
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_hits_total {}", s.hits);
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_misses_total {}", s.misses);
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_bytes {}", s.used_bytes);
        let _ = writeln!(out, "# TYPE cogzilla_gpu_multi_slab_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_gpu_multi_slab_cache_hits_total {}", ms.hits);
        let _ = writeln!(out, "# TYPE cogzilla_gpu_multi_slab_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_gpu_multi_slab_cache_misses_total {}", ms.misses);
        let _ = writeln!(out, "# TYPE cogzilla_gpu_multi_slab_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_gpu_multi_slab_cache_bytes {}", ms.used_bytes);
    }

    if let Some(pg) = engine.pgstac() {
        let s = pg.stats();
        let _ = writeln!(out, "# HELP cogzilla_pgstac_pool_size Current pgSTAC connection pool size.");
        let _ = writeln!(out, "# TYPE cogzilla_pgstac_pool_size gauge");
        let _ = writeln!(out, "cogzilla_pgstac_pool_size {}", s.pool_size);
        let _ = writeln!(out, "# HELP cogzilla_pgstac_pool_available Available idle pgSTAC connections.");
        let _ = writeln!(out, "# TYPE cogzilla_pgstac_pool_available gauge");
        let _ = writeln!(out, "cogzilla_pgstac_pool_available {}", s.pool_available);
        let _ = writeln!(out, "# HELP cogzilla_pgstac_pool_waiting Tasks waiting for a pgSTAC connection.");
        let _ = writeln!(out, "# TYPE cogzilla_pgstac_pool_waiting gauge");
        let _ = writeln!(out, "cogzilla_pgstac_pool_waiting {}", s.pool_waiting);
        let _ = writeln!(out, "# HELP cogzilla_pgstac_cache_entries Total entries across pgSTAC TTL caches.");
        let _ = writeln!(out, "# TYPE cogzilla_pgstac_cache_entries gauge");
        let _ = writeln!(out, "cogzilla_pgstac_cache_entries {}", s.total_cache_entries());
    }
    super::zarr::append_zarr_metrics(&mut out);
    out
}
