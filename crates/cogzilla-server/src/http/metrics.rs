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
    let aimd = engine.aimd.snapshot();
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
        "aimd": {
            "limit": aimd.limit,
            "in_flight": aimd.in_flight,
            "rejected_429": aimd.rejected_429,
        },
        "disk_cache": disk,
        "gpu": gpu_stats,
        "pgstac": pgstac_stats,
        "zarr": super::zarr::zarr_stats_json(),
    })
    .to_string()
}

/// Format Prometheus text exposition (`text/plain; version=0.0.4`) for `GET /metrics`.
pub(super) fn render_prometheus_metrics(engine: &TileEngine) -> String {
    let mut out = String::with_capacity(3072);
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

    let _ = writeln!(out, "# HELP cogzilla_block_cache_hits_total Decoded block cache hits.");
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_hits_total {}", engine.store.cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_block_cache_misses_total Decoded block cache misses.");
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_misses_total {}", engine.store.cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_block_cache_coalesced_total Single-flight coalesced block cache waiters.");
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_block_cache_coalesced_total {}", engine.store.cache.coalesced_waiters.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_block_cache_bytes Resident bytes in the decoded block cache.");
    let _ = writeln!(out, "# TYPE cogzilla_block_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_block_cache_bytes {}", engine.store.cache.current_bytes());

    let _ = writeln!(out, "# HELP cogzilla_tile_cache_hits_total Encoded tile micro-cache hits.");
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_hits_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_hits_total {}", engine.tile_cache.hits.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_tile_cache_misses_total Encoded tile micro-cache misses.");
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_misses_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_misses_total {}", engine.tile_cache.misses.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_tile_cache_coalesced_total Single-flight coalesced tile renders.");
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_coalesced_total counter");
    let _ = writeln!(out, "cogzilla_tile_cache_coalesced_total {}", engine.tile_cache.coalesced.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_tile_cache_bytes Resident bytes in the encoded tile micro-cache.");
    let _ = writeln!(out, "# TYPE cogzilla_tile_cache_bytes gauge");
    let _ = writeln!(out, "cogzilla_tile_cache_bytes {}", engine.tile_cache.current_bytes());

    let _ = writeln!(out, "# HELP cogzilla_upstream_limit Current adaptive upstream I/O concurrency limit.");
    let _ = writeln!(out, "# TYPE cogzilla_upstream_limit gauge");
    let _ = writeln!(out, "cogzilla_upstream_limit {}", engine.store.upstream_limiter.current_limit());
    let _ = writeln!(out, "# HELP cogzilla_upstream_in_flight Current in-flight upstream range requests.");
    let _ = writeln!(out, "# TYPE cogzilla_upstream_in_flight gauge");
    let _ = writeln!(out, "cogzilla_upstream_in_flight {}", engine.store.upstream_limiter.in_flight());
    let _ = writeln!(out, "# HELP cogzilla_upstream_throttles_total Total upstream 429/503 throttle events.");
    let _ = writeln!(out, "# TYPE cogzilla_upstream_throttles_total counter");
    let _ = writeln!(out, "cogzilla_upstream_throttles_total {}", engine.store.upstream_limiter.throttle_events());
    let _ = writeln!(out, "# HELP cogzilla_remote_range_requests_total Total remote HTTP/S3/GCS/Azure range requests.");
    let _ = writeln!(out, "# TYPE cogzilla_remote_range_requests_total counter");
    let _ = writeln!(out, "cogzilla_remote_range_requests_total {}", engine.store.metrics.remote_range_requests.load(Ordering::Relaxed));
    let _ = writeln!(out, "# HELP cogzilla_remote_bytes_fetched_total Total remote bytes fetched.");
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
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_hits_total Local NVMe disk cache hits.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_hits_total {}", s.hits);
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_misses_total Local NVMe disk cache misses.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_misses_total {}", s.misses);
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_bytes Resident bytes in the local NVMe disk cache.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_disk_cache_bytes {}", s.bytes_used);
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_evictions_total Total disk cache LRU evictions.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_evictions_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_evictions_total {}", s.evictions);
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_write_errors_total Total disk cache write errors.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_write_errors_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_write_errors_total {}", s.write_errors);
        let _ = writeln!(out, "# HELP cogzilla_disk_cache_read_errors_total Total disk cache read errors.");
        let _ = writeln!(out, "# TYPE cogzilla_disk_cache_read_errors_total counter");
        let _ = writeln!(out, "cogzilla_disk_cache_read_errors_total {}", s.read_errors);
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
        let _ = writeln!(out, "# HELP cogzilla_gpu_slab_cache_hits_total GPU single-band slab cache hits.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_hits_total {}", s.hits);
        let _ = writeln!(out, "# HELP cogzilla_gpu_slab_cache_misses_total GPU single-band slab cache misses.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_misses_total {}", s.misses);
        let _ = writeln!(out, "# HELP cogzilla_gpu_slab_cache_bytes Resident VRAM bytes in GPU single-band slab cache.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_slab_cache_bytes gauge");
        let _ = writeln!(out, "cogzilla_gpu_slab_cache_bytes {}", s.used_bytes);
        let _ = writeln!(out, "# HELP cogzilla_gpu_multi_slab_cache_hits_total GPU multi-band slab cache hits.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_multi_slab_cache_hits_total counter");
        let _ = writeln!(out, "cogzilla_gpu_multi_slab_cache_hits_total {}", ms.hits);
        let _ = writeln!(out, "# HELP cogzilla_gpu_multi_slab_cache_misses_total GPU multi-band slab cache misses.");
        let _ = writeln!(out, "# TYPE cogzilla_gpu_multi_slab_cache_misses_total counter");
        let _ = writeln!(out, "cogzilla_gpu_multi_slab_cache_misses_total {}", ms.misses);
        let _ = writeln!(out, "# HELP cogzilla_gpu_multi_slab_cache_bytes Resident VRAM bytes in GPU multi-band slab cache.");
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn stats_json_and_prometheus_metrics_have_valid_structure_and_parity() {
        let engine = TileEngine::new(8 * 1024 * 1024);

        let stats: serde_json::Value =
            serde_json::from_str(&render_stats_json(&engine)).expect("valid /stats JSON");
        for key in [
            "tiles_served",
            "cache_hits",
            "tile_cache_hits",
            "upstream_limit",
            "remote_range_requests",
            "aimd",
            "zarr",
        ] {
            assert!(stats.get(key).is_some(), "missing /stats key `{key}`");
        }
        assert!(stats["aimd"]["limit"].is_u64());
        assert!(stats["aimd"]["in_flight"].is_u64());
        assert!(stats["aimd"]["rejected_429"].is_u64());
        assert!(stats["zarr"]["stores_opened"].is_u64());
        assert!(stats["zarr"]["hierarchy_cache_entries"].is_u64());
        assert!(stats["zarr"]["remote_chunk_cache_bytes"].is_u64());

        let text = render_prometheus_metrics(&engine);
        let mut helps = HashSet::new();
        let mut types = HashSet::new();
        let mut samples = HashSet::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let mut parts = rest.splitn(2, ' ');
                let name = parts.next().unwrap();
                let desc = parts.next().unwrap_or("").trim();
                assert!(!desc.is_empty(), "empty # HELP description for {name}");
                assert!(helps.insert(name.to_string()), "duplicate # HELP for {name}");
            } else if let Some(rest) = line.strip_prefix("# TYPE ") {
                let mut parts = rest.split_whitespace();
                let name = parts.next().unwrap();
                let kind = parts.next().unwrap();
                assert!(parts.next().is_none(), "trailing tokens in # TYPE `{line}`");
                match kind {
                    "counter" => assert!(name.ends_with("_total"), "counter `{name}` should end in _total"),
                    "gauge" => assert!(!name.ends_with("_total"), "gauge `{name}` should not end in _total"),
                    other => panic!("unexpected TYPE `{other}` for {name}"),
                }
                assert!(types.insert(name.to_string()), "duplicate # TYPE for {name}");
            } else {
                let mut parts = line.split_whitespace();
                let name = parts.next().unwrap();
                let val = parts.next().unwrap();
                assert!(parts.next().is_none(), "trailing tokens in sample `{line}`");
                assert!(val.parse::<f64>().is_ok_and(f64::is_finite), "non-finite sample `{line}`");
                assert!(samples.insert(name.to_string()), "duplicate sample for {name}");
            }
        }
        assert_eq!(helps, types, "every Prometheus metric must have both # HELP and # TYPE");
        assert_eq!(types, samples, "every Prometheus metric must have a matching sample line");
        assert!(samples.contains("cogzilla_aimd_limit"));
        assert!(samples.contains("cogzilla_zarr_hierarchy_cache_entries"));
        assert!(samples.contains("cogzilla_zarr_remote_chunk_cache_bytes"));
    }
}
