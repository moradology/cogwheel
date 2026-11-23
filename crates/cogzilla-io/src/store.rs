use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use cogzilla_codec::{DecodedTile, apply_mask_to_tile, decode_mask_tile, decode_tile};
use cogzilla_tiff::{CogIndex, IfdMeta, TiffError, parse_cog};
use dashmap::DashMap;

/// Lightweight `'static` clone of `IfdMeta` carrying only the fields `decode_tile` reads
/// (omitting large `tile_offsets`/`tile_byte_counts` vectors) so multi-block decodes can fan out
/// across Tokio worker threads with negligible allocation.
fn clone_for_decode(ifd: &IfdMeta) -> IfdMeta {
    IfdMeta {
        width: ifd.width,
        height: ifd.height,
        tile_width: ifd.tile_width,
        tile_height: ifd.tile_height,
        tiles_across: ifd.tiles_across,
        tiles_down: ifd.tiles_down,
        sample_type: ifd.sample_type,
        samples_per_pixel: ifd.samples_per_pixel,
        planar_config: ifd.planar_config,
        compression: ifd.compression,
        predictor: ifd.predictor,
        photometric: ifd.photometric,
        extra_samples: ifd.extra_samples.clone(),
        jpeg_tables: ifd.jpeg_tables.clone(),
        mask_ifd: None,
        is_stripped: ifd.is_stripped,
        pixel_to_crs: ifd.pixel_to_crs,
        crs_to_pixel: ifd.crs_to_pixel,
        tile_offsets: Box::default(),
        tile_byte_counts: Box::default(),
    }
}

use crate::backend::{CogSource, IoError, IoMetrics, TileByteReq, coalesce_ranges};
use crate::cache::{BlockKey, InFlightGuard, LookupResult, SingleflightBlockCache};
use crate::disk_cache::DiskBlockCache;
use crate::limiter::AimdLimiter;
use crate::s3::S3Config;

const INITIAL_HEADER_FETCH_BYTES: u64 = 128 * 1024;
const RANGE_GAP_THRESHOLD: u64 = 32 * 1024;
const MAX_COALESCED_SPAN: u64 = 4 * 1024 * 1024;
const DEFAULT_MAX_OPEN_URIS: usize = 4096;

pub struct OpenedCog {
    pub id: u64,
    pub uri: String,
    pub index: CogIndex,
    pub source: CogSource,
}

enum OpenSlot {
    Ready {
        cog: Arc<OpenedCog>,
        last_used: AtomicU64,
    },
    InFlight(Arc<tokio::sync::Notify>),
}

/// Manages opened COGs (local mmap + remote HTTP/S3) and the shared singleflight block cache.
pub struct CogStore {
    next_id: AtomicU64,
    datasets_by_name: DashMap<String, Arc<OpenedCog>, ahash::RandomState>,
    datasets_by_uri: DashMap<String, OpenSlot, ahash::RandomState>,
    max_open_uris: usize,
    uri_clock: AtomicU64,
    evicting_uris: AtomicBool,
    pub cache: Arc<SingleflightBlockCache>,
    pub disk_cache: Option<Arc<DiskBlockCache>>,
    pub upstream_limiter: Arc<AimdLimiter>,
    pub client: reqwest::Client,
    pub s3_cfg: Option<S3Config>,
    pub metrics: Arc<IoMetrics>,
}

impl CogStore {
    pub fn new(cache_max_bytes: usize) -> Self {
        let max_open_uris = std::env::var("COGZILLA_MAX_OPEN_URIS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_OPEN_URIS);
        Self::with_limits(cache_max_bytes, max_open_uris, Arc::new(AimdLimiter::from_env()))
    }

    pub fn with_limits(
        cache_max_bytes: usize,
        max_open_uris: usize,
        upstream_limiter: Arc<AimdLimiter>,
    ) -> Self {
        // Without these a stalled S3 response would hang the tile request forever. The read timeout
        // is per read, not per response, so large range bodies are fine as long as bytes keep flowing.
        let env_secs = |key: &str, default: u64| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&s| s > 0)
                .unwrap_or(default)
        };
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(env_secs("GDAL_HTTP_CONNECTTIMEOUT", 10)))
            .read_timeout(Duration::from_secs(env_secs("GDAL_HTTP_TIMEOUT", 30)))
            .build()
            .expect("reqwest client");
        Self {
            next_id: AtomicU64::new(1),
            datasets_by_name: DashMap::with_hasher(ahash::RandomState::new()),
            datasets_by_uri: DashMap::with_hasher(ahash::RandomState::new()),
            max_open_uris: max_open_uris.max(1),
            uri_clock: AtomicU64::new(0),
            evicting_uris: AtomicBool::new(false),
            cache: Arc::new(SingleflightBlockCache::new(cache_max_bytes)),
            disk_cache: None,
            upstream_limiter,
            client,
            s3_cfg: S3Config::from_env(),
            metrics: Arc::new(IoMetrics::default()),
        }
    }

    /// Attach a persistent tier-2 compressed-block disk cache for remote S3/HTTP sources.
    pub fn with_disk_cache(mut self, disk_cache: Arc<DiskBlockCache>) -> Self {
        self.disk_cache = Some(disk_cache);
        self
    }

    #[inline(always)]
    fn uri_tick(&self) -> u64 {
        self.uri_clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Number of currently cached opened-COG URI entries.
    pub fn open_uri_count(&self) -> usize {
        self.datasets_by_uri.len()
    }

    fn evict_uris_if_needed(&self) {
        loop {
            if self.datasets_by_uri.len() <= self.max_open_uris {
                return;
            }
            if self
                .evicting_uris
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return;
            }

            let mut removed_any = false;
            {
                struct Reset<'a>(&'a AtomicBool);
                impl Drop for Reset<'_> {
                    fn drop(&mut self) {
                        self.0.store(false, Ordering::Release);
                    }
                }
                let _reset = Reset(&self.evicting_uris);

                loop {
                    let cur = self.datasets_by_uri.len();
                    if cur <= self.max_open_uris {
                        break;
                    }
                    let excess = cur - self.max_open_uris;
                    let target_drop = excess.max(self.max_open_uris / 4).max(1);

                    let mut candidates: Vec<(u64, String)> = self
                        .datasets_by_uri
                        .iter()
                        .filter_map(|item| match item.value() {
                            OpenSlot::Ready { last_used, .. } => {
                                Some((last_used.load(Ordering::Relaxed), item.key().clone()))
                            }
                            OpenSlot::InFlight(_) => None,
                        })
                        .collect();
                    if candidates.is_empty() {
                        break;
                    }
                    candidates.sort_unstable_by_key(|(stamp, _)| *stamp);

                    let mut dropped = 0usize;
                    let mut skipped: Vec<String> = Vec::new();
                    for (stamp, key) in candidates {
                        if dropped >= target_drop {
                            break;
                        }
                        let removed = self.datasets_by_uri.remove_if(&key, |_, v| {
                            matches!(v, OpenSlot::Ready { last_used, .. }
                                if last_used.load(Ordering::Relaxed) == stamp)
                        });
                        if removed.is_some() {
                            dropped += 1;
                            removed_any = true;
                        } else {
                            skipped.push(key);
                        }
                    }

                    if self.datasets_by_uri.len() > self.max_open_uris {
                        for key in skipped {
                            if self.datasets_by_uri.len() <= self.max_open_uris {
                                break;
                            }
                            if self
                                .datasets_by_uri
                                .remove_if(&key, |_, v| matches!(v, OpenSlot::Ready { .. }))
                                .is_some()
                            {
                                dropped += 1;
                                removed_any = true;
                            }
                        }
                    }

                    if dropped == 0 {
                        break;
                    }
                }
            }

            if !removed_any || self.datasets_by_uri.len() <= self.max_open_uris {
                return;
            }
        }
    }

    pub fn register_local(
        &self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Arc<OpenedCog>, IoError> {
        let uri = path.as_ref().to_string_lossy().to_string();
        let source = CogSource::open_local(&path)?;
        let index = match &source {
            CogSource::LocalMmap { mmap } => parse_cog(mmap)?,
            _ => unreachable!(),
        };
        let cog = Arc::new(OpenedCog {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            uri: uri.clone(),
            index,
            source,
        });
        self.datasets_by_name.insert(name.into(), cog.clone());
        self.datasets_by_uri.insert(
            uri,
            OpenSlot::Ready {
                cog: cog.clone(),
                last_used: AtomicU64::new(self.uri_tick()),
            },
        );
        Ok(cog)
    }

    pub fn get_registered(&self, name: &str) -> Option<Arc<OpenedCog>> {
        self.datasets_by_name.get(name).map(|v| v.clone())
    }

    /// Open (or fetch from metadata cache) any local path, `file://`, `http(s)://`, or `s3://` COG.
    /// Singleflight-deduplicated so a 32-tile cold burst against a new S3 COG issues only 1 header GET.
    pub async fn open_uri(&self, uri: &str) -> Result<Arc<OpenedCog>, IoError> {
        loop {
            if let Some(existing) = self.datasets_by_uri.get(uri) {
                if let OpenSlot::Ready { cog, last_used } = existing.value() {
                    last_used.store(self.uri_tick(), Ordering::Relaxed);
                    return Ok(cog.clone());
                }
            }

            let wait_notify = {
                use dashmap::mapref::entry::Entry;
                match self.datasets_by_uri.entry(uri.to_string()) {
                    Entry::Occupied(occ) => match occ.get() {
                        OpenSlot::Ready { cog, last_used } => {
                            last_used.store(self.uri_tick(), Ordering::Relaxed);
                            return Ok(cog.clone());
                        }
                        OpenSlot::InFlight(n) => Some(n.clone()),
                    },
                    Entry::Vacant(vac) => {
                        vac.insert(OpenSlot::InFlight(Arc::new(tokio::sync::Notify::new())));
                        None
                    }
                }
            };

            if let Some(notify) = wait_notify {
                let fut = notify.notified();
                tokio::pin!(fut);
                fut.as_mut().enable();
                match self.datasets_by_uri.get(uri) {
                    Some(existing) => match existing.value() {
                        OpenSlot::Ready { cog, last_used } => {
                            last_used.store(self.uri_tick(), Ordering::Relaxed);
                            return Ok(cog.clone());
                        }
                        OpenSlot::InFlight(cur_n) if !Arc::ptr_eq(cur_n, &notify) => continue,
                        OpenSlot::InFlight(_) => {}
                    },
                    None => continue,
                }
                fut.await;
                continue;
            }

            // We own the cold open; ensure abort on error or cancellation
            struct OpenAbortGuard<'a> {
                map: &'a DashMap<String, OpenSlot, ahash::RandomState>,
                uri: String,
                done: bool,
            }
            impl Drop for OpenAbortGuard<'_> {
                fn drop(&mut self) {
                    if !self.done {
                        if let Some((_, OpenSlot::InFlight(n))) = self
                            .map
                            .remove_if(&self.uri, |_, v| matches!(v, OpenSlot::InFlight(_)))
                        {
                            n.notify_waiters();
                        }
                    }
                }
            }
            let mut open_guard = OpenAbortGuard {
                map: &self.datasets_by_uri,
                uri: uri.to_string(),
                done: false,
            };

            let source = CogSource::open_remote_limited(
                uri,
                self.client.clone(),
                self.s3_cfg.clone(),
                self.metrics.clone(),
                self.upstream_limiter.clone(),
            )?;

            let index = match &source {
                CogSource::LocalMmap { mmap } => parse_cog(mmap)?,
                CogSource::RemoteS3 { .. } | CogSource::RemoteHttp { .. } => {
                    // Speculative 128 KiB single-RTT header read; at most 1 follow-up if IFDs spill past 128 KiB
                    let mut hdr = source.fetch_range(0, INITIAL_HEADER_FETCH_BYTES).await?.to_vec();
                    loop {
                        match parse_cog(&hdr) {
                            Ok(idx) => break idx,
                            Err(TiffError::NeedMoreBytes { required, .. }) => {
                                let cur_len = hdr.len() as u64;
                                let target = ((required + 65535) / 65536) * 65536;
                                if target <= cur_len {
                                    return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                        required,
                                        available: cur_len,
                                    }));
                                }
                                let more = source.fetch_range(cur_len, target).await?;
                                if more.is_empty() {
                                    return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                        required,
                                        available: cur_len,
                                    }));
                                }
                                hdr.extend_from_slice(&more);
                            }
                            Err(e) => return Err(IoError::Tiff(e)),
                        }
                    }
                }
            };

            let cog = Arc::new(OpenedCog {
                id: self.next_id.fetch_add(1, Ordering::Relaxed),
                uri: uri.to_string(),
                index,
                source,
            });
            open_guard.done = true;
            if let Some(OpenSlot::InFlight(n)) = self.datasets_by_uri.insert(
                uri.to_string(),
                OpenSlot::Ready {
                    cog: cog.clone(),
                    last_used: AtomicU64::new(self.uri_tick()),
                },
            ) {
                n.notify_waiters();
            }
            if self.datasets_by_uri.len() > self.max_open_uris {
                self.evict_uris_if_needed();
            }
            return Ok(cog);
        }
    }

    /// Synchronous zero-allocation fast path for fully cached blocks or a single local `mmap` miss.
    /// Returns `Ok(Some(blocks))` if satisfied immediately without awaiting; `Ok(None)` if async I/O
    /// or multi-block parallel decoding is needed.
    pub fn try_get_blocks_sync(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_indices: &[Option<u32>],
    ) -> Result<Option<Vec<Option<Arc<DecodedTile>>>>, IoError> {
        let ifd = &cog.index.ifds[ifd_idx];
        let mut out = Vec::with_capacity(tile_indices.len());
        let mut single_local_miss: Option<(usize, u32, BlockKey, u64, u32)> = None;

        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            let Some(tile_idx) = opt_idx else {
                out.push(None);
                continue;
            };
            let count = ifd.tile_byte_counts[tile_idx as usize];
            if count == 0 {
                out.push(None);
                continue;
            }
            let key = BlockKey {
                dataset_id: cog.id,
                ifd_idx: ifd_idx as u16,
                tile_idx,
            };
            if let Some(hit) = self.cache.try_get_fast(&key) {
                out.push(Some(hit));
                continue;
            }

            let offset = ifd.tile_offsets[tile_idx as usize];
            if cog.source.local_slice(offset, count).is_some() {
                // If >= 2 local blocks missed, hand off to `get_blocks` so they decode in parallel!
                if single_local_miss.is_some() {
                    return Ok(None);
                }
                single_local_miss = Some((slot, tile_idx, key, offset, count));
                out.push(None);
            } else {
                return Ok(None);
            }
        }

        if let Some((slot, tile_idx, key, offset, count)) = single_local_miss {
            let slice = cog.source.local_slice(offset, count).unwrap();
            match self.cache.acquire(key) {
                LookupResult::Hit(hit) => out[slot] = Some(hit),
                LookupResult::Owner => {
                    let guard = self.cache.guard(key);
                    let mut decoded =
                        decode_tile(slice, ifd, cog.index.byte_order, cog.index.nodata)?;
                    if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                        if mask_ifd.tile_width == ifd.tile_width
                            && mask_ifd.tile_height == ifd.tile_height
                        {
                            let per_plane =
                                (ifd.tiles_across as usize) * (ifd.tiles_down as usize);
                            let mask_idx = (tile_idx as usize) % per_plane.max(1);
                            if let Some((moff, mlen)) = mask_ifd.tile_byte_range(mask_idx) {
                                if let Some(mslice) = cog.source.local_slice(moff, mlen) {
                                    if let Ok(mask) = decode_mask_tile(mslice, mask_ifd) {
                                        apply_mask_to_tile(&mut decoded, &mask);
                                    }
                                }
                            }
                        }
                    }
                    let arc = Arc::new(decoded);
                    guard.complete(arc.clone());
                    out[slot] = Some(arc);
                }
                LookupResult::Wait(_) => {
                    return Ok(None);
                }
            }
        }

        Ok(Some(out))
    }

    /// Async path: satisfies any mix of cache hits, singleflight waiters, local mmap,
    /// and coalesced concurrent remote HTTP/S3 Range requests!
    pub async fn get_blocks(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_indices: &[Option<u32>],
    ) -> Result<Vec<Option<Arc<DecodedTile>>>, IoError> {
        // Fast path: if local or already warm in cache, zero async overhead!
        if let Some(fast) = self.try_get_blocks_sync(cog, ifd_idx, tile_indices)? {
            return Ok(fast);
        }

        let ifd = &cog.index.ifds[ifd_idx];
        let mut results: Vec<Option<Arc<DecodedTile>>> = vec![None; tile_indices.len()];
        // Each owned slot holds an Option<InFlightGuard> so cancellation/panic auto-aborts!
        let mut owned: Vec<(usize, Option<InFlightGuard>, u64, u32)> = Vec::new();
        let mut waiters: Vec<(usize, BlockKey, Arc<tokio::sync::Notify>)> = Vec::new();

        // Deduplicate identical tile_indices within the same call (e.g. chunky multi-band requests)
        let mut seen_owned: HashMap<u32, usize> = HashMap::new();

        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            let Some(tile_idx) = opt_idx else { continue };
            let count = ifd.tile_byte_counts[tile_idx as usize];
            if count == 0 {
                continue;
            }
            if seen_owned.contains_key(&tile_idx) {
                continue;
            }
            let key = BlockKey {
                dataset_id: cog.id,
                ifd_idx: ifd_idx as u16,
                tile_idx,
            };
            match self.cache.acquire(key) {
                LookupResult::Hit(tile) => {
                    results[slot] = Some(tile);
                }
                LookupResult::Owner => {
                    let offset = ifd.tile_offsets[tile_idx as usize];
                    seen_owned.insert(tile_idx, slot);
                    owned.push((slot, Some(self.cache.guard(key)), offset, count));
                }
                LookupResult::Wait(notify) => {
                    waiters.push((slot, key, notify));
                }
            }
        }

        // Fetch & decode all owned misses (parallel multi-core decode when >= 2 blocks!)
        if !owned.is_empty() {
            let is_local = matches!(cog.source, CogSource::LocalMmap { .. });
            if is_local && owned.len() >= 2 && ifd.mask_ifd.is_none() {
                let decode_meta = Arc::new(clone_for_decode(ifd));
                let byte_order = cog.index.byte_order;
                let nodata = cog.index.nodata;
                let mut decode_set = tokio::task::JoinSet::new();

                for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                    let src = cog.source.clone();
                    let meta = decode_meta.clone();
                    decode_set.spawn(async move {
                        let Some(slice) = src.local_slice(offset, count) else {
                            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                required: offset + count as u64,
                                available: 0,
                            }));
                        };
                        let decoded = decode_tile(slice, &meta, byte_order, nodata)?;
                        let arc = Arc::new(decoded);
                        if let Some(guard) = guard_opt {
                            guard.complete(arc.clone());
                        }
                        Ok::<_, IoError>((orig_slot, arc))
                    });
                }

                let mut decode_err: Option<IoError> = None;
                while let Some(res) = decode_set.join_next().await {
                    match res {
                        Ok(Ok((orig_slot, arc))) => {
                            results[orig_slot] = Some(arc);
                        }
                        Ok(Err(e)) => decode_err = Some(e),
                        Err(_) => {}
                    }
                }
                if let Some(e) = decode_err {
                    return Err(e);
                }
            } else {
                // Tier-2 compressed-block disk cache lookup for remote sources (RAM miss -> disk hit)
                if !is_local && ifd.mask_ifd.is_none() {
                    if let Some(dc) = &self.disk_cache {
                        let meta = Arc::new(clone_for_decode(ifd));
                        let byte_order = cog.index.byte_order;
                        let nodata = cog.index.nodata;
                        let mut disk_set = tokio::task::JoinSet::new();
                        let prev_owned = std::mem::take(&mut owned);
                        for item in prev_owned {
                            let (orig_slot, guard_opt, offset, count) = item;
                            let tidx = tile_indices[orig_slot].unwrap();
                            let dc_c = dc.clone();
                            let uri = cog.uri.clone();
                            let m = meta.clone();
                            disk_set.spawn(async move {
                                if let Some(compressed) =
                                    dc_c.get(uri, ifd_idx as u16, tidx).await
                                {
                                    match decode_tile(&compressed, &m, byte_order, nodata) {
                                        Ok(decoded) => {
                                            let arc = Arc::new(decoded);
                                            if let Some(guard) = guard_opt {
                                                guard.complete(arc.clone());
                                            }
                                            return Ok::<_, (usize, Option<InFlightGuard>, u64, u32)>(
                                                (orig_slot, arc),
                                            );
                                        }
                                        Err(_) => {
                                            // Corrupt/unreadable disk entry: fail-open and refetch!
                                            return Err((orig_slot, guard_opt, offset, count));
                                        }
                                    }
                                }
                                Err((orig_slot, guard_opt, offset, count))
                            });
                        }
                        while let Some(res) = disk_set.join_next().await {
                            match res {
                                Ok(Ok((orig_slot, arc))) => {
                                    results[orig_slot] = Some(arc);
                                }
                                Ok(Err(miss)) => {
                                    owned.push(miss);
                                }
                                Err(_) => {}
                            }
                        }
                    }
                }

                if !owned.is_empty() {
                    let reqs: Vec<TileByteReq> = owned
                        .iter()
                        .enumerate()
                        .map(|(i, &(_, _, offset, len))| TileByteReq {
                            slot: i,
                            offset,
                            len,
                        })
                        .collect();
                    let spans = coalesce_ranges(reqs, RANGE_GAP_THRESHOLD, MAX_COALESCED_SPAN);

                    let mut span_futs = Vec::with_capacity(spans.len());
                    for span in spans {
                        let src = cog.source.clone();
                        span_futs.push(async move {
                            let data = src.fetch_range(span.start, span.end).await?;
                            Ok::<_, IoError>((span, data))
                        });
                    }

                    let mut join_set = tokio::task::JoinSet::new();
                    for fut in span_futs {
                        join_set.spawn(fut);
                    }

                    // Parallel decode set for multi-block remote spans (overlaps decode with remaining downloads!)
                    let mut decode_set: tokio::task::JoinSet<
                        Result<(usize, Arc<DecodedTile>), IoError>,
                    > = tokio::task::JoinSet::new();
                    let mut decode_meta_opt: Option<Arc<IfdMeta>> = None;
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata;

                    let mut fetch_err: Option<IoError> = None;
                    while let Some(res) = join_set.join_next().await {
                        match res {
                            Ok(Ok((span, data))) => {
                                let can_parallel =
                                    span.items.len() >= 2 && ifd.mask_ifd.is_none();
                                if can_parallel {
                                    let meta = decode_meta_opt
                                        .get_or_insert_with(|| Arc::new(clone_for_decode(ifd)))
                                        .clone();
                                    for item in span.items {
                                        let (orig_slot, guard_opt, _, _) = &mut owned[item.slot];
                                        let orig_slot = *orig_slot;
                                        let rel_start = (item.offset - span.start) as usize;
                                        let rel_end = rel_start + item.len as usize;
                                        if rel_end > data.len() {
                                            fetch_err =
                                                Some(IoError::Tiff(TiffError::NeedMoreBytes {
                                                    required: item.offset + item.len as u64,
                                                    available: span.start + data.len() as u64,
                                                }));
                                            continue;
                                        }
                                        let slice = data.slice(rel_start..rel_end);
                                        if !is_local {
                                            if let Some(dc) = &self.disk_cache {
                                                if let Some(Some(tidx)) =
                                                    tile_indices.get(orig_slot)
                                                {
                                                    dc.put_async(
                                                        &cog.uri,
                                                        ifd_idx as u16,
                                                        *tidx,
                                                        slice.clone(),
                                                    );
                                                }
                                            }
                                        }
                                        let guard = guard_opt.take();
                                        let m = meta.clone();
                                        decode_set.spawn(async move {
                                            let decoded =
                                                decode_tile(&slice, &m, byte_order, nodata)?;
                                            let arc = Arc::new(decoded);
                                            if let Some(g) = guard {
                                                g.complete(arc.clone());
                                            }
                                            Ok((orig_slot, arc))
                                        });
                                    }
                                } else {
                                    for item in span.items {
                                        let (orig_slot, guard_opt, _, _) = &mut owned[item.slot];
                                        let rel_start = (item.offset - span.start) as usize;
                                        let rel_end = rel_start + item.len as usize;
                                        if rel_end > data.len() {
                                            fetch_err =
                                                Some(IoError::Tiff(TiffError::NeedMoreBytes {
                                                    required: item.offset + item.len as u64,
                                                    available: span.start + data.len() as u64,
                                                }));
                                            continue;
                                        }
                                        if !is_local && ifd.mask_ifd.is_none() {
                                            if let Some(dc) = &self.disk_cache {
                                                if let Some(Some(tidx)) =
                                                    tile_indices.get(*orig_slot)
                                                {
                                                    dc.put_async(
                                                        &cog.uri,
                                                        ifd_idx as u16,
                                                        *tidx,
                                                        data.slice(rel_start..rel_end),
                                                    );
                                                }
                                            }
                                        }
                                        match decode_tile(
                                            &data[rel_start..rel_end],
                                            ifd,
                                            byte_order,
                                            nodata,
                                        ) {
                                            Ok(mut decoded) => {
                                                if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                                                    if mask_ifd.tile_width == ifd.tile_width
                                                        && mask_ifd.tile_height == ifd.tile_height
                                                    {
                                                        if let Some(Some(tidx)) =
                                                            tile_indices.get(*orig_slot)
                                                        {
                                                            let per_plane =
                                                                (ifd.tiles_across as usize)
                                                                    * (ifd.tiles_down as usize);
                                                            let mask_idx =
                                                                (*tidx as usize) % per_plane.max(1);
                                                            if let Some((moff, mlen)) =
                                                                mask_ifd.tile_byte_range(mask_idx)
                                                            {
                                                                if let Ok(mbytes) = cog
                                                                    .source
                                                                    .fetch_range(
                                                                        moff,
                                                                        moff + mlen as u64,
                                                                    )
                                                                    .await
                                                                {
                                                                    if let Ok(mask) =
                                                                        decode_mask_tile(
                                                                            &mbytes, mask_ifd,
                                                                        )
                                                                    {
                                                                        apply_mask_to_tile(
                                                                            &mut decoded,
                                                                            &mask,
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                let arc = Arc::new(decoded);
                                                if let Some(guard) = guard_opt.take() {
                                                    guard.complete(arc.clone());
                                                }
                                                results[*orig_slot] = Some(arc);
                                            }
                                            Err(e) => {
                                                fetch_err = Some(IoError::Codec(e));
                                            }
                                        }
                                    }
                                }
                            }
                            Ok(Err(e)) => fetch_err = Some(e),
                            Err(_) => {}
                        }
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => {
                                results[orig_slot] = Some(arc);
                            }
                            Ok(Err(e)) => fetch_err = Some(e),
                            Err(_) => {}
                        }
                    }

                    if let Some(e) = fetch_err {
                        // Any uncompleted guards in `owned` auto-abort cleanly on Drop!
                        return Err(e);
                    }
                }
            }
        }

        // Resolve any singleflight waiters; if an owner was cancelled/aborted mid-flight,
        // seamlessly take over and fetch those orphaned slots instead of deadlocking!
        let mut retry_slots: Vec<(usize, u32)> = Vec::new();
        for (slot, key, mut notify) in waiters {
            loop {
                let switch_to = {
                    let fut = notify.notified();
                    tokio::pin!(fut);
                    fut.as_mut().enable();
                    match self.cache.peek_state(&key) {
                        Some(Ok(hit)) => {
                            results[slot] = Some(hit);
                            break;
                        }
                        Some(Err(live_notify)) => {
                            if !Arc::ptr_eq(&notify, &live_notify) {
                                Some(live_notify)
                            } else {
                                fut.await;
                                None
                            }
                        }
                        None => {
                            retry_slots.push((slot, key.tile_idx));
                            break;
                        }
                    }
                };
                if let Some(new_notify) = switch_to {
                    notify = new_notify;
                }
            }
        }

        if !retry_slots.is_empty() {
            let retry_indices: Vec<Option<u32>> =
                retry_slots.iter().map(|&(_, tidx)| Some(tidx)).collect();
            let recovered = Box::pin(self.get_blocks(cog, ifd_idx, &retry_indices)).await?;
            for ((slot, _), tile) in retry_slots.into_iter().zip(recovered.into_iter()) {
                results[slot] = tile;
            }
        }

        // Fill any duplicate slots that shared a tile_idx with an owned slot
        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            if results[slot].is_none() {
                if let Some(tile_idx) = opt_idx {
                    if let Some(&first_slot) = seen_owned.get(&tile_idx) {
                        results[slot] = results[first_slot].clone();
                    }
                }
            }
        }

        Ok(results)
    }

    /// Fetch raw compressed bytes for a single COG internal tile, bypassing decoding.
    ///
    /// Used by zero-decode pass-through for WebMercator-aligned JPEG/WebP COGs.
    /// Respects the tier-2 persistent disk cache for remote sources and zero-copy mmap
    /// slicing for local files.
    pub async fn get_raw_block(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_idx: u32,
        offset: u64,
        byte_len: u32,
    ) -> Result<bytes::Bytes, IoError> {
        let is_local = matches!(cog.source, CogSource::LocalMmap { .. });
        if !is_local {
            if let Some(dc) = &self.disk_cache {
                if let Some(cached) = dc.get(cog.uri.clone(), ifd_idx as u16, tile_idx).await {
                    return Ok(cached);
                }
            }
        }
        let raw = cog
            .source
            .fetch_range(offset, offset + byte_len as u64)
            .await?;
        if !is_local {
            if let Some(dc) = &self.disk_cache {
                dc.put_async(&cog.uri, ifd_idx as u16, tile_idx, raw.clone());
            }
        }
        Ok(raw)
    }
}
