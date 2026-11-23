use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cogwheel_codec::{DecodedTile, apply_mask_to_tile, decode_mask_tile, decode_tile};
use cogwheel_tiff::{CogIndex, TiffError, parse_cog};
use dashmap::DashMap;

use crate::backend::{CogSource, IoError, IoMetrics, TileByteReq, coalesce_ranges};
use crate::s3::S3Config;
use crate::cache::{BlockKey, InFlightGuard, LookupResult, SingleflightBlockCache};

const INITIAL_HEADER_FETCH_BYTES: u64 = 128 * 1024;
const RANGE_GAP_THRESHOLD: u64 = 32 * 1024;
const MAX_COALESCED_SPAN: u64 = 4 * 1024 * 1024;

pub struct OpenedCog {
    pub id: u64,
    pub uri: String,
    pub index: CogIndex,
    pub source: CogSource,
}

enum OpenSlot {
    Ready(Arc<OpenedCog>),
    InFlight(Arc<tokio::sync::Notify>),
}

/// Manages opened COGs (local mmap + remote HTTP/S3) and the shared singleflight block cache.
pub struct CogStore {
    next_id: AtomicU64,
    datasets_by_name: DashMap<String, Arc<OpenedCog>, ahash::RandomState>,
    datasets_by_uri: DashMap<String, OpenSlot, ahash::RandomState>,
    pub cache: Arc<SingleflightBlockCache>,
    pub client: reqwest::Client,
    pub s3_cfg: Option<S3Config>,
    pub metrics: Arc<IoMetrics>,
}

impl CogStore {
    pub fn new(cache_max_bytes: usize) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .build()
            .expect("reqwest client");
        Self {
            next_id: AtomicU64::new(1),
            datasets_by_name: DashMap::with_hasher(ahash::RandomState::new()),
            datasets_by_uri: DashMap::with_hasher(ahash::RandomState::new()),
            cache: Arc::new(SingleflightBlockCache::new(cache_max_bytes)),
            client,
            s3_cfg: S3Config::from_env(),
            metrics: Arc::new(IoMetrics::default()),
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
        self.datasets_by_uri.insert(uri, OpenSlot::Ready(cog.clone()));
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
                if let OpenSlot::Ready(cog) = existing.value() {
                    return Ok(cog.clone());
                }
            }

            let wait_notify = {
                use dashmap::mapref::entry::Entry;
                match self.datasets_by_uri.entry(uri.to_string()) {
                    Entry::Occupied(occ) => match occ.get() {
                        OpenSlot::Ready(cog) => return Ok(cog.clone()),
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
                if let Some(existing) = self.datasets_by_uri.get(uri) {
                    if let OpenSlot::Ready(cog) = existing.value() {
                        return Ok(cog.clone());
                    }
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
                        if let Some((_, OpenSlot::InFlight(n))) = self.map.remove(&self.uri) {
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

            let source = CogSource::open_remote(
                uri,
                self.client.clone(),
                self.s3_cfg.clone(),
                self.metrics.clone(),
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
            if let Some(OpenSlot::InFlight(n)) =
                self.datasets_by_uri.insert(uri.to_string(), OpenSlot::Ready(cog.clone()))
            {
                n.notify_waiters();
            }
            return Ok(cog);
        }
    }

    /// Synchronous zero-allocation fast path for local `mmap` or fully cached blocks.
    /// Returns `Ok(Some(blocks))` if satisfied immediately without awaiting; `Ok(None)` if async I/O is needed.
    pub fn try_get_blocks_sync(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_indices: &[Option<u32>],
    ) -> Result<Option<Vec<Option<Arc<DecodedTile>>>>, IoError> {
        let ifd = &cog.index.ifds[ifd_idx];
        let mut out = Vec::with_capacity(tile_indices.len());

        for &opt_idx in tile_indices {
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

            // Local mmap miss -> decode synchronously on this core with zero async overhead!
            let offset = ifd.tile_offsets[tile_idx as usize];
            if let Some(slice) = cog.source.local_slice(offset, count) {
                match self.cache.acquire(key) {
                    LookupResult::Hit(hit) => out.push(Some(hit)),
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
                        out.push(Some(arc));
                    }
                    LookupResult::Wait(_) => {
                        return Ok(None);
                    }
                }
            } else {
                return Ok(None);
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

        // Fetch all owned misses using coalesced concurrent Range requests!
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

            let mut fetch_err: Option<IoError> = None;
            while let Some(res) = join_set.join_next().await {
                match res {
                    Ok(Ok((span, data))) => {
                        for item in span.items {
                            let (orig_slot, guard_opt, _, _) = &mut owned[item.slot];
                            let rel_start = (item.offset - span.start) as usize;
                            let rel_end = rel_start + item.len as usize;
                            if rel_end > data.len() {
                                fetch_err = Some(IoError::Tiff(TiffError::NeedMoreBytes {
                                    required: item.offset + item.len as u64,
                                    available: span.start + data.len() as u64,
                                }));
                                continue;
                            }
                            match decode_tile(
                                &data[rel_start..rel_end],
                                ifd,
                                cog.index.byte_order,
                                cog.index.nodata,
                            ) {
                                Ok(mut decoded) => {
                                    if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                                        if mask_ifd.tile_width == ifd.tile_width
                                            && mask_ifd.tile_height == ifd.tile_height
                                        {
                                            if let Some(Some(tidx)) = tile_indices.get(*orig_slot) {
                                                let per_plane = (ifd.tiles_across as usize)
                                                    * (ifd.tiles_down as usize);
                                                let mask_idx = (*tidx as usize) % per_plane.max(1);
                                                if let Some((moff, mlen)) =
                                                    mask_ifd.tile_byte_range(mask_idx)
                                                {
                                                    if let Ok(mbytes) = cog
                                                        .source
                                                        .fetch_range(moff, moff + mlen as u64)
                                                        .await
                                                    {
                                                        if let Ok(mask) =
                                                            decode_mask_tile(&mbytes, mask_ifd)
                                                        {
                                                            apply_mask_to_tile(&mut decoded, &mask);
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
                    Ok(Err(e)) => fetch_err = Some(e),
                    Err(_) => {}
                }
            }

            if let Some(e) = fetch_err {
                // Any uncompleted guards in `owned` auto-abort cleanly on Drop!
                return Err(e);
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
}
