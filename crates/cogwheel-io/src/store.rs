use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cogwheel_codec::{DecodedTile, decode_tile};
use cogwheel_tiff::{CogIndex, TiffError, parse_cog};
use dashmap::DashMap;

use crate::backend::{CogSource, IoError, IoMetrics, S3Config, TileByteReq, coalesce_ranges};
use crate::cache::{BlockKey, LookupResult, SingleflightBlockCache};

const INITIAL_HEADER_FETCH_BYTES: u64 = 128 * 1024;
const RANGE_GAP_THRESHOLD: u64 = 32 * 1024;
const MAX_COALESCED_SPAN: u64 = 4 * 1024 * 1024;

pub struct OpenedCog {
    pub id: u64,
    pub uri: String,
    pub index: CogIndex,
    pub source: CogSource,
}

/// Manages opened COGs (local mmap + remote HTTP/S3) and the shared singleflight block cache.
pub struct CogStore {
    next_id: AtomicU64,
    datasets_by_name: DashMap<String, Arc<OpenedCog>, ahash::RandomState>,
    datasets_by_uri: DashMap<String, Arc<OpenedCog>, ahash::RandomState>,
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
        self.datasets_by_uri.insert(uri, cog.clone());
        Ok(cog)
    }

    pub fn get_registered(&self, name: &str) -> Option<Arc<OpenedCog>> {
        self.datasets_by_name.get(name).map(|v| v.clone())
    }

    /// Open (or fetch from metadata cache) any local path, `file://`, `http(s)://`, or `s3://` COG.
    pub async fn open_uri(&self, uri: &str) -> Result<Arc<OpenedCog>, IoError> {
        if let Some(existing) = self.datasets_by_uri.get(uri) {
            return Ok(existing.clone());
        }

        let source = CogSource::open_remote(
            uri,
            self.client.clone(),
            self.s3_cfg.clone(),
            self.metrics.clone(),
        )?;

        let index = match &source {
            CogSource::LocalMmap { mmap } => parse_cog(mmap)?,
            CogSource::RemoteHttp { .. } => {
                // Speculative 128 KiB single-RTT header read; at most 1 follow-up if IFD arrays spill past 128 KiB
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
        self.datasets_by_uri.insert(uri.to_string(), cog.clone());
        Ok(cog)
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
                        match decode_tile(slice, ifd, cog.index.byte_order, cog.index.nodata) {
                            Ok(decoded) => {
                                let arc = Arc::new(decoded);
                                self.cache.complete(key, arc.clone());
                                out.push(Some(arc));
                            }
                            Err(e) => {
                                self.cache.abort(&key);
                                return Err(IoError::Codec(e));
                            }
                        }
                    }
                    LookupResult::Wait(_) => {
                        // Another thread is decoding this block; yield to async path to await Notify
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
        let mut owned: Vec<(usize, BlockKey, u64, u32)> = Vec::new();
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
                    owned.push((slot, key, offset, count));
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

            // Execute all coalesced HTTP Range fetches concurrently!
            let mut join_set = tokio::task::JoinSet::new();
            for fut in span_futs {
                join_set.spawn(fut);
            }

            let mut fetch_err: Option<IoError> = None;
            while let Some(res) = join_set.join_next().await {
                match res {
                    Ok(Ok((span, data))) => {
                        for item in span.items {
                            let (orig_slot, key, _, _) = owned[item.slot];
                            let rel_start = (item.offset - span.start) as usize;
                            let rel_end = rel_start + item.len as usize;
                            if rel_end > data.len() {
                                self.cache.abort(&key);
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
                                Ok(decoded) => {
                                    let arc = Arc::new(decoded);
                                    self.cache.complete(key, arc.clone());
                                    results[orig_slot] = Some(arc);
                                }
                                Err(e) => {
                                    self.cache.abort(&key);
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
                for (_, key, _, _) in &owned {
                    self.cache.abort(key);
                }
                return Err(e);
            }
        }

        // Resolve any singleflight waiters
        for (slot, key, notify) in waiters {
            loop {
                if let Some(hit) = self.cache.try_get_fast(&key) {
                    results[slot] = Some(hit);
                    break;
                }
                notify.notified().await;
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
