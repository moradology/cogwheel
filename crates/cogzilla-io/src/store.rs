use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use cogzilla_codec::{
    DecodedTile, apply_alpha_band_to_tile, apply_mask_to_tile, decode_mask_tile_rows,
    decode_tile_rows,
};
use cogzilla_tiff::{CogIndex, IfdMeta, PlanarConfig, TiffError, parse_cog};
use dashmap::DashMap;

/// Return the 0-based alpha plane index if this IFD is a `PlanarConfig::Planar` COG with an
/// alpha channel (`ExtraSamples` tag 338 containing 1 or 2, e.g. Gray+Alpha or RGBA).
fn planar_alpha_plane(ifd: &IfdMeta) -> Option<usize> {
    if ifd.planar_config != PlanarConfig::Planar {
        return None;
    }
    cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
}

/// Lookup result for a paired planar alpha-plane tile or internal mask tile.
enum SidecarTileRange {
    /// Non-empty tile at `(offset, byte_len)`.
    Present(u64, u32),
    /// Sparse (`byte_count == 0`) mask/alpha tile — 100% transparent, zero I/O needed!
    SparseTransparent,
    /// Not applicable (e.g. querying the alpha plane itself) or out of bounds.
    NotApplicable,
}

fn mark_tile_all_nodata(tile: &mut DecodedTile) {
    for b in &mut tile.bands {
        b.data.make_f32_mut().fill(f32::NAN);
        b.stats.all_nodata = true;
        b.stats.no_nodata = false;
    }
}

/// Given a planar color-plane `tile_idx`, return its paired alpha-plane tile range status.
fn planar_alpha_tile_range(ifd: &IfdMeta, tile_idx: u32, alpha_plane: usize) -> SidecarTileRange {
    let per_plane = ((ifd.tiles_across as usize) * (ifd.tiles_down as usize)).max(1);
    let plane = (tile_idx as usize) / per_plane;
    if plane == alpha_plane {
        return SidecarTileRange::NotApplicable;
    }
    let alpha_idx = alpha_plane * per_plane + (tile_idx as usize) % per_plane;
    let Some(&offset) = ifd.tile_offsets.get(alpha_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    let Some(&len) = ifd.tile_byte_counts.get(alpha_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    if len == 0 {
        SidecarTileRange::SparseTransparent
    } else {
        SidecarTileRange::Present(offset, len)
    }
}

fn mask_tile_range(mask_ifd: &cogzilla_tiff::MaskIfdMeta, mask_idx: usize) -> SidecarTileRange {
    let Some(&len) = mask_ifd.tile_byte_counts.get(mask_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    if len == 0 {
        return SidecarTileRange::SparseTransparent;
    }
    match mask_ifd.tile_offsets.get(mask_idx) {
        Some(&off) => SidecarTileRange::Present(off, len),
        None => SidecarTileRange::NotApplicable,
    }
}

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
    /// Bounds concurrent heavy CPU offloads (`spawn_blocking` for >2 MiB blocks, large stats/bbox)
    /// to `num_cpus` so pathological workloads never oversubscribe OS threads or starve Tokio I/O.
    pub heavy_cpu_sem: Arc<tokio::sync::Semaphore>,
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
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 10 {
                    return attempt.error("too many HTTP redirects");
                }
                if let Err(e) = crate::allowlist::check_uri_allowed(attempt.url().as_str()) {
                    return attempt.error(e);
                }
                attempt.follow()
            }))
            .build()
            .expect("reqwest client");
        let heavy_permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
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
            heavy_cpu_sem: Arc::new(tokio::sync::Semaphore::new(heavy_permits)),
            client,
            s3_cfg: S3Config::from_env(),
            metrics: Arc::new(IoMetrics::default()),
        }
    }

    /// Run a heavy synchronous CPU closure on Tokio's blocking pool gated by `heavy_cpu_sem`
    /// (`num_cpus` permits) so large blocks, `/cog/statistics`, and giant bbox crops never stall
    /// Tokio's async I/O reactor threads or oversubscribe the blocking pool.
    pub async fn run_heavy_cpu<F, R>(&self, f: F) -> Result<R, IoError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let _permit = self
            .heavy_cpu_sem
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| IoError::Fs(std::io::Error::other(format!("heavy CPU semaphore closed: {e}"))))?;
        tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| IoError::Fs(std::io::Error::other(format!("heavy CPU task failed: {e}"))))
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
        crate::allowlist::check_uri_allowed(uri)?;
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
                CogSource::RemoteS3 { .. }
                | CogSource::RemoteHttp { .. }
                | CogSource::RemoteGcs { .. }
                | CogSource::RemoteAzure { .. } => {
                    // Speculative 128 KiB single-RTT header read; bounded at 32 MiB to prevent
                    // crafted non-COG / BigTIFF offsets from causing OOM or u64 overflow.
                    const MAX_HEADER_FETCH_BYTES: u64 = 32 * 1024 * 1024;
                    let mut hdr = source.fetch_range(0, INITIAL_HEADER_FETCH_BYTES).await?.to_vec();
                    loop {
                        match parse_cog(&hdr) {
                            Ok(idx) => break idx,
                            Err(TiffError::NeedMoreBytes { required, .. }) => {
                                let cur_len = hdr.len() as u64;
                                if required > MAX_HEADER_FETCH_BYTES {
                                    return Err(IoError::Tiff(TiffError::Corrupt(
                                        "TIFF header/IFD exceeds 32 MiB limit",
                                    )));
                                }
                                let target = required
                                    .checked_add(65535)
                                    .map(|v| (v / 65536) * 65536)
                                    .unwrap_or(MAX_HEADER_FETCH_BYTES)
                                    .min(MAX_HEADER_FETCH_BYTES);
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
                    let rows = ifd.block_height(tile_idx);
                    let mut decoded = decode_tile_rows(
                        slice,
                        ifd,
                        cog.index.byte_order,
                        cog.index.nodata,
                        rows,
                    )?;
                    if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                        if mask_ifd.tile_width == ifd.tile_width
                            && mask_ifd.tile_height == ifd.tile_height
                        {
                            let per_plane =
                                (ifd.tiles_across as usize) * (ifd.tiles_down as usize);
                            let mask_idx = (tile_idx as usize) % per_plane.max(1);
                            match mask_tile_range(mask_ifd, mask_idx) {
                                SidecarTileRange::Present(moff, mlen) => {
                                    let Some(mslice) = cog.source.local_slice(moff, mlen) else {
                                        return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                            required: moff.saturating_add(mlen as u64),
                                            available: 0,
                                        }));
                                    };
                                    let mask = decode_mask_tile_rows(
                                        mslice,
                                        mask_ifd,
                                        mask_ifd.block_height(mask_idx),
                                    )?;
                                    apply_mask_to_tile(&mut decoded, &mask);
                                }
                                SidecarTileRange::SparseTransparent => {
                                    mark_tile_all_nodata(&mut decoded);
                                }
                                SidecarTileRange::NotApplicable => {}
                            }
                        }
                    }
                    if let Some(alpha_plane) = planar_alpha_plane(ifd) {
                        match planar_alpha_tile_range(ifd, tile_idx, alpha_plane) {
                            SidecarTileRange::Present(aoff, alen) => {
                                let Some(aslice) = cog.source.local_slice(aoff, alen) else {
                                    return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                        required: aoff.saturating_add(alen as u64),
                                        available: 0,
                                    }));
                                };
                                let alpha_tile = decode_tile_rows(
                                    aslice,
                                    ifd,
                                    cog.index.byte_order,
                                    None,
                                    rows,
                                )?;
                                if let Some(ab) = alpha_tile.bands.first() {
                                    apply_alpha_band_to_tile(&mut decoded, ab);
                                }
                            }
                            SidecarTileRange::SparseTransparent => {
                                mark_tile_all_nodata(&mut decoded);
                            }
                            SidecarTileRange::NotApplicable => {}
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

        // Fetch & decode all owned misses (bounded multi-core decode when beneficial)
        if !owned.is_empty() {
            let is_local = matches!(cog.source, CogSource::LocalMmap { .. });
            let num_workers = tokio::runtime::Handle::current()
                .metrics()
                .num_workers()
                .max(1);
            let expected_bytes = (ifd.tile_width as usize)
                .saturating_mul(ifd.tile_height as usize)
                .saturating_mul(ifd.samples_per_pixel as usize)
                .saturating_mul(ifd.sample_type.bytes_per_sample());
            const HEAVY_DECODE_BYTES: usize = 2 * 1024 * 1024;
            let heavy_block = expected_bytes > HEAVY_DECODE_BYTES;
            let trivial_block = ifd.compression == cogzilla_tiff::Compression::None
                && ifd.predictor == cogzilla_tiff::Predictor::None
                && expected_bytes <= 256 * 1024;

            let alpha_plane = planar_alpha_plane(ifd);
            let no_sidecar_mask = ifd.mask_ifd.is_none() && alpha_plane.is_none();

            if is_local {
                if heavy_block && no_sidecar_mask {
                    let decode_meta = Arc::new(clone_for_decode(ifd));
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata;
                    let mut decode_set = tokio::task::JoinSet::new();
                    let mut decode_err: Option<IoError> = None;

                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        let rows = tile_indices[orig_slot]
                            .map(|tidx| ifd.block_height(tidx))
                            .unwrap_or(ifd.tile_height);
                        let sem = self.heavy_cpu_sem.clone();
                        let src = cog.source.clone();
                        let meta = decode_meta.clone();
                        decode_set.spawn(async move {
                            let _permit = sem.acquire_owned().await.map_err(|e| {
                                IoError::Fs(std::io::Error::other(format!(
                                    "heavy CPU semaphore closed: {e}"
                                )))
                            })?;
                            let decoded = tokio::task::spawn_blocking(move || {
                                let Some(slice) = src.local_slice(offset, count) else {
                                    return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                        required: offset + count as u64,
                                        available: 0,
                                    }));
                                };
                                Ok::<_, IoError>(decode_tile_rows(
                                    slice, &meta, byte_order, nodata, rows,
                                )?)
                            })
                            .await
                            .map_err(|e| {
                                IoError::Fs(std::io::Error::other(format!(
                                    "heavy local decode task failed: {e}"
                                )))
                            })??;
                            let arc = Arc::new(decoded);
                            if let Some(guard) = guard_opt {
                                guard.complete(arc.clone());
                            }
                            Ok::<_, IoError>((orig_slot, arc))
                        });
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => results[orig_slot] = Some(arc),
                            Ok(Err(e)) => decode_err = Some(e),
                            Err(e) => {
                                decode_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "heavy local decode task failed: {e}"
                                ))));
                            }
                        }
                    }
                    if let Some(e) = decode_err {
                        return Err(e);
                    }
                } else if num_workers >= 2
                    && owned.len() >= 2
                    && !trivial_block
                    && no_sidecar_mask
                {
                    let decode_meta = Arc::new(clone_for_decode(ifd));
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata;
                    let mut decode_set = tokio::task::JoinSet::new();
                    let mut decode_err: Option<IoError> = None;

                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        if decode_set.len() >= num_workers {
                            if let Some(res) = decode_set.join_next().await {
                                match res {
                                    Ok(Ok((s, arc))) => results[s] = Some(arc),
                                    Ok(Err(e)) => decode_err = Some(e),
                                    Err(e) => {
                                        decode_err = Some(IoError::Fs(std::io::Error::other(
                                            format!("local decode task failed: {e}"),
                                        )));
                                    }
                                }
                            }
                        }
                        let rows = tile_indices[orig_slot]
                            .map(|tidx| ifd.block_height(tidx))
                            .unwrap_or(ifd.tile_height);
                        let src = cog.source.clone();
                        let meta = decode_meta.clone();
                        decode_set.spawn(async move {
                            let Some(slice) = src.local_slice(offset, count) else {
                                return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                    required: offset + count as u64,
                                    available: 0,
                                }));
                            };
                            let decoded =
                                decode_tile_rows(slice, &meta, byte_order, nodata, rows)?;
                            let arc = Arc::new(decoded);
                            if let Some(guard) = guard_opt {
                                guard.complete(arc.clone());
                            }
                            Ok::<_, IoError>((orig_slot, arc))
                        });
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => {
                                results[orig_slot] = Some(arc);
                            }
                            Ok(Err(e)) => decode_err = Some(e),
                            Err(e) => {
                                decode_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "local decode task failed: {e}"
                                ))));
                            }
                        }
                    }
                    if let Some(e) = decode_err {
                        return Err(e);
                    }
                } else {
                    // Zero-copy inline mmap decode (covers single-block, 1-vCPU, trivial
                    // uncompressed blocks, and local COGs with an internal mask_ifd).
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata;
                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        let Some(slice) = cog.source.local_slice(offset, count) else {
                            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                required: offset + count as u64,
                                available: 0,
                            }));
                        };
                        let rows = tile_indices[orig_slot]
                            .map(|tidx| ifd.block_height(tidx))
                            .unwrap_or(ifd.tile_height);
                        let mut decoded = decode_tile_rows(slice, ifd, byte_order, nodata, rows)?;
                        if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                            if mask_ifd.tile_width == ifd.tile_width
                                && mask_ifd.tile_height == ifd.tile_height
                            {
                                if let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                    let per_plane = (ifd.tiles_across as usize)
                                        * (ifd.tiles_down as usize);
                                    let mask_idx = (*tidx as usize) % per_plane.max(1);
                                    match mask_tile_range(mask_ifd, mask_idx) {
                                        SidecarTileRange::Present(moff, mlen) => {
                                            let Some(mslice) = cog.source.local_slice(moff, mlen)
                                            else {
                                                return Err(IoError::Tiff(
                                                    TiffError::NeedMoreBytes {
                                                        required: moff + mlen as u64,
                                                        available: 0,
                                                    },
                                                ));
                                            };
                                            let mask = decode_mask_tile_rows(
                                                mslice,
                                                mask_ifd,
                                                mask_ifd.block_height(mask_idx),
                                            )?;
                                            apply_mask_to_tile(&mut decoded, &mask);
                                        }
                                        SidecarTileRange::SparseTransparent => {
                                            mark_tile_all_nodata(&mut decoded);
                                        }
                                        SidecarTileRange::NotApplicable => {}
                                    }
                                }
                            }
                        }
                        if let Some(ap) = alpha_plane {
                            if let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                match planar_alpha_tile_range(ifd, *tidx, ap) {
                                    SidecarTileRange::Present(aoff, alen) => {
                                        let Some(aslice) = cog.source.local_slice(aoff, alen)
                                        else {
                                            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                                required: aoff + alen as u64,
                                                available: 0,
                                            }));
                                        };
                                        let alpha_tile =
                                            decode_tile_rows(aslice, ifd, byte_order, None, rows)?;
                                        if let Some(ab) = alpha_tile.bands.first() {
                                            apply_alpha_band_to_tile(&mut decoded, ab);
                                        }
                                    }
                                    SidecarTileRange::SparseTransparent => {
                                        mark_tile_all_nodata(&mut decoded);
                                    }
                                    SidecarTileRange::NotApplicable => {}
                                }
                            }
                        }
                        let arc = Arc::new(decoded);
                        if let Some(guard) = guard_opt {
                            guard.complete(arc.clone());
                        }
                        results[orig_slot] = Some(arc);
                    }
                }
            } else {
                // Tier-2 compressed-block disk cache lookup for remote sources (RAM miss -> disk hit).
                // Synchronously probe the in-memory index first so 100% cold misses do 0 JoinSet/spawn_blocking hops!
                if no_sidecar_mask {
                    if let Some(dc) = &self.disk_cache {
                        let prev_owned = std::mem::take(&mut owned);
                        let mut disk_candidates = Vec::new();
                        for item in prev_owned {
                            let tidx = tile_indices[item.0].unwrap();
                            if let Some((key, path)) =
                                dc.lookup_path(&cog.uri, ifd_idx as u16, tidx)
                            {
                                disk_candidates.push((item, key, path));
                            } else {
                                owned.push(item);
                            }
                        }

                        if !disk_candidates.is_empty() {
                            let meta = Arc::new(clone_for_decode(ifd));
                            let byte_order = cog.index.byte_order;
                            let nodata = cog.index.nodata;
                            let mut disk_set = tokio::task::JoinSet::new();
                            for (item, key, path) in disk_candidates {
                                let (orig_slot, guard_opt, offset, count) = item;
                                let rows = tile_indices[orig_slot]
                                    .map(|tidx| ifd.block_height(tidx))
                                    .unwrap_or(ifd.tile_height);
                                let dc_c = dc.clone();
                                let m = meta.clone();
                                disk_set.spawn(async move {
                                    let read_res = tokio::task::spawn_blocking(move || {
                                        dc_c.read_verified_sync(key, &path)
                                    })
                                    .await
                                    .ok()
                                    .flatten();
                                    if let Some(compressed) = read_res {
                                        if let Ok(decoded) = decode_tile_rows(
                                            &compressed,
                                            &m,
                                            byte_order,
                                            nodata,
                                            rows,
                                        ) {
                                            let arc = Arc::new(decoded);
                                            if let Some(guard) = guard_opt {
                                                guard.complete(arc.clone());
                                            }
                                            return Ok::<
                                                _,
                                                (usize, Option<InFlightGuard>, u64, u32),
                                            >(
                                                (orig_slot, arc)
                                            );
                                        }
                                    }
                                    // Corrupt/evicted disk entry: fail-open and refetch from remote!
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
                                    Err(e) => {
                                        return Err(IoError::Fs(std::io::Error::other(format!(
                                            "disk cache decode task failed: {e}"
                                        ))));
                                    }
                                }
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
                    let mut spans =
                        coalesce_ranges(reqs, RANGE_GAP_THRESHOLD, MAX_COALESCED_SPAN);

                    // Parallel decode set capped at `num_workers` in-flight so wide requests
                    // overlap download+decode without flooding Tokio's async workers.
                    let mut decode_set: tokio::task::JoinSet<
                        Result<(usize, Arc<DecodedTile>), IoError>,
                    > = tokio::task::JoinSet::new();
                    let mut decode_meta_opt: Option<Arc<IfdMeta>> = None;
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata;
                    let mut fetch_err: Option<IoError> = None;

                    // Fast path: 1 coalesced span -> await fetch_range directly (no JoinSet spawn)
                    let mut single_span = if spans.len() == 1 {
                        let span = spans.pop().unwrap();
                        let data = cog.source.fetch_range(span.start, span.end).await?;
                        Some((span, data))
                    } else {
                        None
                    };

                    let mut join_set = tokio::task::JoinSet::new();
                    if single_span.is_none() {
                        for span in spans {
                            let src = cog.source.clone();
                            join_set.spawn(async move {
                                let data = src.fetch_range(span.start, span.end).await?;
                                Ok::<_, IoError>((span, data))
                            });
                        }
                    }

                    loop {
                        let next_span = if let Some(pair) = single_span.take() {
                            Some(Ok(pair))
                        } else {
                            match join_set.join_next().await {
                                Some(Ok(r)) => Some(r),
                                Some(Err(e)) => {
                                    fetch_err = Some(IoError::Fs(std::io::Error::other(format!(
                                        "remote fetch task failed: {e}"
                                    ))));
                                    continue;
                                }
                                None => None,
                            }
                        };
                        let Some(span_res) = next_span else {
                            break;
                        };
                        match span_res {
                            Ok((span, data)) => {
                                let has_more_fetches = !join_set.is_empty();
                                let can_parallel = no_sidecar_mask
                                    && (heavy_block
                                        || (!trivial_block
                                            && ((num_workers >= 2 && span.items.len() >= 2)
                                                || has_more_fetches)));
                                if can_parallel {
                                    let meta = decode_meta_opt
                                        .get_or_insert_with(|| Arc::new(clone_for_decode(ifd)))
                                        .clone();
                                    for item in span.items {
                                        if decode_set.len() >= num_workers {
                                            if let Some(res) = decode_set.join_next().await {
                                                match res {
                                                    Ok(Ok((s, arc))) => results[s] = Some(arc),
                                                    Ok(Err(e)) => fetch_err = Some(e),
                                                    Err(e) => {
                                                        fetch_err = Some(IoError::Fs(
                                                            std::io::Error::other(format!(
                                                                "remote decode task failed: {e}"
                                                            )),
                                                        ));
                                                    }
                                                }
                                            }
                                        }
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
                                        if let Some(dc) = &self.disk_cache {
                                            if let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                                dc.put_async(
                                                    &cog.uri,
                                                    ifd_idx as u16,
                                                    *tidx,
                                                    slice.clone(),
                                                );
                                            }
                                        }
                                        let rows = tile_indices[orig_slot]
                                            .map(|tidx| ifd.block_height(tidx))
                                            .unwrap_or(ifd.tile_height);
                                        let guard = guard_opt.take();
                                        let m = meta.clone();
                                        let sem = if heavy_block {
                                            Some(self.heavy_cpu_sem.clone())
                                        } else {
                                            None
                                        };
                                        decode_set.spawn(async move {
                                            let decoded = if let Some(sem) = sem {
                                                let _permit =
                                                    sem.acquire_owned().await.map_err(|e| {
                                                        IoError::Fs(std::io::Error::other(format!(
                                                            "heavy CPU semaphore closed: {e}"
                                                        )))
                                                    })?;
                                                tokio::task::spawn_blocking(move || {
                                                    decode_tile_rows(
                                                        &slice, &m, byte_order, nodata, rows,
                                                    )
                                                })
                                                .await
                                                .map_err(|e| {
                                                    IoError::Fs(std::io::Error::other(format!(
                                                        "heavy remote decode task failed: {e}"
                                                    )))
                                                })??
                                            } else {
                                                decode_tile_rows(
                                                    &slice, &m, byte_order, nodata, rows,
                                                )?
                                            };
                                            let arc = Arc::new(decoded);
                                            if let Some(g) = guard {
                                                g.complete(arc.clone());
                                            }
                                            Ok((orig_slot, arc))
                                        });
                                    }
                                } else {
                                    // If this IFD has a separate 1-bit transparency mask_ifd, coalesce
                                    // and fetch all needed mask ranges for this span concurrently first.
                                    let mut mask_bytes_by_slot: Vec<Option<bytes::Bytes>> =
                                        vec![None; span.items.len()];
                                    let mut sparse_transparent_by_slot: Vec<bool> =
                                        vec![false; span.items.len()];
                                    if let Some(mask_ifd) = ifd.mask_ifd.as_deref() {
                                        if mask_ifd.tile_width == ifd.tile_width
                                            && mask_ifd.tile_height == ifd.tile_height
                                        {
                                            let per_plane = (ifd.tiles_across as usize)
                                                * (ifd.tiles_down as usize);
                                            let mut mreqs = Vec::new();
                                            for (idx_in_span, item) in span.items.iter().enumerate()
                                            {
                                                let orig_slot = owned[item.slot].0;
                                                if let Some(Some(tidx)) =
                                                    tile_indices.get(orig_slot)
                                                {
                                                    let mask_idx =
                                                        (*tidx as usize) % per_plane.max(1);
                                                    match mask_tile_range(mask_ifd, mask_idx) {
                                                        SidecarTileRange::Present(moff, mlen) => {
                                                            mreqs.push(TileByteReq {
                                                                slot: idx_in_span,
                                                                offset: moff,
                                                                len: mlen,
                                                            });
                                                        }
                                                        SidecarTileRange::SparseTransparent => {
                                                            sparse_transparent_by_slot[idx_in_span] =
                                                                true;
                                                        }
                                                        SidecarTileRange::NotApplicable => {}
                                                    }
                                                }
                                            }
                                            if !mreqs.is_empty() {
                                                let mspans = coalesce_ranges(
                                                    mreqs,
                                                    RANGE_GAP_THRESHOLD,
                                                    MAX_COALESCED_SPAN,
                                                );
                                                let mut mset = tokio::task::JoinSet::new();
                                                for ms in mspans {
                                                    let src = cog.source.clone();
                                                    mset.spawn(async move {
                                                        let b = src
                                                            .fetch_range(ms.start, ms.end)
                                                            .await?;
                                                        Ok::<_, IoError>((ms, b))
                                                    });
                                                }
                                                while let Some(mres) = mset.join_next().await {
                                                    match mres {
                                                        Ok(Ok((ms, mb))) => {
                                                            for mi in ms.items {
                                                                let rs =
                                                                    (mi.offset - ms.start) as usize;
                                                                let re = rs + mi.len as usize;
                                                                if re <= mb.len() {
                                                                    mask_bytes_by_slot[mi.slot] =
                                                                        Some(mb.slice(rs..re));
                                                                } else {
                                                                    fetch_err = Some(IoError::Tiff(
                                                                        TiffError::NeedMoreBytes {
                                                                            required: mi.offset
                                                                                + mi.len as u64,
                                                                            available: ms.start
                                                                                + mb.len() as u64,
                                                                        },
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                        Ok(Err(e)) => fetch_err = Some(e),
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Fs(
                                                                std::io::Error::other(format!(
                                                                    "mask fetch task failed: {e}"
                                                                )),
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // If this IFD is a planar RGBA COG, coalesce and fetch paired
                                    // alpha-plane tiles for this span concurrently too.
                                    let mut alpha_bytes_by_slot: Vec<Option<bytes::Bytes>> =
                                        vec![None; span.items.len()];
                                    if let Some(ap) = alpha_plane {
                                        let mut areqs = Vec::new();
                                        for (idx_in_span, item) in span.items.iter().enumerate() {
                                            let orig_slot = owned[item.slot].0;
                                            if let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                                match planar_alpha_tile_range(ifd, *tidx, ap) {
                                                    SidecarTileRange::Present(aoff, alen) => {
                                                        areqs.push(TileByteReq {
                                                            slot: idx_in_span,
                                                            offset: aoff,
                                                            len: alen,
                                                        });
                                                    }
                                                    SidecarTileRange::SparseTransparent => {
                                                        sparse_transparent_by_slot[idx_in_span] =
                                                            true;
                                                    }
                                                    SidecarTileRange::NotApplicable => {}
                                                }
                                            }
                                        }
                                        if !areqs.is_empty() {
                                            let aspans = coalesce_ranges(
                                                areqs,
                                                RANGE_GAP_THRESHOLD,
                                                MAX_COALESCED_SPAN,
                                            );
                                            let mut aset = tokio::task::JoinSet::new();
                                            for aspan in aspans {
                                                let src = cog.source.clone();
                                                aset.spawn(async move {
                                                    let b = src
                                                        .fetch_range(aspan.start, aspan.end)
                                                        .await?;
                                                    Ok::<_, IoError>((aspan, b))
                                                });
                                            }
                                            while let Some(ares) = aset.join_next().await {
                                                match ares {
                                                    Ok(Ok((aspan, ab))) => {
                                                        for ai in aspan.items {
                                                            let rs = (ai.offset - aspan.start)
                                                                as usize;
                                                            let re = rs + ai.len as usize;
                                                            if re <= ab.len() {
                                                                alpha_bytes_by_slot[ai.slot] =
                                                                    Some(ab.slice(rs..re));
                                                            } else {
                                                                fetch_err = Some(IoError::Tiff(
                                                                    TiffError::NeedMoreBytes {
                                                                        required: ai.offset
                                                                            + ai.len as u64,
                                                                        available: aspan.start
                                                                            + ab.len() as u64,
                                                                    },
                                                                ));
                                                            }
                                                        }
                                                    }
                                                    Ok(Err(e)) => fetch_err = Some(e),
                                                    Err(e) => {
                                                        fetch_err = Some(IoError::Fs(
                                                            std::io::Error::other(format!(
                                                                "planar alpha fetch task failed: {e}"
                                                            )),
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if fetch_err.is_some() {
                                        continue;
                                    }

                                    for (idx_in_span, item) in span.items.into_iter().enumerate() {
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
                                        if no_sidecar_mask {
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
                                        let rows = tile_indices
                                            .get(*orig_slot)
                                            .and_then(|o| *o)
                                            .map(|tidx| ifd.block_height(tidx))
                                            .unwrap_or(ifd.tile_height);
                                        match decode_tile_rows(
                                            &data[rel_start..rel_end],
                                            ifd,
                                            byte_order,
                                            nodata,
                                            rows,
                                        ) {
                                            Ok(mut decoded) => {
                                                if sparse_transparent_by_slot[idx_in_span] {
                                                    mark_tile_all_nodata(&mut decoded);
                                                } else if let (Some(mask_ifd), Some(mbytes)) = (
                                                    ifd.mask_ifd.as_deref(),
                                                    mask_bytes_by_slot[idx_in_span].as_ref(),
                                                ) {
                                                    let mask_rows = tile_indices
                                                        .get(*orig_slot)
                                                        .and_then(|o| *o)
                                                        .map(|tidx| {
                                                            let per_plane = (ifd.tiles_across
                                                                as usize)
                                                                * (ifd.tiles_down as usize);
                                                            mask_ifd.block_height(
                                                                (tidx as usize) % per_plane.max(1),
                                                            )
                                                        })
                                                        .unwrap_or(mask_ifd.tile_height);
                                                    match decode_mask_tile_rows(
                                                        mbytes, mask_ifd, mask_rows,
                                                    ) {
                                                        Ok(mask) => {
                                                            apply_mask_to_tile(&mut decoded, &mask);
                                                        }
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Codec(e));
                                                            continue;
                                                        }
                                                    }
                                                }
                                                if let Some(abytes) =
                                                    alpha_bytes_by_slot[idx_in_span].as_ref()
                                                {
                                                    match decode_tile_rows(
                                                        abytes, ifd, byte_order, None, rows,
                                                    ) {
                                                        Ok(alpha_tile) => {
                                                            if let Some(ab) =
                                                                alpha_tile.bands.first()
                                                            {
                                                                apply_alpha_band_to_tile(
                                                                    &mut decoded,
                                                                    ab,
                                                                );
                                                            }
                                                        }
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Codec(e));
                                                            continue;
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
                            Err(e) => fetch_err = Some(e),
                        }
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => {
                                results[orig_slot] = Some(arc);
                            }
                            Ok(Err(e)) => fetch_err = Some(e),
                            Err(e) => {
                                fetch_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "remote decode task failed: {e}"
                                ))));
                            }
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
                if let Some(cached) = dc.get(&cog.uri, ifd_idx as u16, tile_idx).await {
                    return Ok(cached);
                }
            }
        }
        let end = offset.checked_add(byte_len as u64).ok_or_else(|| {
            IoError::Tiff(TiffError::NeedMoreBytes {
                required: u64::MAX,
                available: 0,
            })
        })?;
        let raw = cog.source.fetch_range(offset, end).await?;
        let want = byte_len as usize;
        if raw.len() < want {
            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                required: end,
                available: offset.saturating_add(raw.len() as u64),
            }));
        }
        let raw = if raw.len() > want {
            raw.slice(..want)
        } else {
            raw
        };
        if !is_local {
            if let Some(dc) = &self.disk_cache {
                dc.put_async(&cog.uri, ifd_idx as u16, tile_idx, raw.clone());
            }
        }
        Ok(raw)
    }

    /// Fetch a full metadata object (STAC Item JSON, MosaicJSON, etc.) from any supported URI
    /// (`s3://`, `/vsis3/`, `gs://`, `/vsigs/`, `az://`, `abfs(s)://`, `/vsiaz/`, `http(s)://`,
    /// `file://`, or local path) with allowlist checks, dynamic cloud auth (IAM/IRSA/ECS/IMDSv2,
    /// GCS ADC/GCE metadata, Azure SharedKey/SAS/Bearer), AIMD concurrency limiting, transient
    /// retries, and a hard streaming byte cap (`max_bytes`) to prevent OOM.
    pub async fn fetch_uri_bytes(
        &self,
        uri: &str,
        max_bytes: usize,
    ) -> Result<bytes::Bytes, IoError> {
        crate::allowlist::check_uri_allowed(uri)?;

        if let Some((bucket, key)) = crate::s3::parse_s3_uri(uri) {
            let cfg = self
                .s3_cfg
                .clone()
                .unwrap_or_else(crate::s3::S3Config::from_env_full);
            let dyn_creds = crate::backend::shared_dynamic_creds();
            return crate::s3::fetch_s3_object_capped(
                &self.client,
                bucket,
                key,
                &cfg,
                Some(&dyn_creds),
                None,
                max_bytes,
                Some(&self.upstream_limiter),
            )
            .await;
        }

        if let Some((bucket, key)) = crate::cloud::parse_gcs_uri(uri) {
            let cfg = crate::cloud::GcsConfig::from_env();
            let tc = crate::backend::shared_gcs_token_cache();
            return crate::cloud::fetch_gcs_object_capped(
                &self.client,
                bucket,
                key,
                &cfg,
                &tc,
                max_bytes,
                &self.metrics,
                &self.upstream_limiter,
            )
            .await;
        }

        if let Some(target) = crate::cloud::parse_azure_uri(uri, None) {
            let az_cfg = crate::cloud::AzureConfig::from_env();
            return crate::cloud::fetch_azure_object_capped(
                &self.client,
                &target,
                &az_cfg,
                max_bytes,
                &self.metrics,
                &self.upstream_limiter,
            )
            .await;
        }

        if uri.starts_with("http://") || uri.starts_with("https://") {
            let max_retries = std::env::var("GDAL_HTTP_MAX_RETRY")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .unwrap_or(4);
            for attempt in 0..=max_retries {
                let permit = self.upstream_limiter.acquire().await;
                let send_res = self.client.get(uri).send().await;
                let resp = match send_res {
                    Ok(r) => r,
                    Err(e) => {
                        if crate::s3::is_retryable_transport_error(&e) {
                            permit.on_throttle();
                            if attempt < max_retries {
                                tokio::time::sleep(crate::s3::retry_backoff_duration(
                                    attempt, None,
                                ))
                                .await;
                                continue;
                            }
                        }
                        return Err(IoError::Http(e));
                    }
                };
                let status = resp.status().as_u16();
                if crate::s3::is_retryable_status(status) {
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.trim().parse::<u64>().ok());
                    permit.on_throttle();
                    if attempt < max_retries {
                        tokio::time::sleep(crate::s3::retry_backoff_duration(
                            attempt,
                            retry_after,
                        ))
                        .await;
                        continue;
                    }
                    return Err(IoError::HttpStatus {
                        status,
                        url: uri.to_string(),
                    });
                }
                if !resp.status().is_success() {
                    return Err(IoError::HttpStatus {
                        status,
                        url: uri.to_string(),
                    });
                }
                match crate::cloud::read_response_capped(resp, max_bytes).await {
                    Ok(data) => {
                        permit.on_success();
                        return Ok(data);
                    }
                    Err(IoError::Http(e)) => {
                        if crate::s3::is_retryable_transport_error(&e) {
                            permit.on_throttle();
                            if attempt < max_retries {
                                tokio::time::sleep(crate::s3::retry_backoff_duration(
                                    attempt, None,
                                ))
                                .await;
                                continue;
                            }
                        }
                        return Err(IoError::Http(e));
                    }
                    Err(e) => return Err(e),
                }
            }
            return Err(IoError::HttpStatus {
                status: 503,
                url: uri.to_string(),
            });
        }

        let local = uri.strip_prefix("file://").unwrap_or(uri);
        let meta = std::fs::metadata(local)?;
        if meta.len() > max_bytes as u64 {
            return Err(IoError::Fs(std::io::Error::other(format!(
                "local file '{local}' size {} exceeds limit of {max_bytes} bytes",
                meta.len()
            ))));
        }
        let bytes = std::fs::read(local)?;
        Ok(bytes::Bytes::from(bytes))
    }
}
