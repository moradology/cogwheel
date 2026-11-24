use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use bytes::Bytes;
use dashmap::DashMap;
use sha2::{Digest, Sha256};

const MAGIC: [u8; 4] = *b"CGZ1";
const HEADER_LEN: usize = 8;
const WRITE_QUEUE_CAP: usize = 256;

struct DiskEntry {
    path: PathBuf,
    bytes: u64,
    last_used: AtomicU64,
}

struct WriteJob {
    key: [u8; 16],
    data: Bytes,
}

/// Snapshot of tier-2 disk cache counters for `/stats` observability.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_used: u64,
    pub max_bytes: u64,
    pub evictions: u64,
    pub write_errors: u64,
    pub read_errors: u64,
}

/// Persistent, crash-safe, fail-open tier-2 cache storing raw compressed COG blocks on local disk.
///
/// - Keyed by deterministic `SHA-256(uri, ifd_idx, tile_idx)` so cached blocks survive restarts.
/// - Files are written atomically (`tmp` + `rename`) with an 8-byte `[CGZ1 | crc32]` header by a
///   bounded background writer thread so tile responses never wait on disk I/O.
/// - Corrupted or unreadable entries are deleted and treated as misses (fail-open).
/// - LRU eviction enforces `max_bytes`.
pub struct DiskBlockCache {
    dir: PathBuf,
    max_bytes: u64,
    entries: DashMap<[u8; 16], DiskEntry, ahash::RandomState>,
    current_bytes: AtomicU64,
    clock: AtomicU64,
    evicting: AtomicBool,
    tmp_seq: AtomicU64,
    tx: std::sync::mpsc::SyncSender<WriteJob>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub write_errors: AtomicU64,
    pub read_errors: AtomicU64,
}

impl DiskBlockCache {
    /// Open (or create) a persistent compressed-block cache in `dir` capped at `max_bytes`.
    /// Scans existing `*.cgz` files so restarts immediately reuse previously cached blocks.
    pub fn open(dir: impl AsRef<Path>, max_bytes: u64) -> std::io::Result<Arc<Self>> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let (tx, rx) = std::sync::mpsc::sync_channel::<WriteJob>(WRITE_QUEUE_CAP);
        let cache = Arc::new(Self {
            dir: dir.clone(),
            max_bytes: max_bytes.max(1),
            entries: DashMap::with_hasher(ahash::RandomState::new()),
            current_bytes: AtomicU64::new(0),
            clock: AtomicU64::new(1),
            evicting: AtomicBool::new(false),
            tmp_seq: AtomicU64::new(0),
            tx,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
            read_errors: AtomicU64::new(0),
        });

        cache.scan_existing();
        cache.evict_if_needed();

        // Dedicated background writer thread so disk flushes never block Tokio workers or responses.
        let writer_cache = Arc::downgrade(&cache);
        let _ = std::thread::Builder::new()
            .name("cogzilla-disk-cache".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let Some(c) = writer_cache.upgrade() else {
                        break;
                    };
                    c.write_sync(job.key, &job.data);
                }
            });

        Ok(cache)
    }

    #[inline]
    pub fn key_bytes(uri: &str, ifd_idx: u16, tile_idx: u32) -> [u8; 16] {
        let mut h = Sha256::new();
        h.update(uri.as_bytes());
        h.update(b"\0");
        h.update(ifd_idx.to_le_bytes());
        h.update(tile_idx.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        out
    }

    pub fn key_hex(uri: &str, ifd_idx: u16, tile_idx: u32) -> String {
        hex::encode(Self::key_bytes(uri, ifd_idx, tile_idx))
    }

    fn entry_path(&self, key: [u8; 16]) -> PathBuf {
        let key_hex = hex::encode(key);
        self.dir.join(&key_hex[..2]).join(format!("{key_hex}.cgz"))
    }

    #[inline(always)]
    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn scan_existing(&self) {
        let Ok(top) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut found: Vec<(u64, [u8; 16], PathBuf, u64)> = Vec::new();
        for sub in top.flatten() {
            let sub_path = sub.path();
            if !sub_path.is_dir() {
                continue;
            }
            let Ok(files) = fs::read_dir(&sub_path) else {
                continue;
            };
            for entry in files.flatten() {
                let p = entry.path();
                let Some(fname) = p.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if fname.contains(".tmp.") {
                    let _ = fs::remove_file(&p);
                    continue;
                }
                let Some(key_hex) = fname.strip_suffix(".cgz") else {
                    continue;
                };
                if key_hex.len() != 32 {
                    continue;
                }
                let mut key = [0u8; 16];
                if hex::decode_to_slice(key_hex, &mut key).is_err() {
                    continue;
                }
                if let Ok(meta) = entry.metadata() {
                    let sz = meta.len();
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    found.push((mtime, key, p, sz));
                }
            }
        }
        // Assign monotonically increasing logical stamps in mtime order so LRU survives restart
        found.sort_unstable_by_key(|(mtime, _, _, _)| *mtime);
        let mut total = 0u64;
        for (_, key, path, sz) in found {
            let stamp = self.tick();
            self.entries.insert(
                key,
                DiskEntry {
                    path,
                    bytes: sz,
                    last_used: AtomicU64::new(stamp),
                },
            );
            total = total.saturating_add(sz);
        }
        self.current_bytes.store(total, Ordering::Release);
    }

    /// Fast synchronous in-memory index check. Returns `Some((key, path))` if the block is
    /// present in the disk cache index, or increments `misses` and returns `None` immediately
    /// without touching the filesystem or Tokio's blocking pool.
    #[inline]
    pub fn lookup_path(&self, uri: &str, ifd_idx: u16, tile_idx: u32) -> Option<([u8; 16], PathBuf)> {
        let key = Self::key_bytes(uri, ifd_idx, tile_idx);
        let Some(entry) = self.entries.get(&key) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        entry.last_used.store(self.tick(), Ordering::Relaxed);
        Some((key, entry.path.clone()))
    }

    /// Synchronous read + CRC32 verification for a pre-looked-up disk entry (call from
    /// `spawn_blocking` on async paths). Zero-copy slice into the read `Vec<u8>`.
    pub fn read_verified_sync(&self, key: [u8; 16], path: &Path) -> Option<Bytes> {
        let raw = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    self.read_errors.fetch_add(1, Ordering::Relaxed);
                }
                // Always prune missing/unreadable entries (e.g. external /tmp cleanup) so ghost
                // keys don't leak current_bytes or permanently block put_async re-insertion.
                self.remove_entry(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        if raw.len() < HEADER_LEN || raw[..4] != MAGIC {
            self.read_errors.fetch_add(1, Ordering::Relaxed);
            self.remove_entry(key);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let expected_crc = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        if crc32fast::hash(&raw[HEADER_LEN..]) != expected_crc {
            self.read_errors.fetch_add(1, Ordering::Relaxed);
            self.remove_entry(key);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(Bytes::from(raw).slice(HEADER_LEN..))
    }

    /// Synchronous read + CRC32 verification (call from `spawn_blocking` on async paths).
    /// Returns `Some(compressed_bytes)` on a valid hit, or `None` on miss/corruption.
    pub fn get_sync(&self, uri: &str, ifd_idx: u16, tile_idx: u32) -> Option<Bytes> {
        let (key, path) = self.lookup_path(uri, ifd_idx, tile_idx)?;
        self.read_verified_sync(key, &path)
    }

    /// Async wrapper: checks the in-memory index synchronously first (zero blocking-pool hops on
    /// misses), and only dispatches to `spawn_blocking` when a cached file actually exists.
    pub async fn get(self: &Arc<Self>, uri: &str, ifd_idx: u16, tile_idx: u32) -> Option<Bytes> {
        let (key, path) = self.lookup_path(uri, ifd_idx, tile_idx)?;
        let c = self.clone();
        tokio::task::spawn_blocking(move || c.read_verified_sync(key, &path))
            .await
            .ok()
            .flatten()
    }

    /// Queue a non-blocking background write of raw compressed block bytes.
    /// Never blocks or fails the caller; drops and increments `write_errors` if queue is full.
    pub fn put_async(&self, uri: &str, ifd_idx: u16, tile_idx: u32, data: Bytes) {
        if (data.len() + HEADER_LEN) as u64 > self.max_bytes {
            return;
        }
        let key = Self::key_bytes(uri, ifd_idx, tile_idx);
        if self.entries.contains_key(&key) {
            return;
        }
        if self.tx.try_send(WriteJob { key, data }).is_err() {
            self.write_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Synchronous atomic write (also used directly in unit tests).
    pub fn put_sync(&self, uri: &str, ifd_idx: u16, tile_idx: u32, data: &[u8]) {
        if (data.len() + HEADER_LEN) as u64 > self.max_bytes {
            return;
        }
        let key = Self::key_bytes(uri, ifd_idx, tile_idx);
        self.write_sync(key, data);
    }

    fn write_sync(&self, key: [u8; 16], data: &[u8]) {
        let path = self.entry_path(key);
        if let Some(parent) = path.parent() {
            if fs::create_dir_all(parent).is_err() {
                self.write_errors.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("cgz.tmp.{}.{seq}", std::process::id()));

        let crc = crc32fast::hash(data);
        let write_res = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(&MAGIC)?;
            f.write_all(&crc.to_le_bytes())?;
            f.write_all(data)?;
            f.flush()?;
            fs::rename(&tmp_path, &path)?;
            Ok(())
        })();

        if write_res.is_err() {
            let _ = fs::remove_file(&tmp_path);
            self.write_errors.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let new_sz = (HEADER_LEN + data.len()) as u64;
        let old_sz = self
            .entries
            .insert(
                key,
                DiskEntry {
                    path,
                    bytes: new_sz,
                    last_used: AtomicU64::new(self.tick()),
                },
            )
            .map(|e| e.bytes)
            .unwrap_or(0);

        let now = if new_sz >= old_sz {
            let delta = new_sz - old_sz;
            self.current_bytes.fetch_add(delta, Ordering::AcqRel) + delta
        } else {
            let delta = old_sz - new_sz;
            self.current_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| {
                    Some(c.saturating_sub(delta))
                })
                .unwrap_or(0)
                .saturating_sub(delta)
        };

        if now > self.max_bytes {
            self.evict_if_needed();
        }
    }

    fn remove_entry(&self, key: [u8; 16]) {
        if let Some((_, entry)) = self.entries.remove(&key) {
            let _ = fs::remove_file(&entry.path);
            let _ = self
                .current_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |c| {
                    Some(c.saturating_sub(entry.bytes))
                });
        }
    }

    fn evict_if_needed(&self) {
        loop {
            if self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                return;
            }
            if self
                .evicting
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
                let _reset = Reset(&self.evicting);

                let cur = self.current_bytes.load(Ordering::Acquire);
                if cur > self.max_bytes {
                    let excess = cur - self.max_bytes;
                    let target_drop = excess.max(self.max_bytes / 4);

                    let mut candidates: Vec<(u64, [u8; 16])> = self
                        .entries
                        .iter()
                        .map(|e| (e.value().last_used.load(Ordering::Relaxed), *e.key()))
                        .collect();
                    candidates.sort_unstable_by_key(|(stamp, _)| *stamp);

                    let mut dropped = 0u64;
                    for (_, key) in candidates {
                        if dropped >= target_drop {
                            break;
                        }
                        if let Some((_, entry)) = self.entries.remove(&key) {
                            let _ = fs::remove_file(&entry.path);
                            dropped += entry.bytes;
                            removed_any = true;
                            self.evictions.fetch_add(1, Ordering::Relaxed);
                            let _ = self.current_bytes.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |c| Some(c.saturating_sub(entry.bytes)),
                            );
                        }
                    }
                }
            }

            if !removed_any || self.current_bytes.load(Ordering::Acquire) <= self.max_bytes {
                return;
            }
        }
    }

    pub fn stats(&self) -> DiskCacheStats {
        DiskCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_used: self.current_bytes.load(Ordering::Relaxed),
            max_bytes: self.max_bytes,
            evictions: self.evictions.load(Ordering::Relaxed),
            write_errors: self.write_errors.load(Ordering::Relaxed),
            read_errors: self.read_errors.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "cogzilla_disk_cache_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn persists_across_restart_and_detects_corruption() {
        let dir = temp_dir("persist");
        {
            let c = DiskBlockCache::open(&dir, 64 * 1024).unwrap();
            c.put_sync("s3://bucket/a.tif", 0, 7, b"compressed-tile-payload");
            assert_eq!(
                c.get_sync("s3://bucket/a.tif", 0, 7).unwrap().as_ref(),
                b"compressed-tile-payload"
            );
        }
        // Reopen from disk: must hit immediately!
        let c2 = DiskBlockCache::open(&dir, 64 * 1024).unwrap();
        assert_eq!(
            c2.get_sync("s3://bucket/a.tif", 0, 7).unwrap().as_ref(),
            b"compressed-tile-payload"
        );
        assert_eq!(c2.stats().hits, 1);

        // Corrupt a byte on disk: must detect CRC mismatch, delete entry, and fail-open as miss!
        let key = DiskBlockCache::key_bytes("s3://bucket/a.tif", 0, 7);
        let p = c2.entry_path(key);
        let mut raw = fs::read(&p).unwrap();
        raw[HEADER_LEN] ^= 0xFF;
        fs::write(&p, raw).unwrap();

        assert!(c2.get_sync("s3://bucket/a.tif", 0, 7).is_none());
        assert_eq!(c2.stats().read_errors, 1);
        assert!(!p.exists(), "corrupted file should have been deleted");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_eviction_enforces_byte_cap() {
        let dir = temp_dir("evict");
        // Each entry is 8 + 100 = 108 bytes; cap = 450 bytes (~4 entries)
        let c = DiskBlockCache::open(&dir, 450).unwrap();
        let payload = vec![7u8; 100];
        for i in 0..10u32 {
            c.put_sync("s3://bucket/b.tif", 0, i, &payload);
        }
        let st = c.stats();
        assert!(st.bytes_used <= 450, "bytes_used={} > 450", st.bytes_used);
        assert!(st.evictions >= 6, "expected >= 6 evictions, got {}", st.evictions);
        // Most recent entries must survive
        assert!(c.get_sync("s3://bucket/b.tif", 0, 9).is_some());
        let _ = fs::remove_dir_all(&dir);
    }
}
