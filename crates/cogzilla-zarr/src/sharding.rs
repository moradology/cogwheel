//! Zarr v3 `sharding_indexed` codec support (#366).
//!
//! Parses the `sharding_indexed` configuration, fetches only the shard index byte range (caching
//! parsed indices across neighbouring chunks), verifies optional `crc32c`, and fetches only the
//! exact inner-chunk byte ranges through [`ZarrStore::get_range`].

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde_json::Value;

use crate::codecs::decode_chunk_with_shape;
use crate::{ArrayMeta, CodecSpec, Endianness, ZarrError, ZarrStore};

const EMPTY_SENTINEL: u64 = u64::MAX;

/// Where the shard index sits inside a shard object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLocation {
    Start,
    End,
}

/// Parsed configuration of a Zarr v3 `sharding_indexed` codec.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Inner chunk shape (divides the outer shard `meta.chunk_shape`).
    pub inner_chunk_shape: Vec<u64>,
    /// Number of inner chunks per shard along each dimension.
    pub chunks_per_shard: Vec<u64>,
    /// Codec chain for each inner chunk.
    pub inner_codecs: Vec<CodecSpec>,
    /// Codec chain for the shard index itself (`bytes` + optional `crc32c`).
    pub index_codecs: Vec<CodecSpec>,
    pub index_location: IndexLocation,
}

impl ShardingConfig {
    /// Inspect `meta.codecs` and parse `sharding_indexed` if present.
    pub fn from_meta(meta: &ArrayMeta) -> Result<Option<Self>, ZarrError> {
        let Some(spec) = meta.codecs.iter().find(|c| c.name == "sharding_indexed") else {
            return Ok(None);
        };
        let cfg = &spec.configuration;
        let inner_chunk_shape = cfg
            .get("chunk_shape")
            .and_then(Value::as_array)
            .ok_or(ZarrError::MissingField {
                field: "sharding_indexed.configuration.chunk_shape",
            })?
            .iter()
            .map(|v| {
                v.as_u64().ok_or_else(|| ZarrError::BadField {
                    field: "sharding_indexed.configuration.chunk_shape",
                    detail: format!("{v} is not a non-negative integer"),
                })
            })
            .collect::<Result<Vec<u64>, _>>()?;

        if inner_chunk_shape.len() != meta.chunk_shape.len() {
            return Err(ZarrError::BadField {
                field: "sharding_indexed.configuration.chunk_shape",
                detail: format!(
                    "inner chunk rank {} != shard rank {}",
                    inner_chunk_shape.len(),
                    meta.chunk_shape.len()
                ),
            });
        }

        let mut chunks_per_shard = Vec::with_capacity(inner_chunk_shape.len());
        for (&shard_d, &inner_d) in meta.chunk_shape.iter().zip(&inner_chunk_shape) {
            if inner_d == 0 || !shard_d.is_multiple_of(inner_d) {
                return Err(ZarrError::BadField {
                    field: "sharding_indexed.configuration.chunk_shape",
                    detail: format!(
                        "shard dimension {shard_d} is not divisible by inner chunk dimension {inner_d}"
                    ),
                });
            }
            chunks_per_shard.push(shard_d / inner_d);
        }

        let inner_codecs = parse_codec_list(cfg.get("codecs"), "sharding_indexed.codecs")?;
        let index_codecs =
            parse_codec_list(cfg.get("index_codecs"), "sharding_indexed.index_codecs")?;

        let index_location = match cfg.get("index_location").and_then(Value::as_str) {
            None | Some("end") => IndexLocation::End,
            Some("start") => IndexLocation::Start,
            Some(other) => {
                return Err(ZarrError::BadField {
                    field: "sharding_indexed.configuration.index_location",
                    detail: format!("unexpected index_location {other:?}"),
                });
            }
        };

        Ok(Some(Self {
            inner_chunk_shape,
            chunks_per_shard,
            inner_codecs,
            index_codecs,
            index_location,
        }))
    }

    /// Total number of inner chunks in one shard.
    pub fn inner_chunk_count(&self) -> usize {
        self.chunks_per_shard.iter().product::<u64>() as usize
    }

    /// Fixed byte size of the encoded shard index on disk (supports `bytes` + optional `crc32c`).
    pub fn index_encoded_bytes(&self, shard_key: &str) -> Result<u64, ZarrError> {
        let base = (self.inner_chunk_count() as u64)
            .checked_mul(16)
            .ok_or_else(|| ZarrError::Codec {
                codec: "sharding_indexed".into(),
                chunk_key: shard_key.into(),
                detail: "shard index size overflow".into(),
            })?;
        let mut extra = 0u64;
        for c in &self.index_codecs {
            match c.name.as_str() {
                "bytes" => {}
                "crc32c" => extra += 4,
                other => {
                    return Err(ZarrError::Codec {
                        codec: "sharding_indexed".into(),
                        chunk_key: shard_key.into(),
                        detail: format!("unsupported shard index codec {other:?}"),
                    });
                }
            }
        }
        Ok(base + extra)
    }

    fn index_endian(&self) -> Endianness {
        self.index_codecs
            .iter()
            .find(|c| c.name == "bytes")
            .and_then(|c| c.configuration.get("endian").and_then(Value::as_str))
            .map(|e| {
                if e == "big" {
                    Endianness::Big
                } else {
                    Endianness::Little
                }
            })
            .unwrap_or(Endianness::Little)
    }

    fn has_crc32c(&self) -> bool {
        self.index_codecs.iter().any(|c| c.name == "crc32c")
    }
}

/// Parsed shard index: `(offset, nbytes)` per inner chunk (`None` for empty sentinel holes).
#[derive(Debug, Clone)]
pub struct ShardIndex {
    pub entries: Vec<Option<(u64, u64)>>,
}

impl ShardIndex {
    pub fn parse(
        cfg: &ShardingConfig,
        raw: &[u8],
        shard_key: &str,
    ) -> Result<Self, ZarrError> {
        let expected_len = cfg.index_encoded_bytes(shard_key)? as usize;
        if raw.len() != expected_len {
            return Err(ZarrError::Codec {
                codec: "sharding_indexed".into(),
                chunk_key: shard_key.into(),
                detail: format!(
                    "shard index length {} != expected {expected_len}",
                    raw.len()
                ),
            });
        }

        let payload = if cfg.has_crc32c() {
            let (body, crc_bytes) = raw.split_at(raw.len() - 4);
            let expected_crc = u32::from_le_bytes(crc_bytes.try_into().unwrap());
            let actual_crc = crc32c(body);
            if actual_crc != expected_crc {
                return Err(ZarrError::Codec {
                    codec: "sharding_indexed".into(),
                    chunk_key: shard_key.into(),
                    detail: format!(
                        "shard index crc32c mismatch (expected 0x{expected_crc:08x}, got 0x{actual_crc:08x})"
                    ),
                });
            }
            body
        } else {
            raw
        };

        let endian = cfg.index_endian();
        let count = cfg.inner_chunk_count();
        let mut entries = Vec::with_capacity(count);
        for chunk in payload.chunks_exact(16) {
            let off_bytes: [u8; 8] = chunk[0..8].try_into().unwrap();
            let len_bytes: [u8; 8] = chunk[8..16].try_into().unwrap();
            let (off, len) = match endian {
                Endianness::Little => (u64::from_le_bytes(off_bytes), u64::from_le_bytes(len_bytes)),
                Endianness::Big => (u64::from_be_bytes(off_bytes), u64::from_be_bytes(len_bytes)),
            };
            if off == EMPTY_SENTINEL && len == EMPTY_SENTINEL {
                entries.push(None);
            } else {
                entries.push(Some((off, len)));
            }
        }
        Ok(Self { entries })
    }
}

/// Thread-safe cache of parsed shard indices keyed by shard store key.
#[derive(Debug, Default)]
pub struct ShardIndexCache {
    indices: RwLock<HashMap<String, Option<Arc<ShardIndex>>>>,
}

impl ShardIndexCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch (or return cached) parsed shard index for `shard_key`.
    /// Returns `Ok(None)` if the entire shard object does not exist in `store`.
    pub async fn get_or_fetch(
        &self,
        store: &dyn ZarrStore,
        shard_key: &str,
        cfg: &ShardingConfig,
    ) -> Result<Option<Arc<ShardIndex>>, ZarrError> {
        if let Ok(guard) = self.indices.read()
            && let Some(cached) = guard.get(shard_key)
        {
            return Ok(cached.clone());
        }

        let index_bytes = cfg.index_encoded_bytes(shard_key)?;
        let raw_opt = match cfg.index_location {
            IndexLocation::Start => store.get_range(shard_key, 0, index_bytes).await?,
            IndexLocation::End => {
                let Some(total_size) = store.size(shard_key).await? else {
                    if let Ok(mut guard) = self.indices.write() {
                        guard.insert(shard_key.to_string(), None);
                    }
                    return Ok(None);
                };
                if total_size < index_bytes {
                    return Err(ZarrError::Codec {
                        codec: "sharding_indexed".into(),
                        chunk_key: shard_key.into(),
                        detail: format!(
                            "shard size {total_size} smaller than index size {index_bytes}"
                        ),
                    });
                }
                store
                    .get_range(shard_key, total_size - index_bytes, total_size)
                    .await?
            }
        };

        let Some(raw) = raw_opt else {
            if let Ok(mut guard) = self.indices.write() {
                guard.insert(shard_key.to_string(), None);
            }
            return Ok(None);
        };

        let parsed = Arc::new(ShardIndex::parse(cfg, &raw, shard_key)?);
        // Bounds-check every inner chunk range against the shard object size before caching (#386).
        if let Some(total_size) = store.size(shard_key).await? {
            for (i, entry) in parsed.entries.iter().enumerate() {
                if let Some((off, len)) = entry {
                    let end = off.checked_add(*len).ok_or_else(|| ZarrError::Codec {
                        codec: "sharding_indexed".into(),
                        chunk_key: shard_key.into(),
                        detail: format!("shard index entry {i} byte range overflows u64"),
                    })?;
                    if end > total_size {
                        return Err(ZarrError::Codec {
                            codec: "sharding_indexed".into(),
                            chunk_key: shard_key.into(),
                            detail: format!(
                                "shard index entry {i} range {off}..{end} exceeds shard size {total_size}"
                            ),
                        });
                    }
                }
            }
        }
        if let Ok(mut guard) = self.indices.write() {
            guard.insert(shard_key.to_string(), Some(Arc::clone(&parsed)));
        }
        Ok(Some(parsed))
    }
}

/// Fetch and decode a single inner chunk from a sharded v3 array using ranged reads.
///
/// Returns `Ok(None)` if the shard or inner chunk is an unwritten hole (and `fill_value` is set).
pub async fn read_sharded_inner_chunk(
    store: &dyn ZarrStore,
    array_prefix: &str,
    meta: &ArrayMeta,
    cfg: &ShardingConfig,
    index_cache: &ShardIndexCache,
    inner_idx: &[u64],
) -> Result<Option<Vec<u8>>, ZarrError> {
    let ndim = inner_idx.len();
    let mut shard_idx = Vec::with_capacity(ndim);
    let mut local_idx = Vec::with_capacity(ndim);
    for (&idx_d, &cps) in inner_idx.iter().zip(&cfg.chunks_per_shard) {
        shard_idx.push(idx_d / cps);
        local_idx.push(idx_d % cps);
    }

    let rel_shard_key = meta.chunk_key(&shard_idx);
    let prefix = array_prefix.trim_matches('/');
    let shard_key = if prefix.is_empty() {
        rel_shard_key
    } else {
        format!("{prefix}/{rel_shard_key}")
    };

    let Some(shard_index) = index_cache.get_or_fetch(store, &shard_key, cfg).await? else {
        return if meta.fill_value.is_some() {
            Ok(None)
        } else {
            Err(ZarrError::MissingChunk {
                chunk_key: shard_key,
            })
        };
    };

    // Flat C-order index into shard index table
    let mut flat = 0usize;
    for (&loc, &cps) in local_idx.iter().zip(&cfg.chunks_per_shard) {
        flat = flat * (cps as usize) + (loc as usize);
    }
    let Some((offset, nbytes)) = shard_index.entries.get(flat).copied().flatten() else {
        return if meta.fill_value.is_some() {
            Ok(None)
        } else {
            Err(ZarrError::MissingChunk {
                chunk_key: format!("{shard_key}#{local_idx:?}"),
            })
        };
    };

    let end = offset.checked_add(nbytes).ok_or_else(|| ZarrError::Codec {
        codec: "sharding_indexed".into(),
        chunk_key: shard_key.clone(),
        detail: "inner chunk byte range overflow".into(),
    })?;
    let Some(raw_inner) = store.get_range(&shard_key, offset, end).await? else {
        return Err(ZarrError::MissingChunk {
            chunk_key: shard_key,
        });
    };

    let inner_label = format!("{shard_key}#{local_idx:?}");
    let decoded = decode_chunk_with_shape(
        meta,
        &cfg.inner_chunk_shape,
        &cfg.inner_codecs,
        &raw_inner,
        &inner_label,
    )?;
    Ok(Some(decoded))
}

fn parse_codec_list(val: Option<&Value>, field: &'static str) -> Result<Vec<CodecSpec>, ZarrError> {
    let arr = val
        .and_then(Value::as_array)
        .ok_or(ZarrError::MissingField { field })?;
    let mut out = Vec::with_capacity(arr.len());
    for c in arr {
        if let Some(name) = c.as_str() {
            out.push(CodecSpec {
                name: name.to_string(),
                configuration: Value::Null,
            });
        } else {
            let name = c
                .get("name")
                .and_then(Value::as_str)
                .ok_or(ZarrError::MissingField { field })?;
            out.push(CodecSpec {
                name: name.to_string(),
                configuration: c.get("configuration").cloned().unwrap_or(Value::Null),
            });
        }
    }
    Ok(out)
}

/// Castagnoli CRC32C (polynomial `0x82F63B78`) used by Zarr v3 `crc32c` codecs.
pub(crate) fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
        }
    }
    !crc
}
