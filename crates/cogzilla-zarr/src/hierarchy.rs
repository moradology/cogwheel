//! Zarr hierarchy opening: v2/v3 version detection, consolidated metadata, group walking, and
//! dataset introspection.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use crate::{ArrayMeta, ZarrError, ZarrVersion, object_map};

/// Boxed future type for object-safe async [`ZarrStore`] implementations.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Minimal async key-value store abstraction for Zarr metadata and chunk reads.
///
/// `cogzilla-zarr` does not own HTTP/S3/GCS/Azure transport; the server passes an implementation
/// backed by `cogzilla-io`, while unit and integration tests use [`MemoryStore`] or [`FsStore`].
pub trait ZarrStore: Send + Sync {
    /// Fetch raw bytes at `key` (relative to the store root, forward-slash separated, no leading
    /// `/`). Returns `Ok(None)` when the key does not exist.
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>>;

    /// Fetch byte range `[start..end)` of `key`. Default falls back to `get(key)` and slices.
    fn get_range<'a>(
        &'a self,
        key: &'a str,
        start: u64,
        end: u64,
    ) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        Box::pin(async move {
            let Some(all) = self.get(key).await? else {
                return Ok(None);
            };
            let s = usize::try_from(start).unwrap_or(all.len());
            let e = usize::try_from(end).unwrap_or(all.len());
            if s > e || e > all.len() {
                return Err(ZarrError::Io {
                    key: key.to_string(),
                    message: format!("range {start}..{end} out of bounds for len {}", all.len()),
                });
            }
            Ok(Some(all[s..e].to_vec()))
        })
    }

    /// Return the byte length of `key`, or `Ok(None)` if missing.
    fn size<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<u64>, ZarrError>> {
        Box::pin(async move { Ok(self.get(key).await?.map(|b| b.len() as u64)) })
    }

    /// List immediate child directory/node names under `prefix` (empty string `""` for the store
    /// root). Returns `Ok(None)` when the backend does not support directory listing (e.g. plain
    /// HTTP without an index).
    fn list_dir<'a>(
        &'a self,
        _prefix: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, ZarrError>> {
        Box::pin(async { Ok(None) })
    }
}

/// Group metadata (attributes attached to a Zarr group node).
#[derive(Debug, Clone, Default)]
pub struct GroupMeta {
    pub attributes: BTreeMap<String, Value>,
}

/// One node in a Zarr hierarchy: either a group or an array.
#[derive(Debug, Clone)]
pub enum ZarrNode {
    Group(GroupMeta),
    Array(ArrayMeta),
}

/// How [`ZarrHierarchy::open`] discovered the store's structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenMode {
    V3Consolidated,
    V3Walked,
    V3RootArray,
    V2Consolidated,
    V2Walked,
    V2RootArray,
}

/// An opened Zarr v2 or v3 hierarchy (root group + descendants, or a single root array).
#[derive(Debug, Clone)]
pub struct ZarrHierarchy {
    pub version: ZarrVersion,
    pub open_mode: OpenMode,
    /// Map from normalized node path (`""` for root, `"temp"`, `"0/elevation"`, ...) to node.
    pub nodes: BTreeMap<String, ZarrNode>,
}

impl ZarrHierarchy {
    /// Open a Zarr store by inspecting its root metadata keys rather than trusting the URL.
    ///
    /// Order of probes:
    /// 1. `zarr.json` (v3): if consolidated metadata is inline, opens in **1 GET** total;
    ///    if root array, opens in **1 GET**; otherwise falls back to walking subgroups via
    ///    [`ZarrStore::list_dir`].
    /// 2. `.zmetadata` (v2 consolidated): opens the entire hierarchy in **1 metadata hit** (2 GETs
    ///    including the `zarr.json` miss).
    /// 3. `.zgroup` / `.zarray` (v2 unconsolidated): walks subgroups via [`ZarrStore::list_dir`]
    ///    or opens a root array.
    pub async fn open(store: &dyn ZarrStore) -> Result<Self, ZarrError> {
        // 1. Try v3 `zarr.json`
        if let Some(raw) = store.get("zarr.json").await? {
            let root_val: Value = serde_json::from_slice(&raw)?;
            return Self::open_v3(store, &root_val).await;
        }

        // 2. Try v2 `.zmetadata` (consolidated)
        if let Some(raw) = store.get(".zmetadata").await? {
            let cons_val: Value = serde_json::from_slice(&raw)?;
            let hierarchy = Self::parse_v2_consolidated(&cons_val)?;
            tracing::debug!(
                nodes = hierarchy.nodes.len(),
                "opened Zarr v2 hierarchy via consolidated .zmetadata"
            );
            return Ok(hierarchy);
        }

        // 3. Try v2 `.zgroup` (unconsolidated group)
        if let Some(raw) = store.get(".zgroup").await? {
            validate_v2_group(&raw)?;
            let attrs = fetch_v2_attrs(store, "").await?;
            let mut nodes = BTreeMap::new();
            nodes.insert(String::new(), ZarrNode::Group(GroupMeta { attributes: attrs }));
            walk_v2_group(store, "", &mut nodes).await?;
            tracing::debug!(
                nodes = nodes.len(),
                "opened Zarr v2 hierarchy by walking unconsolidated groups"
            );
            return Ok(ZarrHierarchy {
                version: ZarrVersion::V2,
                open_mode: OpenMode::V2Walked,
                nodes,
            });
        }

        // 4. Try v2 `.zarray` (root array with no enclosing group)
        if let Some(raw) = store.get(".zarray").await? {
            let zattrs = store.get(".zattrs").await?;
            let meta = ArrayMeta::from_v2(&raw, zattrs.as_deref())?;
            let mut nodes = BTreeMap::new();
            nodes.insert(String::new(), ZarrNode::Array(meta));
            tracing::debug!("opened Zarr v2 root array");
            return Ok(ZarrHierarchy {
                version: ZarrVersion::V2,
                open_mode: OpenMode::V2RootArray,
                nodes,
            });
        }

        Err(ZarrError::NotAZarrStore {
            searched: "zarr.json, .zmetadata, .zgroup, .zarray",
        })
    }

    async fn open_v3(store: &dyn ZarrStore, root_val: &Value) -> Result<Self, ZarrError> {
        match root_val.get("zarr_format").and_then(Value::as_u64) {
            Some(3) | None => {}
            Some(other) => return Err(ZarrError::UnsupportedVersion(other)),
        }
        let node_type = root_val
            .get("node_type")
            .and_then(Value::as_str)
            .unwrap_or("group");

        if node_type == "array" {
            let meta = ArrayMeta::from_v3_value(root_val)?;
            let mut nodes = BTreeMap::new();
            nodes.insert(String::new(), ZarrNode::Array(meta));
            tracing::debug!("opened Zarr v3 root array");
            return Ok(ZarrHierarchy {
                version: ZarrVersion::V3,
                open_mode: OpenMode::V3RootArray,
                nodes,
            });
        }

        if node_type != "group" {
            return Err(ZarrError::BadField {
                field: "node_type",
                detail: format!("unexpected v3 node_type {node_type:?}"),
            });
        }

        let root_attrs = root_val.get("attributes").map(object_map).unwrap_or_default();
        let mut nodes = BTreeMap::new();
        nodes.insert(
            String::new(),
            ZarrNode::Group(GroupMeta {
                attributes: root_attrs,
            }),
        );

        // Check for inline consolidated metadata
        if let Some(cons) = root_val.get("consolidated_metadata")
            && cons.get("kind").and_then(Value::as_str) == Some("inline")
            && let Some(meta_map) = cons.get("metadata").and_then(Value::as_object)
        {
            for (raw_path, entry) in meta_map {
                let path = normalize_path(raw_path);
                if path.is_empty() {
                    continue;
                }
                let nt = entry
                    .get("node_type")
                    .and_then(Value::as_str)
                    .unwrap_or("group");
                match nt {
                    "array" => {
                        let meta = ArrayMeta::from_v3_value(entry)?;
                        nodes.insert(path, ZarrNode::Array(meta));
                    }
                    "group" => {
                        let attrs = entry.get("attributes").map(object_map).unwrap_or_default();
                        nodes.insert(path, ZarrNode::Group(GroupMeta { attributes: attrs }));
                    }
                    other => {
                        return Err(ZarrError::BadField {
                            field: "consolidated_metadata.metadata[].node_type",
                            detail: format!("unexpected node_type {other:?} at {raw_path:?}"),
                        });
                    }
                }
            }
            tracing::debug!(
                nodes = nodes.len(),
                "opened Zarr v3 hierarchy via consolidated_metadata"
            );
            return Ok(ZarrHierarchy {
                version: ZarrVersion::V3,
                open_mode: OpenMode::V3Consolidated,
                nodes,
            });
        }

        // Unconsolidated v3: walk child directories
        walk_v3_group(store, "", &mut nodes).await?;
        tracing::debug!(
            nodes = nodes.len(),
            "opened Zarr v3 hierarchy by walking unconsolidated groups"
        );
        Ok(ZarrHierarchy {
            version: ZarrVersion::V3,
            open_mode: OpenMode::V3Walked,
            nodes,
        })
    }

    fn parse_v2_consolidated(cons_val: &Value) -> Result<Self, ZarrError> {
        let meta_map = cons_val
            .get("metadata")
            .and_then(Value::as_object)
            .ok_or(ZarrError::MissingField { field: "metadata" })?;

        // Group `.zgroup`, `.zarray`, and `.zattrs` entries by their node path.
        let mut zgroups: BTreeSet<String> = BTreeSet::new();
        let mut zarrays: BTreeMap<String, &Value> = BTreeMap::new();
        let mut zattrs: BTreeMap<String, &Value> = BTreeMap::new();

        for (key, val) in meta_map {
            let trimmed = key.trim_start_matches('/');
            if trimmed == ".zgroup" {
                zgroups.insert(String::new());
            } else if let Some(prefix) = trimmed.strip_suffix("/.zgroup") {
                zgroups.insert(normalize_path(prefix));
            } else if trimmed == ".zarray" {
                zarrays.insert(String::new(), val);
            } else if let Some(prefix) = trimmed.strip_suffix("/.zarray") {
                zarrays.insert(normalize_path(prefix), val);
            } else if trimmed == ".zattrs" {
                zattrs.insert(String::new(), val);
            } else if let Some(prefix) = trimmed.strip_suffix("/.zattrs") {
                zattrs.insert(normalize_path(prefix), val);
            }
        }

        let mut nodes = BTreeMap::new();
        for path in zgroups {
            let attributes = zattrs.get(&path).copied().map(object_map).unwrap_or_default();
            nodes.insert(path, ZarrNode::Group(GroupMeta { attributes }));
        }
        for (path, zarr_val) in zarrays {
            let attr_val = zattrs.get(&path).copied();
            let meta = ArrayMeta::from_v2_value(zarr_val, attr_val)?;
            nodes.insert(path, ZarrNode::Array(meta));
        }

        if nodes.is_empty() {
            return Err(ZarrError::NotAZarrStore {
                searched: ".zmetadata entries (.zgroup / .zarray)",
            });
        }

        Ok(ZarrHierarchy {
            version: ZarrVersion::V2,
            open_mode: OpenMode::V2Consolidated,
            nodes,
        })
    }

    /// True if the hierarchy was opened from consolidated metadata.
    pub fn is_consolidated(&self) -> bool {
        matches!(
            self.open_mode,
            OpenMode::V2Consolidated | OpenMode::V3Consolidated
        )
    }

    /// If the store is a bare root array (no enclosing group), returns its metadata.
    pub fn root_array(&self) -> Option<&ArrayMeta> {
        match self.nodes.get("") {
            Some(ZarrNode::Array(meta)) => Some(meta),
            _ => None,
        }
    }

    /// Look up an array anywhere in the hierarchy by normalized path (`""` for a root array).
    pub fn array(&self, path: &str) -> Option<&ArrayMeta> {
        let key = normalize_path(path);
        match self.nodes.get(&key) {
            Some(ZarrNode::Array(meta)) => Some(meta),
            _ => None,
        }
    }

    /// Select a group subtree, matching TiTiler's `?group=` query parameter.
    ///
    /// `None`, `Some("")`, and `Some("/")` all select the root group.
    pub fn group(&self, group_path: Option<&str>) -> Result<GroupView<'_>, ZarrError> {
        let norm = group_path.map(normalize_path).unwrap_or_default();
        match self.nodes.get(&norm) {
            Some(ZarrNode::Group(meta)) => Ok(GroupView {
                hierarchy: self,
                path: norm,
                meta,
            }),
            _ => Err(ZarrError::GroupNotFound {
                path: group_path.unwrap_or("").to_string(),
            }),
        }
    }
}

/// View over a single group inside a [`ZarrHierarchy`] (root or `?group=` subtree).
#[derive(Debug, Clone)]
pub struct GroupView<'a> {
    hierarchy: &'a ZarrHierarchy,
    path: String,
    meta: &'a GroupMeta,
}

impl<'a> GroupView<'a> {
    /// Normalized path of this group (`""` for root).
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Group-level attributes.
    pub fn attributes(&self) -> &'a BTreeMap<String, Value> {
        &self.meta.attributes
    }

    /// Full store prefix for a child name inside this group (for chunk key construction).
    pub fn child_prefix(&self, name: &str) -> String {
        join_path(&self.path, name)
    }

    /// Direct child arrays in this group (name -> [`ArrayMeta`]).
    pub fn arrays(&self) -> BTreeMap<String, &'a ArrayMeta> {
        let mut out = BTreeMap::new();
        for (node_path, node) in &self.hierarchy.nodes {
            if let Some(child_name) = direct_child(&self.path, node_path)
                && let ZarrNode::Array(meta) = node
            {
                out.insert(child_name.to_string(), meta);
            }
        }
        out
    }

    /// Direct child subgroup names in this group.
    pub fn subgroups(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (node_path, node) in &self.hierarchy.nodes {
            if let Some(child_name) = direct_child(&self.path, node_path)
                && matches!(node, ZarrNode::Group(_))
            {
                out.push(child_name.to_string());
            }
        }
        out
    }

    /// Sorted names of all direct child arrays (variables + coordinates) in this group.
    pub fn variable_names(&self) -> Vec<String> {
        self.arrays().into_keys().collect()
    }

    /// Look up a direct child array by name.
    pub fn array(&self, name: &str) -> Option<&'a ArrayMeta> {
        let full = self.child_prefix(name);
        self.hierarchy.array(&full)
    }

    /// Dimension names and sizes collected across all arrays in this group.
    pub fn dimensions(&self) -> BTreeMap<String, u64> {
        let mut dims = BTreeMap::new();
        for meta in self.arrays().into_values() {
            if let Some(names) = &meta.dimension_names {
                for (name_opt, &size) in names.iter().zip(&meta.shape) {
                    if let Some(name) = name_opt {
                        dims.entry(name.clone()).or_insert(size);
                    }
                }
            }
        }
        dims
    }

    /// Direct child arrays that are not 1-D self-named coordinate arrays or 0-D CRS markers.
    pub fn data_variables(&self) -> Vec<String> {
        let dims = self.dimensions();
        let mut out = Vec::new();
        for (name, meta) in self.arrays() {
            let is_1d_coord = meta.rank() == 1 && dims.contains_key(&name);
            let is_scalar_marker = meta.rank() == 0;
            if !is_1d_coord && !is_scalar_marker {
                out.push(name);
            }
        }
        out
    }
}

async fn walk_v3_group(
    store: &dyn ZarrStore,
    prefix: &str,
    nodes: &mut BTreeMap<String, ZarrNode>,
) -> Result<(), ZarrError> {
    let mut stack = vec![prefix.to_string()];
    while let Some(dir) = stack.pop() {
        let Some(children) = store.list_dir(&dir).await? else {
            continue;
        };
        for child in children {
            let child_path = join_path(&dir, &child);
            let meta_key = join_path(&child_path, "zarr.json");
            let Some(raw) = store.get(&meta_key).await? else {
                continue;
            };
            let val: Value = serde_json::from_slice(&raw)?;
            let nt = val
                .get("node_type")
                .and_then(Value::as_str)
                .unwrap_or("group");
            match nt {
                "array" => {
                    let meta = ArrayMeta::from_v3_value(&val)?;
                    nodes.insert(child_path, ZarrNode::Array(meta));
                }
                "group" => {
                    let attrs = val.get("attributes").map(object_map).unwrap_or_default();
                    nodes.insert(child_path.clone(), ZarrNode::Group(GroupMeta { attributes: attrs }));
                    stack.push(child_path);
                }
                other => {
                    return Err(ZarrError::BadField {
                        field: "node_type",
                        detail: format!("unexpected v3 node_type {other:?} at {child_path:?}"),
                    });
                }
            }
        }
    }
    Ok(())
}

async fn walk_v2_group(
    store: &dyn ZarrStore,
    prefix: &str,
    nodes: &mut BTreeMap<String, ZarrNode>,
) -> Result<(), ZarrError> {
    let mut stack = vec![prefix.to_string()];
    while let Some(dir) = stack.pop() {
        let Some(children) = store.list_dir(&dir).await? else {
            continue;
        };
        for child in children {
            let child_path = join_path(&dir, &child);
            let zarray_key = join_path(&child_path, ".zarray");
            if let Some(raw) = store.get(&zarray_key).await? {
                let zattrs_key = join_path(&child_path, ".zattrs");
                let zattrs = store.get(&zattrs_key).await?;
                let meta = ArrayMeta::from_v2(&raw, zattrs.as_deref())?;
                nodes.insert(child_path, ZarrNode::Array(meta));
                continue;
            }
            let zgroup_key = join_path(&child_path, ".zgroup");
            if let Some(raw) = store.get(&zgroup_key).await? {
                validate_v2_group(&raw)?;
                let attrs = fetch_v2_attrs(store, &child_path).await?;
                nodes.insert(
                    child_path.clone(),
                    ZarrNode::Group(GroupMeta { attributes: attrs }),
                );
                stack.push(child_path);
            }
        }
    }
    Ok(())
}

fn validate_v2_group(raw: &[u8]) -> Result<(), ZarrError> {
    let v: Value = serde_json::from_slice(raw)?;
    match v.get("zarr_format").and_then(Value::as_u64) {
        Some(2) | None => Ok(()),
        Some(other) => Err(ZarrError::UnsupportedVersion(other)),
    }
}

async fn fetch_v2_attrs(
    store: &dyn ZarrStore,
    prefix: &str,
) -> Result<BTreeMap<String, Value>, ZarrError> {
    let key = join_path(prefix, ".zattrs");
    match store.get(&key).await? {
        Some(bytes) => Ok(object_map(&serde_json::from_slice::<Value>(&bytes)?)),
        None => Ok(BTreeMap::new()),
    }
}

fn normalize_path(path: &str) -> String {
    path.trim_matches('/').to_string()
}

fn join_path(prefix: &str, child: &str) -> String {
    let p = prefix.trim_matches('/');
    let c = child.trim_matches('/');
    if p.is_empty() {
        c.to_string()
    } else if c.is_empty() {
        p.to_string()
    } else {
        format!("{p}/{c}")
    }
}

fn direct_child<'a>(parent: &str, candidate: &'a str) -> Option<&'a str> {
    if parent.is_empty() {
        if candidate.is_empty() || candidate.contains('/') {
            None
        } else {
            Some(candidate)
        }
    } else {
        let rest = candidate.strip_prefix(parent)?.strip_prefix('/')?;
        if rest.is_empty() || rest.contains('/') {
            None
        } else {
            Some(rest)
        }
    }
}

/// In-memory [`ZarrStore`] with GET and byte counters for deterministic unit tests.
#[derive(Debug, Default)]
pub struct MemoryStore {
    entries: BTreeMap<String, Vec<u8>>,
    get_count: AtomicUsize,
    hit_count: AtomicUsize,
    range_count: AtomicUsize,
    bytes_read: AtomicUsize,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) {
        let k = normalize_path(&key.into());
        self.entries.insert(k, value.into());
    }

    /// Total number of `get()` calls issued so far (including misses).
    pub fn get_count(&self) -> usize {
        self.get_count.load(Ordering::Relaxed)
    }

    /// Number of `get()` calls that found a key.
    pub fn hit_count(&self) -> usize {
        self.hit_count.load(Ordering::Relaxed)
    }

    /// Total number of `get_range()` calls issued so far.
    pub fn range_count(&self) -> usize {
        self.range_count.load(Ordering::Relaxed)
    }

    /// Total bytes transferred across `get()` and `get_range()`.
    pub fn bytes_read(&self) -> usize {
        self.bytes_read.load(Ordering::Relaxed)
    }
}

impl ZarrStore for MemoryStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        let norm = normalize_path(key);
        self.get_count.fetch_add(1, Ordering::Relaxed);
        let found = self.entries.get(&norm).cloned();
        if let Some(ref b) = found {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(b.len(), Ordering::Relaxed);
        }
        Box::pin(async move { Ok(found) })
    }

    fn get_range<'a>(
        &'a self,
        key: &'a str,
        start: u64,
        end: u64,
    ) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        let norm = normalize_path(key);
        self.range_count.fetch_add(1, Ordering::Relaxed);
        let res = match self.entries.get(&norm) {
            None => Ok(None),
            Some(all) => {
                let s = usize::try_from(start).unwrap_or(all.len());
                let e = usize::try_from(end).unwrap_or(all.len());
                if s > e || e > all.len() {
                    Err(ZarrError::Io {
                        key: norm,
                        message: format!("range {start}..{end} out of bounds for len {}", all.len()),
                    })
                } else {
                    self.bytes_read.fetch_add(e - s, Ordering::Relaxed);
                    Ok(Some(all[s..e].to_vec()))
                }
            }
        };
        Box::pin(async move { res })
    }

    fn size<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<u64>, ZarrError>> {
        let norm = normalize_path(key);
        let sz = self.entries.get(&norm).map(|v| v.len() as u64);
        Box::pin(async move { Ok(sz) })
    }

    fn list_dir<'a>(
        &'a self,
        prefix: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, ZarrError>> {
        let norm = normalize_path(prefix);
        let mut children = BTreeSet::new();
        for key in self.entries.keys() {
            let rest = if norm.is_empty() {
                key.as_str()
            } else if let Some(r) = key.strip_prefix(&norm).and_then(|s| s.strip_prefix('/')) {
                r
            } else {
                continue;
            };
            // Only collect subdirectories (keys with a `/` beneath `prefix`).
            if let Some((first, _)) = rest.split_once('/')
                && !first.is_empty()
            {
                children.insert(first.to_string());
            }
        }
        let list: Vec<String> = children.into_iter().collect();
        Box::pin(async move { Ok(Some(list)) })
    }
}

/// Local filesystem [`ZarrStore`] rooted at a directory.
#[derive(Debug, Clone)]
pub struct FsStore {
    root: PathBuf,
}

impl FsStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl ZarrStore for FsStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        let norm = normalize_path(key);
        let path = self.root.join(&norm);
        Box::pin(async move {
            match std::fs::read(&path) {
                Ok(bytes) => Ok(Some(bytes)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(ZarrError::Io {
                    key: norm,
                    message: e.to_string(),
                }),
            }
        })
    }

    fn get_range<'a>(
        &'a self,
        key: &'a str,
        start: u64,
        end: u64,
    ) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        use std::io::{Read, Seek, SeekFrom};
        let norm = normalize_path(key);
        let path = self.root.join(&norm);
        Box::pin(async move {
            if start > end {
                return Err(ZarrError::Io {
                    key: norm,
                    message: format!("invalid byte range {start}..{end}"),
                });
            }
            let mut f = match std::fs::File::open(&path) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => {
                    return Err(ZarrError::Io {
                        key: norm,
                        message: e.to_string(),
                    });
                }
            };
            let len = usize::try_from(end - start).map_err(|_| ZarrError::Io {
                key: norm.clone(),
                message: format!("byte range {start}..{end} too large"),
            })?;
            f.seek(SeekFrom::Start(start)).map_err(|e| ZarrError::Io {
                key: norm.clone(),
                message: e.to_string(),
            })?;
            let mut buf = vec![0u8; len];
            f.read_exact(&mut buf).map_err(|e| ZarrError::Io {
                key: norm,
                message: e.to_string(),
            })?;
            Ok(Some(buf))
        })
    }

    fn size<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<u64>, ZarrError>> {
        let norm = normalize_path(key);
        let path = self.root.join(&norm);
        Box::pin(async move {
            match std::fs::metadata(&path) {
                Ok(m) => Ok(Some(m.len())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(ZarrError::Io {
                    key: norm,
                    message: e.to_string(),
                }),
            }
        })
    }

    fn list_dir<'a>(
        &'a self,
        prefix: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, ZarrError>> {
        let norm = normalize_path(prefix);
        let dir = if norm.is_empty() {
            self.root.clone()
        } else {
            self.root.join(&norm)
        };
        Box::pin(async move {
            let entries = match std::fs::read_dir(&dir) {
                Ok(rd) => rd,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Some(Vec::new())),
                Err(e) => {
                    return Err(ZarrError::Io {
                        key: norm,
                        message: e.to_string(),
                    });
                }
            };
            let mut dirs = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| ZarrError::Io {
                    key: norm.clone(),
                    message: e.to_string(),
                })?;
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                    && let Some(name) = entry.file_name().to_str()
                {
                    dirs.push(name.to_string());
                }
            }
            dirs.sort();
            Ok(Some(dirs))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleType;

    fn sample_v2_array_json() -> &'static [u8] {
        br#"{"zarr_format":2,"shape":[16,32],"chunks":[8,8],"dtype":"<f4","fill_value":"NaN","order":"C","filters":null,"compressor":null}"#
    }

    fn sample_v3_array_json() -> &'static [u8] {
        br#"{
            "zarr_format": 3,
            "node_type": "array",
            "shape": [16, 32],
            "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "dimension_names": ["y", "x"],
            "attributes": {"units": "K"}
        }"#
    }

    #[tokio::test]
    async fn v3_consolidated_opens_in_exactly_one_get() {
        let mut store = MemoryStore::new();
        let mut meta_obj = serde_json::Map::new();
        for i in 0..20 {
            meta_obj.insert(
                format!("var_{i:02}"),
                serde_json::from_slice(sample_v3_array_json()).unwrap(),
            );
        }
        let root = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"title": "many"},
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": false,
                "metadata": meta_obj,
            }
        });
        store.insert("zarr.json", serde_json::to_vec(&root).unwrap());

        let h = ZarrHierarchy::open(&store).await.unwrap();
        assert_eq!(h.version, ZarrVersion::V3);
        assert_eq!(h.open_mode, OpenMode::V3Consolidated);
        assert!(h.is_consolidated());
        assert_eq!(store.get_count(), 1, "v3 consolidated must issue 1 GET");
        let root_g = h.group(None).unwrap();
        assert_eq!(root_g.variable_names().len(), 20);
        assert_eq!(root_g.dimensions().get("y"), Some(&16));
        assert_eq!(root_g.dimensions().get("x"), Some(&32));
    }

    #[tokio::test]
    async fn v2_consolidated_opens_in_one_metadata_hit() {
        let mut store = MemoryStore::new();
        let mut meta = serde_json::Map::new();
        meta.insert(".zgroup".into(), serde_json::json!({"zarr_format": 2}));
        meta.insert(".zattrs".into(), serde_json::json!({"title": "v2"}));
        for i in 0..15 {
            meta.insert(
                format!("v{i}/.zarray"),
                serde_json::from_slice(sample_v2_array_json()).unwrap(),
            );
            meta.insert(
                format!("v{i}/.zattrs"),
                serde_json::json!({"_ARRAY_DIMENSIONS": ["y", "x"]}),
            );
        }
        let zmeta = serde_json::json!({
            "zarr_consolidated_format": 1,
            "metadata": meta,
        });
        store.insert(".zmetadata", serde_json::to_vec(&zmeta).unwrap());

        let h = ZarrHierarchy::open(&store).await.unwrap();
        assert_eq!(h.version, ZarrVersion::V2);
        assert_eq!(h.open_mode, OpenMode::V2Consolidated);
        assert_eq!(store.hit_count(), 1, "v2 consolidated must issue 1 metadata hit");
        assert_eq!(h.group(None).unwrap().variable_names().len(), 15);
    }

    #[tokio::test]
    async fn v2_and_v3_unconsolidated_walk_nested_groups() {
        // v2 nested
        let mut s2 = MemoryStore::new();
        s2.insert(".zgroup", br#"{"zarr_format":2}"#.as_slice());
        s2.insert("sub/.zgroup", br#"{"zarr_format":2}"#.as_slice());
        s2.insert("sub/temp/.zarray", sample_v2_array_json());
        s2.insert(
            "sub/temp/.zattrs",
            br#"{"_ARRAY_DIMENSIONS":["y","x"]}"#.as_slice(),
        );
        let h2 = ZarrHierarchy::open(&s2).await.unwrap();
        assert_eq!(h2.open_mode, OpenMode::V2Walked);
        assert_eq!(h2.group(None).unwrap().subgroups(), vec!["sub"]);
        let sub2 = h2.group(Some("sub")).unwrap();
        assert_eq!(sub2.variable_names(), vec!["temp"]);
        assert_eq!(sub2.child_prefix("temp"), "sub/temp");

        // v3 nested
        let mut s3 = MemoryStore::new();
        s3.insert(
            "zarr.json",
            br#"{"zarr_format":3,"node_type":"group","attributes":{}}"#.as_slice(),
        );
        s3.insert(
            "level0/zarr.json",
            br#"{"zarr_format":3,"node_type":"group","attributes":{"scale":1}}"#.as_slice(),
        );
        s3.insert("level0/band/zarr.json", sample_v3_array_json());
        let h3 = ZarrHierarchy::open(&s3).await.unwrap();
        assert_eq!(h3.open_mode, OpenMode::V3Walked);
        let g = h3.group(Some("/level0/")).unwrap();
        assert_eq!(g.variable_names(), vec!["band"]);
        assert_eq!(g.array("band").unwrap().dtype, SampleType::Float32);
    }

    #[tokio::test]
    async fn empty_group_and_root_array_and_missing_store() {
        // Empty group
        let mut empty = MemoryStore::new();
        empty.insert(
            "zarr.json",
            br#"{"zarr_format":3,"node_type":"group","attributes":{}}"#.as_slice(),
        );
        let h = ZarrHierarchy::open(&empty).await.unwrap();
        assert!(h.group(None).unwrap().variable_names().is_empty());

        // Root array (v3)
        let mut root_arr = MemoryStore::new();
        root_arr.insert("zarr.json", sample_v3_array_json());
        let ha = ZarrHierarchy::open(&root_arr).await.unwrap();
        assert_eq!(ha.open_mode, OpenMode::V3RootArray);
        assert_eq!(ha.root_array().unwrap().shape, vec![16, 32]);
        assert_eq!(root_arr.get_count(), 1);

        // Root array (v2)
        let mut root_arr_v2 = MemoryStore::new();
        root_arr_v2.insert(".zarray", sample_v2_array_json());
        let ha2 = ZarrHierarchy::open(&root_arr_v2).await.unwrap();
        assert_eq!(ha2.open_mode, OpenMode::V2RootArray);
        assert_eq!(ha2.root_array().unwrap().shape, vec![16, 32]);

        // Not a Zarr store
        let blank = MemoryStore::new();
        let err = ZarrHierarchy::open(&blank).await.unwrap_err();
        assert!(matches!(err, ZarrError::NotAZarrStore { .. }), "{err}");
    }
}
