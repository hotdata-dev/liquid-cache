//! This module contains the cache implementation for the Parquet reader.
//!

use crate::io::ParquetCacheMetadata;
use crate::reader::{LiquidPredicate, extract_multi_column_or};
use crate::sync::{Mutex, RwLock};
use ahash::AHashMap;
use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::buffer::BooleanBuffer;
use arrow_schema::{ArrowError, Field, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use liquid_cache::cache::EvictionPolicy;
use liquid_cache::cache::{
    CacheEntry, CacheExpression, CachePolicy, EntryID, EventTrace, HydrationPolicy, LiquidCache,
    LiquidCacheBuilder,
};
use parquet::arrow::arrow_reader::ArrowPredicate;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

mod column;
mod id;
mod registry;
mod stats;

pub use column::{CachedColumn, CachedColumnRef};
pub(crate) use column::{InsertArrowArrayError, PrefetchOutcome};
pub(crate) use id::ColumnAccessPath;
pub use id::{BatchID, ParquetArrayID};
use registry::{FileKey, FileLease, FileRegistry};

/// Typed lineage expressions for a single file, keyed by file-schema column name.
///
/// Produced by the physical lineage analyzer (local mode) or shipped from
/// the client (Flight mode), and attached to the
/// [`LiquidParquetSource`](crate::LiquidParquetSource) that opens the file.
pub type ColumnLineages = HashMap<String, Arc<CacheExpression>>;

/// The identity of a Parquet object within an object store.
///
/// Object paths are only unique within their object store, so both components
/// are required to keep cached data from different stores isolated.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ParquetFileIdentity {
    object_store_url: ObjectStoreUrl,
    path: String,
}

impl ParquetFileIdentity {
    /// Create an identity from an object store URL and an object path.
    pub fn new(object_store_url: ObjectStoreUrl, path: String) -> Self {
        Self {
            object_store_url,
            path,
        }
    }
}

/// One column of a row group: (file column index, field, lineage expression, is-predicate).
type CachedColumnSpec = (u64, Arc<Field>, Option<Arc<CacheExpression>>, bool);

#[derive(Default, Debug)]
pub(crate) struct RowGroupSnapshots {
    entries: RwLock<AHashMap<EntryID, Arc<CacheEntry>>>,
    selections: RwLock<AHashMap<BatchID, BooleanBuffer>>,
}

impl RowGroupSnapshots {
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<CacheEntry>> {
        self.entries.read().unwrap().get(entry_id).cloned()
    }

    pub(crate) fn insert(&self, entry_id: EntryID, entry: Arc<CacheEntry>) {
        self.entries.write().unwrap().insert(entry_id, entry);
    }

    pub(crate) fn selection(&self, batch_id: BatchID) -> Option<BooleanBuffer> {
        self.selections.read().unwrap().get(&batch_id).cloned()
    }

    pub(crate) fn insert_selection(&self, batch_id: BatchID, selection: BooleanBuffer) {
        self.selections.write().unwrap().insert(batch_id, selection);
    }

    /// Drop every snapshot entry and the selection held for `batch_id`.
    pub(crate) fn release_batch(&self, batch_id: BatchID) {
        let batch = u64::from(*batch_id);
        self.entries
            .write()
            .unwrap()
            .retain(|entry_id, _| ParquetArrayID::from(*entry_id).batch_id_inner() != batch);
        self.selections.write().unwrap().remove(&batch_id);
    }
}

#[cfg(test)]
impl RowGroupSnapshots {
    /// Test-only: (entry count, sum of `CacheEntry::memory_usage_bytes`).
    pub(crate) fn test_entry_count_and_bytes(&self) -> (usize, usize) {
        let entries = self.entries.read().unwrap();
        let bytes = entries.values().map(|e| e.memory_usage_bytes()).sum();
        (entries.len(), bytes)
    }

    /// Test-only: every snapshot entry this map holds.
    pub(crate) fn test_entries(&self) -> Vec<Arc<CacheEntry>> {
        self.entries.read().unwrap().values().cloned().collect()
    }
}

/// Test-only registry of every snapshot map handed to a row group on this
/// thread. `#[tokio::test]` is single-threaded, so each test sees only its own.
#[cfg(test)]
pub(crate) mod snapshot_test_registry {
    use super::RowGroupSnapshots;
    use liquid_cache::cache::{CacheEntry, LiquidCache};
    use std::cell::RefCell;
    use std::sync::{Arc, Weak};

    thread_local! {
        static LIVE: RefCell<Vec<Weak<RowGroupSnapshots>>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn register(snapshots: &Arc<RowGroupSnapshots>) {
        LIVE.with(|live| {
            let mut live = live.borrow_mut();
            let ptr = Arc::as_ptr(snapshots);
            if !live.iter().any(|weak| weak.as_ptr() == ptr) {
                live.push(Arc::downgrade(snapshots));
            }
        });
    }

    /// (live non-empty maps, entries, bytes) over maps that are still alive.
    pub(crate) fn live_totals() -> (usize, usize, usize) {
        LIVE.with(|live| {
            let mut maps = 0;
            let mut entries = 0;
            let mut bytes = 0;
            for weak in live.borrow().iter() {
                if let Some(snapshots) = weak.upgrade() {
                    let (count, size) = snapshots.test_entry_count_and_bytes();
                    if count > 0 {
                        maps += 1;
                    }
                    entries += count;
                    bytes += size;
                }
            }
            (maps, entries, bytes)
        })
    }

    fn data_ptr(entry: &CacheEntry) -> Option<usize> {
        match entry {
            CacheEntry::MemoryArrow(array) => Some(Arc::as_ptr(array) as *const () as usize),
            CacheEntry::MemoryLiquid(array) => Some(Arc::as_ptr(array) as *const () as usize),
            _ => None,
        }
    }

    /// (memory bytes the index holds, bytes only live snapshots hold), each
    /// array counted once.
    pub(crate) fn live_footprint(cache: &LiquidCache) -> (usize, usize) {
        let mut held = std::collections::HashSet::new();
        let mut in_index = 0;
        cache.for_each_entry(|_, entry| {
            if let Some(ptr) = data_ptr(entry)
                && held.insert(ptr)
            {
                in_index += entry.memory_usage_bytes();
            }
        });
        LIVE.with(|live| {
            let mut snapshot_only = 0;
            for weak in live.borrow().iter() {
                if let Some(snapshots) = weak.upgrade() {
                    for entry in snapshots.test_entries() {
                        if let Some(ptr) = data_ptr(&entry)
                            && held.insert(ptr)
                        {
                            snapshot_only += entry.memory_usage_bytes();
                        }
                    }
                }
            }
            (in_index, snapshot_only)
        })
    }
}

#[derive(Default, Debug)]
struct ColumnMaps {
    // invariant: Arc::ptr_eq(map[field.name()], map[field.id()])
    by_id: AHashMap<u64, CachedColumnRef>,
    by_name: AHashMap<String, CachedColumnRef>,
}

/// A row group in the cache.
#[derive(Debug)]
pub struct CachedRowGroup {
    columns: ColumnMaps,
    cache_store: Arc<LiquidCache>,
    snapshots: Arc<RowGroupSnapshots>,
}

impl CachedRowGroup {
    /// Create a new row group.
    /// The column_ids are the indices of the columns in the file schema.
    /// So they may not start from 0.
    fn new(
        cache_store: Arc<LiquidCache>,
        row_group_idx: u64,
        file: FileKey,
        lease: &FileLease,
        columns: &[CachedColumnSpec],
        snapshots: Arc<RowGroupSnapshots>,
    ) -> Self {
        let mut column_maps = ColumnMaps::default();
        for (column_id, field, expression, is_predicate_column) in columns {
            let column_access_path =
                ColumnAccessPath::new(file.slot as u64, row_group_idx, *column_id);
            let column = Arc::new(CachedColumn::new(
                Arc::clone(field),
                Arc::clone(&cache_store),
                column_access_path,
                file.identity,
                lease.clone(),
                expression.clone(),
                *is_predicate_column,
                Arc::clone(&snapshots),
            ));
            column_maps.by_id.insert(*column_id, column.clone());
            column_maps.by_name.insert(field.name().to_string(), column);
        }

        Self {
            columns: column_maps,
            cache_store,
            snapshots,
        }
    }

    /// Returns the batch size configured for this cached row group.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Get a column from the row group.
    pub fn get_column(&self, column_id: u64) -> Option<CachedColumnRef> {
        self.columns.by_id.get(&column_id).cloned()
    }

    /// Get a column from the row group by its field name.
    pub fn get_column_by_name(&self, column_name: &str) -> Option<CachedColumnRef> {
        if let Some(column) = self.columns.by_name.get(column_name) {
            return Some(column.clone());
        }

        // DataFusion may carry qualified names in physical expressions
        // (e.g. "table.col"), while cache fields are keyed by file schema names.
        let unqualified = column_name.rsplit('.').next().unwrap_or(column_name);
        self.columns.by_name.get(unqualified).cloned()
    }

    pub(crate) fn snapshot_selection(&self, batch_id: BatchID) -> Option<BooleanBuffer> {
        self.snapshots.selection(batch_id)
    }

    /// Drop the snapshots held for `batch_id` once the reader is done with it.
    pub(crate) fn release_snapshot_batch(&self, batch_id: BatchID) {
        self.snapshots.release_batch(batch_id);
    }

    /// Evaluate a predicate on a row group.
    #[fastrace::trace]
    pub async fn evaluate_selection_with_predicate(
        &self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
    ) -> Option<Result<BooleanArray, ArrowError>> {
        let column_ids = predicate.predicate_column_ids();

        if column_ids.len() == 1 {
            // If we only have one column, we can short-circuit and try to evaluate the predicate on encoded data.
            let column_id = column_ids[0];
            let cache = self.get_column(column_id as u64)?;
            return cache
                .eval_predicate_with_filter(batch_id, selection, predicate)
                .await;
        } else if column_ids.len() >= 2 {
            // Try to extract multiple column-literal expressions from OR structure
            if let Some(column_exprs) =
                extract_multi_column_or(predicate.physical_expr_physical_column_index())
            {
                let mut combined_buffer: Option<BooleanArray> = None;

                for (col_name, expr) in column_exprs {
                    let column = self.get_column_by_name(col_name)?;
                    let snapshot_liquid = match column.snapshot_entry(batch_id) {
                        Some(entry) => match entry.as_ref() {
                            CacheEntry::MemoryLiquid(array) => Some(Arc::clone(array)),
                            _ => {
                                combined_buffer = None;
                                break;
                            }
                        },
                        None => None,
                    };
                    let expr = expr
                        .transform_up(|expr| {
                            if let Some(column) = expr.downcast_ref::<Column>() {
                                Ok(Transformed::yes(Arc::new(Column::new(column.name(), 0))
                                    as Arc<dyn PhysicalExpr>))
                            } else {
                                Ok(Transformed::no(expr))
                            }
                        })
                        .ok()?
                        .data;
                    let liquid_expr = column.liquid_expr_for_predicate(expr);
                    let liquid_expr = match liquid_expr {
                        Some(expr) => expr,
                        None => {
                            combined_buffer = None;
                            break;
                        }
                    };
                    let entry_id = column.entry_id(batch_id).into();
                    let liquid_array = match snapshot_liquid {
                        Some(array) => Some(array),
                        None => {
                            self.cache_store
                                .try_read_liquid(&entry_id, column.file_identity())
                                .await
                        }
                    };
                    let liquid_array = match liquid_array {
                        None => {
                            combined_buffer = None;
                            break;
                        }
                        Some(array) => array,
                    };
                    let buffer = liquid_array.try_eval_predicate(&liquid_expr, selection);

                    combined_buffer = Some(match combined_buffer {
                        None => buffer,
                        Some(existing) => {
                            arrow::compute::kernels::boolean::or_kleene(&existing, &buffer).ok()?
                        }
                    });
                }

                if let Some(result) = combined_buffer {
                    return Some(Ok(result));
                }
            }
        }
        // A conjunct that reads no column still has to be evaluated, over a batch
        // that carries only the row count the selection implies.
        if column_ids.is_empty() {
            let options =
                RecordBatchOptions::new().with_row_count(Some(selection.count_set_bits()));
            let record_batch =
                RecordBatch::try_new_with_options(Arc::new(Schema::empty()), Vec::new(), &options)
                    .ok()?;
            return Some(predicate.evaluate(record_batch));
        }

        // Otherwise, we need to first convert the data into arrow arrays.
        let mut arrays = Vec::new();
        let mut fields = Vec::new();
        for column_id in column_ids {
            let column = self.get_column(column_id as u64)?;
            let array = column
                .get_arrow_array_with_filter(batch_id, selection)
                .await?;
            arrays.push(array);
            fields.push(column.field());
        }
        let schema = Arc::new(Schema::new(fields));
        let record_batch = RecordBatch::try_new(schema, arrays).unwrap();
        let boolean_array = predicate.evaluate(record_batch).unwrap();
        Some(Ok(boolean_array))
    }
}

pub(crate) type CachedRowGroupRef = Arc<CachedRowGroup>;

/// A file in the cache.
#[derive(Debug)]
pub struct CachedFile {
    cache_store: Arc<LiquidCache>,
    key: FileKey,
    /// Keeps the file registered while this handle, or a row group or column
    /// created from it, is alive.
    lease: FileLease,
    file_schema: SchemaRef,
    lineages: Arc<ColumnLineages>,
}

impl CachedFile {
    fn new(
        cache_store: Arc<LiquidCache>,
        key: FileKey,
        lease: FileLease,
        file_schema: SchemaRef,
        lineages: Arc<ColumnLineages>,
    ) -> Self {
        Self {
            cache_store,
            key,
            lease,
            file_schema,
            lineages,
        }
    }

    /// Create a row group handle scoped to the current query context.
    pub fn create_row_group(
        &self,
        row_group_id: u64,
        predicate_column_ids: Vec<usize>,
    ) -> CachedRowGroupRef {
        self.create_row_group_with_snapshots(row_group_id, predicate_column_ids, Arc::default())
    }

    pub(crate) fn create_row_group_with_snapshots(
        &self,
        row_group_id: u64,
        predicate_column_ids: Vec<usize>,
        snapshots: Arc<RowGroupSnapshots>,
    ) -> CachedRowGroupRef {
        let columns: Vec<CachedColumnSpec> = self
            .file_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                let is_predicate_column = predicate_column_ids.contains(&idx);
                let expression = self.lineages.get(field.name()).cloned();
                (
                    idx as u64,
                    Arc::clone(field),
                    expression,
                    is_predicate_column,
                )
            })
            .collect();

        #[cfg(test)]
        snapshot_test_registry::register(&snapshots);

        Arc::new(CachedRowGroup::new(
            self.cache_store.clone(),
            row_group_id,
            self.key,
            &self.lease,
            &columns,
            snapshots,
        ))
    }

    /// Return the configured cache batch size.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Return the full file schema tracked by the cache entry.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.file_schema)
    }
}

/// A reference to a cached file.
pub(crate) type CachedFileRef = Arc<CachedFile>;

/// The main cache structure.
#[derive(Debug)]
pub struct LiquidCacheParquet {
    /// The files the cache knows, and the key each one is cached under.
    files: Mutex<FileRegistry>,

    cache_store: Arc<LiquidCache>,

    /// The lineage hints the cache keeps per column, owned by `cache_store`.
    /// Held here to drop a forgotten file's hints.
    metadata: Arc<ParquetCacheMetadata>,

    /// Set while a sweep for forgettable files runs, so only one does.
    sweeping: AtomicBool,
}

/// A reference to the main cache structure.
pub type LiquidCacheParquetRef = Arc<LiquidCacheParquet>;

impl LiquidCacheParquet {
    /// Create a new cache for parquet files.
    pub async fn new(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        store: t4::Store,
        cache_policy: Box<dyn CachePolicy>,
        eviction_policy: Box<dyn EvictionPolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
    ) -> Self {
        Self::new_with_eviction_concurrency(
            batch_size,
            max_memory_bytes,
            max_disk_bytes,
            store,
            cache_policy,
            eviction_policy,
            hydration_policy,
            !cfg!(test),
        )
        .await
    }

    /// Create a new cache for parquet files with explicit victim eviction concurrency.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_eviction_concurrency(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        store: t4::Store,
        cache_policy: Box<dyn CachePolicy>,
        eviction_policy: Box<dyn EvictionPolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        evict_victims_concurrently: bool,
    ) -> Self {
        assert!(batch_size.is_power_of_two());
        let metadata = Arc::new(ParquetCacheMetadata::new());
        let cache_storage = LiquidCacheBuilder::new()
            .with_batch_size(batch_size)
            .with_max_memory_bytes(max_memory_bytes)
            .with_max_disk_bytes(max_disk_bytes)
            .with_eviction_policy(eviction_policy)
            .with_cache_policy(cache_policy)
            .with_hydration_policy(hydration_policy)
            .with_metadata(Arc::clone(&metadata) as _)
            .with_store(store)
            .with_evict_victims_concurrently(evict_victims_concurrently)
            .build()
            .await;

        LiquidCacheParquet {
            files: Mutex::new(FileRegistry::default()),
            cache_store: cache_storage,
            metadata,
            sweeping: AtomicBool::new(false),
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(
        &self,
        file_identity: ParquetFileIdentity,
        full_file_schema: SchemaRef,
    ) -> CachedFileRef {
        self.register_or_get_file_with_hints(file_identity, full_file_schema, Arc::default())
    }

    /// Register a file in the cache, attaching typed lineage expressions derived from
    /// the query plan (keyed by file-schema column name).
    pub fn register_or_get_file_with_hints(
        &self,
        file_identity: ParquetFileIdentity,
        full_file_schema: SchemaRef,
        lineages: Arc<ColumnLineages>,
    ) -> CachedFileRef {
        let mut files = self.files.lock().unwrap();
        let (key, lease, new) = files.register(file_identity);
        let sweep = new && files.should_sweep();
        drop(files);
        let file = Arc::new(CachedFile::new(
            self.cache_store.clone(),
            key,
            lease,
            full_file_schema,
            lineages,
        ));
        if sweep {
            self.reclaim_idle_files();
        }
        file
    }

    /// Forget every registered file that no handle refers to and that has
    /// nothing cached, with the lineage hints kept for it. Returns how many
    /// were forgotten.
    ///
    /// Registration runs this whenever the number of registered files has
    /// doubled since the last run, or could take every key slot, so the
    /// registry stays proportional to the files being read or holding cached
    /// data, and idle files usually give their slots back before a new file
    /// has to share one. A forgotten file registers again as a new one: it
    /// had nothing cached to lose.
    pub fn reclaim_idle_files(&self) -> usize {
        /// Clears the flag however the sweep ends, so a panic in one does not
        /// stop every later sweep.
        struct Sweeping<'a>(&'a AtomicBool);
        impl Drop for Sweeping<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        if self.sweeping.swap(true, Ordering::Acquire) {
            return 0;
        }
        let _sweeping = Sweeping(&self.sweeping);
        let idle = self.files.lock().unwrap().idle_files();
        let mut occupied = ahash::AHashSet::new();
        if !idle.is_empty() {
            self.cache_store
                .for_each_entry_identity(|identity| _ = occupied.insert(identity));
        }
        let mut files = self.files.lock().unwrap();
        let (forgotten, freed) = files.forget(&idle, &occupied);
        // Before a freed slot can be handed out again.
        self.metadata.forget_files(&freed);
        files.swept();
        forgotten
    }

    /// How many files are registered with this cache: those being read, or
    /// holding cached data, plus any not yet forgotten by
    /// [`Self::reclaim_idle_files`].
    pub fn leased_file_ids(&self) -> usize {
        self.files.lock().unwrap().len()
    }

    /// How many registrations so far found every value of the cache key's
    /// 16-bit file field taken by a registered file, and had to share one.
    /// Files sharing a value compute the same keys: each is served correctly,
    /// but they cannot share the cache. The count never goes down. It rises
    /// briefly when the registered files come within a sweep's interval of
    /// 65,536, and steadily when more than that are being read or holding
    /// cached data at once.
    pub fn file_ids_over_key_width(&self) -> u64 {
        self.files.lock().unwrap().over_key_width()
    }

    /// How many cache lookups found a key held by another file, and missed
    /// rather than serve it.
    pub fn identity_mismatches(&self) -> u64 {
        self.cache_store.identity_mismatches()
    }

    /// Get the batch size of the cache.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Get the max memory bytes of the cache.
    pub fn max_memory_bytes(&self) -> usize {
        self.cache_store.config().max_memory_bytes()
    }

    /// Get the max disk bytes of the cache.
    pub fn max_disk_bytes(&self) -> usize {
        self.cache_store.config().max_disk_bytes()
    }

    /// Get the memory usage of the cache in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.cache_store.budget().memory_usage_bytes()
    }

    /// Get the disk usage of the cache in bytes.
    pub fn disk_usage_bytes(&self) -> usize {
        self.cache_store.budget().disk_usage_bytes()
    }

    /// Flush the cache trace to a file.
    pub fn flush_trace(&self, to_file: impl AsRef<Path>) {
        self.cache_store.observer().flush_cache_trace(to_file);
    }

    /// Enable the cache trace.
    pub fn enable_trace(&self) {
        self.cache_store.observer().enable_cache_trace();
    }

    /// Disable the cache trace.
    pub fn disable_trace(&self) {
        self.cache_store.observer().disable_cache_trace();
    }

    /// Reset the cache.
    ///
    /// # Safety
    /// This is unsafe because resetting the cache while other threads are using the cache may cause undefined behavior.
    /// You should only call this when no one else is using the cache.
    pub unsafe fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        files.clear();
        self.metadata.clear();
        self.cache_store.reset();
    }

    /// Flush all memory-based entries to disk while preserving their format.
    /// Arrow entries become DiskArrow, Liquid entries become DiskLiquid.
    /// Entries already on disk are left unchanged.
    ///
    /// This is for admin use only.
    /// This has no guarantees that some new entry will not be inserted in the meantime, or some entries are promoted to memory again.
    /// You mostly want to use this when no one else is using the cache.
    pub async fn flush_data(&self) -> Result<(), liquid_cache::cache::CacheFull> {
        self.cache_store.flush_all_to_disk().await
    }

    /// Get the storage of the cache.
    pub fn storage(&self) -> &Arc<LiquidCache> {
        &self.cache_store
    }

    /// Consume the event trace of the cache.
    pub fn consume_event_trace(&self) -> EventTrace {
        self.cache_store.consume_event_trace()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CachedRowGroupRef, LiquidCacheParquet};
    use crate::reader::FilterCandidateBuilder;
    use arrow::array::{Array, ArrayRef, Int32Array, StringViewArray};
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::expressions::Column;
    use liquid_cache::cache::AlwaysHydrate;
    use liquid_cache::cache::TranscodeEvict;
    use liquid_cache::cache_policies::LiquidPolicy;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use std::sync::Arc;

    async fn setup_cache(batch_size: usize, schema: SchemaRef) -> CachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let file = cache.register_or_get_file(
            ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), "test".to_string()),
            schema,
        );
        file.create_row_group(0, vec![])
    }

    async fn setup_liquid_cache(
        batch_size: usize,
        schema: SchemaRef,
        max_memory_bytes: usize,
    ) -> CachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            max_memory_bytes,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        cache
            .register_or_get_file(
                ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), "test".to_string()),
                schema,
            )
            .create_row_group(0, vec![0, 1])
    }

    fn build_predicate(
        schema: &SchemaRef,
        arrays: Vec<ArrayRef>,
        expr: Arc<dyn PhysicalExpr>,
    ) -> LiquidPredicate {
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(schema), None).unwrap();
        writer
            .write(&RecordBatch::try_new(Arc::clone(schema), arrays).unwrap())
            .unwrap();
        writer.close().unwrap();
        let reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&reader, ArrowReaderOptions::new()).unwrap();
        let candidate = FilterCandidateBuilder::new(expr, Arc::clone(schema))
            .build(metadata.metadata())
            .unwrap()
            .unwrap();
        let projection = candidate.projection(metadata.metadata());
        LiquidPredicate::try_new(candidate, projection).unwrap()
    }

    fn equals(name: &str, index: usize, value: ScalarValue) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new(name, index)),
            Operator::Eq,
            Arc::new(Literal::new(value)),
        ))
    }

    fn dummy(len: usize) -> ArrayRef {
        Arc::new(Int32Array::from(vec![0; len]))
    }

    /// Cache `first` and `second` as columns 0 and 1 of a cache they fill,
    /// then a dummy column 2 that transcoding either one alone does not make
    /// room for, and transcoding both does. Eviction takes only as many
    /// victims as the dummy needs room for, so it transcodes both to Liquid.
    async fn insert_liquid_columns(
        row_group: &CachedRowGroupRef,
        batch_id: BatchID,
        first: ArrayRef,
        second: ArrayRef,
    ) {
        let freed = |array: &ArrayRef| {
            let liquid = liquid_cache::liquid_array::LiquidArray::from_arrow_array(array)
                .expect("the column transcodes to Liquid");
            array.get_array_memory_size() - liquid.get_array_memory_size()
        };
        let (freed_first, freed_second) = (freed(&first), freed(&second));
        let (more, less) = (freed_first.max(freed_second), freed_first.min(freed_second));
        let dummy = dummy((more + less / 2) / std::mem::size_of::<i32>());
        let dummy_size = dummy.get_array_memory_size();
        assert!(
            more < dummy_size && dummy_size <= more + less,
            "a {dummy_size}-byte dummy does not need both columns transcoded \
             (they free {freed_first} and {freed_second} bytes)"
        );
        for (column_id, array) in [first, second, dummy].into_iter().enumerate() {
            row_group
                .get_column(column_id as u64)
                .unwrap()
                .insert(batch_id, array)
                .await
                .unwrap();
        }
        assert_eq!(row_group.cache_store.stats().memory_liquid_entries, 2);
    }

    #[tokio::test]
    async fn or_fast_path_evaluates_on_liquid() {
        let batch_size = 1024;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("dummy", DataType::Int32, false),
        ]));
        let a: ArrayRef = Arc::new(Int32Array::from_iter_values(
            (0..batch_size).map(|i| (i % 4) as i32 + 1),
        ));
        let b: ArrayRef = Arc::new(Int32Array::from_iter_values(
            (0..batch_size).map(|i| (i % 5) as i32 * 10),
        ));
        let budget = a.get_array_memory_size() + b.get_array_memory_size();
        let row_group = setup_liquid_cache(batch_size, schema.clone(), budget).await;
        let batch_id = BatchID::from_row_id(0, batch_size);
        insert_liquid_columns(&row_group, batch_id, a.clone(), b.clone()).await;
        let expr = Arc::new(BinaryExpr::new(
            equals("a", 0, ScalarValue::Int32(Some(3))),
            Operator::Or,
            equals("b", 1, ScalarValue::Int32(Some(20))),
        ));
        let mut predicate = build_predicate(&schema, vec![a, b, dummy(batch_size)], expr);
        row_group.cache_store.stats();
        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i % 4 == 2 || i % 5 == 2);
        assert_eq!(result, BooleanArray::new(expected, None));
        assert!(row_group.cache_store.stats().runtime.try_read_liquid_calls >= 2);
    }

    #[tokio::test]
    async fn or_fast_path_on_strings() {
        // Large enough that the Liquid form of each column, with its
        // per-node overhead, is smaller than its Arrow form.
        let batch_size = 1024;
        let schema = Arc::new(Schema::new(vec![
            Field::new("city", DataType::Utf8View, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("dummy", DataType::Int32, false),
        ]));
        let city: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..batch_size).map(|i| if i % 5 == 2 { "Tokyo" } else { "Paris" }),
        ));
        let name: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..batch_size).map(|i| if i % 4 == 1 { "Bob" } else { "Alice" }),
        ));
        let budget = city.get_array_memory_size() + name.get_array_memory_size();
        let row_group = setup_liquid_cache(batch_size, schema.clone(), budget).await;
        let batch_id = BatchID::from_row_id(0, batch_size);
        insert_liquid_columns(&row_group, batch_id, city.clone(), name.clone()).await;
        let expr = Arc::new(BinaryExpr::new(
            equals("name", 1, ScalarValue::Utf8View(Some("Bob".into()))),
            Operator::Or,
            equals("city", 0, ScalarValue::Utf8View(Some("Tokyo".into()))),
        ));
        let mut predicate = build_predicate(&schema, vec![city, name, dummy(batch_size)], expr);
        row_group.cache_store.stats();
        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i % 4 == 1 || i % 5 == 2);
        assert_eq!(result, BooleanArray::new(expected, None));
        assert!(row_group.cache_store.stats().runtime.try_read_liquid_calls >= 2);
    }

    #[tokio::test]
    async fn evaluate_or_on_cached_columns() {
        let batch_size = 4;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_a = row_group.get_column(0).unwrap();
        let col_b = row_group.get_column(1).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));

        assert!(col_a.insert(batch_id, array_a.clone()).await.is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression a = 3 OR b = 20
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 2).into();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn evaluate_three_column_or() {
        let batch_size = 8;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));

        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_a = row_group.get_column(0).unwrap();
        let col_b = row_group.get_column(1).unwrap();
        let col_c = row_group.get_column(2).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]));
        let array_c = Arc::new(Int32Array::from(vec![
            100, 200, 300, 400, 500, 600, 700, 800,
        ]));

        assert!(col_a.insert(batch_id, array_a.clone()).await.is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).await.is_ok());
        assert!(col_c.insert(batch_id, array_c.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b, array_c]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: a = 2 OR b = 40 OR c = 600
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(40)))),
        ));
        let expr_c: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(600)))),
        ));

        // Build nested OR: (a = 2 OR b = 40) OR c = 600
        let expr_ab = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_ab, Operator::Or, expr_c));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        // Expected: row 1 (a=2), row 3 (b=40), row 5 (c=600) -> indices 1, 3, 5
        let expected =
            BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3 || i == 5).into();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn evaluate_string_column_or() {
        let batch_size = 8;

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("city", DataType::Utf8View, false),
        ]));

        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_name = row_group.get_column(0).unwrap();
        let col_city = row_group.get_column(1).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_name = Arc::new(arrow::array::StringViewArray::from(vec![
            "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
        ]));
        let array_city = Arc::new(arrow::array::StringViewArray::from(vec![
            "New York", "London", "Paris", "Tokyo", "Berlin", "Sydney", "Madrid", "Rome",
        ]));

        assert!(col_name.insert(batch_id, array_name.clone()).await.is_ok());
        assert!(col_city.insert(batch_id, array_city.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_name, array_city]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: name = "Bob" OR city = "Tokyo"
        let expr_name: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("name", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8View(Some("Bob".to_string())))),
        ));
        let expr_city: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("city", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8View(Some(
                "Tokyo".to_string(),
            )))),
        ));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(expr_name, Operator::Or, expr_city));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        // Expected: row 1 (name="Bob"), row 3 (city="Tokyo") -> indices 1, 3
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3).into();
        assert_eq!(result, expected);
    }

    /// A cache key holds 16 bits of file id, so the 65,537th file registered
    /// computes the same keys as the first. Its reads must not be served the
    /// first file's data, and its own data must still be cacheable.
    #[tokio::test]
    async fn files_whose_keys_collide_do_not_share_entries() {
        let batch_size = 64;
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let identity =
            |path: String| ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), path);
        let batch_id = BatchID::from_row_id(0, batch_size);

        let first = cache.register_or_get_file(identity("first".to_string()), schema.clone());
        let first_column = first.create_row_group(0, vec![]).get_column(0).unwrap();
        let first_data: ArrayRef = Arc::new(Int32Array::from_iter_values(0..batch_size as i32));
        first_column
            .insert(batch_id, first_data.clone())
            .await
            .unwrap();

        // Held, so all 65,536 files are registered at once.
        let _fillers: Vec<_> = (1..=u16::MAX as u32)
            .map(|i| cache.register_or_get_file(identity(format!("filler-{i}")), schema.clone()))
            .collect();
        assert_eq!(cache.file_ids_over_key_width(), 0);

        let colliding = cache.register_or_get_file(identity("colliding".to_string()), schema);
        assert_eq!(cache.file_ids_over_key_width(), 1);
        let colliding_column = colliding.create_row_group(0, vec![]).get_column(0).unwrap();
        assert_eq!(
            colliding_column.entry_id(batch_id),
            first_column.entry_id(batch_id),
            "the 65,537th file should compute the first file's key"
        );

        assert!(
            colliding_column
                .get_arrow_array_test_only(batch_id)
                .await
                .is_none(),
            "a file was served another file's cached data"
        );

        let colliding_data: ArrayRef =
            Arc::new(Int32Array::from_iter_values((0..batch_size as i32).rev()));
        colliding_column
            .insert(batch_id, colliding_data.clone())
            .await
            .expect("a file could not cache its own data");
        let read = colliding_column
            .get_arrow_array_test_only(batch_id)
            .await
            .unwrap();
        assert_eq!(read.as_ref(), colliding_data.as_ref());
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use liquid_cache::cache::{AlwaysHydrate, Evict};
    use liquid_cache::cache_policies::LiquidPolicy;

    const BATCH: usize = 1024;

    /// A memory-only cache (no disk tier) that holds `batches` batches.
    async fn memory_only_cache(dir: &std::path::Path, batches: usize) -> LiquidCacheParquet {
        let store = t4::mount(dir.join("liquid_cache.t4")).await.unwrap();
        LiquidCacheParquet::new(
            BATCH,
            batch_data(0).get_array_memory_size() * batches,
            0,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(Evict),
            Box::new(AlwaysHydrate::new()),
        )
        .await
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    fn identity(file: usize) -> ParquetFileIdentity {
        ParquetFileIdentity::new(
            ObjectStoreUrl::local_filesystem(),
            format!("table/file-{file}.parquet"),
        )
    }

    fn batch_data(seed: usize) -> ArrayRef {
        Arc::new(Int32Array::from_iter_values(
            (0..BATCH as i32).map(|v| v.wrapping_mul(seed as i32 + 7)),
        ))
    }

    /// Read one file as a filtered scan does: register it, open a row group
    /// with a predicate column, cache a batch, and drop every handle.
    async fn read_file(cache: &LiquidCacheParquet, file: usize) {
        let handle = cache.register_or_get_file(identity(file), schema());
        let row_group = handle.create_row_group(0, vec![0]);
        let column = row_group.get_column(0).unwrap();
        let _ = column.insert(BatchID::from_raw(0), batch_data(file)).await;
    }

    /// A long-running process reads ever new files (a table's data files are
    /// replaced on every compaction) through a cache far smaller than all of
    /// them. Files whose data was evicted and that nothing reads any more
    /// must not stay registered, nor keep their lineage hints.
    #[tokio::test]
    async fn registry_stays_bounded_as_files_come_and_go() {
        const FILES: usize = 10_000;
        const CACHED_BATCHES: usize = 8;
        let dir = tempfile::tempdir().unwrap();
        let cache = memory_only_cache(dir.path(), CACHED_BATCHES).await;
        for file in 0..FILES {
            read_file(&cache, file).await;
        }
        let registered = cache.leased_file_ids();
        let hinted = cache.metadata.hinted_columns();
        println!(
            "after reading {FILES} files: registered={registered} hinted_columns={hinted} \
             cached_entries={}",
            cache.storage().stats().total_entries
        );
        assert!(cache.storage().stats().total_entries <= CACHED_BATCHES);
        assert!(
            registered <= 2 * 1024,
            "{registered} of {FILES} files are still registered, though the cache holds at \
             most {CACHED_BATCHES} batches and no handle is alive"
        );
        assert!(
            hinted <= registered,
            "{hinted} columns keep lineage hints for {registered} registered files"
        );
        assert_eq!(cache.file_ids_over_key_width(), 0);

        // An explicit sweep keeps only the files that still hold data.
        cache.reclaim_idle_files();
        assert!(cache.leased_file_ids() <= CACHED_BATCHES);
        assert!(cache.metadata.hinted_columns() <= CACHED_BATCHES);
    }

    /// Forgetting a file must not cost it its cached data: a file whose
    /// batches are still cached stays registered under the same key, and is
    /// served them when it is read again.
    #[tokio::test]
    async fn a_file_with_cached_data_stays_registered() {
        let dir = tempfile::tempdir().unwrap();
        let cache = memory_only_cache(dir.path(), 8).await;
        read_file(&cache, 0).await;
        assert_eq!(cache.reclaim_idle_files(), 0);
        assert_eq!(cache.leased_file_ids(), 1);
        assert_eq!(cache.metadata.hinted_columns(), 1);

        let handle = cache.register_or_get_file(identity(0), schema());
        let column = handle.create_row_group(0, vec![]).get_column(0).unwrap();
        let cached = column
            .get_arrow_array_test_only(BatchID::from_raw(0))
            .await
            .expect("the file's cached batch was lost");
        assert_eq!(cached.as_ref(), batch_data(0).as_ref());
    }

    /// A row group or column handle outliving its file handle can still
    /// insert, so it keeps the file registered: otherwise its slot could be
    /// handed to another file while it writes under it.
    #[tokio::test]
    async fn a_live_column_keeps_its_file_registered() {
        let dir = tempfile::tempdir().unwrap();
        let cache = memory_only_cache(dir.path(), 8).await;
        let handle = cache.register_or_get_file(identity(0), schema());
        let column = handle.create_row_group(0, vec![]).get_column(1).unwrap();
        drop(handle);

        assert_eq!(cache.reclaim_idle_files(), 0);
        column
            .insert(BatchID::from_raw(0), batch_data(1))
            .await
            .unwrap();
        drop(column);
        assert_eq!(cache.reclaim_idle_files(), 0, "the file now holds data");

        let other = cache.register_or_get_file(identity(1), schema());
        let other_column = other.create_row_group(0, vec![]).get_column(1).unwrap();
        assert!(
            other_column
                .get_arrow_array_test_only(BatchID::from_raw(0))
                .await
                .is_none(),
            "another file was served the first file's data"
        );
    }

    /// A forgotten file's slot goes to the next new file, so a long-running
    /// cache keeps registered files on distinct keys instead of running out
    /// of the key's 16 bits.
    #[tokio::test]
    async fn a_forgotten_file_frees_its_slot() {
        let dir = tempfile::tempdir().unwrap();
        let cache = memory_only_cache(dir.path(), 8).await;
        let first = cache.register_or_get_file(identity(0), schema()).key;
        read_file(&cache, 1).await;
        assert_eq!(cache.reclaim_idle_files(), 1, "only file 0 holds nothing");
        assert_eq!(cache.leased_file_ids(), 1);

        let next = cache.register_or_get_file(identity(2), schema());
        assert_eq!(next.key.slot, first.slot);
        assert_ne!(next.key.identity, first.identity);
        // Read again, the forgotten file registers as a new one.
        let again = cache.register_or_get_file(identity(0), schema());
        assert_ne!(again.key.identity, first.identity);
    }

    #[tokio::test]
    async fn reset_drops_lineage_hints() {
        let dir = tempfile::tempdir().unwrap();
        let cache = memory_only_cache(dir.path(), 8).await;
        let handle = cache.register_or_get_file(identity(0), schema());
        drop(handle.create_row_group(0, vec![0, 1]));
        assert_eq!(cache.metadata.hinted_columns(), 2);
        drop(handle);
        unsafe { cache.reset() };
        assert_eq!(cache.leased_file_ids(), 0);
        assert_eq!(cache.metadata.hinted_columns(), 0);
    }
}
