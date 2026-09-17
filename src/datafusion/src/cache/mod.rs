//! This module contains the cache implementation for the Parquet reader.
//!

use crate::io::ParquetCacheMetadata;
use crate::reader::{LiquidPredicate, extract_multi_column_or};
use ahash::AHashMap;
use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::buffer::BooleanBuffer;
use arrow_schema::{ArrowError, Field, Schema, SchemaRef};
use liquid_cache::cache::squeeze_policies::SqueezePolicy;
use liquid_cache::cache::{
    CacheExpression, CachePolicy, EventTrace, HydrationPolicy, LiquidCache, LiquidCacheBuilder,
};
use parquet::arrow::arrow_reader::ArrowPredicate;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

mod column;
mod file_id;
mod id;
mod stats;

use file_id::{FileId, FileIdPool};

pub(crate) use column::InsertArrowArrayError;
pub use column::{CachedColumn, CachedColumnRef};
pub(crate) use id::ColumnAccessPath;
pub use id::{BatchID, ParquetArrayID};

/// Typed squeeze hints for a single file, keyed by file-schema column name.
///
/// Produced by the physical squeeze-hint analyzer (local mode) or shipped from
/// the client (Flight mode), and attached to the
/// [`LiquidParquetSource`](crate::LiquidParquetSource) that opens the file.
pub type ColumnSqueezeHints = HashMap<String, Arc<CacheExpression>>;

/// One column of a row group: (file column index, field, squeeze hint, is-predicate).
type CachedColumnSpec = (u64, Arc<Field>, Option<Arc<CacheExpression>>, bool);

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
}

impl CachedRowGroup {
    /// Create a new row group.
    /// The column_ids are the indices of the columns in the file schema.
    /// So they may not start from 0.
    fn new(
        cache_store: Arc<LiquidCache>,
        row_group_idx: u64,
        file_id: Arc<FileId>,
        columns: &[CachedColumnSpec],
    ) -> Self {
        let mut column_maps = ColumnMaps::default();
        for (column_id, field, expression, is_predicate_column) in columns {
            let column_access_path =
                ColumnAccessPath::new(file_id.get(), row_group_idx, *column_id);
            let column = Arc::new(CachedColumn::new(
                Arc::clone(field),
                Arc::clone(&cache_store),
                column_access_path,
                Arc::clone(&file_id),
                expression.clone(),
                *is_predicate_column,
            ));
            column_maps.by_id.insert(*column_id, column.clone());
            column_maps.by_name.insert(field.name().to_string(), column);
        }

        Self {
            columns: column_maps,
            cache_store,
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
                    let liquid_expr = column.liquid_expr_for_predicate(Arc::clone(&expr));
                    let liquid_expr = match liquid_expr {
                        Some(expr) => expr,
                        None => {
                            combined_buffer = None;
                            break;
                        }
                    };
                    let entry_id = column.entry_id(batch_id).into();
                    let liquid_array = self
                        .cache_store
                        .try_read_liquid(&entry_id, column.identity())
                        .await;
                    let liquid_array = match liquid_array {
                        None => {
                            combined_buffer = None;
                            break;
                        }
                        Some(array) => array,
                    };
                    // Leave the loop rather than the function, as the two
                    // arms above do: an array that cannot answer the predicate
                    // does not mean the column is unreadable, and the arrow
                    // fallback below may still serve it from the cache.
                    let Some(buffer) = liquid_array.try_eval_predicate(&liquid_expr, selection)
                    else {
                        combined_buffer = None;
                        break;
                    };

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
        // The row count has to be carried explicitly: a column-free conjunct
        // (`NULL`, `false`) projects no arrays, and an array-less batch would
        // otherwise claim zero rows.
        let options = RecordBatchOptions::new().with_row_count(Some(selection.count_set_bits()));
        Some(
            RecordBatch::try_new_with_options(schema, arrays, &options)
                .and_then(|batch| predicate.evaluate(batch)),
        )
    }
}

pub(crate) type CachedRowGroupRef = Arc<CachedRowGroup>;

/// A file in the cache.
#[derive(Debug)]
pub struct CachedFile {
    cache_store: Arc<LiquidCache>,
    /// Held, not copied: the id stays allocated for as long as anything can
    /// still compute a cache key from it.
    file_id: Arc<FileId>,
    file_schema: SchemaRef,
    squeeze_hints: Arc<ColumnSqueezeHints>,
}

impl CachedFile {
    fn new(
        cache_store: Arc<LiquidCache>,
        file_id: Arc<FileId>,
        file_schema: SchemaRef,
        squeeze_hints: Arc<ColumnSqueezeHints>,
    ) -> Self {
        Self {
            cache_store,
            file_id,
            file_schema,
            squeeze_hints,
        }
    }

    /// Create a row group handle scoped to the current query context.
    pub fn create_row_group(
        &self,
        row_group_id: u64,
        predicate_column_ids: Vec<usize>,
    ) -> CachedRowGroupRef {
        let columns: Vec<CachedColumnSpec> = self
            .file_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                let is_predicate_column = predicate_column_ids.contains(&idx);
                let expression = self.squeeze_hints.get(field.name()).cloned();
                (
                    idx as u64,
                    Arc::clone(field),
                    expression,
                    is_predicate_column,
                )
            })
            .collect();

        Arc::new(CachedRowGroup::new(
            self.cache_store.clone(),
            row_group_id,
            Arc::clone(&self.file_id),
            &columns,
        ))
    }

    /// The leased id this file's cache keys are built from.
    #[cfg(test)]
    pub(crate) fn file_id(&self) -> u64 {
        self.file_id.get()
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
    /// Leases the file ids that name cached data. Ids come back when nothing
    /// is reading the file any more, so the number in use tracks what is being
    /// read rather than everything ever read — see [`file_id`].
    file_ids: Arc<FileIdPool>,

    cache_store: Arc<LiquidCache>,
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
        squeeze_policy: Box<dyn SqueezePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
    ) -> Self {
        Self::new_with_squeeze_victim_concurrency(
            batch_size,
            max_memory_bytes,
            max_disk_bytes,
            store,
            cache_policy,
            squeeze_policy,
            hydration_policy,
            !cfg!(test),
        )
        .await
    }

    /// Create a new cache for parquet files with explicit victim squeeze concurrency.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_squeeze_victim_concurrency(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        store: t4::Store,
        cache_policy: Box<dyn CachePolicy>,
        squeeze_policy: Box<dyn SqueezePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        squeeze_victims_concurrently: bool,
    ) -> Self {
        assert!(batch_size.is_power_of_two());
        let metadata = Arc::new(ParquetCacheMetadata::new());
        let cache_storage = LiquidCacheBuilder::new()
            .with_batch_size(batch_size)
            .with_max_memory_bytes(max_memory_bytes)
            .with_max_disk_bytes(max_disk_bytes)
            .with_squeeze_policy(squeeze_policy)
            .with_cache_policy(cache_policy)
            .with_hydration_policy(hydration_policy)
            .with_metadata(metadata)
            .with_store(store)
            .with_squeeze_victims_concurrently(squeeze_victims_concurrently)
            .build()
            .await;

        LiquidCacheParquet {
            file_ids: FileIdPool::new(),
            cache_store: cache_storage,
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(
        &self,
        file_path: String,
        full_file_schema: SchemaRef,
    ) -> CachedFileRef {
        self.register_or_get_file_with_hints(file_path, full_file_schema, Arc::default())
    }

    /// Register a file in the cache, attaching typed squeeze hints derived from
    /// the query plan (keyed by file-schema column name).
    pub fn register_or_get_file_with_hints(
        &self,
        file_path: String,
        full_file_schema: SchemaRef,
        squeeze_hints: Arc<ColumnSqueezeHints>,
    ) -> CachedFileRef {
        Arc::new(CachedFile::new(
            self.cache_store.clone(),
            self.file_ids.acquire(&file_path),
            full_file_schema,
            squeeze_hints,
        ))
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

    /// How many file ids are currently leased.
    ///
    /// This tracks the files being read, not the files ever read. It is the
    /// number that has to stay under the cache key's 16-bit file field, so it
    /// is worth watching: rising without bound means leases are being held by
    /// something that should have let go.
    pub fn leased_file_ids(&self) -> usize {
        self.file_ids.live_count()
    }

    /// How many ids have been handed out that do not fit the cache key's file
    /// field.
    ///
    /// Expected to stay at zero. Above zero, distinct files are computing the
    /// same keys — served correctly, because each entry records which file it
    /// came from, but unable to share the cache.
    pub fn file_ids_over_key_width(&self) -> u64 {
        self.file_ids.over_key_width()
    }

    /// How many cache lookups or writes found a key held by another file.
    ///
    /// The consequence of the counter above, and the one that proves the
    /// aliasing is being caught rather than served.
    pub fn identity_mismatches(&self) -> u64 {
        self.cache_store.stats().identity_mismatches
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
    pub async unsafe fn reset(&self) {
        self.file_ids.reset();
        self.cache_store.reset().await;
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
    use arrow::array::{Array, ArrayRef, Int32Array};
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::expressions::Column;
    use liquid_cache::cache::AlwaysHydrate;
    use liquid_cache::cache::squeeze_policies::TranscodeSqueezeEvict;
    use liquid_cache::cache_policies::LiquidPolicy;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use std::sync::Arc;

    async fn setup_cache(batch_size: usize, schema: SchemaRef) -> CachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let file = cache.register_or_get_file("test".to_string(), schema);
        file.create_row_group(0, vec![])
    }

    /// Recycling a file id must not recycle a file's *name*.
    ///
    /// The id is narrow and reused so the key space cannot run out. If the
    /// identity recorded against each entry were that same id, the next file
    /// to inherit it would be indistinguishable from the one that gave it
    /// back, and would read the entries it left behind — reintroducing the
    /// aliasing the identity exists to catch, at every lease boundary rather
    /// than only past 65,536 files.
    #[tokio::test]
    async fn a_file_inheriting_a_recycled_id_does_not_read_its_predecessors_data() {
        let batch_size = 8;
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;

        let batch_id = BatchID::from_row_id(0, batch_size);
        let filter = BooleanBuffer::new_set(batch_size);
        let first_data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));

        let first_id = {
            let first =
                cache.register_or_get_file("first.parquet".to_string(), Arc::clone(&schema));
            let column = first.create_row_group(0, vec![]).get_column(0).unwrap();
            column
                .insert(batch_id, Arc::clone(&first_data))
                .await
                .unwrap();
            first.file_id()
        }; // lease dropped here, so the id goes back to the pool

        let second = cache.register_or_get_file("second.parquet".to_string(), schema);
        assert_eq!(
            second.file_id(),
            first_id,
            "the id must actually be recycled, or this test proves nothing"
        );

        let column = second.create_row_group(0, vec![]).get_column(0).unwrap();
        assert!(!column.is_cached(batch_id));
        assert!(
            column
                .get_arrow_array_with_filter(batch_id, &filter)
                .await
                .is_none(),
            "inheriting an id must not inherit the entries keyed from it"
        );

        // It must also be able to cache. The predecessor's entries are keyed
        // where this file's belong and nobody can read them any more, so they
        // give way — otherwise a cache under its budget, where nothing is ever
        // evicted, would leave this file permanently uncacheable.
        let second_data: ArrayRef = Arc::new(Int32Array::from(vec![9, 9, 9, 9, 9, 9, 9, 9]));
        column
            .insert(batch_id, Arc::clone(&second_data))
            .await
            .expect("the inheriting file must be able to cache");
        let got = column
            .get_arrow_array_with_filter(batch_id, &filter)
            .await
            .expect("the new owner reads back its own rows");
        assert_eq!(got.as_ref(), second_data.as_ref());
    }

    /// What part of the fix is actually for: a process that reads far more
    /// files than it holds open at once must not exhaust the key's 16-bit file
    /// field. Before ids were leased this counter only ever climbed, so a
    /// long-lived instance wrapped it purely by having *seen* enough files.
    #[tokio::test]
    async fn reading_files_one_after_another_does_not_consume_the_id_space() {
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
        let cache = LiquidCacheParquet::new(
            8,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;

        // Well past the ceiling in total, but only ever one open at a time.
        for i in 0..(u16::MAX as usize + 1_000) {
            let file = cache.register_or_get_file(format!("scan-{i}.parquet"), Arc::clone(&schema));
            assert_eq!(
                file.file_id(),
                0,
                "each file should reuse the id the previous one gave back"
            );
        }

        // And a file opened now still fits the key field.
        let after = cache.register_or_get_file("after.parquet".to_string(), schema);
        assert!(after.file_id() <= u16::MAX as u64);
    }

    /// The bug in its real shape, walked through the actual registration path.
    ///
    /// `ColumnAccessPath` narrows the file id to 16 bits, so the 65,537th
    /// distinct file a process registers is keyed identically to the first.
    /// Before entries recorded their identity, the newcomer read the
    /// incumbent's data — a panic when the column types differed, silently
    /// wrong rows when they matched.
    #[tokio::test]
    async fn a_file_past_the_key_ceiling_does_not_read_the_first_file_s_data() {
        let batch_size = 8;
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeSqueezeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;

        let batch_id = BatchID::from_row_id(0, batch_size);
        let filter = BooleanBuffer::new_set(batch_size);

        // File id 0, with data in the cache.
        let first = cache.register_or_get_file("first.parquet".to_string(), Arc::clone(&schema));
        let first_column = first.create_row_group(0, vec![]).get_column(0).unwrap();
        let first_data: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        first_column
            .insert(batch_id, Arc::clone(&first_data))
            .await
            .unwrap();

        // Burn the rest of the 16-bit id space. The handles are held: ids are
        // leased, so files that are opened and closed hand their id straight
        // back and the ceiling is only reachable with this many files open at
        // once.
        let _fillers: Vec<_> = (1..=u16::MAX as usize)
            .map(|i| cache.register_or_get_file(format!("filler-{i}.parquet"), Arc::clone(&schema)))
            .collect();

        // Id 65536, which narrows to 0.
        let wrapped = cache.register_or_get_file("wrapped.parquet".to_string(), schema);
        let wrapped_column = wrapped.create_row_group(0, vec![]).get_column(0).unwrap();

        assert_eq!(
            usize::from(wrapped_column.entry_id(batch_id)),
            usize::from(first_column.entry_id(batch_id)),
            "the packed keys must actually collide, or this test proves nothing"
        );

        // The newcomer must not be handed the incumbent's rows.
        assert!(!wrapped_column.is_cached(batch_id));
        assert!(
            wrapped_column
                .get_arrow_array_with_filter(batch_id, &filter)
                .await
                .is_none(),
            "a colliding key must read as a miss, not as the other file's data"
        );

        // And the incumbent still reads its own.
        let got = first_column
            .get_arrow_array_with_filter(batch_id, &filter)
            .await
            .expect("the owner's entry is still there");
        assert_eq!(got.as_ref(), first_data.as_ref());
    }

    /// Issue #19: `NOT (s = s)` simplifies to `s IS NULL AND NULL`, so a conjunct
    /// that reads no column reaches the row filter. It has to survive candidate
    /// building and then evaluate against the selection's row count — an
    /// array-less batch would otherwise report zero rows and hand back a mask of
    /// the wrong length, which silently widens the filter.
    #[tokio::test]
    async fn evaluate_column_free_conjunct() {
        let batch_size = 8;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let row_group = setup_cache(batch_size, schema.clone()).await;

        let array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let batch_id = BatchID::from_row_id(0, batch_size);
        let column = row_group.get_column(0).unwrap();
        assert!(column.insert(batch_id, array.clone()).await.is_ok());

        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        let expr: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Boolean(None)));
        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder
            .build(metadata.metadata())
            .unwrap()
            .expect("a column-free conjunct must still produce a candidate");
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();
        assert!(predicate.predicate_column_ids().is_empty());

        // Four of the eight rows are selected, so the mask must be four long.
        let selection = BooleanBuffer::collect_bool(batch_size, |i| i % 2 == 0);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(result.len(), selection.count_set_bits());
        assert_eq!(result.true_count(), 0);
        assert_eq!(result.null_count(), result.len());
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
}
