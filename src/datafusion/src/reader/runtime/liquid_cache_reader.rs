use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::prep_null_mask_filter;
use arrow::record_batch::RecordBatchOptions;
use arrow_schema::{ArrowError, SchemaRef};
use datafusion::physical_plan::metrics::Count;
use futures::{Stream, StreamExt, TryStreamExt, future::BoxFuture, stream::BoxStream};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector,
};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;

use crate::cache::{BatchID, CachedRowGroupRef, InsertArrowArrayError};
use crate::reader::plantime::{LiquidFileReaderFactory, LiquidRowFilter};
use crate::reader::runtime::utils::take_next_batch;
use crate::utils::{boolean_buffer_and_then, row_selector_to_boolean_buffer};

pub(crate) struct LiquidCacheReader {
    state: ReaderState,
    row_filter: Option<LiquidRowFilter>,
}

enum ReaderState {
    Ready(Box<LiquidCacheReaderInner>),
    Processing(
        BoxFuture<
            'static,
            (
                LiquidCacheReaderInner,
                Option<LiquidRowFilter>,
                ProcessResult,
            ),
        >,
    ),
    Finished,
}

enum ProcessResult {
    Emit(Result<RecordBatch, ArrowError>),
    Skip,
}

struct LiquidCacheReaderInner {
    cached_row_group: CachedRowGroupRef,
    current_batch_id: BatchID,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    batch_size: usize,
    projection_columns: Vec<usize>,
    parquet_fallback: ParquetFallback,
    last_pull: Option<(BatchID, RecordBatch)>,
}

pub(crate) struct LiquidCacheReaderConfig {
    pub(crate) selection: RowSelection,
    pub(crate) row_filter: Option<LiquidRowFilter>,
    pub(crate) cached_row_group: CachedRowGroupRef,
    pub(crate) projection_columns: Vec<usize>,
    pub(crate) schema: SchemaRef,
    pub(crate) parquet_fallback: ParquetFallbackConfig,
}

#[derive(Clone)]
pub(crate) struct ParquetFallbackConfig {
    pub(crate) row_group_idx: usize,
    pub(crate) metadata: Arc<ParquetMetaData>,
    pub(crate) reader_factory: Arc<LiquidFileReaderFactory>,
    pub(crate) cache_projection: ProjectionMask,
    pub(crate) cache_column_ids: Vec<usize>,
    pub(crate) cache_batch_size: usize,
    pub(crate) row_count: usize,
    /// Projected columns read for selected rows only and never cached; see
    /// [`ParquetFallback::fetch_selected`]. Empty in the normal mode.
    pub(crate) passthrough_projection: ProjectionMask,
    pub(crate) passthrough_column_ids: Vec<usize>,
    pub(crate) passthrough_batches: Option<Count>,
}

pub(crate) struct ParquetFallback {
    row_group_idx: usize,
    metadata: Arc<ParquetMetaData>,
    reader_factory: Arc<LiquidFileReaderFactory>,
    cache_projection: ProjectionMask,
    cache_column_ids: Vec<usize>,
    cache_batch_size: usize,
    row_count: usize,
    passthrough_projection: ProjectionMask,
    passthrough_column_ids: Vec<usize>,
    passthrough_batches: Option<Count>,
    /// Batches the caller's selection has rows in. `None` reads to the end of
    /// the row group on a miss.
    touched: Option<TouchedBatches>,
    stream: Option<BoxStream<'static, Result<RecordBatch, ParquetError>>>,
    next_batch_id: BatchID,
    /// First batch past what `stream` covers.
    stream_end: usize,
}

/// The cache batches of one row group that a row selection has any row in.
///
/// A fallback read fetches every selected page of its row group up front, so
/// what it selects is what it reads. Bounding each read to one run of
/// adjacent touched batches keeps a sparse selection's misses to its own
/// pages, while a dense selection is still one stream per row group.
#[derive(Clone, Debug)]
pub(crate) struct TouchedBatches {
    touched: Vec<bool>,
}

impl TouchedBatches {
    pub(crate) fn new(selection: &RowSelection, batch_size: usize) -> Self {
        let mut touched = Vec::new();
        let mut row = 0;
        for selector in selection.iter() {
            let end = row + selector.row_count;
            if !selector.skip && selector.row_count > 0 {
                let last = (end - 1) / batch_size;
                if touched.len() <= last {
                    touched.resize(last + 1, false);
                }
                touched[row / batch_size..=last].fill(true);
            }
            row = end;
        }
        Self { touched }
    }

    /// Rows of a `row_count`-row row group that fall in touched batches: what
    /// a cache fill of this selection reads, since misses fill whole batches.
    pub(crate) fn touched_rows(&self, batch_size: usize, row_count: usize) -> usize {
        self.touched
            .iter()
            .enumerate()
            .filter(|(_, touched)| **touched)
            .map(|(batch, _)| {
                let start = (batch * batch_size).min(row_count);
                ((batch + 1) * batch_size).min(row_count) - start
            })
            .sum()
    }

    /// One past the last batch of the touched run starting at `start`; always
    /// covers `start` itself, touched or not.
    fn run_end(&self, start: usize) -> usize {
        let mut end = start + 1;
        while self.touched.get(end).copied().unwrap_or(false) {
            end += 1;
        }
        end
    }
}

impl LiquidCacheReader {
    pub(crate) fn new(config: LiquidCacheReaderConfig) -> Self {
        // The selection is walked in cache-sized windows, not session-sized
        // ones. `current_batch_id` indexes stored chunks, and both the cache
        // read and the parquet fallback turn that id back into rows with the
        // cache batch size; windowing at `datafusion.execution.batch_size`
        // would address different rows than the id names whenever a caller
        // sets it to anything other than the cache's own size. The caller is
        // unaffected either way: `DataSourceExec` re-splits every source
        // stream to the session batch size.
        let batch_size = config.cached_row_group.batch_size();
        let touched = TouchedBatches::new(&config.selection, batch_size);
        let inner = LiquidCacheReaderInner::new(
            batch_size,
            config.selection,
            config.cached_row_group,
            config.projection_columns,
            Arc::clone(&config.schema),
            ParquetFallback::new(config.parquet_fallback).with_touched_batches(touched),
        );
        Self {
            state: ReaderState::Ready(Box::new(inner)),
            row_filter: config.row_filter,
        }
    }
}

impl Stream for LiquidCacheReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let state = std::mem::replace(&mut self.state, ReaderState::Finished);

            match state {
                ReaderState::Processing(mut fut) => match fut.as_mut().poll(cx) {
                    Poll::Pending => {
                        self.state = ReaderState::Processing(fut);
                        return Poll::Pending;
                    }
                    Poll::Ready((inner, row_filter, result)) => {
                        self.row_filter = row_filter;
                        self.state = ReaderState::Ready(Box::new(inner));
                        match result {
                            ProcessResult::Emit(item) => return Poll::Ready(Some(item)),
                            ProcessResult::Skip => continue,
                        }
                    }
                },
                ReaderState::Ready(mut inner) => {
                    match take_next_batch(&mut inner.selection, inner.batch_size) {
                        Some(selection) => {
                            let inner = *inner;
                            let future = inner.next_batch(self.row_filter.take(), selection);
                            self.state = ReaderState::Processing(future);
                            continue;
                        }
                        None => {
                            self.state = ReaderState::Finished;
                            return Poll::Ready(None);
                        }
                    }
                }
                ReaderState::Finished => {
                    self.state = ReaderState::Finished;
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl ParquetFallback {
    pub(crate) fn new(config: ParquetFallbackConfig) -> Self {
        Self {
            row_group_idx: config.row_group_idx,
            metadata: config.metadata,
            reader_factory: config.reader_factory,
            cache_projection: config.cache_projection,
            cache_column_ids: config.cache_column_ids,
            cache_batch_size: config.cache_batch_size,
            row_count: config.row_count,
            passthrough_projection: config.passthrough_projection,
            passthrough_column_ids: config.passthrough_column_ids,
            passthrough_batches: config.passthrough_batches,
            touched: None,
            stream: None,
            next_batch_id: BatchID::from_raw(0),
            stream_end: 0,
        }
    }

    /// Limit miss reads to the batches `touched` names (see [`TouchedBatches`]).
    pub(crate) fn with_touched_batches(mut self, touched: TouchedBatches) -> Self {
        self.restrict_to(touched);
        self
    }

    /// [`Self::with_touched_batches`] in place. Drops any open stream, which
    /// may cover batches outside `touched`.
    pub(crate) fn restrict_to(&mut self, touched: TouchedBatches) {
        self.touched = Some(touched);
        self.stream = None;
    }

    pub(crate) async fn fetch_batch(
        &mut self,
        batch_id: BatchID,
    ) -> Result<RecordBatch, ParquetError> {
        if self.stream.is_none()
            || batch_id != self.next_batch_id
            || usize::from(*batch_id) >= self.stream_end
        {
            self.rebuild_stream(batch_id)?;
        }

        let stream = self.stream.as_mut().expect("fallback stream is present");
        let record_batch = stream.next().await.transpose()?.ok_or_else(|| {
            ParquetError::General(format!(
                "parquet fallback ended before batch {}",
                *batch_id as usize
            ))
        })?;

        self.next_batch_id = batch_id;
        self.next_batch_id.inc();
        Ok(record_batch)
    }

    fn passthrough_position(&self, file_column_id: usize) -> Option<usize> {
        self.passthrough_column_ids
            .iter()
            .position(|id| *id == file_column_id)
    }

    /// Read the passthrough columns for only the rows `selection` keeps in
    /// `batch_id`, without touching the cache. The row selection lets the
    /// parquet reader fetch just the pages holding those rows (the metadata
    /// carries the page index), which is what keeps a wide, selective scan
    /// cheap: the predicate ran against cached columns, and the wide columns are
    /// read only where it matched.
    pub(crate) async fn fetch_selected(
        &mut self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
    ) -> Result<RecordBatch, ParquetError> {
        let start = usize::from(*batch_id) * self.cache_batch_size;
        let end = start + selection.len();
        let mut selectors = Vec::new();
        if start > 0 {
            selectors.push(RowSelector::skip(start));
        }
        let batch_selection =
            RowSelection::from_filters(&[BooleanArray::new(selection.clone(), None)]);
        selectors.extend(Vec::<RowSelector>::from(batch_selection));
        if end < self.row_count {
            selectors.push(RowSelector::skip(self.row_count - end));
        }

        let reader_metadata =
            ArrowReaderMetadata::try_new(Arc::clone(&self.metadata), ArrowReaderOptions::new())?;
        let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
            self.reader_factory.create()?,
            reader_metadata,
        )
        .with_projection(self.passthrough_projection.clone())
        .with_row_groups(vec![self.row_group_idx])
        .with_batch_size(self.cache_batch_size)
        .with_row_selection(RowSelection::from(selectors));
        let batches: Vec<RecordBatch> = builder.build()?.try_collect().await?;
        if let Some(count) = &self.passthrough_batches {
            count.add(1);
        }
        // `builder.schema()` is the whole file's; take the projected one from
        // the output. Callers only ask for a non-empty selection.
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .ok_or_else(|| ParquetError::General("passthrough read returned no rows".into()))?;
        Ok(arrow::compute::concat_batches(&schema, &batches)?)
    }

    fn rebuild_stream(&mut self, batch_id: BatchID) -> Result<(), ParquetError> {
        let reader_metadata =
            ArrowReaderMetadata::try_new(Arc::clone(&self.metadata), ArrowReaderOptions::new())?;
        let start = usize::from(*batch_id);
        let end = self
            .touched
            .as_ref()
            .map_or(usize::MAX, |touched| touched.run_end(start));
        let row_selection = build_row_selection(start..end, self.cache_batch_size, self.row_count);

        let stream = ParquetRecordBatchStreamBuilder::new_with_metadata(
            self.reader_factory.create()?,
            reader_metadata,
        )
        .with_projection(self.cache_projection.clone())
        .with_row_groups(vec![self.row_group_idx])
        .with_batch_size(self.cache_batch_size)
        .with_row_selection(row_selection)
        .build()?
        .boxed();

        self.stream = Some(stream);
        self.next_batch_id = batch_id;
        self.stream_end = end;
        Ok(())
    }
}

/// Select the rows of cache batches `batches` (clamped to the row group).
fn build_row_selection(
    batches: std::ops::Range<usize>,
    batch_size: usize,
    row_count: usize,
) -> RowSelection {
    let start = batches.start.saturating_mul(batch_size).min(row_count);
    let end = batches.end.saturating_mul(batch_size).min(row_count);
    let mut selectors = Vec::new();
    if start > 0 {
        selectors.push(RowSelector::skip(start));
    }
    if end > start {
        selectors.push(RowSelector::select(end - start));
    }
    if row_count > end {
        selectors.push(RowSelector::skip(row_count - end));
    }
    RowSelection::from(selectors)
}

impl LiquidCacheReaderInner {
    fn new(
        batch_size: usize,
        selection: RowSelection,
        cached_row_group: CachedRowGroupRef,
        projection_columns: Vec<usize>,
        schema: SchemaRef,
        parquet_fallback: ParquetFallback,
    ) -> Self {
        Self {
            cached_row_group,
            current_batch_id: BatchID::from_raw(0),
            selection: selection.into(),
            schema,
            batch_size,
            projection_columns,
            parquet_fallback,
            last_pull: None,
        }
    }

    fn next_batch(
        self,
        row_filter: Option<LiquidRowFilter>,
        selection: Vec<RowSelector>,
    ) -> BoxFuture<'static, (Self, Option<LiquidRowFilter>, ProcessResult)> {
        Box::pin(async move {
            let mut inner = self;
            let mut row_filter = row_filter;
            inner.last_pull = None;

            let result = match inner
                .build_predicate_filter(&mut row_filter, selection)
                .await
            {
                Ok(selection) => match inner.read_from_cache(&selection).await {
                    Ok(Some(batch)) => {
                        inner.current_batch_id.inc();
                        ProcessResult::Emit(Ok(batch))
                    }
                    Ok(None) => {
                        inner.current_batch_id.inc();
                        ProcessResult::Skip
                    }
                    Err(e) => ProcessResult::Emit(Err(e)),
                },
                Err(e) => ProcessResult::Emit(Err(e)),
            };

            (inner, row_filter, result)
        })
    }

    #[fastrace::trace]
    async fn build_predicate_filter(
        &mut self,
        row_filter: &mut Option<LiquidRowFilter>,
        selection: Vec<RowSelector>,
    ) -> Result<BooleanBuffer, ArrowError> {
        let input_selection = row_selector_to_boolean_buffer(&selection);

        if let Some(snapshot_selection) = self
            .cached_row_group
            .snapshot_selection(self.current_batch_id)
        {
            // Prefetch stores the filtered selection batch-absolute (already
            // ANDed with this same input), not relative to the input's set
            // bits, so combine positionally. `boolean_buffer_and_then` only
            // agreed when the input selected every row of the batch.
            return Ok(&input_selection & &snapshot_selection);
        }

        let Some(filter) = row_filter.as_mut() else {
            return Ok(input_selection);
        };

        if let Some(selection) = apply_predicates(
            &self.cached_row_group,
            self.current_batch_id,
            input_selection.clone(),
            filter,
        )
        .await?
        {
            return Ok(selection);
        }

        self.read_parquet_batch_and_fill_cache(self.current_batch_id)
            .await?;
        apply_predicates(
            &self.cached_row_group,
            self.current_batch_id,
            input_selection,
            filter,
        )
        .await?
        .ok_or_else(|| {
            ArrowError::ComputeError("predicate unavailable after materialization".to_string())
        })
    }

    #[fastrace::trace]
    async fn read_from_cache(
        &mut self,
        selection: &BooleanBuffer,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        let selected_rows = selection.count_set_bits();
        if selected_rows == 0 {
            return Ok(None);
        }

        if self.projection_columns.is_empty() {
            let options = RecordBatchOptions::new().with_row_count(Some(selected_rows));
            let batch =
                RecordBatch::try_new_with_options(self.schema.clone(), Vec::new(), &options)
                    .unwrap();
            return Ok(Some(batch));
        }

        let mut arrays = Vec::with_capacity(self.projection_columns.len());
        // Passthrough columns for this batch, read on the first cache miss.
        let mut passthrough: Option<RecordBatch> = None;
        for column_idx in self.projection_columns.clone() {
            let column = self
                .cached_row_group
                .get_column(column_idx as u64)
                .ok_or_else(|| {
                    ArrowError::ComputeError(format!(
                        "column {column_idx} not present in liquid cache"
                    ))
                })?;

            let array = column
                .get_arrow_array_with_filter(self.current_batch_id, selection)
                .await;

            let passthrough_position = self.parquet_fallback.passthrough_position(column_idx);
            let array = match (array, passthrough_position) {
                (Some(array), _) => array,
                (None, Some(position)) => {
                    if passthrough.is_none() {
                        passthrough = Some(
                            self.parquet_fallback
                                .fetch_selected(self.current_batch_id, selection)
                                .await
                                .map_err(|e| {
                                    ArrowError::ComputeError(format!(
                                        "parquet passthrough read failed: {e}"
                                    ))
                                })?,
                        );
                    }
                    let batch = passthrough
                        .as_ref()
                        .expect("passthrough batch was just read");
                    Arc::clone(batch.column(position))
                }
                (None, None) => {
                    let record_batch = self
                        .read_parquet_batch_and_fill_cache(self.current_batch_id)
                        .await?;
                    let array = self.parquet_array(&record_batch, column_idx)?;
                    filter_array(array, selection)?
                }
            };

            arrays.push(array);
        }

        Ok(Some(
            RecordBatch::try_new(self.schema.clone(), arrays).unwrap(),
        ))
    }

    async fn read_parquet_batch_and_fill_cache(
        &mut self,
        batch_id: BatchID,
    ) -> Result<RecordBatch, ArrowError> {
        if let Some((pulled_batch_id, record_batch)) = &self.last_pull
            && *pulled_batch_id == batch_id
        {
            return Ok(record_batch.clone());
        }

        let record_batch = self
            .parquet_fallback
            .fetch_batch(batch_id)
            .await
            .map_err(|e| ArrowError::ComputeError(format!("parquet fallback read failed: {e}")))?;

        for (col_idx, file_column_id) in self
            .parquet_fallback
            .cache_column_ids
            .iter()
            .copied()
            .enumerate()
        {
            let column = self
                .cached_row_group
                .get_column(file_column_id as u64)
                .ok_or_else(|| {
                    ArrowError::ComputeError(format!(
                        "column {file_column_id} not present in liquid cache"
                    ))
                })?;
            let array = Arc::clone(record_batch.column(col_idx));

            match column.insert(batch_id, Arc::clone(&array)).await {
                Ok(()) | Err(InsertArrowArrayError::AlreadyCached) => {}
                Err(InsertArrowArrayError::CacheFull) => {
                    column.insert_snapshot(batch_id, array);
                }
            }
        }

        self.last_pull = Some((batch_id, record_batch.clone()));
        Ok(record_batch)
    }

    fn parquet_array(
        &self,
        record_batch: &RecordBatch,
        file_column_id: usize,
    ) -> Result<ArrayRef, ArrowError> {
        let position = self
            .parquet_fallback
            .cache_column_ids
            .iter()
            .position(|column_id| *column_id == file_column_id)
            .ok_or_else(|| {
                ArrowError::ComputeError(format!(
                    "column {file_column_id} not present in parquet fallback projection"
                ))
            })?;

        Ok(Arc::clone(record_batch.column(position)))
    }
}

pub(crate) async fn apply_predicates(
    row_group: &CachedRowGroupRef,
    batch_id: BatchID,
    mut input_selection: BooleanBuffer,
    filter: &mut LiquidRowFilter,
) -> Result<Option<BooleanBuffer>, ArrowError> {
    for predicate in filter.predicates_mut() {
        if input_selection.count_set_bits() == 0 {
            break;
        }

        let Some(boolean_array) = row_group
            .evaluate_selection_with_predicate(batch_id, &input_selection, predicate)
            .await
        else {
            return Ok(None);
        };
        let boolean_array = boolean_array?;
        let boolean_mask = if boolean_array.null_count() == 0 {
            boolean_array.into_parts().0
        } else {
            prep_null_mask_filter(&boolean_array).into_parts().0
        };
        input_selection = boolean_buffer_and_then(&input_selection, &boolean_mask);
    }

    Ok(Some(input_selection))
}

fn filter_array(array: ArrayRef, selection: &BooleanBuffer) -> Result<ArrayRef, ArrowError> {
    let selection_array = BooleanArray::new(selection.clone(), None);
    arrow::compute::filter(array.as_ref(), &selection_array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache::LiquidCacheParquet,
        reader::plantime::LiquidFileReaderFactory,
        reader::{FilterCandidateBuilder, LiquidPredicate, LiquidRowFilter},
    };
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::record_batch::RecordBatch;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::{
        listing::PartitionedFile, physical_plan::parquet::DefaultParquetFileReaderFactory,
    };
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::{
        logical_expr::Operator,
        physical_expr::PhysicalExpr,
        physical_expr::expressions::{BinaryExpr, Column, Literal},
        scalar::ScalarValue,
    };
    use futures::{StreamExt, pin_mut};
    use liquid_cache::cache::{AlwaysHydrate, Evict};
    use liquid_cache::cache_policies::LiquidPolicy;
    use object_store::local::LocalFileSystem;
    use parquet::arrow::{
        ArrowWriter, ProjectionMask,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector},
    };
    use parquet::file::metadata::PageIndexPolicy;
    use std::fs::File;
    use std::sync::Arc;

    struct TestRowGroup {
        row_group: CachedRowGroupRef,
        schema: SchemaRef,
        fallback: ParquetFallbackConfig,
        _tmp_dir: tempfile::TempDir,
    }

    struct ReaderRequest {
        selection: RowSelection,
        row_filter: Option<LiquidRowFilter>,
        projection_columns: Vec<usize>,
        schema: SchemaRef,
    }

    impl TestRowGroup {
        fn reader(&self, request: ReaderRequest) -> LiquidCacheReader {
            LiquidCacheReader::new(LiquidCacheReaderConfig {
                selection: request.selection,
                row_filter: request.row_filter,
                cached_row_group: Arc::clone(&self.row_group),
                projection_columns: request.projection_columns,
                schema: request.schema,
                parquet_fallback: self.fallback.clone(),
            })
        }
    }

    async fn make_row_group(batch_size: usize, batches: &[Vec<i32>]) -> TestRowGroup {
        let tmp_dir = tempfile::tempdir().unwrap();
        let field = Arc::new(Field::new("col0", DataType::Int32, false));
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let parquet_path = tmp_dir.path().join("data.parquet");
        let file = File::create(&parquet_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        for values in batches {
            let array: ArrayRef = Arc::new(Int32Array::from(values.clone()));
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();
            writer.write(&batch).unwrap();
        }
        writer.close().unwrap();

        let metadata_file = File::open(&parquet_path).unwrap();
        let reader_metadata =
            ArrowReaderMetadata::load(&metadata_file, ArrowReaderOptions::new()).unwrap();
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let partitioned_file = PartitionedFile::new(
            "data.parquet",
            std::fs::metadata(&parquet_path).unwrap().len(),
        );
        let metrics = ExecutionPlanMetricsSet::new();
        let reader_factory = Arc::new(LiquidFileReaderFactory {
            factory: Arc::new(DefaultParquetFileReaderFactory::new(object_store)),
            partition_index: 0,
            partitioned_file,
            metadata_size_hint: None,
            metrics,
        });
        let projection = ProjectionMask::roots(
            reader_metadata.metadata().file_metadata().schema_descr(),
            [0],
        );

        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(Evict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let file = cache.register_or_get_file(
            crate::cache::ParquetFileIdentity::new(
                datafusion::execution::object_store::ObjectStoreUrl::parse("test-runtime:///")
                    .unwrap(),
                "test".to_string(),
            ),
            schema.clone(),
        );
        let row_group = file.create_row_group(0, vec![]);
        let column = row_group.get_column(0).unwrap();

        for (idx, values) in batches.iter().enumerate() {
            let array: ArrayRef = Arc::new(Int32Array::from(values.clone()));
            column
                .insert(BatchID::from_raw(idx as u16), array)
                .await
                .expect("cache insert");
        }

        TestRowGroup {
            row_group,
            schema,
            fallback: ParquetFallbackConfig {
                row_group_idx: 0,
                metadata: Arc::clone(reader_metadata.metadata()),
                reader_factory,
                cache_projection: projection,
                cache_column_ids: vec![0],
                cache_batch_size: batch_size,
                row_count: flatten_batches(batches).len(),
                passthrough_projection: ProjectionMask::none(1),
                passthrough_column_ids: vec![],
                passthrough_batches: None,
            },
            _tmp_dir: tmp_dir,
        }
    }

    /// One row group of `0..rows` with nothing cached, written in pages of
    /// `page_rows` with no dictionary, so a page-index row selection can skip
    /// pages and `bytes_scanned` shows which ones were read.
    async fn make_uncached_row_group(
        cache_batch_size: usize,
        rows: i32,
        page_rows: usize,
    ) -> (TestRowGroup, ExecutionPlanMetricsSet, u64) {
        use parquet::file::properties::WriterProperties;

        let tmp_dir = tempfile::tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col0",
            DataType::Int32,
            false,
        )]));
        let parquet_path = tmp_dir.path().join("data.parquet");
        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(page_rows)
            .set_write_batch_size(page_rows)
            .set_dictionary_enabled(false)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(&parquet_path).unwrap(),
            Arc::clone(&schema),
            Some(props),
        )
        .unwrap();
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..rows));
        writer
            .write(&RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap())
            .unwrap();
        writer.close().unwrap();

        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader_metadata =
            ArrowReaderMetadata::load(&File::open(&parquet_path).unwrap(), options).unwrap();
        let metadata = Arc::clone(reader_metadata.metadata());
        assert_eq!(metadata.num_row_groups(), 1);
        let chunk_bytes = metadata.row_group(0).column(0).compressed_size() as u64;

        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tmp_dir.path()).unwrap());
        let metrics = ExecutionPlanMetricsSet::new();
        let reader_factory = Arc::new(LiquidFileReaderFactory {
            factory: Arc::new(DefaultParquetFileReaderFactory::new(object_store)),
            partition_index: 0,
            partitioned_file: PartitionedFile::new(
                "data.parquet",
                std::fs::metadata(&parquet_path).unwrap().len(),
            ),
            metadata_size_hint: None,
            metrics: metrics.clone(),
        });
        let projection = ProjectionMask::roots(metadata.file_metadata().schema_descr(), [0]);

        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            cache_batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(Evict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let file = cache.register_or_get_file(
            crate::cache::ParquetFileIdentity::new(
                datafusion::execution::object_store::ObjectStoreUrl::parse("test-sparse:///")
                    .unwrap(),
                "sparse".to_string(),
            ),
            Arc::clone(&schema),
        );
        let row_group = file.create_row_group(0, vec![]);
        let test = TestRowGroup {
            row_group,
            schema,
            fallback: ParquetFallbackConfig {
                row_group_idx: 0,
                metadata,
                reader_factory,
                cache_projection: projection,
                cache_column_ids: vec![0],
                cache_batch_size,
                row_count: rows as usize,
                passthrough_projection: ProjectionMask::none(1),
                passthrough_column_ids: vec![],
                passthrough_batches: None,
            },
            _tmp_dir: tmp_dir,
        };
        (test, metrics, chunk_bytes)
    }

    fn bytes_scanned(metrics: &ExecutionPlanMetricsSet) -> u64 {
        metrics
            .clone_inner()
            .sum_by_name("bytes_scanned")
            .map(|v| v.as_usize() as u64)
            .unwrap_or(0)
    }

    /// A selection that keeps one row in each of a few far-apart cache
    /// batches must read only those batches' pages on a cold cache. Each miss
    /// used to rebuild the fallback stream from its batch to the end of the
    /// row group, and the parquet reader fetches every selected page of a row
    /// group up front, so four touched batches read about twice the column.
    #[tokio::test]
    async fn sparse_selection_reads_only_touched_batches() {
        const BATCH: usize = 16;
        const BATCHES: usize = 64;
        let (test, metrics, chunk_bytes) =
            make_uncached_row_group(BATCH, (BATCH * BATCHES) as i32, BATCH).await;
        let wanted = [0usize, 21 * BATCH + 3, 42 * BATCH + 7, 63 * BATCH + 15];
        let mut selectors = Vec::new();
        let mut at = 0;
        for row in wanted {
            if row > at {
                selectors.push(RowSelector::skip(row - at));
            }
            selectors.push(RowSelector::select(1));
            at = row + 1;
        }
        if at < BATCH * BATCHES {
            selectors.push(RowSelector::skip(BATCH * BATCHES - at));
        }

        let reader = test.reader(ReaderRequest {
            selection: RowSelection::from(selectors),
            row_filter: None,
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });
        let got: Vec<i32> = collect_batches(reader)
            .iter()
            .flat_map(as_i32_values)
            .collect();
        assert_eq!(got, wanted.map(|r| r as i32).to_vec());

        // 4 of 64 pages; allow 2x for page headers and the column's first page.
        let read = bytes_scanned(&metrics);
        assert!(
            read * 8 < chunk_bytes,
            "read {read} bytes of a {chunk_bytes}-byte column for 4 of {BATCHES} batches"
        );
        // Every touched batch was filled into the cache, nothing else.
        let column = test.row_group.get_column(0).unwrap();
        for batch in 0..BATCHES {
            let touched = wanted.iter().any(|row| row / BATCH == batch);
            assert_eq!(
                column.contains(BatchID::from_raw(batch as u16)),
                touched,
                "batch {batch}"
            );
        }
    }

    /// A dense selection still streams the row group once.
    #[tokio::test]
    async fn dense_selection_reads_the_column_once() {
        const BATCH: usize = 16;
        const BATCHES: usize = 64;
        let (test, metrics, chunk_bytes) =
            make_uncached_row_group(BATCH, (BATCH * BATCHES) as i32, BATCH).await;
        let reader = test.reader(ReaderRequest {
            selection: RowSelection::from(vec![RowSelector::select(BATCH * BATCHES)]),
            row_filter: None,
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });
        let got: Vec<i32> = collect_batches(reader)
            .iter()
            .flat_map(as_i32_values)
            .collect();
        assert_eq!(got, (0..(BATCH * BATCHES) as i32).collect::<Vec<_>>());
        assert!(bytes_scanned(&metrics) <= chunk_bytes);
    }

    fn flatten_batches(batches: &[Vec<i32>]) -> Vec<i32> {
        batches.iter().flat_map(|b| b.iter().copied()).collect()
    }

    fn collect_batches(reader: LiquidCacheReader) -> Vec<RecordBatch> {
        futures::executor::block_on(
            reader
                .map(|batch| batch.expect("valid record batch"))
                .collect::<Vec<_>>(),
        )
    }

    fn as_i32_values(batch: &RecordBatch) -> Vec<i32> {
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column");
        array.iter().map(|v| v.expect("non-null")).collect()
    }

    fn build_filter(
        schema: SchemaRef,
        values: &[i32],
        expr: Arc<dyn PhysicalExpr>,
    ) -> LiquidRowFilter {
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array.clone()]).unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new()).unwrap();

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        LiquidRowFilter::new(vec![predicate])
    }

    fn make_gt_filter(schema: SchemaRef, values: &[i32], literal: i32) -> LiquidRowFilter {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(literal)))),
        ));
        build_filter(schema, values, expr)
    }

    #[tokio::test]
    async fn reads_batches_in_order() {
        let batch_size = 2;
        let test = make_row_group(batch_size, &[vec![1, 2], vec![3, 4]]).await;
        let selection = RowSelection::from(vec![RowSelector::select(4)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: None,
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 2);
        assert_eq!(as_i32_values(&batches[0]), vec![1, 2]);
        assert_eq!(as_i32_values(&batches[1]), vec![3, 4]);
    }

    #[tokio::test]
    async fn skips_unselected_batches() {
        let batch_size = 2;
        let test = make_row_group(batch_size, &[vec![1, 2], vec![3, 4]]).await;
        let selection = RowSelection::from(vec![RowSelector::skip(2), RowSelector::select(2)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: None,
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches[0]), vec![3, 4]);
    }

    #[tokio::test]
    async fn empty_projection_emits_schema_only_batches() {
        let batch_size = 2;
        let test = make_row_group(batch_size, &[vec![10, 11]]).await;
        let selection = RowSelection::from(vec![RowSelector::select(2)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: None,
            projection_columns: Vec::new(),
            schema: Arc::new(Schema::new(Vec::<Field>::new())),
        });

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 2);
    }

    #[tokio::test]
    async fn predicate_filters_rows_across_batches() {
        let batches = vec![vec![1, 2], vec![3, 4]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let test = make_row_group(batch_size, &batches).await;
        let filter = make_gt_filter(Arc::clone(&test.schema), &all_values, 2);
        let selection = RowSelection::from(vec![RowSelector::select(4)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: Some(filter),
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches[0]), vec![3, 4]);
    }

    fn make_or_filter(
        schema: SchemaRef,
        values: &[i32],
        gt_literal: i32,
        lt_literal: i32,
    ) -> LiquidRowFilter {
        let cond1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(gt_literal)))),
        ));
        let cond2: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Lt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(lt_literal)))),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(cond1, Operator::Or, cond2));
        build_filter(schema, values, expr)
    }

    #[tokio::test]
    async fn predicate_filters_or_rows() {
        let batches = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let test = make_row_group(batch_size, &batches).await;
        let filter = make_or_filter(Arc::clone(&test.schema), &all_values, 4, 2);
        let selection = RowSelection::from(vec![RowSelector::select(6)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: Some(filter),
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let batches = collect_batches(reader);
        assert_eq!(batches.len(), 2);
        assert_eq!(as_i32_values(&batches[0]), vec![1]);
        assert_eq!(as_i32_values(&batches[1]), vec![5, 6]);
    }

    #[tokio::test]
    async fn predicate_combines_with_selection() {
        let batches = vec![vec![1, 2, 3, 4]];
        let batch_size = 4;
        let all_values = flatten_batches(&batches);
        let test = make_row_group(batch_size, &batches).await;
        let filter = make_gt_filter(Arc::clone(&test.schema), &all_values, 2);
        let selection = RowSelection::from(vec![
            RowSelector::skip(1),
            RowSelector::select(2),
            RowSelector::skip(1),
        ]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: Some(filter),
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let mut batches = collect_batches(reader);
        assert_eq!(batches.len(), 1);
        assert_eq!(as_i32_values(&batches.pop().unwrap()), vec![3]);
    }

    #[tokio::test]
    async fn predicate_can_filter_all_rows() {
        let batches = vec![vec![1, 2]];
        let batch_size = 2;
        let all_values = flatten_batches(&batches);
        let test = make_row_group(batch_size, &batches).await;
        let filter = make_gt_filter(Arc::clone(&test.schema), &all_values, 10);
        let selection = RowSelection::from(vec![RowSelector::select(2)]);

        let reader = test.reader(ReaderRequest {
            selection,
            row_filter: Some(filter),
            projection_columns: vec![0],
            schema: Arc::clone(&test.schema),
        });

        let next_batch = futures::executor::block_on(async {
            pin_mut!(reader);
            reader.next().await
        });

        assert!(next_batch.is_none());
    }
}
