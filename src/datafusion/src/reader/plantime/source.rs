use super::LiquidMorselizer;
use crate::cache::{ColumnLineages, LiquidCacheParquetRef};
use datafusion::{
    common::ScalarValue,
    common::{internal_err, tree_node::TreeNodeRecursion},
    config::{ConfigOptions, TableParquetOptions},
    datasource::{
        physical_plan::{
            FileScanConfig, FileSource, ParquetFileReaderFactory, ParquetSource,
            parquet::{DefaultParquetFileReaderFactory, can_expr_be_pushed_down_with_schemas},
        },
        table_schema::TableSchema,
    },
    error::Result,
    physical_expr::projection::ProjectionExprs,
    physical_expr::utils::conjunction,
    physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr, expressions::Column},
    physical_expr_adapter::DefaultPhysicalExprAdapterFactory,
    physical_plan::{
        DisplayFormatType, PhysicalExpr, SortOrderPushdownResult, apply_expression_roots,
        filter_pushdown::{FilterPushdownPropagation, PushedDown, PushedDownPredicate},
        metrics::ExecutionPlanMetricsSet,
    },
};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::morsel::Morselizer;
use object_store::ObjectStore;
use std::{
    fmt::{self, Formatter},
    sync::Arc,
};

/// The data source for LiquidCache
#[derive(Clone)]
pub struct LiquidParquetSource {
    metrics: ExecutionPlanMetricsSet,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_parquet_options: TableParquetOptions,
    liquid_cache: LiquidCacheParquetRef,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    table_schema: TableSchema,
    parquet_file_reader_factory: Option<Arc<dyn ParquetFileReaderFactory>>,
    span: Option<Arc<fastrace::Span>>,
    lineages: Arc<ColumnLineages>,
    prefetch: bool,
    /// The sort a parent `SortExec` asked this scan to approximate (set by
    /// [`FileSource::try_pushdown_sort`]). Drives [`FileSource::reorder_files`],
    /// which orders the shared work queue so a TopK sees its best files first.
    sort_order_for_reorder: Option<LexOrdering>,
}

impl LiquidParquetSource {
    fn reorder_filters(&self) -> bool {
        self.table_parquet_options.global.reorder_filters
    }

    fn bloom_filter_on_read(&self) -> bool {
        self.table_parquet_options.global.bloom_filter_on_read
    }

    /// Set the span for the LiquidParquetSource
    pub fn with_span(&self, span: fastrace::Span) -> Self {
        Self {
            span: Some(Arc::new(span)),
            ..self.clone()
        }
    }

    /// Set the table schema for the LiquidParquetSource
    pub fn with_table_schema(&self, table_schema: TableSchema) -> Self {
        Self {
            table_schema,
            ..self.clone()
        }
    }

    /// Attach typed lineage expressions (keyed by file-schema column name) derived
    /// from the query plan. These flow to the cache when the file is opened.
    pub fn with_lineages(&self, lineages: Arc<ColumnLineages>) -> Self {
        Self {
            lineages,
            ..self.clone()
        }
    }

    /// Enable or disable row-group prefetching.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// The typed lineage expressions currently attached to this source.
    pub fn lineages(&self) -> &Arc<ColumnLineages> {
        &self.lineages
    }

    /// Set predicate information.
    pub fn with_predicate(mut self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Create a new LiquidParquetSource from a ParquetSource
    pub fn from_parquet_source(source: ParquetSource, liquid_cache: LiquidCacheParquetRef) -> Self {
        let predicate = source.filter();
        let parquet_file_reader_factory = source.parquet_file_reader_factory().cloned();

        let table_schema = source.table_schema().clone();
        let projection = source.projection().cloned().unwrap_or_else(|| {
            let table_schema = table_schema.table_schema();
            ProjectionExprs::from_indices(
                &(0..table_schema.fields().len()).collect::<Vec<_>>(),
                table_schema,
            )
        });
        let mut v = Self {
            table_schema,
            table_parquet_options: source.table_parquet_options().clone(),
            batch_size: Some(liquid_cache.batch_size()),
            liquid_cache,
            projection,
            metrics: source.metrics().clone(),
            parquet_file_reader_factory,
            predicate: None,
            span: None,
            lineages: Arc::default(),
            prefetch: true,
            // A `ParquetSource` keeps its own pushed sort private, so it cannot
            // be carried over here. `LocalModeOptimizer` pushes the sorts down
            // again once the scans are liquid.
            sort_order_for_reorder: None,
        };

        if let Some(predicate) = predicate {
            v = v.with_predicate(predicate);
        }

        v
    }

    /// Get the predicate for the LiquidParquetSource
    pub fn predicate(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }

    /// The sort this scan was asked to approximate, if any.
    pub fn sort_order_for_reorder(&self) -> Option<&LexOrdering> {
        self.sort_order_for_reorder.as_ref()
    }
}

/// Order `files` so the files most likely to hold a TopK's rows come first:
/// by the per-file `min` of each leading plain-column sort key, in the key's
/// direction (largest `min` first for a descending key). Files without that
/// statistic go last.
///
/// This mirrors `reorder_files_by_min_statistics` in
/// `datafusion-datasource-parquet`, which is crate-private. Sharing the same
/// order with DataFusion's own reader keeps a liquid scan's TopK behaviour the
/// same as a plain parquet scan's: the files that tighten the dynamic filter
/// the most are opened first, and every file the filter then rules out is
/// skipped without being opened.
fn reorder_files_by_min_statistics(
    mut files: Vec<PartitionedFile>,
    sort_order: Option<&LexOrdering>,
    table_schema: &arrow_schema::Schema,
) -> Vec<PartitionedFile> {
    let Some(sort_order) = sort_order else {
        return files;
    };
    // The longest prefix of plain columns that resolve in the table schema;
    // the leading key is required.
    let mut keys: Vec<(usize, bool)> = Vec::new();
    for sort_expr in sort_order.iter() {
        let Some(col) = sort_expr.expr.downcast_ref::<Column>() else {
            break;
        };
        match table_schema.index_of(col.name()) {
            Ok(idx) => keys.push((idx, sort_expr.options.descending)),
            Err(_) => break,
        }
    }
    if keys.is_empty() {
        return files;
    }

    let file_min = |file: &PartitionedFile, idx: usize| -> Option<ScalarValue> {
        file.statistics
            .as_ref()?
            .column_statistics
            .get(idx)?
            .min_value
            .get_value()
            .cloned()
    };
    files.sort_by(|a, b| {
        for &(idx, descending) in &keys {
            let ord = match (file_min(a, idx), file_min(b, idx)) {
                (Some(va), Some(vb)) => {
                    let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
                    if descending { cmp.reverse() } else { cmp }
                }
                // Files with statistics first, whatever the direction.
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    files
}

impl FileSource for LiquidParquetSource {
    fn create_file_opener(
        &self,
        _object_store: Arc<dyn ObjectStore>,
        _base_config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn datafusion::datasource::physical_plan::FileOpener>> {
        internal_err!(
            "LiquidParquetSource::create_file_opener called but it supports the Morsel API, please use that instead"
        )
    }

    fn create_morselizer(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Result<Box<dyn Morselizer>> {
        let expr_adapter_factory = base_config
            .expr_adapter_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory) as _);

        let parquet_file_reader_factory = self
            .parquet_file_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(DefaultParquetFileReaderFactory::new(object_store)));

        let execution_span = self
            .span
            .clone()
            .map(|span| fastrace::Span::enter_with_parent(format!("opener_{partition}"), &span));
        Ok(Box::new(LiquidMorselizer {
            partition_index: partition,
            projection: self.projection.clone(),
            batch_size: self
                .batch_size
                .expect("Batch size must be set before creating LiquidMorselizer"),
            predicate: self.predicate.clone(),
            table_schema: self.table_schema.clone(),
            metrics: self.metrics.clone(),
            liquid_cache: self.liquid_cache.clone(),
            parquet_file_reader_factory,
            object_store_url: base_config.object_store_url.clone(),
            reorder_filters: self.reorder_filters(),
            enable_bloom_filter: self.bloom_filter_on_read(),
            expr_adapter_factory,
            span: execution_span.map(Arc::new),
            lineages: Arc::clone(&self.lineages),
            prefetch: self.prefetch,
        }))
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.predicate.clone()
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        let mut source = self.clone();
        source.projection = self.projection.try_merge(projection)?;
        Ok(Some(Arc::new(source)))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        "liquid_parquet"
    }

    fn fmt_extra(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if let Some(predicate) = self.filter() {
                    write!(f, ", predicate={predicate}")?;
                }
                if let Some(sort_order) = &self.sort_order_for_reorder {
                    write!(f, ", sort_order_for_reorder=[{sort_order}]")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => Ok(()),
        }
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let filters: Vec<_> = filters
            .into_iter()
            .map(|filter| {
                if can_expr_be_pushed_down_with_schemas(&filter, self.table_schema.file_schema()) {
                    PushedDownPredicate::supported(filter)
                } else {
                    PushedDownPredicate::unsupported(filter)
                }
            })
            .collect();

        if filters
            .iter()
            .all(|filter| matches!(filter.discriminant, PushedDown::No))
        {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let supported = filters
            .iter()
            .filter_map(|filter| match filter.discriminant {
                PushedDown::Yes => Some(Arc::clone(&filter.predicate)),
                PushedDown::No => None,
            });
        let predicate = conjunction(self.predicate.iter().cloned().chain(supported));
        let source = Arc::new(self.clone().with_predicate(predicate));

        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            filters.iter().map(|filter| filter.discriminant).collect(),
        )
        .with_updated_node(source))
    }

    /// Approximate a sort for a parent TopK by ordering the files it reads.
    ///
    /// Mirrors `ParquetSource::try_pushdown_sort`, minus the row-group work
    /// this reader does not do: when the request's leading key is a plain
    /// column of the file schema, answer `Inexact` and remember the request,
    /// so [`Self::reorder_files`] puts the files with the best statistics at
    /// the front of the shared work queue. The parent `SortExec` stays, so the
    /// result is the same; the TopK's dynamic filter just tightens after the
    /// first files instead of the last.
    ///
    /// Everything else stays `Unsupported`, which leaves today's behaviour:
    /// - a request the source's own ordering already satisfies, where
    ///   `FileScanConfig` sorts the file groups by statistics and may drop the
    ///   `SortExec` (an `Inexact` answer here would forbid that);
    /// - a request whose proper prefix the ordering satisfies, so the
    ///   `SortExec`'s prefix-aware early stop still applies (the same reason
    ///   `ParquetSource` declines it);
    /// - a leading key that is not a plain file column, which statistics
    ///   cannot order.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
        eq_properties: &EquivalenceProperties,
    ) -> Result<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        let Some(sort_order) = LexOrdering::new(order.iter().cloned()) else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        if eq_properties.ordering_satisfy(order.iter().cloned())? {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        for prefix_len in 1..order.len() {
            if eq_properties.ordering_satisfy(order[..prefix_len].iter().cloned())? {
                return Ok(SortOrderPushdownResult::Unsupported);
            }
        }
        let leading_is_file_column = sort_order
            .first()
            .expr
            .downcast_ref::<Column>()
            .is_some_and(|col| {
                self.table_schema
                    .file_schema()
                    .field_with_name(col.name())
                    .is_ok()
            });
        if !leading_is_file_column {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        let mut source = self.clone();
        source.sort_order_for_reorder = Some(sort_order);
        Ok(SortOrderPushdownResult::Inexact {
            inner: Arc::new(source) as Arc<dyn FileSource>,
        })
    }

    fn reorder_files(&self, files: Vec<PartitionedFile>) -> Vec<PartitionedFile> {
        reorder_files_by_min_statistics(
            files,
            self.sort_order_for_reorder.as_ref(),
            self.table_schema.table_schema(),
        )
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(
            self.predicate
                .iter()
                .chain(self.projection.iter().map(|projection| &projection.expr)),
            f,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::SortOptions;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::{ColumnStatistics, Statistics, stats::Precision};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("ts", DataType::Int64, true),
        ])
    }

    /// A file whose `ts` column has `min` (`None` = no statistic).
    fn file(name: &str, ts_min: Option<i64>) -> PartitionedFile {
        let file = PartitionedFile::new(name, 100);
        let Some(min) = ts_min else {
            return file;
        };
        let ts = ColumnStatistics {
            min_value: Precision::Exact(ScalarValue::Int64(Some(min))),
            ..ColumnStatistics::new_unknown()
        };
        file.with_statistics(Arc::new(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics::new_unknown(), ts],
        }))
    }

    fn names(files: &[PartitionedFile]) -> Vec<&str> {
        files
            .iter()
            .map(|f| f.object_meta.location.as_ref())
            .collect()
    }

    fn ts_order(descending: bool) -> LexOrdering {
        LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("ts", 1)),
            SortOptions {
                descending,
                nulls_first: false,
            },
        )])
        .unwrap()
    }

    fn files() -> Vec<PartitionedFile> {
        vec![
            file("mid", Some(50)),
            file("none", None),
            file("old", Some(10)),
            file("new", Some(90)),
        ]
    }

    #[test]
    fn descending_puts_the_largest_min_first_and_missing_stats_last() {
        let order = ts_order(true);
        let sorted = reorder_files_by_min_statistics(files(), Some(&order), &schema());
        assert_eq!(names(&sorted), ["new", "mid", "old", "none"]);
    }

    #[test]
    fn ascending_puts_the_smallest_min_first_and_missing_stats_last() {
        let order = ts_order(false);
        let sorted = reorder_files_by_min_statistics(files(), Some(&order), &schema());
        assert_eq!(names(&sorted), ["old", "mid", "new", "none"]);
    }

    #[test]
    fn no_sort_order_keeps_the_given_order() {
        let sorted = reorder_files_by_min_statistics(files(), None, &schema());
        assert_eq!(names(&sorted), ["mid", "none", "old", "new"]);
    }

    #[test]
    fn a_leading_expression_keeps_the_given_order() {
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("ts", 1)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int64(Some(1)))),
        ));
        let order =
            LexOrdering::new(vec![PhysicalSortExpr::new(expr, SortOptions::default())]).unwrap();
        let sorted = reorder_files_by_min_statistics(files(), Some(&order), &schema());
        assert_eq!(names(&sorted), ["mid", "none", "old", "new"]);
    }

    #[test]
    fn a_column_missing_from_the_schema_keeps_the_given_order() {
        let order = LexOrdering::new(vec![PhysicalSortExpr::new(
            Arc::new(Column::new("gone", 5)),
            SortOptions::default(),
        )])
        .unwrap();
        let sorted = reorder_files_by_min_statistics(files(), Some(&order), &schema());
        assert_eq!(names(&sorted), ["mid", "none", "old", "new"]);
    }
}
