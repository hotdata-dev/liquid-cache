//! Optimizers for the Parquet module

mod squeeze_hint;

use std::sync::Arc;

use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileScanConfig, ParquetSource},
        source::DataSource,
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};

pub(crate) use squeeze_hint::HintAnalyzer;
pub use squeeze_hint::SqueezeHintMap;

use crate::{LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnSqueezeHints};

/// Admission predicate deciding whether a parquet scan is routed through
/// LiquidCache. Called once per parquet `DataSourceExec` with that scan's
/// [`FileScanConfig`]; returning `false` leaves the scan as a vanilla
/// DataFusion read. The caller owns the policy (e.g. estimated scan size vs.
/// the cache budget); [`estimate_projected_bytes`] is a ready-made building
/// block for size-based filters.
pub type ScanAdmissionFilter = Arc<dyn Fn(&FileScanConfig) -> bool + Send + Sync>;

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed squeeze hints from the full physical plan
/// (via the squeeze-hint analyzer) and attaches each scan's hints to its source.
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    /// Optional admission predicate. When set, a scan is only wrapped by
    /// LiquidCache if the predicate returns `true`; scans it rejects are left
    /// as vanilla DataFusion reads. `None` means cache every scan.
    scan_filter: Option<ScanAdmissionFilter>,
}

impl std::fmt::Debug for LocalModeOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalModeOptimizer")
            .field("cache", &self.cache)
            .field("scan_filter", &self.scan_filter.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            scan_filter: None,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            scan_filter: None,
        }
    }

    /// Set an admission predicate that decides, per parquet scan, whether the
    /// scan is routed through LiquidCache. Scans the predicate rejects are read
    /// directly from the underlying parquet source, bypassing the cache. See
    /// [`ScanAdmissionFilter`] and [`estimate_projected_bytes`].
    pub fn with_scan_filter(mut self, filter: ScanAdmissionFilter) -> Self {
        self.scan_filter = Some(filter);
        self
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        let analysis = HintAnalyzer::analyze(&plan);
        let cache = self.cache.clone();
        let scan_filter = self.scan_filter.clone();
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnSqueezeHints| {
            // Let the admission predicate veto a scan, leaving it as a vanilla
            // parquet read that bypasses the cache.
            if let Some(filter) = &scan_filter
                && let Some(config) = parquet_file_scan_config(node)
                && !filter(config)
            {
                return None;
            }
            convert_parquet_scan(node, &cache, hints)
        };
        Ok(squeeze_hint::rewrite_with_hints(
            plan,
            &mut convert,
            &analysis,
        ))
    }

    fn name(&self) -> &str {
        "LocalModeLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rewrite the data source plan to use liquid cache, attaching `hints` (keyed by
/// file-schema column name) to every parquet scan it rewrites.
///
/// This is the entry point used by the cache server, where hints are derived on
/// the client (which has the full plan) and shipped alongside the pushed
/// fragment, which is always single-scan.
pub fn rewrite_data_source_plan_with_hints(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: &ColumnSqueezeHints,
) -> Arc<dyn ExecutionPlan> {
    plan.transform_up(
        |node| match convert_parquet_scan(&node, cache, hints.clone()) {
            Some(new_node) => Ok(Transformed::new(
                new_node,
                true,
                TreeNodeRecursion::Continue,
            )),
            None => Ok(Transformed::no(node)),
        },
    )
    .unwrap()
    .data
}

/// Rewrite the data source plan to use liquid cache (no squeeze hints).
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
) -> Arc<dyn ExecutionPlan> {
    rewrite_data_source_plan_with_hints(plan, cache, &ColumnSqueezeHints::default())
}

/// If `node` is a parquet `DataSourceExec`, return its [`FileScanConfig`].
fn parquet_file_scan_config(node: &Arc<dyn ExecutionPlan>) -> Option<&FileScanConfig> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, _) = data_source_exec.downcast_to_file_source::<ParquetSource>()?;
    Some(file_scan_config)
}

/// Estimate the compressed bytes a parquet scan will read, accounting for
/// column projection: the total on-disk size of the scan's files scaled by the
/// fraction of columns the query actually projects.
///
/// Intended as a building block for size-based [`ScanAdmissionFilter`]s. Two
/// caveats on accuracy: it assumes columns are roughly uniform in size (a
/// single very wide column projected alone is under-counted), and it does not
/// account for row-group/page pruning, which is decided at execution time.
/// File- and partition-level pruning is already reflected, since the scan's
/// file groups only contain files that survived planning.
pub fn estimate_projected_bytes(config: &FileScanConfig) -> u64 {
    let total_bytes: u64 = config
        .file_groups
        .iter()
        .flat_map(|g| g.files())
        .map(|f| f.object_meta.size)
        .sum();
    let total_cols = config.file_schema().fields().len().max(1);
    // On error, assume the full projection so the estimate stays conservative
    // (larger → more likely to be gated out). Clamp to the file column count:
    // the projected schema can also include (near-zero-byte) partition columns,
    // which must not push the ratio above 1.
    let projected_cols = config
        .projected_schema()
        .map(|schema| schema.fields().len())
        .unwrap_or(total_cols)
        .min(total_cols);
    ((total_bytes as u128 * projected_cols as u128) / total_cols as u128) as u64
}

/// If `node` is a `DataSourceExec` over a `ParquetSource`, return an equivalent
/// node backed by [`LiquidParquetSource`] carrying `hints`.
fn convert_parquet_scan(
    node: &Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: ColumnSqueezeHints,
) -> Option<Arc<dyn ExecutionPlan>> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, parquet_source) =
        data_source_exec.downcast_to_file_source::<ParquetSource>()?;

    let new_source =
        LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone())
            .with_squeeze_hints(Arc::new(hints));

    let mut new_config = file_scan_config.clone();
    new_config.file_source = Arc::new(new_source);
    let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
    Some(Arc::new(DataSourceExec::new(new_file_source)))
}

#[cfg(test)]
mod tests {
    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };

    use crate::LiquidCacheParquet;

    use super::*;

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        let expected_schema = plan.schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let liquid_cache = Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeSqueezeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        );
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let source = data_source.downcast_ref::<FileScanConfig>().unwrap();
                    let file_source = source.file_source();
                    let _parquet_source =
                        file_source.downcast_ref::<LiquidParquetSource>().unwrap();
                    let schema = source.file_schema().as_ref();
                    assert_eq!(schema, expected_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        rewrite_plan_inner(plan.clone()).await;
    }

    async fn build_cache() -> LiquidCacheParquetRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeSqueezeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        )
    }

    /// True if any parquet scan in `plan` was rewritten to `LiquidParquetSource`.
    fn has_liquid_source(plan: &Arc<dyn ExecutionPlan>) -> bool {
        let mut found = false;
        plan.apply(|node| {
            if let Some(exec) = node.downcast_ref::<DataSourceExec>()
                && let Some(cfg) = exec.data_source().downcast_ref::<FileScanConfig>()
                && cfg
                    .file_source()
                    .downcast_ref::<LiquidParquetSource>()
                    .is_some()
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    /// Extract the [`FileScanConfig`] of the first parquet scan in `plan`.
    fn scan_config_of(plan: &Arc<dyn ExecutionPlan>) -> FileScanConfig {
        let mut found = None;
        plan.apply(|node| {
            if let Some(config) = parquet_file_scan_config(node) {
                found = Some(config.clone());
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found.expect("plan should contain a parquet scan")
    }

    /// A scan the admission filter rejects is left as a plain `ParquetSource`;
    /// one it accepts (and the default no-filter case) is wrapped in
    /// `LiquidParquetSource`.
    #[tokio::test]
    async fn test_scan_filter_pass_through() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let plan = ctx
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let config = ConfigOptions::default();

        // Filter rejects the scan → bypasses the cache.
        let rejected = LocalModeOptimizer::new(build_cache().await)
            .with_scan_filter(Arc::new(|_| false))
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&rejected),
            "rejected scan should stay a plain ParquetSource"
        );

        // Filter accepts the scan → cached.
        let accepted = LocalModeOptimizer::new(build_cache().await)
            .with_scan_filter(Arc::new(|_| true))
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            has_liquid_source(&accepted),
            "accepted scan should be wrapped in LiquidParquetSource"
        );

        // No filter → cache every scan (default).
        let default = LocalModeOptimizer::new(build_cache().await)
            .optimize(plan, &config)
            .unwrap();
        assert!(
            has_liquid_source(&default),
            "with no filter every scan should be cached"
        );
    }

    /// The projection-aware estimate shrinks when fewer columns are read: a
    /// single-column projection must estimate strictly fewer bytes than a
    /// full-table scan of the same files.
    #[tokio::test]
    async fn test_estimate_projected_bytes_scales_with_projection() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();

        let all_cols = scan_config_of(
            &ctx.sql("SELECT * FROM nano_hits")
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap(),
        );
        let one_col = scan_config_of(
            &ctx.sql("SELECT \"URL\" FROM nano_hits")
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap(),
        );

        let all_bytes = estimate_projected_bytes(&all_cols);
        let one_bytes = estimate_projected_bytes(&one_col);

        assert!(all_bytes > 0, "full-table estimate should be non-zero");
        assert!(
            one_bytes < all_bytes,
            "single-column projection ({one_bytes}) should estimate fewer bytes \
             than the full scan ({all_bytes})"
        );
    }
}
