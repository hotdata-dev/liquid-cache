//! Optimizers for the Parquet module

mod squeeze_hint;

use std::sync::Arc;

use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{physical_plan::ParquetSource, source::DataSource},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};

pub(crate) use squeeze_hint::HintAnalyzer;
pub use squeeze_hint::SqueezeHintMap;

use crate::{LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnSqueezeHints};

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed squeeze hints from the full physical plan
/// (via the squeeze-hint analyzer) and attaches each scan's hints to its source.
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    /// When set, parquet scans whose total file size exceeds this threshold
    /// are left as vanilla DataFusion reads instead of being wrapped by
    /// LiquidCache. `None` means cache every scan.
    max_scan_bytes: Option<u64>,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            max_scan_bytes: None,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            max_scan_bytes: None,
        }
    }

    /// Set the maximum total file size (in bytes) for a parquet scan to be
    /// routed through LiquidCache. Scans exceeding this are read directly
    /// from the underlying parquet source, bypassing the cache.
    pub fn with_max_scan_bytes(mut self, max_bytes: u64) -> Self {
        self.max_scan_bytes = Some(max_bytes);
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
        let max_scan_bytes = self.max_scan_bytes;
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnSqueezeHints| {
            // Leave oversized scans as vanilla parquet reads, bypassing the cache.
            if let Some(max_bytes) = max_scan_bytes
                && parquet_scan_total_bytes(node).is_some_and(|total| total > max_bytes)
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

/// Total size in bytes of all files scanned by a parquet `DataSourceExec`, or
/// `None` if `node` is not such a scan.
fn parquet_scan_total_bytes(node: &Arc<dyn ExecutionPlan>) -> Option<u64> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, _) = data_source_exec.downcast_to_file_source::<ParquetSource>()?;
    Some(
        file_scan_config
            .file_groups
            .iter()
            .flat_map(|g| g.files())
            .map(|f| f.object_meta.size)
            .sum(),
    )
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

    /// A parquet scan whose total file size exceeds `max_scan_bytes` is left as
    /// a plain `ParquetSource`; one under the threshold is still wrapped in
    /// `LiquidParquetSource`.
    #[tokio::test]
    async fn test_max_scan_bytes_pass_through() {
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

        // Threshold below the file size → scan bypasses the cache.
        let capped = LocalModeOptimizer::new(build_cache().await)
            .with_max_scan_bytes(1)
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&capped),
            "oversized scan should stay a plain ParquetSource"
        );

        // Threshold above the file size → scan is cached as usual.
        let uncapped = LocalModeOptimizer::new(build_cache().await)
            .with_max_scan_bytes(u64::MAX)
            .optimize(plan, &config)
            .unwrap();
        assert!(
            has_liquid_source(&uncapped),
            "under-threshold scan should be wrapped in LiquidParquetSource"
        );
    }
}
