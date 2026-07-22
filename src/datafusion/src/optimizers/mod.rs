//! Optimizers for the Parquet module

mod squeeze_hint;

use std::sync::Arc;

use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::memory::DataSourceExec,
    common::{
        Statistics,
        pruning::PrunableStatistics,
        stats::Precision,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    config::ConfigOptions,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{FileScanConfig, FileSource, ParquetSource},
        source::DataSource,
    },
    physical_expr::utils::collect_columns,
    physical_optimizer::{PhysicalOptimizerRule, pruning::PruningPredicate},
    physical_plan::ExecutionPlan,
};

pub(crate) use squeeze_hint::HintAnalyzer;
pub use squeeze_hint::SqueezeHintMap;

use crate::{LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnSqueezeHints};

/// Parameters for the footprint-based admission gate.
///
/// A parquet scan is routed through LiquidCache only if its estimated in-memory
/// liquid footprint stays within `budget × tolerance`; larger scans are left as
/// vanilla parquet reads (which, if the object store is a cached mount, read
/// from it).
///
/// The estimate multiplies the raw required parquet bytes by `expansion`
/// (parquet -> liquid in-memory blow-up) and `safety` (extra margin); both are
/// `>= 1.0` so the estimate is conservative (over-counts). `tolerance` (`>= 1.0`)
/// is the one knob applied to the *budget*: it encodes that LiquidCache compacts
/// in RAM and still beats the fallback mount until footprint reaches ~5x the
/// budget, so the gate tolerates that overcommit before bypassing.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionGate {
    /// Parquet-bytes -> liquid-in-memory-bytes multiplier (>= 1.0). Inflates the
    /// estimate (conservative direction).
    pub expansion: f64,
    /// Extra safety margin on the estimate (>= 1.0). Conservative direction.
    pub safety: f64,
    /// Overcommit tolerance: how far the estimated liquid footprint may exceed
    /// the budget before the scan is denied, in multiples of the budget.
    /// LiquidCache compacts in RAM up to ~5x over budget before it starts
    /// thrashing, so caching still wins in that band. This is the one *relaxing*
    /// knob, so it is clamped to `[1.0, 5.0]` (5.0 = the measured crossover).
    pub tolerance: f64,
}

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed squeeze hints from the full physical plan
/// (via the squeeze-hint analyzer) and attaches each scan's hints to its source.
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    /// When set, a scan whose estimated footprint exceeds the cache budget is
    /// left as a vanilla parquet read instead of being wrapped by LiquidCache.
    /// `None` means cache every scan.
    admission: Option<AdmissionGate>,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            admission: None,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            admission: None,
        }
    }

    /// Enable the footprint-based admission gate. A parquet scan is cached only
    /// when its estimated liquid footprint (raw required bytes x `expansion` x
    /// `safety`) stays within `budget × tolerance`; otherwise it is read directly
    /// from the parquet source, bypassing the cache. See [`AdmissionGate`].
    ///
    /// Inputs are sanitized so a misconfigured value can never make the gate
    /// unsound: `expansion`/`safety` are forced finite and `>= 1.0` (their only
    /// effect is to inflate the estimate, the conservative direction), and
    /// `tolerance` — the one *relaxing* knob — is forced finite and clamped to
    /// `[1.0, 5.0]` (5.0 = the measured compaction crossover), defaulting to 3.0
    /// if non-finite.
    pub fn with_admission_gate(mut self, expansion: f64, safety: f64, tolerance: f64) -> Self {
        let estimate_factor = |v: f64| if v.is_finite() && v >= 1.0 { v } else { 1.0 };
        let tolerance = if tolerance.is_finite() {
            tolerance.clamp(1.0, 5.0)
        } else {
            3.0
        };
        self.admission = Some(AdmissionGate {
            expansion: estimate_factor(expansion),
            safety: estimate_factor(safety),
            tolerance,
        });
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
        let admission = self.admission;
        let budget = self.cache.max_memory_bytes() as u64;
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnSqueezeHints| {
            // Leave scans whose estimated liquid footprint exceeds the budget as
            // vanilla parquet reads, so oversized scans don't thrash the cache.
            if let Some(gate) = admission
                && let Some((cfg, src)) = parquet_scan_parts(node)
                && should_bypass(cfg, src, gate, budget)
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

/// If `node` is a parquet `DataSourceExec`, return its `FileScanConfig` and
/// `ParquetSource`.
fn parquet_scan_parts(node: &Arc<dyn ExecutionPlan>) -> Option<(&FileScanConfig, &ParquetSource)> {
    let dse = node.downcast_ref::<DataSourceExec>()?;
    dse.downcast_to_file_source::<ParquetSource>()
}

/// Estimate the raw required-column parquet bytes a scan reads: the sum, over
/// the files that survive the scan's predicate, of the byte sizes of the columns
/// it materializes. This is the byte-accurate, filter-aware size the admission
/// decision is built on (see [`should_bypass`]); the expansion/safety/tolerance
/// factors are applied there, not here.
///
/// "Required columns" is the output projection **unioned with the predicate
/// columns**, since LiquidCache materializes both. Byte sizing uses only
/// `Exact` per-column sizes; anything else falls back to the whole file size,
/// so the estimate never *under*-counts (which could wrongly admit an oversized
/// scan and thrash the cache). The sum saturates rather than overflowing on
/// pathological file lists.
fn estimate_required_bytes(cfg: &FileScanConfig, src: &ParquetSource) -> u64 {
    let num_file_cols = cfg.file_schema().fields().len();
    // Full table schema (file + partition columns). The pushed-down predicate
    // and `PartitionedFile` stats are expressed against it, so file pruning must
    // use it; byte accounting still charges only file columns.
    let table_schema = cfg.file_source.table_schema().table_schema().clone();

    let mut required: Vec<usize> = {
        // Removed in df58; we are pinned to df54. Returns file-column indices.
        #[allow(deprecated)]
        match cfg.file_column_projection_indices() {
            Some(cols) => cols,
            None => (0..num_file_cols).collect(),
        }
    };
    if let Some(pred) = src.filter() {
        for col in collect_columns(&pred) {
            let idx = col.index();
            // Partition columns are appended after file columns in the table
            // schema and are literals (never materialized in LiquidCache), so
            // only file columns contribute to the footprint.
            if idx < num_file_cols {
                required.push(idx);
            }
        }
    }
    required.sort_unstable();
    required.dedup();

    let files: Vec<&PartitionedFile> = cfg.file_groups.iter().flat_map(|g| g.files()).collect();
    if files.is_empty() {
        return 0;
    }

    let surviving = surviving_files(src, &table_schema, &files);

    files
        .iter()
        .zip(surviving.iter())
        .filter(|(_, keep)| **keep)
        .map(|(f, _)| file_required_bytes(f.statistics.as_deref(), f.object_meta.size, &required))
        .fold(0u64, u64::saturating_add)
}

/// Decide whether a parquet scan should bypass the cache (read as vanilla
/// parquet) rather than be transcoded into LiquidCache.
///
/// The scan's estimated liquid footprint is `raw × expansion × safety`, where
/// `raw` is [`estimate_required_bytes`]. It bypasses when that footprint exceeds
/// `budget × tolerance` — equivalently, when the pressure ratio
/// `footprint / budget` exceeds `tolerance`. The comparison is done in finite
/// `f64` space to avoid the overflow/truncation of integer `budget × tolerance`.
///
/// `budget == 0` (cache disabled / unsized) bypasses any non-empty scan and
/// admits only zero-footprint ones.
fn should_bypass(
    cfg: &FileScanConfig,
    src: &ParquetSource,
    gate: AdmissionGate,
    budget: u64,
) -> bool {
    let raw = estimate_required_bytes(cfg, src);
    // Multipliers are already sanitized to finite, >= 1.0 in `with_admission_gate`.
    let footprint = (raw as f64) * gate.expansion * gate.safety;
    if budget == 0 {
        return footprint > 0.0;
    }
    footprint / (budget as f64) > gate.tolerance
}

/// Boolean per file: `true` if the file may match the predicate (keep it),
/// `false` if the predicate's stats prove it cannot match (prune it).
/// Conservative: no predicate, missing/mismatched stats, or any pruning error
/// keeps files. `table_schema` (file + partition columns) is used so predicates
/// on partition columns resolve, matching `PartitionedFile::statistics`.
fn surviving_files(
    src: &ParquetSource,
    table_schema: &SchemaRef,
    files: &[&PartitionedFile],
) -> Vec<bool> {
    let Some(pred) = src.filter() else {
        return vec![true; files.len()];
    };
    let pruning = match PruningPredicate::try_new(pred, table_schema.clone()) {
        Ok(p) => p,
        Err(_) => return vec![true; files.len()],
    };
    let expected = table_schema.fields().len();
    let stats: Vec<Arc<Statistics>> = files
        .iter()
        .map(|f| match &f.statistics {
            // Only trust stats whose width matches the table schema; otherwise
            // treat as unknown (kept, not pruned) to avoid a schema mismatch.
            Some(s) if s.column_statistics.len() == expected => s.clone(),
            _ => Arc::new(Statistics::new_unknown(table_schema)),
        })
        .collect();
    let prunable = PrunableStatistics::new(stats, table_schema.clone());
    match pruning.prune(&prunable) {
        Ok(mask) if mask.len() == files.len() => mask,
        _ => vec![true; files.len()],
    }
}

/// Bytes the `required` columns of one file contribute to the footprint.
///
/// Uses only `Exact` per-column byte sizes. If the file has no stats, or any
/// required column lacks an exact size, fall back to the whole-file size — a
/// deliberate over-estimate, since under-counting could wrongly admit an
/// oversized scan.
fn file_required_bytes(stats: Option<&Statistics>, object_size: u64, required: &[usize]) -> u64 {
    let Some(stats) = stats else {
        return object_size;
    };
    let mut sum: u64 = 0;
    for &c in required {
        match stats.column_statistics.get(c).map(|cs| &cs.byte_size) {
            Some(Precision::Exact(n)) => sum += *n as u64,
            _ => return object_size,
        }
    }
    sum
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
        build_cache_with_budget(1_000_000).await
    }

    async fn build_cache_with_budget(max_memory_bytes: usize) -> LiquidCacheParquetRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                max_memory_bytes,
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

    /// The admission gate bypasses a scan whose estimated footprint exceeds the
    /// budget (large expansion here forces that), and caches one that fits.
    #[tokio::test]
    async fn test_admission_gate_pass_through() {
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

        // A huge expansion inflates the estimate past the 1 MB budget → bypass.
        let capped = LocalModeOptimizer::new(build_cache().await)
            .with_admission_gate(1e9, 1.0, 1.0)
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&capped),
            "oversized estimate should stay a plain ParquetSource"
        );

        // With a large budget the footprint fits (expansion 1.0) → cached.
        let uncapped = LocalModeOptimizer::new(build_cache_with_budget(usize::MAX).await)
            .with_admission_gate(1.0, 1.0, 1.0)
            .optimize(plan, &config)
            .unwrap();
        assert!(
            has_liquid_source(&uncapped),
            "fitting scan should be wrapped in LiquidParquetSource"
        );
    }
}

/// Pure unit tests for the footprint byte-math (no cache / no t4 mount, so they
/// run everywhere). These cover the correctness-critical fallback direction:
/// under-counting could wrongly admit an oversized scan, so anything not backed
/// by an `Exact` per-column byte size must fall back to the whole-file size.
#[cfg(test)]
mod footprint_tests {
    use super::file_required_bytes;
    use datafusion::common::{ColumnStatistics, Statistics, stats::Precision};

    fn col(byte_size: Precision<usize>) -> ColumnStatistics {
        let mut c = ColumnStatistics::new_unknown();
        c.byte_size = byte_size;
        c
    }

    fn stats(cols: Vec<ColumnStatistics>) -> Statistics {
        Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: cols,
        }
    }

    #[test]
    fn no_stats_falls_back_to_full_file() {
        assert_eq!(file_required_bytes(None, 5000, &[0, 1]), 5000);
    }

    #[test]
    fn sums_only_required_exact_columns() {
        let s = stats(vec![
            col(Precision::Exact(100)),
            col(Precision::Exact(200)),
            col(Precision::Exact(400)),
        ]);
        // required = cols 0 and 2 → 100 + 400, ignoring col 1.
        assert_eq!(file_required_bytes(Some(&s), 9999, &[0, 2]), 500);
    }

    #[test]
    fn inexact_required_column_falls_back_to_full_file() {
        let s = stats(vec![
            col(Precision::Exact(100)),
            col(Precision::Inexact(200)),
        ]);
        // col 1 inexact → not a safe upper bound → whole file.
        assert_eq!(file_required_bytes(Some(&s), 7000, &[0, 1]), 7000);
        // col 0 alone is exact → its bytes only.
        assert_eq!(file_required_bytes(Some(&s), 7000, &[0]), 100);
    }

    #[test]
    fn missing_required_column_falls_back_to_full_file() {
        let s = stats(vec![col(Precision::Exact(100))]);
        // required col 5 doesn't exist → conservative whole file.
        assert_eq!(file_required_bytes(Some(&s), 3000, &[0, 5]), 3000);
    }

    #[test]
    fn absent_byte_size_falls_back_to_full_file() {
        let s = stats(vec![col(Precision::Absent)]);
        assert_eq!(file_required_bytes(Some(&s), 2000, &[0]), 2000);
    }
}
