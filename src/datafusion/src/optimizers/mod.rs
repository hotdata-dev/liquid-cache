//! Optimizers for the Parquet module

mod squeeze_hint;

use std::collections::HashSet;
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
/// A parquet scan is routed through LiquidCache only if its estimated liquid
/// footprint stays within the admission threshold; larger scans are left as
/// vanilla parquet reads (which, if the object store is a cached mount, read
/// from it).
///
/// The threshold spans both cache tiers, weighted differently:
/// `memory × tolerance + disk`. A scan that overflows
/// RAM spills to the on-disk liquid tier rather than thrashing, so disk capacity
/// counts toward what fits — at face value, since only the RAM tier has the
/// measured compaction overcommit. With the disk tier off it is just
/// `memory × tolerance`.
///
/// The estimate multiplies the raw required parquet bytes by `expansion`
/// (parquet -> liquid in-memory blow-up) and `safety` (extra margin); both are
/// `>= 1.0` so the estimate is conservative (over-counts).
#[derive(Debug, Clone, Copy)]
pub struct AdmissionGate {
    /// Parquet-bytes -> liquid-in-memory-bytes multiplier (>= 1.0). Inflates the
    /// estimate (conservative direction).
    pub expansion: f64,
    /// Extra safety margin on the estimate (>= 1.0). Conservative direction.
    pub safety: f64,
    /// Overcommit tolerance on the *memory* tier: how far the estimated liquid
    /// footprint may exceed the RAM budget before it counts against the scan, in
    /// multiples of the RAM budget. LiquidCache compacts in RAM up to ~5x over
    /// budget before it starts thrashing, so caching still wins in that band.
    /// This is the one *relaxing* knob, so it is clamped to `[1.0, 5.0]` (5.0 =
    /// the measured crossover). It applies only to memory; the disk tier is
    /// counted at face value (`memory × tolerance + disk`), never scaled by it.
    pub tolerance: f64,
    /// Fail-loud mode. When `true`, a panic during footprint estimation is *not*
    /// caught — it aborts the query — so estimation bugs surface immediately.
    /// When `false`, the panic is caught, logged at ERROR, and the scan is cached
    /// normally so the query survives. The caller chooses the default.
    pub strict: bool,
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
    /// `safety`) stays within the admission threshold (`memory × tolerance +
    /// disk`); otherwise it is read directly from the parquet source, bypassing
    /// the cache. See [`AdmissionGate`].
    ///
    /// Inputs are sanitized so a misconfigured value can never make the gate
    /// unsound: `expansion`/`safety` are forced finite and `>= 1.0` (their only
    /// effect is to inflate the estimate, the conservative direction), and
    /// `tolerance` — the one *relaxing* knob — is forced finite and clamped to
    /// `[1.0, 5.0]` (5.0 = the measured compaction crossover), defaulting to 3.0
    /// if non-finite. `strict` (see [`AdmissionGate::strict`]) turns off the
    /// panic guard so estimation bugs fail loud.
    pub fn with_admission_gate(
        mut self,
        expansion: f64,
        safety: f64,
        tolerance: f64,
        strict: bool,
    ) -> Self {
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
            strict,
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
        // The gate sizes against both cache tiers, not just RAM: when a scan
        // overflows memory its entries spill to the on-disk liquid tier (NVMe)
        // instead of thrashing. They are weighted differently (see
        // `admission_threshold`) — RAM carries the compaction overcommit, disk
        // counts at face value — so they are passed through separately. With the
        // disk tier off (`max_disk_bytes == 0`) the threshold is exactly the RAM
        // budget, so gate behaviour is unchanged there.
        let memory_budget = self.cache.max_memory_bytes() as u64;
        let disk_budget = self.cache.max_disk_bytes() as u64;
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnSqueezeHints| {
            // Leave scans whose estimated liquid footprint exceeds the budget as
            // vanilla parquet reads, so oversized scans don't thrash the cache.
            if let Some(gate) = admission
                && let Some((cfg, src)) = parquet_scan_parts(node)
                && should_bypass_guarded(cfg, src, gate, memory_budget, disk_budget)
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

/// A scan's footprint estimate plus the breakdown behind it, for the admission
/// decision ([`should_bypass`]) and its diagnostic log line.
#[derive(Debug, Default, Clone, Copy)]
struct FootprintEstimate {
    /// Raw required-column parquet bytes (the decision input).
    raw_bytes: u64,
    /// Number of file columns the scan materializes (projection ∪ predicate).
    required_cols: usize,
    /// Distinct physical files charged (survived pruning, deduped across the
    /// byte-range splits of the same file).
    charged_files: usize,
    /// Raw `PartitionedFile` count before dedupe (one file may be split into
    /// several byte-range partitions for parallelism).
    partitioned_files: usize,
    /// Surviving files charged the *whole file* size because they lacked
    /// per-column byte sizes. A non-zero count means the catalog has no
    /// `column_size_bytes` for this table, so the estimate is coarse and
    /// over-counts — the signal that the DuckLake write-side size stat is
    /// missing for this table.
    fallback_files: usize,
}

/// Estimate the raw required-column parquet bytes a scan reads: the sum, over
/// the files that survive the scan's predicate, of the byte sizes of the columns
/// it materializes. This is the byte-accurate, filter-aware size the admission
/// decision is built on (see [`should_bypass`]); the expansion/safety/tolerance
/// factors are applied there, not here.
///
/// "Required columns" is the output projection **unioned with the predicate
/// columns**, since LiquidCache materializes both. Byte sizing uses `Exact` or
/// `Inexact` per-column sizes (DuckLake records real column sizes but labels
/// them `Inexact`); a column with an `Absent` size or a file with no stats
/// falls back to the whole file size. The sum saturates rather than overflowing
/// on pathological file lists.
fn estimate_required_bytes(cfg: &FileScanConfig, src: &ParquetSource) -> FootprintEstimate {
    let num_file_cols = cfg.file_schema().fields().len();
    // Full table schema (file + partition columns). The pushed-down predicate
    // and `PartitionedFile` stats are expressed against it, so file pruning must
    // use it; byte accounting still charges only file columns.
    let table_schema = cfg.file_source.table_schema().table_schema().clone();

    // Columns the scan projects. `column_indices()` collects the source columns
    // referenced by each projection expression, so it is correct for compound
    // projections (e.g. `a * b` reads a and b) and — unlike the deprecated
    // `FileScanConfig::file_column_projection_indices` /
    // `ProjectionExprs::ordered_column_indices` — does not panic on a
    // non-column projection expression. Indices are table-schema-relative; keep
    // only file columns (partition columns are literals, never materialized).
    let mut required: Vec<usize> = match src.projection() {
        Some(p) => p
            .column_indices()
            .into_iter()
            .filter(|&i| i < num_file_cols)
            .collect(),
        None => (0..num_file_cols).collect(),
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
    let required_cols = required.len();
    if files.is_empty() {
        return FootprintEstimate {
            required_cols,
            ..FootprintEstimate::default()
        };
    }

    let surviving = surviving_files(src, &table_schema, &files);

    // Dedupe by physical file identity (path + size). DataFusion splits one file
    // into several byte-range `PartitionedFile`s for parallelism, each a *clone*
    // of the whole file's statistics (and whole-file object size), differing only
    // in `range`. Counting every range would multiply a single file's footprint
    // by the split count, so charge each distinct file once — pruning is
    // file-granular, so a surviving file means caching its whole required columns
    // regardless of how it was split. Size is part of the key so two genuinely
    // distinct objects that share a path are never collapsed.
    let mut seen: HashSet<(object_store::path::Path, u64)> = HashSet::new();
    let mut raw_bytes = 0u64;
    let mut charged_files = 0usize;
    let mut fallback_files = 0usize;
    for (f, keep) in files.iter().zip(surviving.iter()) {
        if !*keep {
            continue;
        }
        if !seen.insert((f.object_meta.location.clone(), f.object_meta.size)) {
            continue;
        }
        charged_files += 1;
        let (bytes, fell_back) =
            file_required_bytes(f.statistics.as_deref(), f.object_meta.size, &required);
        if fell_back {
            fallback_files += 1;
        }
        raw_bytes = raw_bytes.saturating_add(bytes);
    }

    FootprintEstimate {
        raw_bytes,
        required_cols,
        charged_files,
        partitioned_files: files.len(),
        fallback_files,
    }
}

/// The admission threshold in bytes: the largest estimated liquid footprint the
/// cache admits before a scan is bypassed.
///
/// The two tiers are weighted differently. The memory tier carries the
/// compaction overcommit — `tolerance`, the measured RAM crossover where
/// LiquidCache still beats the fallback mount despite spilling. The on-disk
/// liquid tier is counted at **face value (1×)**: a scan that overflows RAM
/// spills to NVMe without thrashing, so its capacity needs no overcommit — and
/// extending the RAM crossover factor to disk has no evidence behind it. Hence
/// `memory × tolerance + disk` rather than `(memory + disk) × tolerance`.
///
/// Computed in `f64` so the `memory × tolerance` product cannot overflow `u64`.
/// A zero threshold (both tiers unsized / cache disabled) bypasses any non-empty
/// scan and admits only zero-footprint ones.
fn admission_threshold(memory_budget: u64, disk_budget: u64, tolerance: f64) -> f64 {
    (memory_budget as f64) * tolerance + (disk_budget as f64)
}

/// Decide whether a parquet scan should bypass the cache (read as vanilla
/// parquet) rather than be transcoded into LiquidCache.
///
/// The scan's estimated liquid footprint is `raw × expansion × safety`, where
/// `raw` is [`estimate_required_bytes`]. It bypasses when that footprint exceeds
/// the [`admission_threshold`] (`memory × tolerance + disk`). The comparison is
/// done in finite `f64` space to avoid integer overflow.
fn should_bypass(
    cfg: &FileScanConfig,
    src: &ParquetSource,
    gate: AdmissionGate,
    memory_budget: u64,
    disk_budget: u64,
) -> bool {
    let est = estimate_required_bytes(cfg, src);
    // Multipliers are already sanitized to finite, >= 1.0 in `with_admission_gate`.
    let footprint = (est.raw_bytes as f64) * gate.expansion * gate.safety;
    let threshold = admission_threshold(memory_budget, disk_budget, gate.tolerance);
    let bypass = footprint > threshold;

    // One line per admission decision. Without this the gate is a black box and
    // every tuning cycle costs a benchmark run to infer decisions from side
    // effects. `fallback_files > 0` is the flag that the catalog has no
    // per-column sizes for this table (estimate is coarse / over-counts).
    let path = cfg
        .file_groups
        .iter()
        .flat_map(|g| g.files())
        .next()
        .map(|f| f.object_meta.location.as_ref())
        .unwrap_or("<none>");
    log::info!(
        target: "liquid_cache::admission",
        "admission {verdict}: file={path} projected_cols={cols} \
         charged_files={charged} partitioned_files={total} \
         fallback_files={fb} raw_bytes={raw} footprint_bytes={fp} \
         memory_bytes={mem} disk_bytes={disk} threshold_bytes={thr} \
         expansion={exp} safety={saf} tolerance={tol}",
        verdict = if bypass { "BYPASS" } else { "ADMIT" },
        cols = est.required_cols,
        charged = est.charged_files,
        total = est.partitioned_files,
        fb = est.fallback_files,
        raw = est.raw_bytes,
        fp = footprint as u64,
        mem = memory_budget,
        disk = disk_budget,
        thr = threshold as u64,
        exp = gate.expansion,
        saf = gate.safety,
        tol = gate.tolerance,
    );

    bypass
}

/// [`should_bypass`], with an optional panic guard.
///
/// The gate is a pure performance optimization: caching a scan or reading it as
/// vanilla parquet yields identical results. So if footprint estimation ever
/// panics — e.g. a DataFusion API that panics on an unusual plan shape, the
/// class of bug that `ordered_column_indices` was — a non-strict gate must not
/// let it abort the query.
///
/// Either way the panic is caught and logged at ERROR with its message (never
/// silently swallowed), and the log advises flipping the admission gate's
/// `strict` flag for the alternative behavior:
///
/// - `gate.strict == true`: log, then re-raise so the panic aborts the query and
///   the bug surfaces immediately. The log advises turning the flag *off* to
///   keep queries running while it is fixed.
/// - `gate.strict == false`: log, then cache the scan normally so the query
///   survives. The log advises turning the flag *on* to fail loud instead.
fn should_bypass_guarded(
    cfg: &FileScanConfig,
    src: &ParquetSource,
    gate: AdmissionGate,
    memory_budget: u64,
    disk_budget: u64,
) -> bool {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        should_bypass(cfg, src, gate, memory_budget, disk_budget)
    }));
    match result {
        Ok(bypass) => bypass,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("<non-string panic payload>");
            if gate.strict {
                log::error!(
                    "liquid-cache admission gate panicked during footprint estimation: \
                     {msg}. Aborting query (admission gate is in strict mode). Configure the \
                     admission gate with strict=false to fall back to caching and keep queries \
                     running while this is fixed."
                );
                std::panic::resume_unwind(payload);
            }
            log::error!(
                "liquid-cache admission gate panicked during footprint estimation: {msg}; \
                 caching scan normally. Configure the admission gate with strict=true to fail \
                 loud instead."
            );
            false
        }
    }
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
/// Uses per-column byte sizes that are either `Exact` or `Inexact`. DuckLake
/// records the real compressed on-disk column size but always labels it
/// `Inexact` (catalog stats can go stale after deletes/compaction), so
/// rejecting `Inexact` would make the gate fall back to the whole-file size on
/// *every* DuckLake scan — charging all columns for a single-column read and
/// bypassing everything. `Inexact` is a real measurement, not a guess; the
/// caller's `expansion`/`safety` margin absorbs modest drift, and even a large
/// stale-low under-count only risks a too-eager admit (perf), never wrong
/// results.
///
/// If the file has no stats, or any required column's size is `Absent`, fall
/// back to the whole-file size — a deliberate over-estimate.
///
/// Returns `(bytes, fell_back)`, where `fell_back` is `true` when the whole-file
/// over-estimate was used (surfaced in the decision log as the "no per-column
/// sizes in the catalog" signal).
fn file_required_bytes(
    stats: Option<&Statistics>,
    object_size: u64,
    required: &[usize],
) -> (u64, bool) {
    let Some(stats) = stats else {
        return (object_size, true);
    };
    let mut sum: u64 = 0;
    for &c in required {
        match stats.column_statistics.get(c).map(|cs| &cs.byte_size) {
            Some(Precision::Exact(n) | Precision::Inexact(n)) => {
                sum = sum.saturating_add(*n as u64)
            }
            _ => return (object_size, true),
        }
    }
    (sum, false)
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
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
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

    /// Regression: a `get_field` on a struct column is pushed into the scan
    /// projection as a non-`Column` expression (via
    /// `enable_leaf_expression_pushdown`). The footprint gate must estimate such
    /// a scan without panicking (the old code called the deprecated
    /// `ordered_column_indices`, which `.expect`s a bare column and killed the
    /// query). Runs everywhere: it builds a physical plan and calls the estimate
    /// directly, so it needs no cache / t4 mount.
    #[tokio::test]
    async fn estimate_survives_non_column_scan_projection() {
        use arrow::array::{ArrayRef, Int64Array, RecordBatch, StructArray};
        use arrow_schema::{DataType, Field, Fields};
        use datafusion::physical_expr::expressions::Column;
        use parquet::arrow::ArrowWriter;

        // A struct column `s {a, b}` plus a flat column `p`.
        let struct_fields = Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("s", DataType::Struct(struct_fields.clone()), false),
            Field::new("p", DataType::Int64, false),
        ]));
        let s = StructArray::new(
            struct_fields,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef,
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(s) as ArrayRef,
                Arc::new(Int64Array::from(vec![7, 8, 9])),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("structs.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let ctx = SessionContext::new();
        ctx.register_parquet("t", path.to_str().unwrap(), Default::default())
            .await
            .unwrap();
        let plan = ctx
            .sql("SELECT s.a FROM t")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();

        // Locate the parquet scan and assert its projection really does carry a
        // non-column expression — otherwise the test wouldn't exercise the bug.
        let mut parts = None;
        plan.apply(|node| {
            if let Some((cfg, src)) = parquet_scan_parts(node) {
                let has_expr = src
                    .projection()
                    .map(|p| p.iter().any(|e| e.expr.downcast_ref::<Column>().is_none()))
                    .unwrap_or(false);
                assert!(
                    has_expr,
                    "expected a non-column scan projection to exercise the gate; \
                     if this fails, DataFusion changed leaf-expression pushdown"
                );
                parts = Some((cfg.clone(), src.clone()));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        let (cfg, src) = parts.expect("no parquet scan in plan");

        // The call that used to panic. It must return a finite estimate; `s`
        // (the struct column read by `s.a`) is the one required file column.
        let est = estimate_required_bytes(&cfg, &src);
        // Whole-file fallback (no per-column Exact stats here) → non-zero, finite.
        assert!(est.raw_bytes > 0, "estimate should be a real byte count");
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
        build_cache_with_mem_disk(max_memory_bytes, 0).await
    }

    async fn build_cache_with_mem_disk(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
    ) -> LiquidCacheParquetRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = crate::test_utils::mount_test_store(tmp_dir.path()).await;
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                max_memory_bytes,
                max_disk_bytes,
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
            .with_admission_gate(1e9, 1.0, 1.0, false)
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&capped),
            "oversized estimate should stay a plain ParquetSource"
        );

        // With a large budget the footprint fits (expansion 1.0) → cached.
        let uncapped = LocalModeOptimizer::new(build_cache_with_budget(usize::MAX).await)
            .with_admission_gate(1.0, 1.0, 1.0, false)
            .optimize(plan, &config)
            .unwrap();
        assert!(
            has_liquid_source(&uncapped),
            "fitting scan should be wrapped in LiquidParquetSource"
        );
    }

    /// The budget counts the on-disk liquid tier, not just memory: the same scan
    /// that a memory-only budget bypasses is admitted once the disk tier has room
    /// for it (evicted entries spill to disk instead of thrashing).
    #[tokio::test]
    async fn test_admission_gate_counts_disk_tier() {
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

        // A huge expansion inflates the estimate past the 1 MB memory budget, and
        // with no disk tier there is nowhere else for it to fit → bypass.
        let mem_only = LocalModeOptimizer::new(build_cache_with_mem_disk(1_000_000, 0).await)
            .with_admission_gate(1e9, 1.0, 1.0, false)
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&mem_only),
            "with a memory-only budget the oversized estimate should bypass"
        );

        // Same 1 MB memory and same estimate, but now a large disk tier — the
        // budget is memory + disk, so the scan fits and is cached.
        let with_disk =
            LocalModeOptimizer::new(build_cache_with_mem_disk(1_000_000, usize::MAX).await)
                .with_admission_gate(1e9, 1.0, 1.0, false)
                .optimize(plan, &config)
                .unwrap();
        assert!(
            has_liquid_source(&with_disk),
            "the disk tier's capacity should count toward the budget and admit the scan"
        );
    }
}

/// Pure unit tests for the footprint byte-math (no cache / no t4 mount, so they
/// run everywhere). These cover the fallback direction: a column with an
/// `Absent` size (or a file with no stats) falls back to the whole-file size,
/// while `Exact`/`Inexact` per-column sizes are counted.
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
        assert_eq!(file_required_bytes(None, 5000, &[0, 1]), (5000, true));
    }

    #[test]
    fn sums_only_required_exact_columns() {
        let s = stats(vec![
            col(Precision::Exact(100)),
            col(Precision::Exact(200)),
            col(Precision::Exact(400)),
        ]);
        // required = cols 0 and 2 → 100 + 400, ignoring col 1. No fallback.
        assert_eq!(file_required_bytes(Some(&s), 9999, &[0, 2]), (500, false));
    }

    #[test]
    fn inexact_required_column_is_counted() {
        let s = stats(vec![
            col(Precision::Exact(100)),
            col(Precision::Inexact(200)),
        ]);
        // Inexact is a real (possibly-stale) size, so it counts (no fallback).
        // DuckLake always labels byte_size Inexact; rejecting it would bypass all.
        assert_eq!(file_required_bytes(Some(&s), 7000, &[0, 1]), (300, false));
        assert_eq!(file_required_bytes(Some(&s), 7000, &[1]), (200, false));
    }

    #[test]
    fn absent_among_sized_columns_falls_back_to_full_file() {
        let s = stats(vec![
            col(Precision::Exact(100)),
            col(Precision::Inexact(200)),
            col(Precision::Absent),
        ]);
        // Any Absent required column → whole file (fallback), even mixed with sized ones.
        assert_eq!(
            file_required_bytes(Some(&s), 8000, &[0, 1, 2]),
            (8000, true)
        );
        // Dropping the absent column, Exact + Inexact are summed (no fallback).
        assert_eq!(file_required_bytes(Some(&s), 8000, &[0, 1]), (300, false));
    }

    #[test]
    fn missing_required_column_falls_back_to_full_file() {
        let s = stats(vec![col(Precision::Exact(100))]);
        // required col 5 doesn't exist → conservative whole file (fallback).
        assert_eq!(file_required_bytes(Some(&s), 3000, &[0, 5]), (3000, true));
    }

    #[test]
    fn absent_byte_size_falls_back_to_full_file() {
        let s = stats(vec![col(Precision::Absent)]);
        assert_eq!(file_required_bytes(Some(&s), 2000, &[0]), (2000, true));
    }
}

/// Pure unit tests for the admission threshold arithmetic (no cache / no t4
/// mount, so they run everywhere — including where `direct_io` is unavailable).
#[cfg(test)]
mod threshold_tests {
    use super::admission_threshold;

    #[test]
    fn memory_carries_tolerance_disk_is_face_value() {
        // memory × tolerance + disk = 10×3 + 100 = 130. This pins the exact
        // formula: it is NOT (memory+disk)×tolerance (=330), max(mem,disk) (=100),
        // nor disk alone (=100).
        assert_eq!(admission_threshold(10, 100, 3.0), 130.0);
    }

    #[test]
    fn disk_off_is_memory_times_tolerance() {
        // With no disk tier the threshold is exactly the RAM budget × tolerance,
        // so gate behaviour matches the pre-disk model.
        assert_eq!(admission_threshold(1_000, 0, 3.0), 3_000.0);
    }

    #[test]
    fn disk_gets_no_overcommit() {
        // A pure-disk budget contributes at 1×, never scaled by tolerance.
        assert_eq!(admission_threshold(0, 500, 5.0), 500.0);
    }

    #[test]
    fn both_tiers_unsized_is_zero() {
        // Zero threshold => any non-empty scan bypasses (cache disabled/unsized).
        assert_eq!(admission_threshold(0, 0, 3.0), 0.0);
    }

    #[test]
    fn large_memory_times_tolerance_does_not_overflow() {
        // memory × tolerance is computed in f64, so a budget near u64::MAX cannot
        // wrap (the bug an integer `budget * tolerance` would have).
        let t = admission_threshold(u64::MAX, 0, 5.0);
        assert!(t > u64::MAX as f64);
        assert!(t.is_finite());
    }
}
