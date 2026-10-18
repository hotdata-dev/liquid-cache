//! Optimizers for the Parquet module

mod lineage;

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
        physical_plan::{
            FileScanConfig, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, ParquetRowSelection, RowGroupAccess},
        },
        source::DataSource,
        table_schema::TableSchema,
    },
    physical_expr::{PhysicalExpr, projection::ProjectionExprs, utils::collect_columns},
    physical_optimizer::{PhysicalOptimizerRule, pruning::PruningPredicateBuilder},
    physical_plan::ExecutionPlan,
};

pub(crate) use lineage::HintAnalyzer;
pub use lineage::LineageHints;

use crate::{
    LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnLineages, reader::TouchedBatches,
};

/// Default [`LocalModeOptimizer::with_sparse_selection_bypass_min_columns`]:
/// a scan reading one column besides its filter's (a histogram's time
/// column, say) is still served by the cache.
pub const DEFAULT_SPARSE_SELECTION_BYPASS_MIN_COLUMNS: usize = 2;

/// Default [`LocalModeOptimizer::with_sparse_selection_ratio`].
pub const DEFAULT_SPARSE_SELECTION_RATIO: f64 = 0.05;

/// Parameters for the footprint-based admission gate.
///
/// A parquet scan is routed through LiquidCache only if its estimated liquid
/// footprint stays within the admission threshold; larger scans are left as
/// vanilla parquet reads (which, if the object store is a cached mount, read
/// from it).
///
/// The threshold spans both cache tiers, weighted differently:
/// `memory × tolerance + disk`. A scan that overflows RAM spills to the on-disk
/// liquid tier rather than thrashing, so disk capacity counts toward what fits —
/// at face value, since only the RAM tier has the measured compaction
/// overcommit. With the disk tier off it is just `memory × tolerance`.
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
    /// When a scan's full projection is too big to admit but its pushed-down
    /// predicate's columns fit, admit it in predicate-columns-only mode instead
    /// of bypassing: the predicate columns are cached and the predicate runs
    /// on them, and the other projected columns are read from parquet for just
    /// the matched rows (never cached). Turns a selective wide scan's full read
    /// of its filter columns into a cache hit.
    pub predicate_only: bool,
}

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed lineage expressions from the full physical plan
/// (via the lineage analyzer) and attaches each scan's hints to its source.
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    prefetch: bool,
    /// When set, a scan whose estimated footprint exceeds the cache budget is
    /// left as a vanilla parquet read instead of being wrapped by LiquidCache.
    /// `None` means cache every scan.
    admission: Option<AdmissionGate>,
    /// See [`Self::with_sparse_selection_ratio`].
    sparse_selection_ratio: f64,
    /// See [`Self::with_sparse_selection_bypass`].
    sparse_selection_bypass: bool,
    /// See [`Self::with_sparse_selection_bypass_min_columns`].
    sparse_selection_bypass_min_columns: usize,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            prefetch: true,
            admission: None,
            sparse_selection_ratio: DEFAULT_SPARSE_SELECTION_RATIO,
            sparse_selection_bypass: false,
            sparse_selection_bypass_min_columns: DEFAULT_SPARSE_SELECTION_BYPASS_MIN_COLUMNS,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self::new(cache)
    }

    /// Enable or disable row-group prefetching.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
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
            predicate_only: true,
        });
        self
    }

    /// Enable or disable the gate's predicate-columns-only admission (see
    /// [`AdmissionGate::predicate_only`]); on by default. No-op without a gate.
    pub fn with_predicate_only_admission(mut self, enabled: bool) -> Self {
        if let Some(gate) = &mut self.admission {
            gate.predicate_only = enabled;
        }
        self
    }

    /// A scan whose files' row selections (a `ParquetAccessPlan` or
    /// `ParquetRowSelection` extension, e.g. from a secondary index) keep fewer
    /// than this fraction of its rows is *sparse*. A sparse scan is never
    /// admitted with its whole projection: a cache miss fills whole cache
    /// batches of every cached column, so a few hundred scattered rows would
    /// read and cache whole batches of wide columns. It is admitted
    /// predicate-only (the other columns are read for the selected rows only),
    /// or bypassed when it has no predicate to cache.
    ///
    /// Applies with or without the admission gate. `0.0` turns it off; values
    /// are clamped to `[0.0, 1.0]`, and a non-finite one means the default
    /// ([`DEFAULT_SPARSE_SELECTION_RATIO`]).
    pub fn with_sparse_selection_ratio(mut self, ratio: f64) -> Self {
        self.sparse_selection_ratio = if ratio.is_finite() {
            ratio.clamp(0.0, 1.0)
        } else {
            DEFAULT_SPARSE_SELECTION_RATIO
        };
        self
    }

    /// Bypass a wide sparse scan (see [`Self::with_sparse_selection_ratio`])
    /// instead of admitting it predicate-only: read it as vanilla parquet,
    /// whose access plan fetches only the selected rows' pages, rather than
    /// fetching each matched row's batch of every other column on its own.
    /// Wide means it projects at least
    /// [`Self::with_sparse_selection_bypass_min_columns`] columns its
    /// predicate does not read; a narrower sparse scan is admitted as before.
    /// Off by default.
    pub fn with_sparse_selection_bypass(mut self, enabled: bool) -> Self {
        self.sparse_selection_bypass = enabled;
        self
    }

    /// Columns beyond its predicate's a sparse scan must project to be
    /// bypassed (see [`Self::with_sparse_selection_bypass`]); default
    /// [`DEFAULT_SPARSE_SELECTION_BYPASS_MIN_COLUMNS`]. `0` bypasses every
    /// sparse scan.
    pub fn with_sparse_selection_bypass_min_columns(mut self, columns: usize) -> Self {
        self.sparse_selection_bypass_min_columns = columns;
        self
    }

    fn admission_policy(&self) -> AdmissionPolicy {
        AdmissionPolicy {
            gate: self.admission,
            memory_budget: self.cache.max_memory_bytes() as u64,
            disk_budget: self.cache.max_disk_bytes() as u64,
            cache_batch_size: self.cache.batch_size(),
            sparse_selection_ratio: self.sparse_selection_ratio,
            sparse_selection_bypass: self.sparse_selection_bypass,
            sparse_selection_bypass_min_columns: self.sparse_selection_bypass_min_columns,
        }
    }
}

/// Everything the admission decision needs besides the scan itself.
#[derive(Debug, Clone, Copy)]
struct AdmissionPolicy {
    gate: Option<AdmissionGate>,
    memory_budget: u64,
    disk_budget: u64,
    /// Rows per cache batch: misses fill whole batches, so a row selection's
    /// footprint is its touched batches, not its rows.
    cache_batch_size: usize,
    sparse_selection_ratio: f64,
    sparse_selection_bypass: bool,
    sparse_selection_bypass_min_columns: usize,
}

/// The admission gate's decision for one parquet scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    /// Cache the scan's whole projection (the normal liquid path).
    Admit,
    /// Cache only the predicate's columns; read the rest from parquet for the
    /// matched rows.
    AdmitPredicateOnly,
    /// Leave the scan as a vanilla parquet read.
    Bypass,
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        let analysis = HintAnalyzer::analyze(&plan);
        let cache = self.cache.clone();
        let prefetch = self.prefetch;
        // The gate sizes against both cache tiers, not just RAM: when a scan
        // overflows memory its entries spill to the on-disk liquid tier (NVMe)
        // instead of thrashing. They are weighted differently (see
        // `admission_threshold`) — RAM carries the compaction overcommit, disk
        // counts at face value — so they are passed through separately. With the
        // disk tier off (`max_disk_bytes == 0`) the threshold is exactly the RAM
        // budget, so gate behaviour is unchanged there.
        let policy = self.admission_policy();
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnLineages| {
            // Leave scans whose estimated liquid footprint exceeds the budget as
            // vanilla parquet reads, so oversized scans don't thrash the cache.
            let admission = match parquet_scan_parts(node) {
                Some((cfg, src)) => admission_guarded(cfg, src, policy),
                None => Admission::Admit,
            };
            match admission {
                Admission::Bypass => None,
                Admission::Admit => convert_parquet_scan(node, &cache, hints, prefetch, false),
                Admission::AdmitPredicateOnly => {
                    convert_parquet_scan(node, &cache, hints, prefetch, true)
                }
            }
        };
        Ok(lineage::rewrite_with_hints(plan, &mut convert, &analysis))
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
    hints: &ColumnLineages,
) -> Arc<dyn ExecutionPlan> {
    plan.transform_up(
        |node| match convert_parquet_scan(&node, cache, hints.clone(), true, false) {
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

/// Rewrite the data source plan to use liquid cache (no lineage expressions).
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
) -> Arc<dyn ExecutionPlan> {
    rewrite_data_source_plan_with_hints(plan, cache, &ColumnLineages::default())
}

/// If `node` is a parquet `DataSourceExec`, return its `FileScanConfig` and
/// `ParquetSource`.
fn parquet_scan_parts(node: &Arc<dyn ExecutionPlan>) -> Option<(&FileScanConfig, &ParquetSource)> {
    let dse = node.downcast_ref::<DataSourceExec>()?;
    dse.downcast_to_file_source::<ParquetSource>()
}

/// A scan's footprint estimate plus the breakdown behind it, for the admission
/// decision ([`admission`]) and its diagnostic log line.
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
    /// per-column size stat for this table, so the estimate is coarse and
    /// over-counts.
    fallback_files: usize,
    /// Charged files carrying a row selection (see [`FileRowCoverage`]).
    selected_files: usize,
    /// Rows the charged files' selections keep, over the rows they hold.
    /// `None` when some charged file's row count is unknown.
    selected_rows: Option<(u64, u64)>,
}

impl FootprintEstimate {
    /// Whether the files' row selections keep fewer than `ratio` of the rows.
    /// A scan without selections, or with unknown row counts, is not sparse.
    fn is_sparse(&self, ratio: f64) -> bool {
        match self.selected_rows {
            Some((selected, total)) if self.selected_files > 0 && total > 0 => {
                (selected as f64) < ratio * (total as f64)
            }
            _ => false,
        }
    }
}

/// How much of one file a scan reads, given the row selection attached to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileRowCoverage {
    /// Rows the selection keeps.
    selected: u64,
    /// Rows in the cache batches those rows fall in: what filling the cache
    /// for them reads.
    touched: u64,
    /// Rows in the file.
    total: u64,
    has_selection: bool,
}

impl FileRowCoverage {
    /// The coverage of `file`'s `ParquetAccessPlan` or `ParquetRowSelection`
    /// extension, or the whole file without one. `None` when the size cannot
    /// be told: no row count and a selection that does not imply one.
    fn of(file: &PartitionedFile, cache_batch_size: usize) -> Option<Self> {
        let num_rows = file
            .statistics
            .as_ref()
            .and_then(|stats| stats.num_rows.get_value().copied())
            .map(|rows| rows as u64);
        if let Some(plan) = file.extension::<ParquetAccessPlan>() {
            return Self::of_access_plan(plan, num_rows, cache_batch_size);
        }
        if let Some(selection) = file.extension::<ParquetRowSelection>() {
            // Spans row groups whose sizes are unknown here, so batches are
            // counted from the file start: an estimate.
            let selection = selection.selection();
            let rows: usize = selection.iter().map(|s| s.row_count).sum();
            let touched = TouchedBatches::new(selection, cache_batch_size)
                .touched_rows(cache_batch_size, rows);
            return Some(Self {
                selected: selection.row_count() as u64,
                touched: touched as u64,
                total: num_rows.unwrap_or(0).max(rows as u64),
                has_selection: true,
            });
        }
        num_rows.map(|rows| Self {
            selected: rows,
            touched: rows,
            total: rows,
            has_selection: false,
        })
    }

    /// A `Selection` row group carries its own size. `Scan` and `Skip` ones
    /// share what the file's row count leaves; without a row count, `Skip`
    /// groups count as empty (over-states the kept fraction, the conservative
    /// side) and a `Scan` group makes the coverage unknown.
    fn of_access_plan(
        plan: &ParquetAccessPlan,
        num_rows: Option<u64>,
        cache_batch_size: usize,
    ) -> Option<Self> {
        let (mut selected, mut touched, mut known) = (0u64, 0u64, 0u64);
        let (mut scan_groups, mut other_groups) = (0u64, 0u64);
        for access in plan.inner() {
            match access {
                RowGroupAccess::Selection(selection) => {
                    let rows: usize = selection.iter().map(|s| s.row_count).sum();
                    known += rows as u64;
                    selected += selection.row_count() as u64;
                    touched += TouchedBatches::new(selection, cache_batch_size)
                        .touched_rows(cache_batch_size, rows) as u64;
                }
                RowGroupAccess::Scan => scan_groups += 1,
                RowGroupAccess::Skip => other_groups += 1,
            }
        }
        let total = match num_rows {
            Some(rows) => {
                let rest = rows.saturating_sub(known);
                let scanned = rest
                    .saturating_mul(scan_groups)
                    .checked_div(scan_groups + other_groups)
                    .unwrap_or(0);
                selected += scanned;
                touched += scanned;
                rows.max(known)
            }
            None if scan_groups == 0 => known,
            None => return None,
        };
        Some(Self {
            selected,
            touched,
            total,
            has_selection: true,
        })
    }

    /// Scale `bytes` (the file's required-column bytes) to the cache batches
    /// the selection touches.
    fn scale(&self, bytes: u64) -> u64 {
        if !self.has_selection || self.total == 0 || self.touched >= self.total {
            return bytes;
        }
        ((bytes as u128 * self.touched as u128) / self.total as u128) as u64
    }
}

/// Estimate the raw required-column parquet bytes a scan reads: the sum, over
/// the files that survive the scan's predicate, of the byte sizes of the columns
/// it materializes. This is the byte-accurate, filter-aware size the admission
/// decision is built on (see [`admission`]); the expansion/safety/tolerance
/// factors are applied there, not here.
///
/// "Required columns" is the output projection **unioned with the predicate
/// columns**, since LiquidCache materializes both. Byte sizing uses `Exact` or
/// `Inexact` per-column sizes; a column with an `Absent` size or a file with no
/// stats falls back to the whole file size. The sum saturates rather than
/// overflowing on pathological file lists.
///
/// With `predicate_only`, only the predicate's columns count: the footprint of
/// a scan admitted in [`Admission::AdmitPredicateOnly`] mode.
///
/// A file carrying a row selection is charged only for the cache batches the
/// selection touches (see [`FileRowCoverage`]), with `cache_batch_size` rows
/// per batch.
fn estimate_required_bytes(
    cfg: &FileScanConfig,
    src: &ParquetSource,
    predicate_only: bool,
    cache_batch_size: usize,
) -> FootprintEstimate {
    let num_file_cols = cfg.file_schema().fields().len();
    // Full table schema (file + partition columns). The pushed-down predicate
    // and `PartitionedFile` stats are expressed against it, so file pruning must
    // use it; byte accounting still charges only file columns.
    let table_schema = cfg.file_source.table_schema().table_schema().clone();

    // Columns the scan projects. `column_indices()` collects the source columns
    // referenced by each projection expression, so it is correct for compound
    // projections (e.g. `a * b` reads a and b) and does not panic on a
    // non-column projection expression. Indices are table-schema-relative; keep
    // only file columns (partition columns are literals, never materialized).
    let mut required: Vec<usize> = match src.projection() {
        _ if predicate_only => Vec::new(),
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
    let mut selected_files = 0usize;
    let mut selected_rows = Some((0u64, 0u64));
    for (f, keep) in files.iter().zip(surviving.iter()) {
        if !*keep {
            continue;
        }
        if !seen.insert((f.object_meta.location.clone(), f.object_meta.size)) {
            continue;
        }
        charged_files += 1;
        let (mut bytes, fell_back) =
            file_required_bytes(f.statistics.as_deref(), f.object_meta.size, &required);
        if fell_back {
            fallback_files += 1;
        }
        let coverage = FileRowCoverage::of(f, cache_batch_size);
        if let Some(coverage) = coverage {
            bytes = coverage.scale(bytes);
            selected_files += usize::from(coverage.has_selection);
        }
        selected_rows = selected_rows
            .zip(coverage)
            .map(|((selected, total), coverage)| {
                (selected + coverage.selected, total + coverage.total)
            });
        raw_bytes = raw_bytes.saturating_add(bytes);
    }

    FootprintEstimate {
        raw_bytes,
        required_cols,
        charged_files,
        partitioned_files: files.len(),
        fallback_files,
        selected_files,
        selected_rows,
    }
}

/// Whether any file of the scan carries a row selection, checked before the
/// (costlier) footprint estimate so an ungated scan without one stays free.
fn has_row_selection(cfg: &FileScanConfig) -> bool {
    cfg.file_groups.iter().flat_map(|g| g.files()).any(|f| {
        f.extension::<ParquetAccessPlan>().is_some()
            || f.extension::<ParquetRowSelection>().is_some()
    })
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
///
/// A scan that would bypass is instead admitted predicate-only when
/// `gate.predicate_only` is set, it has a pushed-down predicate that reads
/// fewer columns than the whole scan, and the predicate columns' footprint
/// fits the same threshold.
///
/// A sparse scan (see [`LocalModeOptimizer::with_sparse_selection_ratio`]) is
/// never admitted whole while it projects columns its predicate does not
/// read: it is admitted predicate-only if that fits, else bypassed. Without a
/// gate every footprint fits and only that rule applies.
fn admission(cfg: &FileScanConfig, src: &ParquetSource, policy: AdmissionPolicy) -> Admission {
    if policy.gate.is_none() && (policy.sparse_selection_ratio <= 0.0 || !has_row_selection(cfg)) {
        return Admission::Admit;
    }
    let threshold = policy.gate.map_or(f64::INFINITY, |gate| {
        admission_threshold(policy.memory_budget, policy.disk_budget, gate.tolerance)
    });
    // Multipliers are already sanitized to finite, >= 1.0 in `with_admission_gate`.
    let (expansion, safety) = policy
        .gate
        .map_or((1.0, 1.0), |gate| (gate.expansion, gate.safety));
    let predicate_only_allowed = policy.gate.is_none_or(|gate| gate.predicate_only);
    let fits = |est: &FootprintEstimate| {
        let footprint = (est.raw_bytes as f64) * expansion * safety;
        (footprint, footprint <= threshold)
    };

    let est = estimate_required_bytes(cfg, src, false, policy.cache_batch_size);
    let sparse = est.is_sparse(policy.sparse_selection_ratio);
    let (footprint, full_fits) = fits(&est);
    let pred = src
        .filter()
        .map(|_| estimate_required_bytes(cfg, src, true, policy.cache_batch_size))
        .filter(|pred| pred.required_cols > 0);
    // The predicate reads every column the scan does: caching it whole caches
    // only predicate columns anyway.
    let pred_covers_scan = pred
        .as_ref()
        .is_some_and(|pred| pred.required_cols >= est.required_cols);

    // Columns the scan projects that its predicate does not read: what a
    // predicate-only read fetches per matched batch.
    let extra_cols = est
        .required_cols
        .saturating_sub(pred.as_ref().map_or(0, |pred| pred.required_cols));
    let (verdict, est, footprint) = if sparse
        && policy.sparse_selection_bypass
        && extra_cols >= policy.sparse_selection_bypass_min_columns
    {
        (Admission::Bypass, est, footprint)
    } else if full_fits && (!sparse || pred_covers_scan) {
        (Admission::Admit, est, footprint)
    } else {
        match pred {
            Some(pred) if predicate_only_allowed && !pred_covers_scan => {
                let (pred_footprint, pred_fits) = fits(&pred);
                if pred_fits {
                    (Admission::AdmitPredicateOnly, pred, pred_footprint)
                } else {
                    (Admission::Bypass, est, footprint)
                }
            }
            _ => (Admission::Bypass, est, footprint),
        }
    };

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
    let (selected, total) = est.selected_rows.unwrap_or((0, 0));
    log::info!(
        target: "liquid_cache::admission",
        "admission {verdict}: file={path} projected_cols={cols} \
         charged_files={charged} partitioned_files={total_files} \
         fallback_files={fb} selected_files={sel_files} selected_rows={selected} \
         total_rows={total} sparse={sparse} raw_bytes={raw} footprint_bytes={fp} \
         memory_bytes={mem} disk_bytes={disk} threshold_bytes={thr} \
         expansion={expansion} safety={safety} tolerance={tolerance} gated={gated}",
        verdict = match verdict {
            Admission::Admit => "ADMIT",
            Admission::AdmitPredicateOnly => "ADMIT_PREDICATE_ONLY",
            Admission::Bypass => "BYPASS",
        },
        cols = est.required_cols,
        charged = est.charged_files,
        total_files = est.partitioned_files,
        fb = est.fallback_files,
        sel_files = est.selected_files,
        raw = est.raw_bytes,
        fp = footprint as u64,
        mem = policy.memory_budget,
        disk = policy.disk_budget,
        thr = threshold.min(u64::MAX as f64) as u64,
        tolerance = policy.gate.map_or(1.0, |gate| gate.tolerance),
        gated = policy.gate.is_some(),
    );

    verdict
}

/// [`admission`], with an optional panic guard.
///
/// The gate is a pure performance optimization: caching a scan or reading it as
/// vanilla parquet yields identical results. So if footprint estimation ever
/// panics — e.g. a DataFusion API that panics on an unusual plan shape — a
/// non-strict gate must not let it abort the query.
///
/// Either way the panic is caught and logged at ERROR with its message (never
/// silently swallowed), and the log advises flipping the admission gate's
/// `strict` flag for the alternative behavior:
///
/// - `gate.strict == true`: log, then re-raise so the panic aborts the query and
///   the bug surfaces immediately. The log advises turning the flag *off* to
///   keep queries running while it is fixed.
/// - `gate.strict == false` (or no gate): log, then cache the scan normally so
///   the query survives. The log advises turning the flag *on* to fail loud
///   instead.
fn admission_guarded(
    cfg: &FileScanConfig,
    src: &ParquetSource,
    policy: AdmissionPolicy,
) -> Admission {
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| admission(cfg, src, policy)));
    match result {
        Ok(verdict) => verdict,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("<non-string panic payload>");
            if policy.gate.is_some_and(|gate| gate.strict) {
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
            Admission::Admit
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
    let pruning = match PruningPredicateBuilder::new()
        .with_file_schema(table_schema.clone())
        .try_build(pred)
    {
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
/// Uses per-column byte sizes that are either `Exact` or `Inexact`. A catalog
/// may record the real compressed on-disk column size and still label it
/// `Inexact` (stats can go stale after deletes/compaction), so rejecting
/// `Inexact` would make the gate fall back to the whole-file size on every such
/// scan — charging all columns for a single-column read and bypassing
/// everything. `Inexact` is a real measurement, not a guess; the caller's
/// `expansion`/`safety` margin absorbs modest drift, and even a large stale-low
/// under-count only risks a too-eager admit (perf), never wrong results.
///
/// If the file has no stats, or any required column's size is `Absent`, fall
/// back to the whole-file size — a deliberate over-estimate.
///
/// Returns `(bytes, fell_back)`, where `fell_back` is `true` when the whole-file
/// over-estimate was used.
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

/// The virtual columns this scan reads that the liquid path cannot produce.
///
/// The liquid read path has no notion of DataFusion's virtual columns. It carries
/// the [`TableSchema`] across faithfully but never produces one, so a scan that
/// reads a virtual column gets back a batch which simply lacks it, and a predicate
/// over one cannot be rewritten against the file schemas either. Such a scan has
/// to stay on `ParquetSource`, which derives virtual columns from the parquet
/// reader.
///
/// Declining a scan costs it the cache, so this stays narrow: a virtual column the
/// scan actually reads, not the mere presence of one on the table. A provider that
/// declares a row-position column on every table keeps the cache for the queries
/// that never project one. No projection at all is the one broad case, and it is
/// not a guess — the scan then reads the whole table schema, virtual columns
/// included.
///
/// Positional reads are what reaches here: applying positional deletes, and row
/// lineage, both project a reader-produced physical row position, which is an
/// absolute index into the file. A plausible but shifted position would associate
/// a delete with the wrong row, so declining is the sound answer while the liquid
/// reader cannot generate positions itself.
fn unproducible_virtual_columns(
    table_schema: &TableSchema,
    projection: Option<&ProjectionExprs>,
    filter: Option<&Arc<dyn PhysicalExpr>>,
) -> Option<String> {
    let virtual_columns = table_schema.virtual_columns();
    if virtual_columns.is_empty() {
        return None;
    }

    let mut needed: Vec<&str> = Vec::new();
    match projection {
        None => needed.extend(virtual_columns.iter().map(|field| field.name().as_str())),
        Some(projection) => {
            let mut read: HashSet<String> = projection
                .expr_iter()
                .flat_map(|expr| collect_columns(&expr))
                .map(|column| column.name().to_string())
                .collect();
            // The pushed-down predicate is rewritten against the file schemas in
            // the reader too, so a conjunct over a virtual column fails exactly as
            // a projection over one does. The row filter's own check cannot catch
            // it: that resolves against the table schema, which does hold the
            // virtual columns.
            if let Some(filter) = filter {
                read.extend(
                    collect_columns(filter)
                        .into_iter()
                        .map(|column| column.name().to_string()),
                );
            }
            needed.extend(
                virtual_columns
                    .iter()
                    .map(|field| field.name().as_str())
                    .filter(|name| read.contains(*name)),
            );
        }
    }

    if needed.is_empty() {
        return None;
    }
    Some(needed.join("`, `"))
}

/// If `node` is a `DataSourceExec` over a `ParquetSource`, return an equivalent
/// node backed by [`LiquidParquetSource`] carrying `hints`.
fn convert_parquet_scan(
    node: &Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: ColumnLineages,
    prefetch: bool,
    cache_predicate_columns_only: bool,
) -> Option<Arc<dyn ExecutionPlan>> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, parquet_source) =
        data_source_exec.downcast_to_file_source::<ParquetSource>()?;

    let pushed_filter = parquet_source.filter();
    if let Some(names) = unproducible_virtual_columns(
        parquet_source.table_schema(),
        parquet_source.projection(),
        pushed_filter.as_ref(),
    ) {
        // At info, like the admission gate's BYPASS line: this silently turns the
        // cache off for a scan, and the only symptom is that queries stop getting
        // faster.
        log::info!(
            "liquid_cache scan BYPASS: the read path cannot produce virtual column(s) `{names}`"
        );
        return None;
    }

    let new_source =
        LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone())
            .with_lineages(Arc::new(hints))
            .with_prefetch(prefetch)
            .with_cache_predicate_columns_only(cache_predicate_columns_only);

    let mut new_config = file_scan_config.clone();
    new_config.file_source = Arc::new(new_source);
    let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
    Some(Arc::new(DataSourceExec::new(new_file_source)))
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use arrow::{array::Int32Array, record_batch::RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        common::{ScalarValue, stats::Precision},
        datasource::physical_plan::{FileScanConfig, FileSource},
        logical_expr::Operator,
        physical_expr::expressions::{BinaryExpr, Column, Literal},
        physical_plan::{
            PhysicalExpr, collect, display::DisplayableExecutionPlan, filter_pushdown::PushedDown,
        },
        prelude::SessionContext,
    };
    use liquid_cache::{
        cache::{AlwaysHydrate, TranscodeEvict},
        cache_policies::LiquidPolicy,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use crate::LiquidCacheParquet;

    use super::*;

    async fn make_cache(path: &Path) -> LiquidCacheParquetRef {
        make_cache_with_mem_disk(path, 1000000, usize::MAX).await
    }

    /// The admission gate reads both tier budgets off the cache, so its tests
    /// need them set independently rather than via [`make_cache`]'s defaults.
    async fn make_cache_with_mem_disk(
        path: &Path,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
    ) -> LiquidCacheParquetRef {
        let store = t4::mount(path.join("liquid_cache.t4")).await.unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                max_memory_bytes,
                max_disk_bytes,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeEvict),
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

    async fn nano_hits_plan() -> Arc<dyn ExecutionPlan> {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        ctx.sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// Declining a scan costs it the cache, so the guard must key on what the scan
    /// reads, not on what the table declares. A row-position column present on the
    /// table but absent from the projection keeps the cache; no projection at all
    /// reads the whole table schema and does not.
    #[test]
    fn only_a_virtual_column_the_scan_reads_costs_the_cache() {
        use arrow_schema::Fields;
        use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
        use datafusion::physical_expr::projection::ProjectionExpr;

        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int64, true),
        ]));
        let row_pos = Field::new("__ducklake_row_pos", DataType::Int64, true);

        // No virtual columns at all: nothing to refuse, whatever the projection.
        let plain = TableSchema::builder(Arc::clone(&file_schema)).build();
        assert_eq!(unproducible_virtual_columns(&plain, None, None), None);

        let positional = TableSchema::builder(Arc::clone(&file_schema))
            .with_virtual_columns(Fields::from(vec![row_pos.clone()]))
            .build();
        let full = positional.table_schema();

        // Declared but not read: still cached.
        let only_id =
            ProjectionExprs::new(vec![ProjectionExpr::new(col("id", full).unwrap(), "id")]);
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&only_id), None),
            None
        );

        // Read: refused, and named.
        let reads_pos = ProjectionExprs::new(vec![ProjectionExpr::new(
            col("__ducklake_row_pos", full).unwrap(),
            "__ducklake_row_pos",
        )]);
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&reads_pos), None).as_deref(),
            Some("__ducklake_row_pos")
        );

        // No projection reads the whole table schema, virtual columns included.
        assert_eq!(
            unproducible_virtual_columns(&positional, None, None).as_deref(),
            Some("__ducklake_row_pos")
        );

        // Read only by the pushed-down predicate: refused too. The reader rewrites
        // the predicate against the file schemas as well, and the row filter's own
        // check passes it because that resolves against the table schema.
        let pos_predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("__ducklake_row_pos", full).unwrap(),
            Operator::Gt,
            lit(0i64),
        ));
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&only_id), Some(&pos_predicate))
                .as_deref(),
            Some("__ducklake_row_pos")
        );
    }

    /// The guard at its call site, with an ordinary scan as a positive control, so
    /// deleting it from `convert_parquet_scan` fails a test instead of silently
    /// restoring a plan that cannot execute.
    #[tokio::test]
    async fn a_scan_reading_a_virtual_column_stays_on_parquet_source() {
        use arrow_schema::Fields;
        use datafusion::datasource::physical_plan::FileScanConfigBuilder;
        use datafusion::execution::object_store::ObjectStoreUrl;

        let file_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let scan = |table_schema: TableSchema| -> Arc<dyn ExecutionPlan> {
            let source = Arc::new(ParquetSource::new(table_schema)) as Arc<dyn FileSource>;
            let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
                .with_file(PartitionedFile::new("t.parquet", 16))
                .build();
            Arc::new(DataSourceExec::new(Arc::new(config)))
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = make_cache(tmp_dir.path()).await;

        // Positive control: an ordinary scan is still handed to the cache.
        let plain = scan(TableSchema::builder(Arc::clone(&file_schema)).build());
        assert!(
            convert_parquet_scan(&plain, &cache, ColumnLineages::default(), true, false).is_some(),
            "an ordinary scan must still convert to the liquid source"
        );

        // `ParquetSource::new` projects the whole table schema, so the positional
        // scan reads the row-position column and cannot be served.
        let positional = scan(
            TableSchema::builder(file_schema)
                .with_virtual_columns(Fields::from(vec![Field::new(
                    "__ducklake_row_pos",
                    DataType::Int64,
                    true,
                )]))
                .build(),
        );
        assert!(
            convert_parquet_scan(&positional, &cache, ColumnLineages::default(), true, false)
                .is_none(),
            "a scan reading a virtual column must stay on ParquetSource"
        );
    }

    /// The admission gate bypasses a scan whose estimated footprint exceeds the
    /// budget (large expansion here forces that), and caches one that fits.
    #[tokio::test]
    async fn admission_gate_bypasses_an_oversized_scan() {
        let plan = nano_hits_plan().await;
        let config = ConfigOptions::default();
        let capped_dir = tempfile::tempdir().unwrap();
        let uncapped_dir = tempfile::tempdir().unwrap();

        // A huge expansion inflates the estimate past the 1 MB budget → bypass.
        let capped = LocalModeOptimizer::new(
            make_cache_with_mem_disk(capped_dir.path(), 1_000_000, 0).await,
        )
        .with_admission_gate(1e9, 1.0, 1.0, false)
        .optimize(plan.clone(), &config)
        .unwrap();
        assert!(
            !has_liquid_source(&capped),
            "oversized estimate should stay a plain ParquetSource"
        );

        // With a large budget the footprint fits (expansion 1.0) → cached.
        let uncapped = LocalModeOptimizer::new(
            make_cache_with_mem_disk(uncapped_dir.path(), usize::MAX, 0).await,
        )
        .with_admission_gate(1.0, 1.0, 1.0, false)
        .optimize(plan, &config)
        .unwrap();
        assert!(
            has_liquid_source(&uncapped),
            "fitting scan should be wrapped in LiquidParquetSource"
        );
    }

    /// Sum of the named metric over every node in `plan`.
    fn sum_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        let mut total = 0;
        plan.apply(|node| {
            if let Some(metrics) = node.metrics() {
                total += metrics.sum_by_name(name).map(|v| v.as_usize()).unwrap_or(0);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        total
    }

    /// The production shape: `SELECT * ... WHERE user_id = ? ORDER BY ts DESC
    /// LIMIT n` over a table whose wide column makes the whole projection too
    /// big to admit, while the filter column fits. The gate must admit it
    /// predicate-only rather than bypass, the rows must match vanilla parquet,
    /// the filter column must be served from the cache, and the wide column
    /// must be read from parquet for the matched rows only and never cached.
    #[tokio::test]
    async fn wide_selective_scan_is_admitted_predicate_only() {
        use arrow::array::{Int64Array, StringArray};
        use datafusion::prelude::SessionConfig;

        const ROWS: usize = 4 * 8192;
        const USERS: usize = 4093;
        // Wide-ish ids so the filter column, like production's, costs real bytes.
        let user = |i: usize| format!("user-{:0>58}", i % USERS);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("ts", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from_iter_values((0..ROWS).map(user))),
                Arc::new(Int64Array::from_iter_values((0..ROWS).map(|i| i as i64))),
                Arc::new(StringArray::from_iter_values(
                    (0..ROWS).map(|i| format!("{i:0>256}")),
                )),
            ],
        )
        .unwrap();
        // Small pages so the payload read can skip pages without a match.
        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(128)
            .set_write_batch_size(128)
            .set_dictionary_enabled(false)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let session_config =
            SessionConfig::new().set_bool("datafusion.execution.parquet.pushdown_filters", true);
        let ctx = SessionContext::new_with_config(session_config);
        ctx.register_parquet("t", path.to_str().unwrap(), Default::default())
            .await
            .unwrap();
        let sql = format!(
            "SELECT * FROM t WHERE user_id = '{}' ORDER BY ts DESC LIMIT 500",
            user(7)
        );
        let sql = sql.as_str();
        let expected = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let expected_rows: usize = expected.iter().map(|b| b.num_rows()).sum();
        assert_eq!(expected_rows, (0..ROWS).filter(|i| i % USERS == 7).count());

        // Bytes vanilla parquet reads for this query, the bypass baseline.
        let vanilla_plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        collect(Arc::clone(&vanilla_plan), ctx.task_ctx())
            .await
            .unwrap();
        let vanilla_bytes = sum_metric(&vanilla_plan, "bytes_scanned");

        // A budget between the filter column's footprint and the whole scan's.
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let (cfg, src) = {
            let mut parts = None;
            plan.apply(|node| {
                if let Some((cfg, src)) = parquet_scan_parts(node) {
                    parts = Some((cfg.clone(), src.clone()));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
            parts.expect("parquet scan in plan")
        };
        let full = estimate_required_bytes(&cfg, &src, false, 8192);
        let pred = estimate_required_bytes(&cfg, &src, true, 8192);
        assert_eq!(
            full.fallback_files + pred.fallback_files,
            0,
            "per-column sizes expected"
        );
        assert!(pred.raw_bytes * 4 < full.raw_bytes, "{pred:?} vs {full:?}");
        let budget = ((pred.raw_bytes + full.raw_bytes) / 2) as usize;

        let cache = make_cache_with_mem_disk(dir.path(), budget, 0).await;
        let optimizer =
            LocalModeOptimizer::new(Arc::clone(&cache)).with_admission_gate(1.0, 1.0, 1.0, true);
        let config = ConfigOptions::default();

        // Kill switch: with predicate-only admission off, the scan bypasses.
        let off = LocalModeOptimizer::new(Arc::clone(&cache))
            .with_admission_gate(1.0, 1.0, 1.0, true)
            .with_predicate_only_admission(false)
            .optimize(plan.clone(), &config)
            .unwrap();
        assert!(
            !has_liquid_source(&off),
            "full footprint exceeds the budget"
        );

        let to_strings = |batches: &[RecordBatch]| {
            arrow::util::pretty::pretty_format_batches(batches)
                .unwrap()
                .to_string()
        };
        // The scan's metrics set is shared by every plan optimized from `plan`,
        // so per-run bytes are deltas.
        let mut bytes_before = 0;
        let mut last_bytes = 0;
        // Run twice: the first fills the cache, the second reads it.
        for _ in 0..2 {
            let liquid = optimizer.optimize(plan.clone(), &config).unwrap();
            assert!(liquid_source(&liquid).cache_predicate_columns_only());
            let got = collect(Arc::clone(&liquid), ctx.task_ctx()).await.unwrap();
            assert_eq!(to_strings(&got), to_strings(&expected));
            assert!(sum_metric(&liquid, "passthrough_batches") > 0);
            let bytes_after = sum_metric(&liquid, "bytes_scanned");
            last_bytes = bytes_after - bytes_before;
            bytes_before = bytes_after;
        }

        // The filter column is resident; the wide payload never was.
        let file = cache.register_or_get_file(
            crate::cache::ParquetFileIdentity::new(
                cfg.object_store_url.clone(),
                cfg.file_groups[0].files()[0]
                    .object_meta
                    .location
                    .to_string(),
            ),
            cfg.file_schema().clone(),
        );
        let row_group = file.create_row_group(0, vec![]);
        let user_id = row_group.get_column_by_name("user_id").unwrap();
        let payload = row_group.get_column_by_name("payload").unwrap();
        for batch in 0..(ROWS / 8192) as u16 {
            let batch = crate::cache::BatchID::from_raw(batch);
            assert!(
                user_id.contains(batch),
                "user_id batch {batch:?} not cached"
            );
            assert!(
                !payload.contains(batch),
                "payload batch {batch:?} was cached"
            );
        }

        // Warm, parquet serves only the matched rows' pages of the other
        // columns; vanilla also reads the whole filter column.
        assert!(
            last_bytes * 4 < vanilla_bytes,
            "warm predicate-only read {last_bytes} vs vanilla {vanilla_bytes}"
        );
    }

    /// Rows per cache batch in [`make_cache_with_mem_disk`] caches.
    const CACHE_BATCH: usize = 8192;
    const NEEDLE_ROWS: usize = 4 * CACHE_BATCH;

    /// Sorts between two ordinary ids.
    const NEEDLE: &str = "user-0000000000000000000000000000000000000000000000000000002000x";

    /// Rows of the needle file whose `user_id` is [`NEEDLE`]: ten in cache batch
    /// 0 and ten in batch 2, so a selection of them touches two of four batches.
    fn needle_rows() -> Vec<usize> {
        (100..110)
            .chain(2 * CACHE_BATCH + 50..2 * CACHE_BATCH + 60)
            .collect()
    }

    /// `user_id`, `ts`, and a wide `payload`, in 128-row pages with the page
    /// index, as one row group.
    fn write_needle_file(path: &Path) {
        use arrow::array::{Int64Array, StringArray};

        let needles = needle_rows();
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("ts", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        // Scattered ids, so every page's min/max spans the needle and page
        // statistics cannot do the index's job.
        let user = |i: usize| {
            if needles.contains(&i) {
                NEEDLE.to_string()
            } else {
                format!("user-{:0>58}", (i * 7919) % 4093)
            }
        };
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from_iter_values((0..NEEDLE_ROWS).map(user))),
                Arc::new(Int64Array::from_iter_values(
                    (0..NEEDLE_ROWS).map(|i| i as i64),
                )),
                Arc::new(StringArray::from_iter_values(
                    (0..NEEDLE_ROWS).map(|i| format!("{i:0>256}")),
                )),
            ],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(128)
            .set_write_batch_size(128)
            .set_dictionary_enabled(false)
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// The index's view of the needle rows: one row group, selecting them.
    fn needle_access_plan() -> ParquetAccessPlan {
        use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

        let mut selectors = Vec::new();
        let mut at = 0;
        for row in needle_rows() {
            if row > at {
                selectors.push(RowSelector::skip(row - at));
            }
            selectors.push(RowSelector::select(1));
            at = row + 1;
        }
        selectors.push(RowSelector::skip(NEEDLE_ROWS - at));
        ParquetAccessPlan::new(vec![RowGroupAccess::Selection(RowSelection::from(
            selectors,
        ))])
    }

    struct NeedleTable {
        ctx: SessionContext,
        dir: tempfile::TempDir,
        file_size: usize,
    }

    impl NeedleTable {
        async fn new() -> Self {
            use datafusion::prelude::SessionConfig;

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("needle.parquet");
            write_needle_file(&path);
            let file_size = std::fs::metadata(&path).unwrap().len() as usize;
            let config = SessionConfig::new()
                .with_target_partitions(1)
                .set_bool("datafusion.execution.parquet.pushdown_filters", true);
            let ctx = SessionContext::new_with_config(config);
            ctx.register_parquet("t", path.to_str().unwrap(), Default::default())
                .await
                .unwrap();
            Self {
                ctx,
                dir,
                file_size,
            }
        }

        async fn plan(&self, sql: &str) -> Arc<dyn ExecutionPlan> {
            self.ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap()
        }

        async fn run(&self, plan: Arc<dyn ExecutionPlan>) -> (String, usize) {
            let batches = collect(Arc::clone(&plan), self.ctx.task_ctx())
                .await
                .unwrap();
            let rows = arrow::util::pretty::pretty_format_batches(&batches)
                .unwrap()
                .to_string();
            (rows, sum_metric(&plan, "bytes_scanned"))
        }

        async fn cache(&self, name: &str) -> LiquidCacheParquetRef {
            let dir = self.dir.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            make_cache_with_mem_disk(&dir, usize::MAX, 0).await
        }
    }

    /// Attach `extension` to every file of every parquet scan in `plan`.
    fn with_file_extension<T: Clone + Send + Sync + 'static>(
        plan: Arc<dyn ExecutionPlan>,
        extension: T,
    ) -> Arc<dyn ExecutionPlan> {
        use datafusion::datasource::physical_plan::FileGroup;

        plan.transform_up(|node| {
            let Some((cfg, _)) = parquet_scan_parts(&node) else {
                return Ok(Transformed::no(node));
            };
            let mut cfg = cfg.clone();
            cfg.file_groups = cfg
                .file_groups
                .iter()
                .map(|group| {
                    FileGroup::new(
                        group
                            .files()
                            .iter()
                            .map(|f| f.clone().with_extension(extension.clone()))
                            .collect(),
                    )
                })
                .collect();
            let node: Arc<dyn ExecutionPlan> = Arc::new(DataSourceExec::new(Arc::new(cfg)));
            Ok(Transformed::yes(node))
        })
        .unwrap()
        .data
    }

    /// Force one liquid path on every parquet scan of `plan`, bypassing the gate.
    fn force_liquid(
        plan: Arc<dyn ExecutionPlan>,
        cache: &LiquidCacheParquetRef,
        predicate_only: bool,
    ) -> Arc<dyn ExecutionPlan> {
        plan.transform_up(|node| {
            Ok(
                match convert_parquet_scan(
                    &node,
                    cache,
                    ColumnLineages::default(),
                    true,
                    predicate_only,
                ) {
                    Some(node) => Transformed::yes(node),
                    None => Transformed::no(node),
                },
            )
        })
        .unwrap()
        .data
    }

    const NEEDLE_FILTERED: &str = "SELECT * FROM t WHERE user_id = \
        'user-0000000000000000000000000000000000000000000000000000002000x'";
    const NEEDLE_UNFILTERED: &str = "SELECT * FROM t";

    /// A sparse access plan gives the same rows as no access plan on every
    /// path (vanilla parquet, liquid admitted, liquid predicate-only), with the
    /// filter kept and with it dropped. Cold, each liquid path reads only the
    /// touched cache batches: half the file when it caches every column, and
    /// a small fraction when it caches only the filter column. Warm, the
    /// selection is served without re-reading the cached columns.
    #[tokio::test]
    async fn sparse_access_plan_matches_on_every_path() {
        let table = NeedleTable::new().await;
        let (expected, _) = table.run(table.plan(NEEDLE_FILTERED).await).await;
        assert_eq!(expected.matches(NEEDLE).count(), needle_rows().len());
        let (_, scan_bytes) = table.run(table.plan(NEEDLE_UNFILTERED).await).await;
        assert!(
            scan_bytes * 10 > table.file_size * 9,
            "a full scan reads the file"
        );

        for sql in [NEEDLE_FILTERED, NEEDLE_UNFILTERED] {
            let selected =
                || async { with_file_extension(table.plan(sql).await, needle_access_plan()) };

            let (rows, _) = table.run(selected().await).await;
            assert_eq!(rows, expected, "bypass, {sql}");

            for predicate_only in [false, true] {
                let cache = table
                    .cache(&format!("{predicate_only}-{}", sql.len()))
                    .await;
                let mut bytes = Vec::new();
                for _ in 0..2 {
                    let plan = force_liquid(selected().await, &cache, predicate_only);
                    assert!(has_liquid_source(&plan));
                    let (rows, read) = table.run(plan).await;
                    assert_eq!(rows, expected, "predicate_only={predicate_only}, {sql}");
                    bytes.push(read);
                }
                let (cold, warm) = (bytes[0], bytes[1]);
                let context = format!(
                    "predicate_only={predicate_only} {sql}: cold {cold} warm {warm} of {}",
                    table.file_size
                );
                // Two of four cache batches; whole-row-group re-reads put this
                // at 1.5x the file.
                assert!(cold * 10 < table.file_size * 6, "{context}");
                if predicate_only && sql == NEEDLE_FILTERED {
                    // The filter column for two batches, the rest for 20 rows.
                    assert!(cold * 4 < table.file_size, "{context}");
                }
                // Warm reads at most the uncached columns' selected pages.
                assert!(warm * 20 < table.file_size, "{context}");
            }
        }
    }

    /// The same rows from a file-wide `ParquetRowSelection` as from the
    /// equivalent access plan, on the liquid path, and an error when a file
    /// carries both rather than a silent pick.
    #[tokio::test]
    async fn file_row_selection_is_honoured() {
        let table = NeedleTable::new().await;
        let (expected, _) = table.run(table.plan(NEEDLE_FILTERED).await).await;
        // One row group, so its selection is the file's.
        let selection = ParquetRowSelection::new(overall_needle_selection());
        let cache = table.cache("row-selection").await;
        let plan = force_liquid(
            with_file_extension(table.plan(NEEDLE_UNFILTERED).await, selection.clone()),
            &cache,
            false,
        );
        let (rows, _) = table.run(plan).await;
        assert_eq!(rows, expected);

        let both = force_liquid(
            with_file_extension(
                with_file_extension(table.plan(NEEDLE_UNFILTERED).await, selection),
                needle_access_plan(),
            ),
            &cache,
            false,
        );
        let err = collect(both, table.ctx.task_ctx()).await.unwrap_err();
        assert!(err.to_string().contains("not both"), "{err}");
    }

    fn overall_needle_selection() -> parquet::arrow::arrow_reader::RowSelection {
        match &needle_access_plan().inner()[0] {
            RowGroupAccess::Selection(selection) => selection.clone(),
            _ => unreachable!(),
        }
    }

    /// The gate's verdict for one scan of `plan`.
    fn verdict(plan: Arc<dyn ExecutionPlan>, optimizer: LocalModeOptimizer) -> Option<bool> {
        let optimized = optimizer.optimize(plan, &ConfigOptions::default()).unwrap();
        has_liquid_source(&optimized)
            .then(|| liquid_source(&optimized).cache_predicate_columns_only())
    }

    /// A sparse selection is cached predicate-only with a filter and bypassed
    /// without one, gated or not; the ratio's kill switch and a dense
    /// selection keep today's whole-projection admission.
    #[tokio::test]
    async fn sparse_selection_is_never_admitted_whole() {
        let table = NeedleTable::new().await;
        let cache = table.cache("gate").await;
        let gated =
            || LocalModeOptimizer::new(Arc::clone(&cache)).with_admission_gate(1.0, 1.0, 1.0, true);
        let ungated = || LocalModeOptimizer::new(Arc::clone(&cache));
        let filtered = with_file_extension(table.plan(NEEDLE_FILTERED).await, needle_access_plan());
        let unfiltered =
            with_file_extension(table.plan(NEEDLE_UNFILTERED).await, needle_access_plan());

        // Some(true) = predicate-only, Some(false) = admitted whole, None = bypass.
        for optimizer in [&gated as &dyn Fn() -> LocalModeOptimizer, &ungated] {
            assert_eq!(verdict(filtered.clone(), optimizer()), Some(true));
            assert_eq!(verdict(unfiltered.clone(), optimizer()), None);
            let off = optimizer().with_sparse_selection_ratio(0.0);
            assert_eq!(verdict(filtered.clone(), off), Some(false));
            // 20 of 32768 rows; a ratio below that is not sparse.
            let low = optimizer().with_sparse_selection_ratio(0.0001);
            assert_eq!(verdict(filtered.clone(), low), Some(false));
        }
        // The predicate-only kill switch leaves a gated sparse scan nowhere to go.
        let no_pred_only = gated().with_predicate_only_admission(false);
        assert_eq!(verdict(filtered.clone(), no_pred_only), None);
        // Without a selection nothing changes.
        assert_eq!(
            verdict(table.plan(NEEDLE_FILTERED).await, gated()),
            Some(false)
        );
        assert_eq!(
            verdict(table.plan(NEEDLE_UNFILTERED).await, ungated()),
            Some(false)
        );
    }

    /// With sparse-selection bypass on, a wide sparse scan (two or more
    /// columns beyond its predicate's) is read as vanilla parquet, gated or
    /// not; a narrow one keeps the cache's path, and a dense one is
    /// unaffected. Each answers as vanilla parquet does.
    #[tokio::test]
    async fn sparse_selection_bypass_bypasses_only_wide_sparse_scans() {
        const NEEDLE: &str = "'user-0000000000000000000000000000000000000000000000000000002000x'";
        let wide = format!("SELECT * FROM t WHERE user_id = {NEEDLE}");
        // A histogram's shape: the filter column and one more.
        let narrow = format!("SELECT user_id, ts FROM t WHERE user_id = {NEEDLE}");
        let filter_only = format!("SELECT user_id FROM t WHERE user_id = {NEEDLE}");
        let table = NeedleTable::new().await;
        let cache = table.cache("bypass").await;
        let gated = || {
            LocalModeOptimizer::new(Arc::clone(&cache))
                .with_admission_gate(1.0, 1.0, 1.0, true)
                .with_sparse_selection_bypass(true)
        };
        let ungated =
            || LocalModeOptimizer::new(Arc::clone(&cache)).with_sparse_selection_bypass(true);
        let t = &table;
        let selected = |sql: String| async move {
            with_file_extension(t.plan(&sql).await, needle_access_plan())
        };
        let unfiltered = selected(NEEDLE_UNFILTERED.to_string()).await;
        // Some(true) = predicate-only, Some(false) = admitted whole, None = bypass.
        for optimizer in [&gated as &dyn Fn() -> LocalModeOptimizer, &ungated] {
            assert_eq!(verdict(selected(wide.clone()).await, optimizer()), None);
            assert_eq!(verdict(unfiltered.clone(), optimizer()), None);
            assert_eq!(
                verdict(selected(narrow.clone()).await, optimizer()),
                Some(true)
            );
            assert_eq!(
                verdict(selected(filter_only.clone()).await, optimizer()),
                Some(false)
            );
            // Zero columns: every sparse scan bypasses.
            let all = optimizer().with_sparse_selection_bypass_min_columns(0);
            assert_eq!(verdict(selected(narrow.clone()).await, all), None);
            // 20 of 32768 rows; a ratio below that is not sparse.
            let low = optimizer().with_sparse_selection_ratio(0.0001);
            assert_eq!(verdict(selected(wide.clone()).await, low), Some(false));
        }
        assert_eq!(verdict(table.plan(&wide).await, ungated()), Some(false));
        // Same rows on the cached and the bypassed path as on vanilla parquet.
        for sql in [&wide, &narrow, &filter_only] {
            let (vanilla, _) = table.run(selected(sql.clone()).await).await;
            let optimized = gated()
                .optimize(selected(sql.clone()).await, &ConfigOptions::default())
                .unwrap();
            let (rows, _) = table.run(optimized).await;
            assert_eq!(rows, vanilla, "{sql}");
            assert!(vanilla.contains("2000x"), "{vanilla}");
        }
    }

    /// The footprint charges a selected file for the cache batches its
    /// selection touches, not the whole file.
    #[tokio::test]
    async fn estimate_scales_to_touched_batches() {
        let table = NeedleTable::new().await;
        let parts = |plan: Arc<dyn ExecutionPlan>| {
            let mut parts = None;
            plan.apply(|node| {
                if let Some((cfg, src)) = parquet_scan_parts(node) {
                    parts = Some((cfg.clone(), src.clone()));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
            parts.unwrap()
        };
        let (cfg, src) = parts(table.plan(NEEDLE_FILTERED).await);
        let full = estimate_required_bytes(&cfg, &src, false, CACHE_BATCH);
        let (cfg, src) = parts(with_file_extension(
            table.plan(NEEDLE_FILTERED).await,
            needle_access_plan(),
        ));
        let selected = estimate_required_bytes(&cfg, &src, false, CACHE_BATCH);
        assert_eq!(full.fallback_files, 0);
        assert_eq!(selected.raw_bytes, full.raw_bytes / 2);
        assert_eq!(selected.selected_rows, Some((20, NEEDLE_ROWS as u64)));
        assert!(selected.is_sparse(0.05));
        assert!(!full.is_sparse(0.05));
    }

    /// The budget counts the on-disk liquid tier, not just memory: the same scan
    /// that a memory-only budget bypasses is admitted once the disk tier has room
    /// for it (evicted entries spill to disk instead of thrashing).
    #[tokio::test]
    async fn admission_gate_counts_the_disk_tier() {
        let plan = nano_hits_plan().await;
        let config = ConfigOptions::default();
        let mem_only_dir = tempfile::tempdir().unwrap();
        let with_disk_dir = tempfile::tempdir().unwrap();

        // A huge expansion inflates the estimate past the 1 MB memory budget, and
        // with no disk tier there is nowhere else for it to fit → bypass.
        let mem_only = LocalModeOptimizer::new(
            make_cache_with_mem_disk(mem_only_dir.path(), 1_000_000, 0).await,
        )
        .with_admission_gate(1e9, 1.0, 1.0, false)
        .optimize(plan.clone(), &config)
        .unwrap();
        assert!(
            !has_liquid_source(&mem_only),
            "with a memory-only budget the oversized estimate should bypass"
        );

        // Same 1 MB memory and same estimate, but now a large disk tier — the
        // threshold is memory × tolerance + disk, so the scan fits and is cached.
        let with_disk = LocalModeOptimizer::new(
            make_cache_with_mem_disk(with_disk_dir.path(), 1_000_000, usize::MAX).await,
        )
        .with_admission_gate(1e9, 1.0, 1.0, false)
        .optimize(plan, &config)
        .unwrap();
        assert!(
            has_liquid_source(&with_disk),
            "the disk tier's capacity should count toward the budget and admit the scan"
        );
    }

    /// A scan projection can hold a non-column expression (`SELECT s.a` pushes a
    /// struct field access into the scan). The estimator must read it without
    /// panicking — the class of bug that made the gate's panic guard necessary.
    #[tokio::test]
    async fn estimate_survives_non_column_scan_projection() {
        use arrow::array::{ArrayRef, Int64Array, StructArray};
        use arrow_schema::Fields;

        let struct_fields = Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Struct(struct_fields.clone()), false),
            Field::new("p", DataType::Int64, false),
        ]));
        let struct_array = StructArray::new(
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
                Arc::new(struct_array) as ArrayRef,
                Arc::new(Int64Array::from(vec![7, 8, 9])),
            ],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("structs.parquet");
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
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
        let est = estimate_required_bytes(&cfg, &src, false, 8192);
        assert!(est.raw_bytes > 0, "estimate should be a real byte count");
    }

    fn liquid_source(plan: &Arc<dyn ExecutionPlan>) -> LiquidParquetSource {
        let mut source = None;
        plan.apply(|node| {
            if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                let config = plan.data_source().downcast_ref::<FileScanConfig>().unwrap();
                source = Some(
                    config
                        .file_source()
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap()
                        .clone(),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        source.unwrap()
    }

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let expected_schema = plan.schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let liquid_cache = make_cache(tmp_dir.path()).await;
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

        rewritten
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
        let rewritten = rewrite_plan_inner(plan).await;

        let displayed = DisplayableExecutionPlan::new(rewritten.as_ref())
            .indent(true)
            .to_string();
        assert!(displayed.contains("predicate="), "{displayed}");

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let statistics = plan.data_source().partition_statistics(None)?;
                    assert!(!matches!(statistics.num_rows, Precision::Exact(_)));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        // Supported filters are conjoined onto the predicate; unsupported ones
        // are handed back to the parent.
        let source = liquid_source(&rewritten);
        let url_index = source.table_schema().file_schema().index_of("URL").unwrap();
        let supported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("URL", url_index)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "https://example.com".into(),
            )))),
        ));
        let unsupported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("missing", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("value".into())))),
        ));
        let result = source
            .try_pushdown_filters(
                vec![supported, unsupported],
                &datafusion::config::ConfigOptions::new(),
            )
            .unwrap();
        assert!(matches!(
            result.filters.as_slice(),
            [PushedDown::Yes, PushedDown::No]
        ));
        let predicate = result.updated_node.unwrap().filter().unwrap().to_string();
        assert!(predicate.contains(" AND "), "{predicate}");
        assert!(predicate.contains("https://example.com"), "{predicate}");
        assert!(!predicate.contains("missing"), "{predicate}");
    }

    fn write_bloom_file(path: &Path) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let properties = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            schema.clone(),
            Some(properties),
        )
        .unwrap();
        for values in [[1, 2, 4], [1, 3, 4]] {
            writer
                .write(
                    &RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(values.to_vec()))],
                    )
                    .unwrap(),
                )
                .unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn prunes_row_group_with_bloom_filter() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("bloom.parquet");
        write_bloom_file(&parquet_path);

        let ctx = SessionContext::new();
        ctx.register_parquet("t", parquet_path.to_str().unwrap(), Default::default())
            .await
            .unwrap();
        let plan = ctx
            .sql("SELECT * FROM t WHERE a = 2")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let cache = make_cache(tmp_dir.path()).await;
        let rewritten = rewrite_data_source_plan(plan, &cache);
        let metrics = liquid_source(&rewritten).metrics().clone();

        let batches = collect(rewritten, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

        let metric = metrics
            .clone_inner()
            .sum_by_name("row_groups_pruned_bloom_filter")
            .unwrap();
        let datafusion::physical_plan::metrics::MetricValue::PruningMetrics {
            pruning_metrics, ..
        } = metric
        else {
            panic!("unexpected metric: {metric:?}");
        };
        assert_eq!(pruning_metrics.pruned(), 1);
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
        // A catalog that always labels byte_size Inexact would otherwise make
        // every scan fall back to the whole file and bypass the cache.
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
/// mount, so they run everywhere).
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
