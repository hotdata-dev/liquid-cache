use arrow_schema::{DataType, Field, Schema};
use liquid_cache::{
    cache::{
        CacheStats,
        squeeze_policies::{Evict, SqueezePolicy, TranscodeEvict, TranscodeSqueezeEvict},
    },
    cache_policies::LiquidPolicy,
};
use liquid_cache_datafusion::LiquidCacheParquetRef;
use std::{fmt, path::Path, sync::Arc};
use tempfile::TempDir;

use arrow::util::pretty::pretty_format_batches;
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    },
    error::Result,
    physical_plan::{ExecutionPlan, collect, display::DisplayableExecutionPlan},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

use crate::LiquidCacheLocalBuilder;
mod batch_size_alignment;
mod column_free_conjunct;
mod date_optimizer;
mod filter_limit;
mod nested_filter;
mod page_index;
mod squeeze;
mod unevaluable_conjunct;
mod variants;

const TEST_FILE: &str = "../../examples/nano_hits.parquet";
const OPENOBSERVE_FILE: &str = "../../dev/test_parquet/openobserve.parquet";

#[derive(Debug, Clone)]
struct QueryOutcome {
    values: String,
    plan: String,
    stats: CacheStatsSummary,
}

#[derive(Debug, Clone)]
struct CacheStatsSummary {
    stats: CacheStats,
    entries_after_first_run: usize,
}

impl CacheStatsSummary {
    fn from_stats(stats: CacheStats, entries_after_first_run: usize) -> Self {
        Self {
            stats,
            entries_after_first_run,
        }
    }

    fn has_cache_hits(&self) -> bool {
        let runtime = &self.stats.runtime;
        runtime.get_with_selection > 0
            || runtime.try_read_liquid_calls > 0
            || runtime.get > 0
            || runtime.eval_predicate > 0
    }

    fn entries_reused(&self) -> bool {
        self.stats.total_entries == self.entries_after_first_run
    }
}

impl fmt::Display for CacheStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "entries.total: {}", self.stats.total_entries)?;
        writeln!(
            f,
            "entries.after_first_run: {}",
            self.entries_after_first_run
        )?;
        writeln!(
            f,
            "entries.memory.arrow: {}",
            self.stats.memory_arrow_entries
        )?;
        writeln!(
            f,
            "entries.memory.liquid: {}",
            self.stats.memory_liquid_entries
        )?;
        writeln!(
            f,
            "entries.memory.squeezed_liquid: {}",
            self.stats.memory_squeezed_liquid_entries
        )?;
        writeln!(f, "entries.disk.liquid: {}", self.stats.disk_liquid_entries)?;
        writeln!(f, "entries.disk.arrow: {}", self.stats.disk_arrow_entries)?;
        writeln!(f, "usage.memory_bytes: {}", self.stats.memory_usage_bytes)?;
        writeln!(f, "usage.disk_bytes: {}", self.stats.disk_usage_bytes)?;
        // Use the Display implementation for runtime stats
        write!(f, "{}", self.stats.runtime)
    }
}

/// Session config for the tests that read [`TEST_FILE`] and pin cache traces,
/// entry counts or IO counts.
///
/// DataFusion 55 lowered `repartition_file_min_size` from 10 MiB to 1 MiB, so the
/// 2.3 MB test file is now split into one scan partition per `target_partitions`
/// instead of being read by a single one. Several scan partitions hit the shared
/// cache concurrently, which makes admission and eviction order — and with it
/// every trace and byte count these tests assert — depend on scheduling and on
/// the host's core count. Raise the threshold back above the file size so the
/// scan stays single-partition and the snapshots stay reproducible.
pub(super) fn cache_test_config() -> SessionConfig {
    let mut config = SessionConfig::new();
    config.options_mut().optimizer.repartition_file_min_size = 16 * 1024 * 1024;
    config
}

async fn create_session_context_with_liquid_cache(
    squeeze_policy: Box<dyn SqueezePolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> Result<(SessionContext, LiquidCacheParquetRef)> {
    // These tests snapshot exact cache contents and counters. A repartitioned
    // file scan populates the cache concurrently, so insertion order (and, for
    // LIMIT queries, which partitions finish before cancellation) is not a
    // stable property to snapshot.
    let mut config = SessionConfig::new().with_repartition_file_scans(false);
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_prefetch(false)
        .with_max_memory_bytes(cache_size_bytes)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_squeeze_policy(squeeze_policy)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await?;

    // Register the test parquet file
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    Ok((ctx, cache))
}

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn get_result(ctx: &SessionContext, sql: &str) -> String {
    let plan = get_physical_plan(sql, ctx).await;
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

async fn run_io_profile(prefetch: bool, cache_dir: &Path) -> (String, u64, u64, u64) {
    let config = SessionConfig::new().with_repartition_file_scans(false);
    let builder = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(64 * 1024 * 1024)
        .with_cache_dir(cache_dir.to_path_buf());
    let builder = if prefetch {
        builder
    } else {
        builder.with_prefetch(false)
    };
    let (ctx, cache) = builder.build(config).await.unwrap();
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();
    let sql = r#"SELECT "WatchID" FROM hits WHERE "SearchPhrase" LIKE '%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%'"#;

    let first = get_result(&ctx, sql).await;
    cache.flush_data().await.unwrap();
    cache.storage().stats();
    let second = get_result(&ctx, sql).await;
    let runtime = cache.storage().stats().runtime;
    assert_eq!(first, second);

    (
        second,
        runtime.read_io_count,
        runtime.get,
        runtime.eval_predicate,
    )
}

#[tokio::test]
async fn prefetch_matches_lazy_io() {
    let lazy_dir = TempDir::new().unwrap();
    let prefetch_dir = TempDir::new().unwrap();

    let lazy = run_io_profile(false, lazy_dir.path()).await;
    let prefetch = run_io_profile(true, prefetch_dir.path()).await;

    assert_eq!(lazy, prefetch);
}

async fn run_sql_with_cache(
    sql: &str,
    squeeze_policy: Box<dyn SqueezePolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> QueryOutcome {
    let (ctx, cache) =
        create_session_context_with_liquid_cache(squeeze_policy, cache_size_bytes, cache_dir)
            .await
            .unwrap();

    let plan = get_physical_plan(sql, &ctx).await;
    let displayable = DisplayableExecutionPlan::new(plan.as_ref());
    let plan_string = format!("{}", displayable.tree_render());

    // Clear any historical runtime counters before prefetching the cache.
    cache.storage().stats();

    let first_run = get_result(&ctx, sql).await;
    let entries_after_first_run = cache.storage().stats().total_entries;
    let second_run = get_result(&ctx, sql).await;

    assert_eq!(first_run, second_run);

    let stats_after_second_run = cache.storage().stats();
    let stats = CacheStatsSummary::from_stats(stats_after_second_run, entries_after_first_run);

    QueryOutcome {
        values: second_run,
        plan: plan_string,
        stats,
    }
}

async fn test_runner(sql: &str, reference: &str, cache_dir: &Path) {
    let cache_sizes = [10 * 1024, 1024 * 1024, usize::MAX]; // 10KB, 1MB, unlimited

    for cache_size in cache_sizes {
        let squeeze_policies: Vec<Box<dyn SqueezePolicy>> = vec![
            Box::new(TranscodeSqueezeEvict),
            Box::new(Evict),
            Box::new(TranscodeEvict),
        ];
        for squeeze_policy in squeeze_policies {
            let QueryOutcome { values, .. } =
                run_sql_with_cache(sql, squeeze_policy, cache_size, cache_dir).await;
            assert_eq!(
                values, reference,
                "Results differ, cache_size: {cache_size}"
            );
        }
    }
}

#[tokio::test]
async fn test_url_prefix_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_url_selection_and_ordering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 300,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_os_selection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_referer_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_single_column_filter_projection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID" from hits where "WatchID" = 6978470580070504163"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    test_runner(sql, &reference, cache_dir.path()).await;
}

/// Runs on x86_64 only, because the snapshot pins `usage.memory_bytes` exactly
/// and aarch64 reports 935 bytes less at every one of the three measurement
/// points (1000915 -> 999980, 1036304 -> 1035369), reproducibly.
///
/// The split is by architecture, not by OS. Measured:
///
/// | target              | usage.memory_bytes |
/// |---------------------|--------------------|
/// | x86_64-linux        | 1000915 (recorded) |
/// | aarch64-linux       | 999980             |
/// | aarch64-darwin      | 999980             |
///
/// aarch64-linux and aarch64-darwin agree exactly, so the OS is not the
/// variable — gating on `target_os` would still fail on Graviton or on any ARM
/// Linux runner.
///
/// The whole delta is the FSST-compressed payload — `RawFsstBuffer::values.len()`.
/// Componentwise, everything else is byte-identical across the two architectures
/// (the arrow entries, the fastlanes bit-packed dictionary keys at 17504, the
/// prefix keys, the compact offsets, the struct sizes, and the 537585 bytes of
/// uncompressed FSST input). Only the compressed output moves: 254655 on aarch64
/// against 255590 on x86_64.
///
/// Cause: `fsst-rs` 0.5.11 drains a hash map of symbol candidates into a
/// `BinaryHeap` (`builder.rs:796`), and `Candidate`'s ordering key is just
/// `(gain, symbol.len())` (`builder.rs:835-837`) — the symbol bytes are excluded.
/// Two distinct symbols with equal gain and equal length therefore compare
/// `Equal`, so which one wins is decided by heap insertion order, i.e. hash-map
/// iteration order, which is not stable across architectures (hashbrown selects
/// an SSE2, NEON or generic probe implementation per target). Different
/// tie-break, different 255-symbol table, different compressed length. The real
/// fix belongs upstream: make `Candidate`'s ordering total by including the
/// symbol bytes as a final tie-breaker.
///
/// Note this means liquid-encoded bytes are NOT identical across architectures —
/// `to_bytes` writes `values` verbatim — so compression ratios and capacity
/// figures do not transfer between arm64 and x86_64. `usage.disk_bytes` staying
/// at 35000 is not evidence against that; the two disk-resident entries are
/// different, much smaller columns than the one that moves.
///
/// The byte-exact snapshot therefore runs on x86_64 only, keeping its full
/// strength and the `cargo insta` workflow there. Everywhere else the test still
/// runs and bounds the same figures to within 1% — see the bottom of this test.
#[tokio::test]
async fn test_provide_schema2() {
    use std::fmt::Write as _;

    let cache_dir = TempDir::new().unwrap();
    let df_ctx = SessionContext::new();
    let mut config = cache_test_config();
    config.options_mut().execution.target_partitions = 4;
    let (liquid_ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_prefetch(false)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_memory_bytes(1024 * 1024)
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(config)
        .await
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");
    let table_path = ListingTableUrl::parse(OPENOBSERVE_FILE).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("_timestamp", DataType::Int64, false),
        Field::new("log", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, true),
        Field::new("kubernetes_namespace_name", DataType::Utf8, false),
    ]));

    df_ctx
        .register_listing_table(
            "default",
            &table_path,
            listing_options.clone(),
            Some(schema.clone()),
            None,
        )
        .await
        .unwrap();
    liquid_ctx
        .register_listing_table("default", &table_path, listing_options, Some(schema), None)
        .await
        .unwrap();

    let queries = [
        "SELECT * from default where log like '%hhj%' order by _timestamp",
        "SELECT date_bin(interval '10 second', to_timestamp_micros(_timestamp), to_timestamp('2001-01-01T00:00:00')) AS zo_sql_key, count(*) AS zo_sql_num from default WHERE log like '%hhj%' or message like '%hhj%' GROUP BY zo_sql_key ORDER BY zo_sql_key",
        "SELECT _timestamp, kubernetes_namespace_name from default order by _timestamp desc limit 100",
    ];

    let mut snapshot = String::new();

    for (idx, sql) in queries.iter().enumerate() {
        let df_results = df_ctx.sql(sql).await.unwrap().collect().await.unwrap();

        let plan = get_physical_plan(sql, &liquid_ctx).await;
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_string = format!("{}", displayable.tree_render());

        // Reset runtime counters so we measure hits from the prefetch run onwards.
        cache.storage().stats();

        let first_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), first_liquid_run[0].columns());

        let entries_after_first_run = cache.storage().stats().total_entries;
        let second_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), second_liquid_run[0].columns());

        let stats = CacheStatsSummary::from_stats(cache.storage().stats(), entries_after_first_run);

        assert!(stats.has_cache_hits());
        assert!(stats.entries_reused());

        writeln!(snapshot, "query[{idx}]: {sql}").unwrap();
        writeln!(snapshot, "plan: \n{}", plan_string).unwrap();
        writeln!(snapshot, "stats:\n{}", stats).unwrap();

        if idx + 1 != queries.len() {
            snapshot.push('\n');
        }
    }

    // FSST breaks equal-gain symbol ties using target-specific HashMap iteration order.
    // Canonicalize the known AArch64 totals to x86_64 while leaving unexpected totals visible.
    #[cfg(target_arch = "aarch64")]
    let snapshot = snapshot
        .replace("usage.memory_bytes: 999980", "usage.memory_bytes: 1000915")
        .replace("usage.memory_bytes: 1035369", "usage.memory_bytes: 1036304");

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    insta::assert_snapshot!(snapshot);

    // On any other target the byte-exact snapshot cannot match, because FSST
    // picks a different symbol table (see above) and only the x86_64/aarch64
    // totals are known. Bound the figures instead of skipping the test: what this
    // test covers besides the plan text — the DataFusion-vs-liquid column
    // equality, the cache hits, the tier split, the `Utf8`-declared schema over a
    // `string_view` file — is architecture-independent and worth running.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    assert_memory_bytes_within_1pct(
        &snapshot,
        include_str!("snapshots/liquid_cache_datafusion_local__tests__provide_schema2.snap"),
    );
}

/// Checks each `usage.memory_bytes` line in `snapshot` against the figure the
/// committed x86_64 snapshot records for the same query, allowing 1%.
///
/// `recorded` is the `.snap` file itself rather than a hand-copied array, so
/// regenerating the snapshot on x86_64 with `cargo insta accept` cannot leave
/// this assertion silently checking stale numbers.
///
/// The known architecture difference is ~0.1% (935 bytes in ~1 MiB), so 1% has an
/// order of magnitude of headroom while still catching the kind of regression that
/// matters — a buffer counted twice, or a tier accounted at the wrong size.
#[cfg(not(target_arch = "x86_64"))]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn assert_memory_bytes_within_1pct(snapshot: &str, recorded: &str) {
    let actual = memory_bytes(snapshot);
    let expected = memory_bytes(recorded);

    // Both sides are parsed with the same predicate, so a changed prefix would
    // empty both and make the length check below pass on nothing.
    assert!(
        !expected.is_empty(),
        "found no `usage.memory_bytes` readings in the committed snapshot; the \
         stats format has changed and this assertion is no longer reading anything"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "expected {} memory_bytes readings, found {}: {actual:?}",
        expected.len(),
        actual.len()
    );

    for (idx, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let drift = actual.abs_diff(expected);
        assert!(
            drift * 100 <= expected,
            "query[{idx}]: memory_bytes {actual} is more than 1% from the x86_64 \
             figure {expected} (off by {drift}); the architecture difference should \
             be ~0.1%, so this is a real accounting change"
        );
    }
}

/// Pulls every `usage.memory_bytes` figure out of a stats snapshot, in order.
///
/// Works on both the live snapshot and a committed `.snap` file: insta writes the
/// snapshot body unindented after its YAML header, so the same prefix matches.
#[cfg(not(target_arch = "x86_64"))]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn memory_bytes(snapshot: &str) -> Vec<u64> {
    snapshot
        .lines()
        .filter_map(|line| line.strip_prefix("usage.memory_bytes: "))
        .map(|value| value.trim().parse().expect("memory_bytes must be a number"))
        .collect()
}

#[tokio::test]
async fn test_provide_schema_with_filter() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        cache_dir.path(),
    )
    .await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::assert_snapshot!(format!(
        "plan: \n{}\nvalues: \n{}\nstats:\n{}",
        plan, values, stats
    ));

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build(cache_test_config())
        .await
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

    let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet").unwrap();
    let schema = Schema::new(vec![
        Field::new("WatchID", DataType::Int64, true),
        Field::new("EventTime", DataType::Int64, true),
        Field::new("OS", DataType::Int16, true),
    ]);

    ctx.register_listing_table(
        "hits",
        &table_path,
        listing_options.clone(),
        Some(Arc::new(schema)),
        None,
    )
    .await
    .unwrap();

    let results = ctx.sql(sql).await.unwrap().collect().await.unwrap();

    let formatted_results = pretty_format_batches(&results).unwrap().to_string();
    if formatted_results != reference {
        println!("formatted_results: \n{formatted_results}");
        println!("reference: \n{reference}");
    }
    assert_eq!(formatted_results, reference);
}

/// Covers the multi-partition scan path against the shared cache.
///
/// The tests above disable file-scan repartitioning outright so
/// the scan stays single-partition and their traces stay reproducible. That is
/// is deliberate, but it also means nothing else exercises several scan
/// partitions admitting into one cache concurrently — which is exactly what a
/// default DataFusion 55 deployment does for any file over 1 MiB, since DF 55
/// lowered the threshold from 10 MiB to 1 MiB.
///
/// So this test leaves `repartition_file_min_size` at the DF 55 default and
/// asserts only order-independent properties: the result rows compared as a
/// sorted multiset (against a single-partition run of the same query), and that
/// the warm run hits the cache. No trace, byte count or row order is pinned, so
/// it cannot reintroduce the snapshot flakiness the pin defends against.
#[tokio::test]
async fn test_multi_partition_scan_shares_cache() {
    /// Rows as an order-independent multiset.
    async fn sorted_rows(ctx: &SessionContext, sql: &str) -> Vec<String> {
        let plan = get_physical_plan(sql, ctx).await;
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let mut rows = pretty_format_batches(&batches)
            .unwrap()
            .to_string()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        rows.sort();
        rows
    }

    async fn build_ctx(
        config: SessionConfig,
        cache_dir: &Path,
    ) -> (SessionContext, LiquidCacheParquetRef) {
        let (ctx, cache) = LiquidCacheLocalBuilder::new()
            .with_max_memory_bytes(1024 * 1024)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build(config)
            .await
            .unwrap();
        ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
            .await
            .unwrap();
        (ctx, cache)
    }

    let sql = r#"select "OS", COUNT(*) from hits where "URL" like '%tours%' group by "OS""#;

    // Multi-partition: DF 55 default `repartition_file_min_size` (1 MiB) against
    // the 2.3 MB test file, so `target_partitions` really does split the scan.
    let multi_dir = TempDir::new().unwrap();
    let mut multi_config = SessionConfig::new();
    multi_config.options_mut().execution.target_partitions = 4;
    let (multi_ctx, cache) = build_ctx(multi_config, multi_dir.path()).await;

    // Guard the premise: if a future default makes the scan single-partition
    // again, this test would silently stop covering concurrent admission.
    let scan_partitions = {
        let mut node = get_physical_plan(sql, &multi_ctx).await;
        while let Some(child) = node.children().first() {
            node = Arc::clone(child);
        }
        node.properties().partitioning.partition_count()
    };
    assert!(
        scan_partitions > 1,
        "expected a multi-partition scan, got {scan_partitions}"
    );

    // Clear historical counters, then warm the cache and read it back.
    cache.storage().stats();
    let first_run = sorted_rows(&multi_ctx, sql).await;
    let entries_after_first_run = cache.storage().stats().total_entries;
    let second_run = sorted_rows(&multi_ctx, sql).await;
    let stats = CacheStatsSummary::from_stats(cache.storage().stats(), entries_after_first_run);

    assert_eq!(first_run, second_run);
    assert!(
        stats.has_cache_hits(),
        "warm multi-partition run did not read from the cache"
    );

    // Same answer as the single-partition path the snapshot tests pin.
    let single_dir = TempDir::new().unwrap();
    let mut single_config = cache_test_config();
    single_config.options_mut().execution.target_partitions = 4;
    let (single_ctx, _single_cache) = build_ctx(single_config, single_dir.path()).await;
    assert_eq!(sorted_rows(&single_ctx, sql).await, second_run);
}

#[tokio::test]
async fn test_repartitioned_file_scan_cache_correctness() {
    let reference_cache_dir = TempDir::new().unwrap();
    let parallel_cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let reference = run_sql_with_cache(
        sql,
        Box::new(TranscodeSqueezeEvict),
        1024 * 1024,
        reference_cache_dir.path(),
    )
    .await
    .values;

    // DataFusion 55 lowered repartition_file_min_size from 10 MiB to 1 MiB,
    // which splits the 2.3 MiB fixture into four concurrent scan partitions.
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(1024 * 1024)
        .with_cache_dir(parallel_cache_dir.path().to_path_buf())
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    let plan = get_physical_plan(sql, &ctx).await;
    let plan = format!(
        "{}",
        DisplayableExecutionPlan::new(plan.as_ref()).tree_render()
    );
    assert!(
        plan.contains("files: 4"),
        "expected a repartitioned scan:\n{plan}"
    );

    assert_eq!(get_result(&ctx, sql).await, reference);
    let entries_after_first_run = cache.storage().stats().total_entries;
    assert_eq!(get_result(&ctx, sql).await, reference);

    let stats = cache.storage().stats();
    assert!(stats.runtime.get_with_selection > 0);
    assert!(stats.total_entries >= entries_after_first_run);
}
