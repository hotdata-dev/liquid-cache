//! Regression tests for `NOT IN` (null-aware anti join) over cached scans.
//!
//! `HashJoinExec` publishes a filter built from its build-side keys
//! (`key >= min AND key <= max AND key IN (..)`) and pushes it into the
//! probe-side scan. A null-aware anti join cannot survive that: its answer
//! depends on what the probe side *contains*, and the filter prunes exactly
//! the rows that carry the signal — NULL satisfies no comparison, and a probe
//! side holding no matching keys prunes away to nothing and reads as empty.
//!
//! So `a NOT IN (SELECT b ..)` returned the rows SQL tri-state semantics
//! require it to drop. The bug is in DataFusion's join, not in the cache — a
//! plain parquet scan is equally wrong, and a `MemTable` (which accepts no
//! pushdown) is the only thing that answers correctly — but it lands on every
//! LiquidCache query, so `NullAwareJoinDynamicFilterGuard` detaches the filter
//! from null-aware joins. See
//! <https://github.com/hotdata-dev/liquid-cache/issues/16>.
//!
//! The expected values below are plain SQL semantics, not a differential
//! against vanilla DataFusion, because vanilla-over-parquet shares the bug.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::{ExecutionPlan, joins::HashJoinExec};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// Outer table `t0`, one nullable `a` column.
fn write_outer(dir: &Path, vals: &[Option<i64>]) {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    write(
        &dir.join("t0.parquet"),
        schema,
        vec![Arc::new(Int64Array::from(vals.to_vec()))],
    );
}

/// Subquery table `t1`, a nullable `a` keyed by the `s` tag the filter selects on.
fn write_sub(dir: &Path, vals: &[(Option<i64>, &str)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let a = Int64Array::from(vals.iter().map(|(a, _)| *a).collect::<Vec<_>>());
    let s = StringArray::from(vals.iter().map(|(_, s)| *s).collect::<Vec<_>>());
    write(
        &dir.join("t1.parquet"),
        schema,
        vec![Arc::new(a), Arc::new(s)],
    );
}

fn write(path: &Path, schema: Arc<Schema>, cols: Vec<Arc<dyn Array>>) {
    let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A LiquidCache context over `t0`/`t1` in `dir`.
async fn liquid_ctx(dir: &Path) -> SessionContext {
    let cache_dir = dir.join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir)
        .build(config)
        .await
        .unwrap();
    for t in ["t0", "t1"] {
        ctx.register_parquet(
            t,
            dir.join(format!("{t}.parquet")).to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
    }
    ctx
}

/// The outer `a` values the query returned, sorted, with NULL as `None`.
async fn result(ctx: &SessionContext, sql: &str) -> Vec<Option<i64>> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out: Vec<Option<i64>> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            (0..col.len())
                .map(move |i| (!col.is_null(i)).then(|| col.value(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    out
}

/// Asserts `sql` on both the cold and the warm (cache-populated) run, so a
/// correct answer that only survives one of them still fails.
async fn assert_result(ctx: &SessionContext, sql: &str, expected: &[Option<i64>]) {
    for run in ["cold", "warm"] {
        assert_eq!(result(ctx, sql).await, expected, "{run} run: {sql}");
    }
}

const NOT_IN: &str =
    "SELECT a FROM t0 WHERE t0.a NOT IN (SELECT t1.a FROM t1 WHERE t1.s LIKE 'keep%')";

/// The reported bug. The subquery keeps a NULL, so `NOT IN` is never true and
/// every outer row must be dropped — the dynamic filter used to prune the
/// subquery's NULL away, leaving the join to answer as if it had none.
#[tokio::test]
async fn probe_side_null_filters_every_row() {
    let dir = TempDir::new().unwrap();
    // No subquery value matches an outer value, so only the NULL can decide.
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(
        dir.path(),
        &[(Some(1), "keep"), (None, "keep"), (Some(2), "drop")],
    );

    assert_result(&liquid_ctx(dir.path()).await, NOT_IN, &[]).await;
}

/// The same shape with the NULL filtered out of the subquery: an ordinary anti
/// join, so every outer row survives. Pins that the guard did not break the
/// non-NULL path.
#[tokio::test]
async fn probe_side_without_null_keeps_every_row() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(
        dir.path(),
        &[(Some(1), "keep"), (None, "drop"), (Some(2), "keep")],
    );

    let expected: Vec<Option<i64>> = (1000..1109).map(Some).collect();
    assert_result(&liquid_ctx(dir.path()).await, NOT_IN, &expected).await;
}

/// A matching subquery value must still be excluded — the anti join itself has
/// to keep working, not just its NULL handling.
#[tokio::test]
async fn matching_probe_value_is_excluded() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &[Some(1), Some(2), Some(3)]);
    write_sub(dir.path(), &[(Some(2), "keep"), (Some(3), "drop")]);

    assert_result(&liquid_ctx(dir.path()).await, NOT_IN, &[Some(1), Some(3)]).await;
}

/// A NULL *outer* key against a non-empty subquery: `NULL NOT IN (1)` is NULL,
/// so the outer NULL row is dropped while the non-matching row survives. The
/// dynamic filter used to prune the subquery to nothing here, which made the
/// join read the probe side as empty and wrongly emit the NULL row.
#[tokio::test]
async fn build_side_null_dropped_against_non_empty_probe() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &[None, Some(1000)]);
    write_sub(dir.path(), &[(Some(1), "keep")]);

    assert_result(&liquid_ctx(dir.path()).await, NOT_IN, &[Some(1000)]).await;
}

/// `x NOT IN (<empty>)` is vacuously true for every `x`, NULL included.
#[tokio::test]
async fn empty_probe_keeps_null_outer_key() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &[None, Some(1000)]);
    write_sub(dir.path(), &[(Some(1), "drop")]);

    assert_result(&liquid_ctx(dir.path()).await, NOT_IN, &[None, Some(1000)]).await;
}

fn find_join(plan: &Arc<dyn ExecutionPlan>) -> Option<&HashJoinExec> {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        return Some(join);
    }
    plan.children().into_iter().find_map(find_join)
}

/// The guard is surgical: it detaches the dynamic filter from the null-aware
/// join only, leaving the pruning optimization in place everywhere else.
#[tokio::test]
async fn guard_only_strips_null_aware_joins() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(dir.path(), &[(Some(1), "keep"), (None, "keep")]);
    let ctx = liquid_ctx(dir.path()).await;

    for (sql, null_aware) in [
        (NOT_IN, true),
        ("SELECT t0.a FROM t0 JOIN t1 ON t0.a = t1.a", false),
    ] {
        let (state, logical) = ctx.sql(sql).await.unwrap().into_parts();
        let plan = state.create_physical_plan(&logical).await.unwrap();
        let join = find_join(&plan).expect("query should plan a hash join");
        assert_eq!(join.null_aware, null_aware, "{sql}");
        assert_eq!(
            join.dynamic_filter_expr().is_none(),
            null_aware,
            "dynamic filter should be stripped from null-aware joins only: {sql}"
        );
    }
}
