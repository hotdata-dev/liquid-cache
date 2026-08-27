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
//! LiquidCache query, so `NoDynamicFiltersForNullAwareJoins` keeps dynamic
//! filters out of any plan holding a null-aware join. See
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

/// Table `t2`, used as the far side of a join enclosing the anti join.
fn write_probe(dir: &Path, vals: &[Option<i64>]) {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    write(
        &dir.join("t2.parquet"),
        schema,
        vec![Arc::new(Int64Array::from(vals.to_vec()))],
    );
}

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
    // `t2` only exists for the tests that write it.
    for t in ["t0", "t1", "t2"] {
        let path = dir.join(format!("{t}.parquet"));
        if !path.exists() {
            continue;
        }
        ctx.register_parquet(t, path.to_str().unwrap(), ParquetReadOptions::default())
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

/// A join *above* the anti join pushes its own dynamic filter straight through
/// into the same probe subtree, because `HashJoinExec` reports both sides of a
/// `LeftAnti` as preserved for pushdown. Detaching only the anti join's own
/// filter left this wrong, which is why the guard works at plan scope.
#[tokio::test]
async fn enclosing_join_cannot_prune_the_probe_side() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(dir.path(), &[(Some(1), "keep"), (None, "keep")]);
    // `t2` matches one outer value, so the enclosing join's filter is narrow
    // enough to prune the whole subquery file away.
    write_probe(dir.path(), &[Some(1050)]);
    let ctx = liquid_ctx(dir.path()).await;

    // The anti join drops every outer row, so the enclosing join has nothing to
    // match and both orderings must return no rows.
    for sql in [
        "SELECT x.a FROM t2 JOIN ({NOT_IN}) x ON t2.a = x.a",
        "SELECT x.a FROM ({NOT_IN}) x JOIN t2 ON t2.a = x.a",
    ] {
        assert_result(&ctx, &sql.replace("{NOT_IN}", NOT_IN), &[]).await;
    }
}

fn find_joins(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<(bool, bool)>) {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        out.push((join.null_aware, join.dynamic_filter_expr().is_some()));
    }
    for child in plan.children() {
        find_joins(child, out);
    }
}

async fn joins_of(ctx: &SessionContext, sql: &str) -> Vec<(bool, bool)> {
    let (state, logical) = ctx.sql(sql).await.unwrap().into_parts();
    let plan = state.create_physical_plan(&logical).await.unwrap();
    let mut out = Vec::new();
    find_joins(&plan, &mut out);
    out
}

/// Dynamic filters survive everywhere except in a plan that holds a null-aware
/// join — there they are suppressed plan-wide, since one join's filter reaches
/// another's probe side.
#[tokio::test]
async fn dynamic_filters_suppressed_only_alongside_a_null_aware_join() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(dir.path(), &[(Some(1), "keep"), (None, "keep")]);
    write_probe(dir.path(), &[Some(1050)]);
    let ctx = liquid_ctx(dir.path()).await;

    // A plain join keeps its dynamic filter.
    assert_eq!(
        joins_of(&ctx, "SELECT t0.a FROM t0 JOIN t2 ON t0.a = t2.a").await,
        vec![(false, true)]
    );
    // The null-aware join never gets one.
    assert_eq!(joins_of(&ctx, NOT_IN).await, vec![(true, false)]);
    // Nor does the join above it, which is the case the plan-scoped guard adds.
    let nested = format!("SELECT x.a FROM t2 JOIN ({NOT_IN}) x ON t2.a = x.a");
    let joins = joins_of(&ctx, &nested).await;
    assert_eq!(
        joins.len(),
        2,
        "expected an enclosing join and an anti join"
    );
    assert!(
        joins.iter().all(|(_, has_filter)| !has_filter),
        "no join in a null-aware plan may carry a dynamic filter: {joins:?}"
    );
}

/// A `NOT IN` alongside a range predicate on the same column is still wrong,
/// and no physical rule can fix it: DataFusion's logical `push_down_filter`
/// infers the join key and copies `t1.a > ..` into the subquery, dropping the
/// NULL that makes `NOT IN` never true. Wrong in vanilla DataFusion over a
/// `MemTable` too, with every dynamic filter disabled. Un-ignore when upstream
/// stops inferring join-key predicates across a null-aware join.
#[tokio::test]
#[ignore = "upstream logical-optimizer bug, not reachable from a physical rule"]
async fn not_in_with_range_predicate_on_the_join_key() {
    let dir = TempDir::new().unwrap();
    write_outer(dir.path(), &(1000..1109).map(Some).collect::<Vec<_>>());
    write_sub(dir.path(), &[(Some(1), "keep"), (None, "keep")]);
    let ctx = liquid_ctx(dir.path()).await;

    let sql = format!("SELECT a FROM ({NOT_IN}) x WHERE x.a > 1050");
    assert_result(&ctx, &sql, &[]).await;
}
