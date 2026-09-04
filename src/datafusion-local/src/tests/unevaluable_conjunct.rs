//! Regression tests for issue #23 (and #21, one instance of it): a pushed-down
//! conjunct the liquid row filter cannot evaluate.
//!
//! `build_row_filter` splits the pushed-down predicate into conjuncts and builds
//! one `FilterCandidate` per conjunct. `PushdownChecker` refuses a conjunct that
//! references a column absent from the table schema. Such a refusal used to be
//! dropped silently, leaving the scan applying a *strictly weaker* filter than
//! the query asked for — and since DataFusion removes the `FilterExec` when it
//! pushes a predicate down, nothing re-applies the dropped conjunct.
//!
//! A nested column is not one of these: it pushes down like any other, so a
//! struct in the predicate costs the scan nothing.
//!
//! The scan is now declined at plan time instead, so the predicate stays with
//! the reader that planned it.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTableUrl};
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// Eight rows: `id` 0..8, `st` a `struct<a int>` whose `a` mirrors `id`. Exactly
/// one row has `st.a = 3`.
fn write_t(path: &Path) {
    let struct_fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("st", DataType::Struct(struct_fields.clone()), false),
    ]));

    let id: Int64Array = (0..8i64).collect::<Vec<_>>().into();
    let a: ArrayRef = Arc::new((0..8i32).collect::<Int32Array>());
    let st = StructArray::new(struct_fields, vec![a], None);

    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(id), Arc::new(st)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn liquid_ctx(cache_dir: &Path) -> SessionContext {
    liquid_ctx_with_cache(cache_dir).await.0
}

async fn liquid_ctx_with_cache(
    cache_dir: &Path,
) -> (
    SessionContext,
    liquid_cache_datafusion::LiquidCacheParquetRef,
) {
    std::fs::create_dir_all(cache_dir).unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build(SessionConfig::new())
        .await
        .unwrap();
    (ctx, cache)
}

async fn ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        out.extend(column.iter().map(|v| v.unwrap()));
    }
    out.sort_unstable();
    out
}

async fn plan_of(ctx: &SessionContext, sql: &str) -> String {
    let (state, plan) = ctx.sql(sql).await.unwrap().into_parts();
    let plan = state.create_physical_plan(&plan).await.unwrap();
    format!(
        "{}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    )
}

/// The repro from #21: one conjunct is pushable (`id >= 0`, matching all eight
/// rows) and one is not (`st.a = 3`, matching one). Dropping the second returned
/// all eight rows.
#[tokio::test]
async fn nested_column_conjunct_is_still_applied() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t.parquet");
    write_t(&parquet);
    let ctx = liquid_ctx(&dir.path().join("cache")).await;
    ctx.register_parquet(
        "t",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql = "SELECT id FROM t WHERE id >= 0 AND st.a = 3";

    // A nested conjunct is evaluable now, so the scan is taken over and keeps the
    // cache. What must not change is the answer: the conjunct is applied, not
    // dropped.
    let plan = plan_of(&ctx, sql).await;
    assert!(
        plan.contains("liquid_parquet"),
        "a nested conjunct should no longer cost the scan its cache:\n{plan}"
    );

    // The first pass reads through the parquet fallback and would fill the cache;
    // the second is served from it, a separate evaluation path.
    for pass in ["cold", "warm"] {
        assert_eq!(ids(&ctx, sql).await, vec![3], "{pass}");
    }
}

/// Every conjunct being unevaluable was the case the old code thought was safe:
/// it returned `None` only when *all* candidates failed. That was wrong too —
/// `None` means the scan applies no filter at all, and the `FilterExec` that
/// would have caught it is gone. `id > 100 OR st.a = 3` is a single conjunct
/// touching `st`, so it takes that path, and it matches exactly one row.
#[tokio::test]
async fn sole_unevaluable_conjunct_is_still_applied() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t.parquet");
    write_t(&parquet);
    let ctx = liquid_ctx(&dir.path().join("cache")).await;
    ctx.register_parquet(
        "t",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql = "SELECT id FROM t WHERE id > 100 OR st.a = 3";
    for pass in ["cold", "warm"] {
        assert_eq!(ids(&ctx, sql).await, vec![3], "{pass}");
    }
}

/// A struct column costs the scan nothing, wherever it appears. The row filter
/// used to refuse any nested column outright — a bar inherited from DataFusion's
/// own `row_filter.rs` — which declined the whole scan to `ParquetSource` and so
/// lost the cache for every filtered query on a table carrying one. A column the
/// cache cannot transcode is simply held as Arrow and the predicate evaluates
/// against that, so nothing here needs declining.
#[tokio::test]
async fn a_nested_column_never_costs_the_cache() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t.parquet");
    write_t(&parquet);
    let ctx = liquid_ctx(&dir.path().join("cache")).await;
    ctx.register_parquet(
        "t",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    for sql in [
        "SELECT st.a FROM t WHERE id >= 0",
        "SELECT id FROM t WHERE id >= 0",
        "SELECT id FROM t",
        "SELECT id FROM t WHERE st.a = 3",
        "SELECT id FROM t WHERE id >= 0 AND st.a = 3",
    ] {
        let plan = plan_of(&ctx, sql).await;
        assert!(
            plan.contains("liquid_parquet"),
            "`{sql}` lost the cache:\n{plan}"
        );
    }

    // And the nested predicates still return the right rows through the cache.
    for sql in [
        "SELECT id FROM t WHERE st.a = 3",
        "SELECT id FROM t WHERE id >= 0 AND st.a = 3",
    ] {
        for pass in ["cold", "warm"] {
            assert_eq!(ids(&ctx, sql).await, vec![3], "{sql} ({pass})");
        }
    }
}

/// The cache is not merely *kept* on a nested predicate, it is *used*: the struct
/// column is admitted and the warm pass reads it back from the cache.
///
/// What such a scan does not get is the encoded-data fast path
/// (`eval_predicate`), because a struct root does not resolve to one cache column
/// id — see `convert_parquet_scan`'s docs. Pinned here so that closing that gap
/// shows up as a deliberate change to this test rather than passing unnoticed.
#[tokio::test]
async fn a_nested_predicate_is_served_from_the_cache() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t.parquet");
    write_t(&parquet);
    let (ctx, cache) = liquid_ctx_with_cache(&dir.path().join("cache")).await;
    ctx.register_parquet(
        "t",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let sql = "SELECT id FROM t WHERE st.a = 3";
    cache.storage().stats();
    assert_eq!(ids(&ctx, sql).await, vec![3], "cold");
    assert_eq!(ids(&ctx, sql).await, vec![3], "warm");

    let stats = cache.storage().stats();
    assert!(
        stats.total_entries > 0,
        "the struct column was never admitted: {stats:?}"
    );
    assert!(
        stats.runtime.get_with_selection > 0 || stats.runtime.get > 0,
        "the warm pass did not read the cache: {stats:?}"
    );
}

/// The other `PushdownChecker` refusal: a conjunct on a column that is not in the
/// file schema. The table declares `extra`, the file does not have it, so every
/// row's `extra` is NULL and `extra = 3` is never TRUE.
///
/// The scan is still cached here — the opener's physical-expr adapter resolves
/// `extra` to a literal NULL before the row filter is built, so nothing is
/// dropped — but the answer has to come out right either way, and a filter that
/// dropped the conjunct would return all eight rows.
#[tokio::test]
async fn conjunct_on_column_outside_file_schema_is_still_applied() {
    let dir = TempDir::new().unwrap();
    let table_dir = dir.path().join("t");
    std::fs::create_dir_all(&table_dir).unwrap();
    write_t(&table_dir.join("t.parquet"));
    let ctx = liquid_ctx(&dir.path().join("cache")).await;

    // A declared schema wider than the file: `extra` exists in the table schema
    // only. `st` is left out so this test exercises the missing-column path alone.
    let declared = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("extra", DataType::Int64, true),
    ]));
    let listing_options =
        ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
    ctx.register_listing_table(
        "t",
        &ListingTableUrl::parse(table_dir.to_str().unwrap()).unwrap(),
        listing_options,
        Some(declared),
        None,
    )
    .await
    .unwrap();

    let sql = "SELECT id FROM t WHERE id >= 0 AND extra = 3";
    for pass in ["cold", "warm"] {
        assert_eq!(ids(&ctx, sql).await, Vec::<i64>::new(), "{pass}");
    }
}
