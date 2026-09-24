//! A pushed-down conjunct the liquid row filter may not be able to evaluate.
//!
//! `build_row_filter` splits the pushed-down predicate into conjuncts and builds
//! one `FilterCandidate` per conjunct. A conjunct that is refused and then
//! dropped leaves the scan applying a *strictly weaker* filter than the query
//! asked for — and since DataFusion removes the `FilterExec` when it pushes a
//! predicate down, nothing re-applies what was dropped.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTableUrl};
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
    std::fs::create_dir_all(cache_dir).unwrap();
    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build(SessionConfig::new())
        .await
        .unwrap();
    ctx
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

/// One pushable conjunct (`id >= 0`, all eight rows) and one nested
/// (`st.a = 3`, one row). Dropping the second returns all eight.
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
    // Cold reads through the source and fills the cache; warm is served from it,
    // a separate evaluation path.
    for pass in ["cold", "warm"] {
        assert_eq!(ids(&ctx, sql).await, vec![3], "{pass}");
    }
}

/// A single conjunct that mixes a pushable and a nested column through `OR`, so
/// the whole predicate is one candidate. Refusing it and returning no filter at
/// all means the scan applies nothing, and the `FilterExec` is already gone.
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

/// A conjunct on a column that is not in the file schema. The table declares
/// `extra`, the file lacks it, so every row's `extra` is NULL and `extra = 3` is
/// never TRUE. A filter that dropped the conjunct would return all eight rows.
#[tokio::test]
async fn conjunct_on_column_outside_file_schema_is_still_applied() {
    let dir = TempDir::new().unwrap();
    let table_dir = dir.path().join("t");
    std::fs::create_dir_all(&table_dir).unwrap();
    write_t(&table_dir.join("t.parquet"));
    let ctx = liquid_ctx(&dir.path().join("cache")).await;

    // A declared schema wider than the file: `extra` exists in the table schema
    // only. `st` is left out so this exercises the missing-column path alone.
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
