//! Regression tests for issue #19: a pushed-down conjunct that references no column.
//!
//! DataFusion simplifies `NOT (s = s)` to `s IS NULL AND NULL`. The bare `NULL`
//! is a conjunct with an empty column set, and the row-filter builder used to
//! drop every candidate whose column set was empty. The liquid scan is the only
//! place the pushed-down predicate is applied — DataFusion has already removed
//! the `FilterExec` — so dropping a conjunct *widens* the filter: rows where the
//! predicate is UNKNOWN came back as if it were TRUE.
//!
//! That shows up as a ternary-logic partitioning violation. `WHERE p`,
//! `WHERE NOT p` and `WHERE p IS NULL` must partition the table, since every row
//! satisfies exactly one of the three; with the conjunct dropped the three
//! buckets returned more rows than the table holds.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// The predicate from the issue. Over [`write_t1`]'s data it is TRUE for eight
/// rows, UNKNOWN for four, and FALSE for none:
///
/// - `id BETWEEN 1 AND 7` is FALSE everywhere — `id` starts at 10.
/// - `f <= 333.0` is TRUE for the first four rows only.
/// - `s = s` is TRUE where `s` is set and UNKNOWN where it is NULL.
///
/// So the last four NULL-`s` rows are UNKNOWN, and belong to the `p IS NULL`
/// bucket alone.
const P: &str = "(f <= 333.0 OR s = s) OR id BETWEEN 1 AND 7";

/// Twelve rows. `s` is NULL on even rows, `f` climbs past the 333 threshold at
/// row four, and `id` stays clear of the 1..7 range.
fn write_t1(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("f", DataType::Float64, false),
        Field::new("s", DataType::Utf8, true),
    ]));

    let id: Int64Array = (0..12).map(|i| i + 10).collect();
    let f: Float64Array = (0..12).map(|i| i as f64 * 100.0).collect();
    let s: StringArray = (0..12)
        .map(|i| (i % 2 == 1).then(|| format!("s{i}")))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(id), Arc::new(f), Arc::new(s)],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn liquid_ctx(cache_dir: &Path, parquet: &Path) -> SessionContext {
    std::fs::create_dir_all(cache_dir).unwrap();
    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build(SessionConfig::new())
        .await
        .unwrap();
    ctx.register_parquet(
        "t1",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx
}

/// The `s` values matching `where`, sorted, with NULL spelled out.
async fn projected_rows(ctx: &SessionContext, r#where: &str) -> Vec<String> {
    let batches = ctx
        .sql(&format!("SELECT s FROM t1 {}", r#where))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..column.len() {
            rows.push(match column.is_valid(i) {
                true => column.value(i).to_string(),
                false => "NULL".to_string(),
            });
        }
    }
    rows.sort();
    rows
}

/// The scan projects `s` while the filter reads `f`, `s` and `id`, so the
/// predicate columns are materialized separately from the projected one.
#[tokio::test]
async fn ternary_partitions_cover_every_row_once() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t1.parquet");
    write_t1(&parquet);
    let ctx = liquid_ctx(&dir.path().join("cache"), &parquet).await;

    // Twice: the first pass fills the cache, the second reads it back.
    for pass in 0..2 {
        let all = projected_rows(&ctx, "").await;
        let mut partitioned = projected_rows(&ctx, &format!("WHERE {P}")).await;
        partitioned.extend(projected_rows(&ctx, &format!("WHERE NOT ({P})")).await);
        partitioned.extend(projected_rows(&ctx, &format!("WHERE ({P}) IS NULL")).await);
        partitioned.sort();

        assert_eq!(
            all, partitioned,
            "pass {pass}: the three buckets are not a partition"
        );
    }
}

/// The direct reading of the same bug: `NOT p` is never TRUE here, because `p`
/// is TRUE or UNKNOWN for every row. Dropping the column-free `NULL` conjunct
/// left `f > 333 AND s IS NULL AND id NOT BETWEEN 1 AND 7`, which matches four.
#[tokio::test]
async fn negation_of_an_unknown_predicate_matches_nothing() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t1.parquet");
    write_t1(&parquet);
    let ctx = liquid_ctx(&dir.path().join("cache"), &parquet).await;

    for pass in 0..2 {
        let matched = projected_rows(&ctx, &format!("WHERE NOT ({P})")).await;
        assert!(matched.is_empty(), "pass {pass}: NOT p matched {matched:?}");

        assert_eq!(projected_rows(&ctx, &format!("WHERE {P}")).await.len(), 8);
        assert_eq!(
            projected_rows(&ctx, &format!("WHERE ({P}) IS NULL")).await,
            vec!["NULL"; 4]
        );
    }
}
