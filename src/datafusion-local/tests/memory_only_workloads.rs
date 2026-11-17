//! Workloads against a cache with no disk tier, once its memory is full.
//!
//! Each test builds the cache as a memory-only deployment would:
//! `LiquidCacheLocalBuilder` with the builder's default policies, the disk
//! tier set to zero, and an admission gate of expansion 4, safety 1,
//! tolerance 1, strict. Every scan passes the gate on its own; together the
//! tables hold several times the budget, so the cache turns over.
//!
//! Every query's result is checked against DataFusion reading the same files
//! with no cache, so a query that silently returns wrong rows fails as surely
//! as one that errors.
//!
//! These are integration tests so the crates build as they do in production
//! (concurrent eviction is on outside `cfg(test)`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use liquid_cache_datafusion::LiquidCacheParquetRef;
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use parquet::arrow::ArrowWriter;

/// The cache's memory budget.
const MEMORY_BYTES: usize = 8 << 20;
/// Rows per table: three columns, about 1.4 MiB of parquet, so a scan's gate
/// estimate (parquet bytes × 4) stays under the budget.
const ROWS: u64 = 60_000;

/// The type of a table's first column, `p`, which the filters read.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Timestamp with a time zone, which Liquid does not encode.
    TimestampTz,
    /// Boolean, which Liquid does not encode.
    Boolean,
    /// Int64, which Liquid encodes.
    Int64,
    /// Utf8, which Liquid encodes.
    Utf8,
}

impl Kind {
    /// A filter on `p` that keeps about half the rows.
    fn predicate(self) -> &'static str {
        match self {
            Kind::TimestampTz => "p > TIMESTAMP '2033-01-01T00:00:00Z'",
            Kind::Boolean => "p",
            Kind::Int64 => "p > 0",
            Kind::Utf8 => "p > 'm'",
        }
    }
}

fn mix(i: u64, seed: u64) -> i64 {
    (i.wrapping_add(seed << 40))
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(17) as i64
}

fn write_table(path: &Path, seed: u64, kind: Kind) {
    let (p_field, p): (Field, ArrayRef) = match kind {
        Kind::TimestampTz => (
            Field::new(
                "p",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..ROWS).map(|i| mix(i, seed + 99).rem_euclid(4_000_000_000_000_000)),
                )
                .with_timezone("UTC"),
            ),
        ),
        Kind::Boolean => (
            Field::new("p", DataType::Boolean, false),
            Arc::new(BooleanArray::from_iter(
                (0..ROWS).map(|i| Some(mix(i, seed + 99) & 1 == 0)),
            )),
        ),
        Kind::Int64 => (
            Field::new("p", DataType::Int64, false),
            Arc::new(Int64Array::from_iter_values(
                (0..ROWS).map(|i| mix(i, seed + 99)),
            )),
        ),
        Kind::Utf8 => (
            Field::new("p", DataType::Utf8, false),
            Arc::new(StringArray::from_iter_values((0..ROWS).map(|i| {
                let letter = (b'a' + mix(i, seed + 99).rem_euclid(26) as u8) as char;
                format!("{letter}{}", mix(i, seed + 5).rem_euclid(1000))
            }))),
        ),
    };
    let schema = Arc::new(Schema::new(vec![
        p_field,
        Field::new("v", DataType::Int64, false),
        Field::new("w", DataType::Int64, false),
    ]));
    let v: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..ROWS).map(|i| mix(i, seed)),
    ));
    let w: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..ROWS).map(|i| mix(i, seed + 7)),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![p, v, w]).unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

struct Fixture {
    cached: SessionContext,
    plain: SessionContext,
    cache: LiquidCacheParquetRef,
    tables: Vec<(String, Kind)>,
    _data: tempfile::TempDir,
    _cache_dir: tempfile::TempDir,
}

/// `tables` tables, their kinds cycling through `kinds`, registered with a
/// memory-only cache and with a plain context to check results against.
async fn fixture(tables: usize, kinds: &[Kind]) -> Fixture {
    let data = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let (cached, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(MEMORY_BYTES)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_disk_bytes(0)
        .with_admission_gate(4.0, 1.0, 1.0, true)
        .build(SessionConfig::new())
        .await
        .unwrap();
    let plain = SessionContext::new();
    let mut names = Vec::new();
    for t in 0..tables {
        let kind = kinds[t % kinds.len()];
        let name = format!("t{t}");
        let path: PathBuf = data.path().join(format!("{name}.parquet"));
        write_table(&path, t as u64, kind);
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        assert!(
            size * 4 <= MEMORY_BYTES,
            "{name} ({size} bytes) would not pass the admission gate"
        );
        for ctx in [&cached, &plain] {
            ctx.register_parquet(&name, path.to_str().unwrap(), ParquetReadOptions::default())
                .await
                .unwrap();
        }
        names.push((name, kind));
    }
    Fixture {
        cached,
        plain,
        cache,
        tables: names,
        _data: data,
        _cache_dir: cache_dir,
    }
}

/// The SQL each test runs against a table: a filter on `p` and on the two
/// Int64 columns, so a batch's predicate columns and its projected columns
/// are cached one after another.
fn sql(table: &str, kind: Kind) -> String {
    format!(
        "SELECT COUNT(*) AS n, SUM(v) AS sv, MIN(w) AS mw FROM {table} \
         WHERE {} AND v > -4000000000000000000 AND w < 4000000000000000000",
        kind.predicate()
    )
}

async fn run(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// Run `table`'s query on the cache and check it against the plain context.
async fn query_and_check(f: &Fixture, table: &str, kind: Kind) {
    let sql = sql(table, kind);
    let got = run(&f.cached, &sql)
        .await
        .unwrap_or_else(|e| panic!("{table} ({kind:?}) failed on the cache: {e}"));
    let want = run(&f.plain, &sql).await.unwrap();
    assert_eq!(got, want, "{table} ({kind:?}) returned a different result");
}

fn entry_ids(cache: &LiquidCacheParquetRef) -> HashSet<usize> {
    let mut ids = HashSet::new();
    cache.storage().for_each_entry(|id, _| {
        ids.insert(usize::from(*id));
    });
    ids
}

/// The cache charges exactly the memory its entries hold. Called once no query
/// is running.
fn assert_accounting_exact(cache: &LiquidCacheParquetRef, after: &str) -> usize {
    let used = cache.storage().stats().memory_usage_bytes;
    let mut held = 0;
    cache
        .storage()
        .for_each_entry(|_, entry| held += entry.memory_usage_bytes());
    assert_eq!(
        used, held,
        "after {after} the cache charges {used} bytes for entries holding {held}"
    );
    used
}

/// The accounting is exact and within the budget. An insert that raced the
/// removal of its key can charge one entry over a full budget, until a later
/// insert that grows evicts; so after concurrent queries, call this only
/// after a sequential query has run.
fn assert_within_budget(cache: &LiquidCacheParquetRef, after: &str) {
    let used = assert_accounting_exact(cache, after);
    assert!(
        used <= MEMORY_BYTES,
        "after {after} the cache charges {used} bytes, over its {MEMORY_BYTES}"
    );
}

/// A filter that reads a column Liquid cannot encode used to fail once the
/// cache was full: filling the cache with a batch's later columns dropped its
/// earlier ones, and the predicate found nothing to read. Every column type
/// must keep its queries correct.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn filtered_queries_on_every_column_type_stay_correct_once_full() {
    for kind in [Kind::TimestampTz, Kind::Boolean, Kind::Int64, Kind::Utf8] {
        let f = fixture(16, &[kind]).await;
        for _pass in 0..3 {
            for (table, kind) in f.tables.clone() {
                query_and_check(&f, &table, kind).await;
                assert_within_budget(&f.cache, &table);
            }
        }
    }
}

/// Many more tables than the cache holds, queried in turn for several passes:
/// the steady state of a long-running deployment. The table queried last must
/// always be cached, however long the cache has been full.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_long_rotation_keeps_caching_the_table_queried_last() {
    let kinds = [Kind::Int64, Kind::Utf8, Kind::TimestampTz, Kind::Boolean];
    let f = fixture(24, &kinds).await;
    let mut cached_after_query = Vec::new();
    for pass in 0..4 {
        for (table, kind) in f.tables.clone() {
            let before = entry_ids(&f.cache);
            query_and_check(&f, &table, kind).await;
            assert_within_budget(&f.cache, &table);
            let after = entry_ids(&f.cache);
            if pass > 0 {
                cached_after_query.push((table.clone(), after.difference(&before).count()));
            }
        }
    }
    let stalled: Vec<_> = cached_after_query
        .iter()
        .filter(|(_, new_entries)| *new_entries == 0)
        .collect();
    assert!(
        stalled.is_empty(),
        "once full, these queries cached nothing of their table: {stalled:?}"
    );
}

/// The same rotation from eight queries at once. Results stay correct, the
/// budget holds, and afterwards a table not queried yet is cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_queries_keep_the_cache_admitting() {
    let kinds = [Kind::Int64, Kind::Utf8, Kind::TimestampTz, Kind::Boolean];
    let f = Arc::new(fixture(25, &kinds).await);
    let (last, last_kind) = f.tables.last().cloned().unwrap();
    let mut tasks = Vec::new();
    for worker in 0..8 {
        let f = Arc::clone(&f);
        tasks.push(tokio::spawn(async move {
            for round in 0..24 {
                let (table, kind) = f.tables[(worker * 5 + round) % 24].clone();
                query_and_check(&f, &table, kind).await;
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    assert_accounting_exact(&f.cache, "the concurrent queries");

    let before = entry_ids(&f.cache);
    query_and_check(&f, &last, last_kind).await;
    let new_entries = entry_ids(&f.cache).difference(&before).count();
    assert!(
        new_entries > 0,
        "after the concurrent queries, {last} cached nothing"
    );
    assert_within_budget(&f.cache, &last);
}

/// Queries cancelled part way, many times over, must not leave the cache
/// unable to make room: afterwards a table not queried yet is cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_queries_do_not_freeze_the_cache() {
    let kinds = [Kind::Int64, Kind::Utf8, Kind::TimestampTz, Kind::Boolean];
    let f = Arc::new(fixture(13, &kinds).await);
    let (last, last_kind) = f.tables.last().cloned().unwrap();
    for round in 0..120u64 {
        let (table, kind) = f.tables[(round % 12) as usize].clone();
        let sql = sql(&table, kind);
        let cached = f.cached.clone();
        let task = tokio::spawn(async move { run(&cached, &sql).await });
        // Cut each query off at a different point.
        tokio::time::sleep(Duration::from_micros(200 + (round * 397) % 4000)).await; // sleep-ok: the cancellation point is the fixture
        task.abort();
        let _ = task.await;
    }
    assert_accounting_exact(&f.cache, "the cancelled queries");

    for (table, kind) in f.tables[..12].iter().cloned() {
        query_and_check(&f, &table, kind).await;
    }
    let before = entry_ids(&f.cache);
    query_and_check(&f, &last, last_kind).await;
    let new_entries = entry_ids(&f.cache).difference(&before).count();
    assert!(
        new_entries > 0,
        "after the cancelled queries, {last} cached nothing"
    );
    assert_within_budget(&f.cache, &last);
}
