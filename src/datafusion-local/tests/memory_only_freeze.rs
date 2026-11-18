//! Does a cache with no disk tier keep admitting new data once it is full?
//!
//! `LiquidCacheLocalBuilder` with the builder's default policies, the disk
//! tier set to zero, and an admission gate of expansion 4, safety 1,
//! tolerance 1, strict.
//!
//! Eviction moves an Arrow entry to Liquid in memory, and a Liquid entry to
//! disk. With no disk, a Liquid victim is dropped instead, so a full memory
//! tier still makes room. The scenario: queries over a few tables (A) fill the
//! cache, then queries over another table (B) run repeatedly. Every scan
//! passes the admission gate on its own; together they exceed the budget.
//!
//! This is an integration test so the crates build as they do in production
//! (concurrent eviction is on outside `cfg(test)`).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use liquid_cache::cache::EntryID;
use liquid_cache_datafusion::LiquidCacheParquetRef;
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use parquet::arrow::ArrowWriter;

/// The cache's memory budget.
const MEMORY_BYTES: usize = 8 << 20;
/// Rows per table. One Int64 column of values that do not compress, so a
/// table is about 2 MiB of parquet, of Arrow, and of Liquid alike: a scan's
/// gate estimate (parquet bytes × 4) stays just under the budget.
const ROWS: u64 = 200_000;
/// Tables queried first, to fill the cache. Together they hold more than the
/// budget.
const FILL_TABLES: usize = 6;
/// Times each phase queries each of its tables.
const PASSES: usize = 3;
/// Cache entries one table makes: one per batch of the builder's default
/// batch size (8192 rows).
const TABLE_BATCHES: usize = ROWS.div_ceil(8192) as usize;

/// The admission gate.
const EXPANSION: f64 = 4.0;
const SAFETY: f64 = 1.0;
const TOLERANCE: f64 = 1.0;
const STRICT: bool = true;

fn write_table(path: &Path, seed: u64) {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let values: ArrayRef = Arc::new(Int64Array::from_iter_values((0..ROWS).map(|i| {
        (i.wrapping_add(seed << 40))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17) as i64
    })));
    let batch = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// What the cache held after a phase, and what the phase asked of it.
#[derive(Debug, Default)]
struct Phase {
    memory_usage_bytes: usize,
    memory_arrow_entries: usize,
    memory_liquid_entries: usize,
    disk_entries: usize,
    /// Entries cached for the fill tables still held.
    fill_entries_held: usize,
    /// Entries cached for table B held.
    b_entries_held: usize,
    /// Inserts the cache refused: an entry whose insert failed and never
    /// succeeded within the same query, summed over the phase's queries.
    refused_inserts: usize,
}

fn entry_ids(cache: &LiquidCacheParquetRef) -> HashSet<usize> {
    let mut ids = HashSet::new();
    cache.storage().for_each_entry(|id: &EntryID, _| {
        ids.insert(usize::from(*id));
    });
    ids
}

/// Inserts refused during the queries since the trace was last drained.
fn refused_inserts(cache: &LiquidCacheParquetRef) -> usize {
    let trace = format!("{}", cache.consume_event_trace());
    let mut failed = HashSet::new();
    let mut succeeded = HashSet::new();
    for line in trace.lines() {
        let entry = || {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("entry="))
                .unwrap()
                .to_string()
        };
        if line.starts_with("event=insert_failed") {
            failed.insert(entry());
        } else if line.starts_with("event=insert_success") {
            succeeded.insert(entry());
        }
    }
    failed.difference(&succeeded).count()
}

async fn query(ctx: &SessionContext, table: &str) {
    let batches = ctx
        .sql(&format!("SELECT SUM(v) AS s, COUNT(*) AS n FROM {table}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

fn snapshot(
    cache: &LiquidCacheParquetRef,
    fill_ids: &HashSet<usize>,
    refused_inserts: usize,
) -> Phase {
    let stats = cache.storage().stats();
    let ids = entry_ids(cache);
    Phase {
        memory_usage_bytes: stats.memory_usage_bytes,
        memory_arrow_entries: stats.memory_arrow_entries,
        memory_liquid_entries: stats.memory_liquid_entries,
        disk_entries: stats.disk_arrow_entries + stats.disk_liquid_entries,
        fill_entries_held: ids.intersection(fill_ids).count(),
        b_entries_held: ids.difference(fill_ids).count(),
        refused_inserts,
    }
}

/// Fill the cache from the fill tables, then query table B. Returns the cache
/// after the fill phase and after the B phase.
async fn fill_then_query_another_table(max_disk_bytes: usize) -> (Phase, Phase) {
    let data = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(MEMORY_BYTES)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_disk_bytes(max_disk_bytes)
        .with_admission_gate(EXPANSION, SAFETY, TOLERANCE, STRICT)
        .build(SessionConfig::new())
        .await
        .unwrap();
    cache.enable_trace();

    let fill: Vec<String> = (0..FILL_TABLES).map(|t| format!("a{t}")).collect();
    for (seed, table) in fill
        .iter()
        .chain(std::iter::once(&"b".to_string()))
        .enumerate()
    {
        let path = data.path().join(format!("{table}.parquet"));
        write_table(&path, seed as u64);
        let size = std::fs::metadata(&path).unwrap().len() as f64;
        assert!(
            size * EXPANSION * SAFETY <= MEMORY_BYTES as f64 * TOLERANCE,
            "{table} ({size} bytes) would not pass the admission gate"
        );
        ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .unwrap();
    }

    let _ = cache.consume_event_trace();
    let mut refused = 0;
    for _ in 0..PASSES {
        for table in &fill {
            query(&ctx, table).await;
            refused += refused_inserts(&cache);
        }
    }
    let fill_ids = entry_ids(&cache);
    let after_fill = snapshot(&cache, &fill_ids, refused);
    assert!(
        after_fill.fill_entries_held > 0,
        "the fill tables were not cached at all"
    );

    let mut refused = 0;
    for _ in 0..PASSES {
        query(&ctx, "b").await;
        refused += refused_inserts(&cache);
    }
    let after_b = snapshot(&cache, &fill_ids, refused);

    println!(
        "max_disk_bytes={max_disk_bytes} max_memory_bytes={MEMORY_BYTES}\n  \
         after fill: {after_fill:?}\n  after B:    {after_b:?}"
    );
    (after_fill, after_b)
}

/// With no disk tier, a table queried after the cache filled is cached, as it
/// is with a disk tier: the entries cached first are dropped to make room.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_memory_only_cache_admits_a_new_table_once_full() {
    let (after_fill, after_b) = fill_then_query_another_table(0).await;

    assert!(
        after_fill.fill_entries_held < FILL_TABLES * TABLE_BATCHES,
        "the cache holds every fill entry; the test does not fill the cache"
    );
    assert!(
        after_b.fill_entries_held < after_fill.fill_entries_held,
        "no fill entry was dropped to make room for table B"
    );
    assert_eq!(
        after_b.refused_inserts, 0,
        "inserts for table B were refused"
    );
    assert_eq!(
        after_b.b_entries_held,
        TABLE_BATCHES,
        "a memory-only cache full of Liquid entries cached {} of table B's \
         {TABLE_BATCHES} batches: {} inserts refused over {PASSES} queries, \
         while the fill tables keep {} entries ({} Liquid, {} Arrow, {} of \
         {MEMORY_BYTES} bytes)",
        after_b.b_entries_held,
        after_b.refused_inserts,
        after_b.fill_entries_held,
        after_b.memory_liquid_entries,
        after_b.memory_arrow_entries,
        after_b.memory_usage_bytes,
    );
}

/// The control: with a disk tier (the builder's default of ten times
/// memory), Liquid entries spill to disk and table B is cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cache_with_a_disk_tier_admits_a_new_table_once_full() {
    let (after_fill, after_b) = fill_then_query_another_table(10 * MEMORY_BYTES).await;

    assert!(
        after_fill.disk_entries > 0,
        "the fill tables never reached disk"
    );
    assert_eq!(
        after_b.b_entries_held, TABLE_BATCHES,
        "table B was not cached"
    );
    assert_eq!(after_b.refused_inserts, 0);
}
