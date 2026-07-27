//! The parquet page index (ColumnIndex / OffsetIndex) is an optional footer
//! structure. Scans must not depend on it being present: it drives page-level
//! pruning, which degrades to row-group granularity when it is missing.

use std::{fs::File, io::Write, path::Path, sync::Arc};

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::{
    arrow::ArrowWriter,
    arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
    file::metadata::{PageIndexPolicy, ParquetMetaDataWriter},
    file::properties::{EnabledStatistics, WriterProperties},
    file::writer::TrackedWrite,
};
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// Write a parquet file that advertises a column index but no offset index.
///
/// A file carrying *no* page index at all is not enough to exercise this: the
/// reader skips page index parsing entirely when no column chunk declares an
/// index range, so even `PageIndexPolicy::Required` accepts it. The failure
/// needs an index range to exist while a chunk's offset index is absent.
///
/// Writers expose no knob for that combination, so the file is produced by
/// writing a fully indexed one and re-encoding only its footer with each
/// chunk's offset-index pointer cleared. Data pages and the original column
/// index blobs are kept byte-for-byte, so the surviving column-index pointers
/// still resolve.
fn write_parquet_without_offset_index(dir: &Path) -> String {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let ids: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
    let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, names]).expect("batch");

    // Page-level statistics make the writer emit both index structures.
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut indexed = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut indexed, schema, Some(props)).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");

    let base_path = dir.join("indexed.parquet");
    std::fs::write(&base_path, &indexed).expect("write base file");
    let base = File::open(&base_path).expect("open base file");
    let metadata = ArrowReaderMetadata::load(
        &base,
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional),
    )
    .expect("load base metadata");
    let metadata = metadata.metadata().as_ref().clone();
    assert!(
        metadata.column_index().is_some() && metadata.offset_index().is_some(),
        "base fixture should be fully indexed"
    );

    // Keep everything up to the footer — data pages and index blobs alike.
    // Layout is `... | footer | footer_len:u32 | "PAR1"`.
    let len = indexed.len();
    let footer_len = u32::from_le_bytes(indexed[len - 8..len - 4].try_into().unwrap()) as usize;
    let prefix_end = len - 8 - footer_len;

    // Clear each chunk's offset-index pointer, and drop the parsed indexes so
    // the writer re-emits neither blob. Column-index pointers stay, keeping the
    // file's page index range non-empty.
    let row_groups: Vec<_> = metadata
        .row_groups()
        .iter()
        .map(|rg| {
            let columns: Vec<_> = rg
                .columns()
                .iter()
                .map(|c| {
                    c.clone()
                        .into_builder()
                        .set_offset_index_offset(None)
                        .set_offset_index_length(None)
                        .build()
                        .expect("rebuild column chunk")
                })
                .collect();
            rg.clone()
                .into_builder()
                .set_column_metadata(columns)
                .build()
                .expect("rebuild row group")
        })
        .collect();
    let stripped = metadata
        .into_builder()
        .set_row_groups(row_groups)
        .set_offset_index(None)
        .set_column_index(None)
        .build();

    let mut out = Vec::new();
    let mut tracked = TrackedWrite::new(&mut out);
    tracked.write_all(&indexed[..prefix_end]).expect("prefix");
    ParquetMetaDataWriter::new_with_tracked(tracked, &stripped)
        .finish()
        .expect("rewrite footer");

    let path = dir.join("no_offset_index.parquet");
    std::fs::write(&path, &out).expect("write fixture");
    path.to_str().expect("utf8 path").to_string()
}

/// Guards the premise: the fixture must be the shape that `Required` rejects,
/// otherwise the scan test below would pass vacuously.
#[test]
fn fixture_is_rejected_by_required_policy() {
    let dir = TempDir::new().unwrap();
    let path = write_parquet_without_offset_index(dir.path());
    let file = File::open(&path).unwrap();

    let err = ArrowReaderMetadata::load(
        &file,
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required),
    )
    .expect_err("fixture should be rejected under Required");
    assert!(
        err.to_string().contains("missing offset index"),
        "unexpected error: {err}"
    );
}

/// Scanning such a file through LiquidCache must succeed. The predicate matters:
/// it is what builds the page-pruning predicate that consults the index.
///
/// Requires `direct_io`, so this runs on Linux/CI rather than every dev machine
/// — the same constraint as the rest of the cache-backed tests here.
#[tokio::test]
async fn scans_file_without_offset_index() {
    let dir = TempDir::new().unwrap();
    let file_path = write_parquet_without_offset_index(dir.path());
    let cache_dir = TempDir::new().unwrap();

    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 1;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(1024 * 1024)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .build(config)
        .await
        .unwrap();

    ctx.register_parquet("no_offset_index", &file_path, ParquetReadOptions::default())
        .await
        .unwrap();

    async fn run(ctx: &SessionContext) -> String {
        let batches = ctx
            .sql("SELECT id, name FROM no_offset_index WHERE id > 2 ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .expect("scanning a file without an offset index should succeed");
        pretty_format_batches(&batches).unwrap().to_string()
    }

    let expected = "\
+----+------+
| id | name |
+----+------+
| 3  | c    |
| 4  | d    |
| 5  | e    |
+----+------+";

    assert_eq!(run(&ctx).await, expected);
    // Again, now served from the cache, covering the warm path too.
    assert_eq!(run(&ctx).await, expected);

    // The scan must actually have gone through LiquidCache. Without this the
    // test would still pass if the plan stopped being rewritten to
    // `LiquidParquetSource` — i.e. it would silently stop covering the reader
    // whose policy this test exists to pin.
    let stats = cache.storage().stats();
    assert!(
        stats.total_entries > 0,
        "expected the scan to populate LiquidCache, got {stats:?}"
    );
}
