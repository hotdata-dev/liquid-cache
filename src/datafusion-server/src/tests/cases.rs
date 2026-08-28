use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use liquid_cache::cache::squeeze_policies::TranscodeSqueezeEvict;

use crate::tests::run_sql;

fn gen_parquet(dir: impl AsRef<Path>) -> PathBuf {
    use arrow::array::UInt32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let temp_path = dir.as_ref().join("parquet_page_index.parquet");
    let file = File::create(&temp_path).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, false)]));
    let id_array = UInt32Array::from_iter_values(0..200_000);
    let id_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(id_array)]).unwrap();
    let props = WriterProperties::builder()
        .set_offset_index_disabled(false)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&id_batch).unwrap();
    writer.into_inner().unwrap();
    temp_path
}

/// Eight rows, `st` a `struct<a int>` mirroring `id`; one row has `st.a = 3`.
fn gen_struct_parquet(dir: impl AsRef<Path>) -> PathBuf {
    use arrow::array::{ArrayRef, Int32Array, Int64Array, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let temp_path = dir.as_ref().join("struct.parquet");
    let struct_fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("st", DataType::Struct(struct_fields.clone()), false),
    ]));

    let id: ArrayRef = Arc::new((0..8i64).collect::<Int64Array>());
    let a: ArrayRef = Arc::new((0..8i32).collect::<Int32Array>());
    let st: ArrayRef = Arc::new(StructArray::new(struct_fields, vec![a], None));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![id, st]).unwrap();

    let file = File::create(&temp_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.into_inner().unwrap();
    temp_path
}

/// Issue #23 on the server side. The client ships a fragment whose predicate is
/// already fully pushed — the `FilterExec` was removed on the client, and the
/// server cannot ask DataFusion to put one back — so `register_plan`'s rewrite is
/// the only place the scan can be declined. Exercises that declined scans
/// actually execute on the server, as vanilla parquet reads.
#[tokio::test]
async fn nested_column_conjunct_is_applied_on_the_server() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file = gen_struct_parquet(&temp_dir);

    // `id >= 0` matches all eight rows and `st.a = 3` matches one, so a dropped
    // conjunct would show up as all eight ids coming back.
    let result = run_sql(
        "SELECT id FROM hits WHERE id >= 0 AND st.a = 3",
        Box::new(TranscodeSqueezeEvict),
        1000,
        file.to_str().unwrap(),
    )
    .await;
    assert_eq!(
        result,
        ["+----+", "| id |", "+----+", "| 3  |", "+----+"].join("\n")
    );
}

#[tokio::test]
async fn test_parquet_with_page_index() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file = gen_parquet(&temp_dir);
    let file_path = file.to_str().unwrap();

    let result = run_sql(
        "SELECT * FROM hits WHERE id = 0",
        Box::new(TranscodeSqueezeEvict),
        1000,
        file_path,
    )
    .await;
    insta::assert_snapshot!(result);
}
