//! What a Liquid entry is charged, against the heap it really holds.
//!
//! The cache bounds its memory by summing `get_array_memory_size()` over its
//! entries, so a charge below what an entry holds lets the cache outgrow its
//! budget. These tests measure the heap with a counting allocator. It is
//! installed for this test binary only, and counts per thread, so the tests
//! hold with others running in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use arrow::array::{Array, AsArray};
use arrow::array::{
    ArrayRef, BinaryArray, Date32Array, Decimal128Array, DictionaryArray, Float64Array, Int32Array,
    Int64Array, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::UInt16Type;
use datafusion_common::ScalarValue;
use datafusion_expr_common::operator::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use liquid_cache::cache::LiquidExpr;
use liquid_cache::liquid_array::{LiquidArray, eval_predicate_on_array};

struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn add(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() + bytes as isize));
}

fn sub(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as isize));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            sub(layout.size());
            add(new_size);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes this thread has allocated and not freed.
fn live() -> isize {
    LIVE.with(Cell::get)
}

const ROWS: usize = 8192;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn null_at(row: usize, nullable: bool) -> bool {
    nullable && row % 7 == 3
}

type Maker = Box<dyn Fn(bool) -> ArrayRef>;

/// Column shapes a cache sees, each made with and without nulls where that
/// is meaningful: random and narrow integers, floats, dates and timestamps,
/// decimals, strings of every cardinality, binary, and the degenerate
/// constant, all-null and sequential columns.
fn shapes() -> Vec<(&'static str, Maker, bool)> {
    vec![
        (
            "Int32 random < 1e6",
            Box::new(|nullable| {
                let mut rng = Rng(1);
                Arc::new(Int32Array::from_iter((0..ROWS).map(|row| {
                    let value = (rng.next() % 1_000_000) as i32;
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Int64 random",
            Box::new(|nullable| {
                let mut rng = Rng(2);
                Arc::new(Int64Array::from_iter((0..ROWS).map(|row| {
                    let value = rng.next() as i64;
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Float64 two decimals",
            Box::new(|nullable| {
                let mut rng = Rng(3);
                Arc::new(Float64Array::from_iter((0..ROWS).map(|row| {
                    let value = (rng.next() % 10_000_000) as f64 / 100.0;
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Timestamp(us) random steps",
            Box::new(|nullable| {
                let mut rng = Rng(5);
                let mut time = 1_700_000_000_000_000i64;
                Arc::new(TimestampMicrosecondArray::from_iter((0..ROWS).map(|row| {
                    time += (rng.next() % 5_000_000) as i64;
                    (!null_at(row, nullable)).then_some(time)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Timestamp(us) ms ramp",
            Box::new(|_| {
                let mut rng = Rng(15);
                Arc::new(TimestampMicrosecondArray::from_iter_values((0..ROWS).map(
                    |row| 1_700_000_000_000_000 + row as i64 * 1000 + (rng.next() % 997) as i64,
                ))) as ArrayRef
            }),
            false,
        ),
        (
            "Date32",
            Box::new(|nullable| {
                let mut rng = Rng(6);
                Arc::new(Date32Array::from_iter((0..ROWS).map(|row| {
                    let value = 10_000 + (rng.next() % 10_000) as i32;
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Decimal128(15, 2)",
            Box::new(|nullable| {
                let mut rng = Rng(7);
                Arc::new(
                    Decimal128Array::from_iter((0..ROWS).map(|row| {
                        let value = (rng.next() % 100_000_000) as i128;
                        (!null_at(row, nullable)).then_some(value)
                    }))
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
                ) as ArrayRef
            }),
            true,
        ),
        (
            "Utf8 100 distinct",
            Box::new(|nullable| {
                let mut rng = Rng(8);
                Arc::new(StringArray::from_iter((0..ROWS).map(|row| {
                    let value = format!("category-value-{:03}", rng.next() % 100);
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Utf8 3 distinct",
            Box::new(|_| {
                let values = ["open", "closed", "pending"];
                Arc::new(StringArray::from_iter_values(
                    (0..ROWS).map(|row| values[(row * 7 + row / 3) % 3]),
                )) as ArrayRef
            }),
            false,
        ),
        (
            "Utf8 hex16",
            Box::new(|nullable| {
                let mut rng = Rng(9);
                Arc::new(StringArray::from_iter((0..ROWS).map(|row| {
                    let value = format!("{:016x}", rng.next());
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Utf8 urls",
            Box::new(|_| {
                let mut rng = Rng(16);
                Arc::new(StringArray::from_iter_values((0..ROWS).map(|row| {
                    format!(
                        "https://example.com/path/1/{row}/item?id={}",
                        rng.next() >> 33
                    )
                }))) as ArrayRef
            }),
            false,
        ),
        (
            "Utf8View text",
            Box::new(|nullable| {
                let mut rng = Rng(11);
                let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];
                Arc::new(StringViewArray::from_iter((0..ROWS).map(|row| {
                    let value = format!(
                        "the {} {} jumped over {} {}",
                        words[(rng.next() % 6) as usize],
                        words[(rng.next() % 6) as usize],
                        words[(rng.next() % 6) as usize],
                        rng.next() % 1000
                    );
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Dictionary<UInt16, Utf8> 200 distinct",
            Box::new(|_| {
                let mut rng = Rng(17);
                let values: Vec<String> = (0..ROWS)
                    .map(|_| format!("dictionary-value-{:05}", rng.next() % 200))
                    .collect();
                let dictionary: DictionaryArray<UInt16Type> =
                    values.iter().map(String::as_str).collect();
                Arc::new(dictionary) as ArrayRef
            }),
            false,
        ),
        (
            "Binary random 16 B",
            Box::new(|nullable| {
                let mut rng = Rng(12);
                Arc::new(BinaryArray::from_iter((0..ROWS).map(|row| {
                    let mut value = rng.next().to_le_bytes().to_vec();
                    value.extend_from_slice(&rng.next().to_le_bytes());
                    (!null_at(row, nullable)).then_some(value)
                }))) as ArrayRef
            }),
            true,
        ),
        (
            "Int64 constant",
            Box::new(|_| Arc::new(Int64Array::from(vec![42i64; ROWS])) as ArrayRef),
            false,
        ),
        (
            "Int32 all null",
            Box::new(|_| Arc::new(Int32Array::from(vec![None::<i32>; ROWS])) as ArrayRef),
            false,
        ),
        (
            "Int64 sequential",
            Box::new(|_| Arc::new(Int64Array::from_iter_values(0..ROWS as i64)) as ArrayRef),
            false,
        ),
        (
            "Utf8 constant",
            Box::new(|_| {
                Arc::new(StringArray::from_iter_values((0..ROWS).map(|_| "user"))) as ArrayRef
            }),
            false,
        ),
    ]
}

fn fixtures() -> Vec<(String, ArrayRef)> {
    let mut fixtures = Vec::new();
    for (name, make, has_nulls) in shapes() {
        fixtures.push((name.to_string(), make(false)));
        if has_nulls {
            fixtures.push((format!("{name} with nulls"), make(true)));
        }
    }
    fixtures
}

/// The heap a freshly transcoded entry holds, and what it is charged.
fn measure(source: &ArrayRef) -> (isize, isize) {
    // Build lazily initialized statics on a throwaway entry first.
    drop(LiquidArray::from_arrow_array(source).unwrap());
    let base = live();
    let liquid = Arc::new(LiquidArray::from_arrow_array(source).unwrap());
    let real = live() - base;
    let charged = liquid.get_array_memory_size() as isize;
    drop(liquid);
    (real, charged)
}

/// `a` is within 10% plus 256 bytes of `b`, or below it.
fn at_most_about(a: isize, b: isize) -> bool {
    a * 10 <= b * 11 + 2560
}

/// An entry's charge is all that bounds how many entries the cache keeps, so
/// it must cover the heap the entry holds: here within 10% plus 256 bytes.
/// It must not overshoot by more either, or the cache keeps fewer entries
/// than its budget allows.
#[test]
fn charge_matches_the_heap_an_entry_holds() {
    let mut failures = Vec::new();
    for (name, source) in fixtures() {
        let (real, charged) = measure(&source);
        println!(
            "{name:<40} charged {charged:>8} real {real:>8} ({:.2}x)",
            real as f64 / charged as f64
        );
        if !at_most_about(real, charged) || !at_most_about(charged, real) {
            failures.push(format!(
                "{name}: holds {real} bytes, charged {charged} ({:.2}x)",
                real as f64 / charged as f64
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "charges more than 10% + 256 bytes off the heap held:\n  {}",
        failures.join("\n  ")
    );
}

/// Small string batches, where an FSST node's fixed overhead is a large part
/// of the entry. The charge must still cover the heap held. It may overshoot:
/// a full symbol table (255 symbols, as high-entropy strings produce) holds
/// about 2.6 KB outside the block rather than the 4.8 KB charged for one, and
/// over-charging only costs cache space, where under-charging overruns memory.
#[test]
fn charge_matches_the_heap_of_small_string_batches() {
    let mut failures = Vec::new();
    for rows in [128, 512, 2048] {
        let mut rng = Rng(91);
        let hex16: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..rows).map(|_| format!("{:016x}", rng.next())),
        ));
        let urls: ArrayRef =
            Arc::new(StringArray::from_iter_values((0..rows).map(|row| {
                format!("https://example.com/page/{}?q={}", row % 97, row)
            })));
        for (name, source) in [("Utf8 hex16", hex16), ("Utf8 urls", urls)] {
            let (real, charged) = measure(&source);
            println!(
                "{name} x{rows:<5} charged {charged:>8} real {real:>8} ({:.2}x)",
                real as f64 / charged as f64
            );
            if !at_most_about(real, charged) {
                failures.push(format!(
                    "{name} x{rows}: holds {real} bytes, charged {charged} ({:.2}x)",
                    real as f64 / charged as f64
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "entries hold more than 10% + 256 bytes over their charge:\n  {}",
        failures.join("\n  ")
    );
}

/// A column, and how to make a literal of its type.
type StringColumn = (&'static str, ArrayRef, fn(String) -> ScalarValue);

fn fsst_columns() -> Vec<StringColumn> {
    let strings = |words: usize| -> Vec<String> {
        let mut rng = Rng(77);
        (0..ROWS)
            .map(|_| (0..words).map(|_| format!("{:016x}", rng.next())).collect())
            .collect()
    };
    vec![
        (
            "Utf8 hex16",
            Arc::new(StringArray::from_iter_values(strings(1))),
            |value| ScalarValue::Utf8(Some(value)),
        ),
        (
            "Utf8 hex128",
            Arc::new(StringArray::from_iter_values(strings(8))),
            |value| ScalarValue::Utf8(Some(value)),
        ),
        (
            "Utf8View hex128",
            Arc::new(StringViewArray::from_iter_values(strings(8))),
            |value| ScalarValue::Utf8View(Some(value)),
        ),
        (
            "Dictionary<UInt16, Utf8> hex16 over 4096 values",
            {
                let values = strings(1);
                let dictionary: DictionaryArray<UInt16Type> =
                    (0..ROWS).map(|row| values[row % 4096].as_str()).collect();
                Arc::new(dictionary)
            },
            |value| ScalarValue::Utf8(Some(value)),
        ),
    ]
}

/// The heap a cached entry gains from evaluating `op` against one of its own
/// values, once the result is dropped, and how much its charge grew.
fn growth_from_predicate(
    source: &ArrayRef,
    literal: ScalarValue,
    op: Operator,
) -> (isize, isize, LiquidArray) {
    let liquid = LiquidArray::from_arrow_array(source).unwrap();
    let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("c", 0)),
        op,
        Arc::new(Literal::new(literal)),
    ));
    let expr = LiquidExpr::try_new(expr, source.data_type()).unwrap();
    let selection = BooleanBuffer::new_set(ROWS);
    let base = live();
    let charged = liquid.get_array_memory_size() as isize;
    let matches = liquid.try_eval_predicate(&expr, &selection);
    assert!(
        matches.true_count() > 0,
        "the literal is a value of the column"
    );
    assert_eq!(matches, eval_predicate_on_array(source.clone(), &expr));
    drop(matches);
    let grown = live() - base;
    let charge_grown = liquid.get_array_memory_size() as isize - charged;
    (grown, charge_grown, liquid)
}

/// A predicate is a read. Evaluating one must not leave the cached entry
/// holding more than it is charged for: FSST `=` and `!=` used to build a
/// 160 KiB compressor for the literal and keep it in the entry.
#[test]
fn predicates_do_not_grow_an_entry_past_its_charge() {
    let mut failures = Vec::new();
    for (name, source, scalar) in fsst_columns() {
        assert!(
            LiquidArray::from_arrow_array(&source).unwrap().uses_fsst(),
            "{name} is not FSST-compressed, so this test no longer exercises FSST"
        );
        let needle = source
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|strings| strings.value(100).to_string())
            .or_else(|| {
                source
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .map(|strings| strings.value(100).to_string())
            })
            .unwrap_or_else(|| {
                let dictionary = source.as_dictionary::<UInt16Type>();
                let values = dictionary.values().as_string::<i32>();
                values
                    .value(dictionary.keys().value(100) as usize)
                    .to_string()
            });
        for op in [Operator::Eq, Operator::NotEq, Operator::Lt] {
            // Build lazily initialized statics on a throwaway entry first.
            drop(growth_from_predicate(&source, scalar(needle.clone()), op));
            let (grown, charge_grown, liquid) =
                growth_from_predicate(&source, scalar(needle.clone()), op);
            println!("{name:<48} {op:<3} heap +{grown:>7} charge +{charge_grown:>7}");
            if grown > charge_grown + 4096 {
                failures.push(format!(
                    "{name} {op}: entry grew {grown} bytes, charge grew {charge_grown}"
                ));
            }
            drop(liquid);
        }
    }
    assert!(
        failures.is_empty(),
        "predicates grew entries past their charge:\n  {}",
        failures.join("\n  ")
    );
}
