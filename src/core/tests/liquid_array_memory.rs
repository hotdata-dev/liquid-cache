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

use arrow::array::{
    ArrayRef, BinaryArray, Date32Array, Decimal128Array, DictionaryArray, Float64Array, Int32Array,
    Int64Array, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow::datatypes::UInt16Type;
use liquid_cache::liquid_array::LiquidArray;

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
