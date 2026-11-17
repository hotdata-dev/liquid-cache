//! Size audit of `LiquidArray`: what the cache is charged for one array
//! (`get_array_memory_size`) against what the allocator says it holds.
//!
//! Every `audit_*` test asserts the behaviour a caller budgeting memory relies
//! on, so a failure is a finding, not a broken test. Run:
//!
//! ```text
//! cargo +1.96.0 test -p liquid-cache --lib audit_ -- --nocapture --test-threads=1
//! ```
//!
//! Heap numbers are per-thread deltas (see `crate::audit_alloc`), so the
//! assertions hold with other tests running in parallel too.

use std::sync::Arc;

use arrow::array::{
    BinaryArray, Date32Array, Decimal128Array, DictionaryArray, Float64Array, Int32Array,
    Int64Array, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow::datatypes::UInt16Type;
use datafusion_physical_expr::PhysicalExpr;

use super::*;
use crate::audit_alloc::{thread_live as live, thread_peak, thread_reset_peak};

const N: usize = 8192;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn null_at(i: usize, nullable: bool) -> bool {
    nullable && i % 7 == 3
}

fn encodings(liquid: &LiquidArray) -> String {
    let mut out: Vec<String> = Vec::new();
    for node in liquid.array.depth_first_traversal() {
        let id = node.encoding_id().as_ref().to_string();
        let id = id.trim_start_matches("vortex.").to_string();
        out.push(id);
    }
    out.join(">")
}

type Maker = Box<dyn Fn(bool) -> ArrayRef>;

fn cases() -> Vec<(&'static str, Maker, bool)> {
    // (name, maker, has_nullable_variant)
    let mut v: Vec<(&'static str, Maker, bool)> = Vec::new();
    v.push((
        "Int32 rand<1e6",
        Box::new(|n| {
            let mut r = Rng(1);
            Arc::new(Int32Array::from_iter((0..N).map(|i| {
                let x = (r.next() % 1_000_000) as i32;
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Int64 rand full",
        Box::new(|n| {
            let mut r = Rng(2);
            Arc::new(Int64Array::from_iter((0..N).map(|i| {
                let x = r.next() as i64;
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Int64 rand<1e6",
        Box::new(|n| {
            let mut r = Rng(22);
            Arc::new(Int64Array::from_iter((0..N).map(|i| {
                let x = (r.next() % 1_000_000) as i64;
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Float64 2-decimals",
        Box::new(|n| {
            let mut r = Rng(3);
            Arc::new(Float64Array::from_iter((0..N).map(|i| {
                let x = (r.next() % 10_000_000) as f64 / 100.0;
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Float64 rand bits",
        Box::new(|n| {
            let mut r = Rng(4);
            Arc::new(Float64Array::from_iter((0..N).map(|i| {
                let x = (r.next() as f64) / (u64::MAX as f64);
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Timestamp(us)",
        Box::new(|n| {
            let mut r = Rng(5);
            let mut t = 1_700_000_000_000_000i64;
            Arc::new(TimestampMicrosecondArray::from_iter((0..N).map(|i| {
                t += (r.next() % 5_000_000) as i64;
                (!null_at(i, n)).then_some(t)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Date32",
        Box::new(|n| {
            let mut r = Rng(6);
            Arc::new(Date32Array::from_iter((0..N).map(|i| {
                let x = 10_000 + (r.next() % 10_000) as i32;
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Decimal128(15,2)",
        Box::new(|n| {
            let mut r = Rng(7);
            Arc::new(
                Decimal128Array::from_iter((0..N).map(|i| {
                    let x = (r.next() % 100_000_000) as i128;
                    (!null_at(i, n)).then_some(x)
                }))
                .with_precision_and_scale(15, 2)
                .unwrap(),
            ) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Utf8 lowcard(100)",
        Box::new(|n| {
            let mut r = Rng(8);
            Arc::new(StringArray::from_iter((0..N).map(|i| {
                let x = format!("category-value-{:03}", r.next() % 100);
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Utf8 highcard hex16",
        Box::new(|n| {
            let mut r = Rng(9);
            Arc::new(StringArray::from_iter((0..N).map(|i| {
                let x = format!("{:016x}", r.next());
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Utf8View long hex48",
        Box::new(|n| {
            let mut r = Rng(10);
            Arc::new(StringViewArray::from_iter((0..N).map(|i| {
                let x = format!("{:016x}{:016x}{:016x}", r.next(), r.next(), r.next());
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Utf8View long text",
        Box::new(|n| {
            let mut r = Rng(11);
            let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"];
            Arc::new(StringViewArray::from_iter((0..N).map(|i| {
                let x = format!(
                    "the {} {} jumped over {} {}",
                    words[(r.next() % 7) as usize],
                    words[(r.next() % 7) as usize],
                    words[(r.next() % 7) as usize],
                    r.next() % 1000
                );
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    v.push((
        "Binary rand 16B",
        Box::new(|n| {
            let mut r = Rng(12);
            Arc::new(BinaryArray::from_iter((0..N).map(|i| {
                let mut x = r.next().to_le_bytes().to_vec();
                x.extend_from_slice(&r.next().to_le_bytes());
                (!null_at(i, n)).then_some(x)
            }))) as ArrayRef
        }),
        true,
    ));
    // Degenerate
    v.push((
        "Int64 constant",
        Box::new(|_| Arc::new(Int64Array::from(vec![42i64; N])) as ArrayRef),
        false,
    ));
    v.push((
        "Int32 constant",
        Box::new(|_| Arc::new(Int32Array::from(vec![42i32; N])) as ArrayRef),
        false,
    ));
    v.push((
        "Int32 all-null",
        Box::new(|_| Arc::new(Int32Array::from(vec![None::<i32>; N])) as ArrayRef),
        false,
    ));
    v.push((
        "Utf8 all-null",
        Box::new(|_| Arc::new(StringArray::from(vec![None::<&str>; N])) as ArrayRef),
        false,
    ));
    v.push((
        "Int64 sequential",
        Box::new(|_| Arc::new(Int64Array::from_iter_values(0..N as i64)) as ArrayRef),
        false,
    ));
    v.push((
        "Int32 sequential",
        Box::new(|_| Arc::new(Int32Array::from_iter_values(0..N as i32)) as ArrayRef),
        false,
    ));
    v.push((
        "Utf8 3-distinct",
        Box::new(|_| {
            let vals = ["open", "closed", "pending"];
            Arc::new(StringArray::from_iter_values((0..N).map(|i| vals[(i * 7 + i / 3) % 3])))
                as ArrayRef
        }),
        false,
    ));
    v.push((
        "Utf8 constant",
        Box::new(|_| Arc::new(StringArray::from_iter_values((0..N).map(|_| "user"))) as ArrayRef),
        false,
    ));
    v.push((
        "Utf8View constant long",
        Box::new(|_| {
            Arc::new(StringViewArray::from_iter_values(
                (0..N).map(|_| "a value longer than the inline twelve bytes"),
            )) as ArrayRef
        }),
        false,
    ));
    v.push((
        "Float64 constant",
        Box::new(|_| Arc::new(Float64Array::from(vec![1.5f64; N])) as ArrayRef),
        false,
    ));
    v.push((
        "Timestamp constant",
        Box::new(|_| {
            Arc::new(TimestampMicrosecondArray::from(vec![1_700_000_000_000_000i64; N]))
                as ArrayRef
        }),
        false,
    ));
    v
}

/// The generator `tests/audit_cache_heap.rs` (and the earlier
/// `alloc_vs_budget` measurement) draws its arrays from.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn lcg_seed(i: usize) -> u64 {
    (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// `alloc_vs_budget`'s `Kind::TimestampUs`: a millisecond ramp plus 0..997 us
/// of noise.
fn avb_timestamp(i: usize) -> ArrayRef {
    let mut s = lcg_seed(i);
    Arc::new(TimestampMicrosecondArray::from_iter_values((0..N).map(
        |r| 1_700_000_000_000_000 + (r as i64) * 1000 + (lcg(&mut s) % 997) as i64,
    )))
}

/// `alloc_vs_budget`'s `Kind::Utf8LowCard`: 16 distinct 19-byte strings.
fn avb_utf8_lowcard(i: usize) -> ArrayRef {
    let mut s = lcg_seed(i);
    Arc::new(StringArray::from_iter_values(
        (0..N).map(|_| format!("category-value-{:04}", lcg(&mut s) % 16)),
    ))
}

/// `alloc_vs_budget`'s `Kind::Utf8HighCard`.
fn avb_utf8_highcard(i: usize) -> ArrayRef {
    let mut s = lcg_seed(i);
    Arc::new(StringArray::from_iter_values((0..N).map(|r| {
        format!("https://example.com/path/{}/{}/item?id={}", i, r, lcg(&mut s))
    })))
}

/// `alloc_vs_budget`'s `Kind::DictUtf8`: u16 keys over 200 distinct strings.
fn avb_dict_utf8(i: usize) -> ArrayRef {
    let mut s = lcg_seed(i);
    let values: Vec<String> = (0..N)
        .map(|_| format!("dictionary-value-{:05}", lcg(&mut s) % 200))
        .collect();
    let dict: DictionaryArray<UInt16Type> = values.iter().map(|v| v.as_str()).collect();
    Arc::new(dict)
}

type Fixture = (String, Box<dyn Fn() -> ArrayRef>);

/// Every column shape the audit covers, nullable and not.
fn fixtures() -> Vec<Fixture> {
    let mut out: Vec<Fixture> = Vec::new();
    for (name, make, has_nullable) in cases() {
        let make = std::rc::Rc::new(make);
        for nullable in [false, true] {
            if nullable && !has_nullable {
                continue;
            }
            let make = make.clone();
            out.push((
                format!("{name}{}", if nullable { " +nulls" } else { "" }),
                Box::new(move || make(nullable)),
            ));
        }
    }
    out.push(("avb Timestamp(us) ramp".into(), Box::new(|| avb_timestamp(1))));
    out.push(("avb Utf8 lowcard(16)".into(), Box::new(|| avb_utf8_lowcard(1))));
    out.push(("avb Utf8 highcard url".into(), Box::new(|| avb_utf8_highcard(1))));
    out.push(("avb Dict<u16,Utf8>(200)".into(), Box::new(|| avb_dict_utf8(1))));
    out
}

/// A `LiquidArray` built exactly as `from_arrow_array` builds it, keeping the
/// address range of the one serialized block it is decoded from.
struct Built {
    liquid: LiquidArray,
    block: std::ops::Range<usize>,
    /// The block itself; a clone of the handle the array holds, not a copy.
    bytes: Bytes,
}

fn build_with_block(source: &ArrayRef) -> Built {
    let compressed = LiquidArray::compress(source).unwrap();
    // The body of `LiquidArray::compacted`.
    let bytes = Bytes::from(compressed.to_bytes().into_boxed_slice());
    let start = bytes.as_ptr() as usize;
    let block = start..start + bytes.len();
    Built {
        liquid: LiquidArray::from_bytes(bytes.clone()),
        block,
        bytes,
    }
}

struct Placement {
    encoding: String,
    buffer: String,
    len: usize,
    addr: usize,
    /// The alignment the buffer asks for.
    align: usize,
    inside: bool,
    /// For a buffer copied out of the block: where its bytes still sit in it.
    original_offset: Option<usize>,
}

/// Where each buffer of `built` lives: inside the serialized block, or in an
/// allocation of its own.
fn placements(built: &Built) -> Vec<Placement> {
    let mut out = Vec::new();
    for node in built.liquid.array.depth_first_traversal() {
        let encoding = node.encoding_id().as_ref().to_string();
        let names = node.buffer_names();
        for (i, buffer) in node.buffers().into_iter().enumerate() {
            let addr = buffer.as_slice().as_ptr() as usize;
            let len = buffer.len();
            let inside =
                len == 0 || (addr >= built.block.start && addr + len <= built.block.end);
            let original_offset = (!inside && len >= 8)
                .then(|| {
                    built
                        .bytes
                        .windows(len)
                        .position(|window| window == buffer.as_slice())
                })
                .flatten();
            out.push(Placement {
                encoding: encoding.clone(),
                buffer: names.get(i).cloned().unwrap_or_else(|| format!("#{i}")),
                len,
                addr,
                align: 1usize << buffer.alignment().exponent(),
                inside,
                original_offset,
            });
        }
    }
    out
}

fn outside_bytes(placements: &[Placement]) -> usize {
    placements.iter().filter(|p| !p.inside).map(|p| p.len).sum()
}

/// What one fixture's array is charged, and what it holds.
struct Sizes {
    arrow: usize,
    charged: isize,
    real: isize,
    serialized: isize,
    encodings: String,
}

fn measure(make: &dyn Fn() -> ArrayRef) -> Sizes {
    // Warm every lazy path for this type on a throwaway array.
    {
        let source = make();
        let liquid = LiquidArray::from_arrow_array(&source).unwrap();
        let _ = liquid.to_bytes();
        let _ = liquid.get_array_memory_size();
        let _ = encodings(&liquid);
    }
    let base = live();
    let source = make();
    let arrow = source.get_array_memory_size();
    let liquid = LiquidArray::from_arrow_array(&source).unwrap();
    drop(source);
    let real = live() - base;
    let sizes = Sizes {
        arrow,
        charged: liquid.get_array_memory_size() as isize,
        real,
        serialized: liquid.to_bytes().len() as isize,
        encodings: encodings(&liquid),
    };
    assert_eq!(liquid.len(), N);
    drop(liquid);
    // A few dozen bytes can stay behind (interned ids, not the array).
    assert!(live() - base < 256, "the measurement leaked {}", live() - base);
    sizes
}

/// Item 2. The cache charges a Liquid entry `get_array_memory_size()`. That
/// charge is the only thing bounding how many entries the cache keeps, so it
/// has to cover what the entry holds: here, within 10% plus 256 bytes.
#[test]
fn audit_liquid_charge_covers_real_heap() {
    println!(
        "\n{:<28}|{:>9}|{:>9}|{:>6}|{:>9}|{:>9}|{:>9}|{:>5}| encoding tree (dfs)",
        "fixture", "charged", "real", "r/c", "real-chg", "ser", "arrow", "ok"
    );
    let mut failures = Vec::new();
    let mut total = 0;
    for (name, make) in fixtures() {
        let s = measure(&make);
        let ok = s.real * 10 <= s.charged * 11 + 2560;
        total += 1;
        println!(
            "{:<28}|{:>9}|{:>9}|{:>6.2}|{:>9}|{:>9}|{:>9}|{:>5}| {}",
            name,
            s.charged,
            s.real,
            s.real as f64 / s.charged as f64,
            s.real - s.charged,
            s.serialized,
            s.arrow,
            if ok { "pass" } else { "FAIL" },
            s.encodings,
        );
        if !ok {
            failures.push(format!(
                "{name}: real {} > charged {} * 1.10 + 256 ({:.2}x)",
                s.real,
                s.charged,
                s.real as f64 / s.charged as f64
            ));
        }
    }
    println!("size_of::<LiquidArray>() = {}", size_of::<LiquidArray>());
    assert!(
        failures.is_empty(),
        "real_heap <= charged * 1.10 + 256 failed for {} of {total} fixtures:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Item 3. A fresh `LiquidArray` is decoded from one serialized block, which
/// it keeps alive whole. A buffer the decoder copied out of that block is
/// held twice, and `nbytes()` counts it once.
#[test]
fn audit_decode_copies_no_buffer_out_of_the_serialized_block() {
    println!(
        "\n{:<28}|{:>9}|{:>9}|{:>9}| buffers outside the block",
        "fixture", "ser", "charged", "outside"
    );
    let mut failures = Vec::new();
    let mut total = 0;
    for (name, make) in fixtures() {
        let built = build_with_block(&make());
        let placements = placements(&built);
        let outside = outside_bytes(&placements);
        let detail: Vec<String> = placements
            .iter()
            .filter(|p| !p.inside)
            .map(|p| {
                format!(
                    "{}.{} len={} align={}{}",
                    p.encoding.trim_start_matches("vortex."),
                    p.buffer,
                    p.len,
                    p.align,
                    match p.original_offset {
                        Some(offset) => format!(
                            " (in block at addr%align={})",
                            (built.block.start + offset) % p.align
                        ),
                        None => String::new(),
                    }
                )
            })
            .collect();
        total += 1;
        println!(
            "{:<28}|{:>9}|{:>9}|{:>9}| {}",
            name,
            built.block.len(),
            built.liquid.get_array_memory_size(),
            outside,
            if detail.is_empty() { "-".to_string() } else { detail.join("; ") },
        );
        if outside != 0 {
            failures.push(format!("{name}: {outside} bytes in separate allocations"));
        }
    }
    assert!(
        failures.is_empty(),
        "bytes in separate allocations == 0 failed for {} of {total} fixtures:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Item 4, a report rather than an assertion: where the bytes beyond the
/// charge live, for the two timestamp fixtures that disagreed and for
/// low-cardinality strings.
#[test]
fn audit_report_where_uncharged_bytes_live() {
    let subjects: Vec<Fixture> = vec![
        (
            "Timestamp(us) size_audit: cumulative random steps < 5s".into(),
            Box::new(|| {
                let mut r = Rng(5);
                let mut t = 1_700_000_000_000_000i64;
                Arc::new(TimestampMicrosecondArray::from_iter_values((0..N).map(|_| {
                    t += (r.next() % 5_000_000) as i64;
                    t
                }))) as ArrayRef
            }),
        ),
        (
            "Timestamp(us) alloc_vs_budget: 1ms ramp + noise < 997us".into(),
            Box::new(|| avb_timestamp(1)),
        ),
        (
            "Utf8 lowcard(100) size_audit".into(),
            Box::new(|| {
                let mut r = Rng(8);
                Arc::new(StringArray::from_iter_values(
                    (0..N).map(|_| format!("category-value-{:03}", r.next() % 100)),
                )) as ArrayRef
            }),
        ),
        (
            "Utf8 lowcard(16) alloc_vs_budget".into(),
            Box::new(|| avb_utf8_lowcard(1)),
        ),
    ];
    for (name, make) in subjects {
        let s = measure(&make);
        let built = build_with_block(&make());
        let placements = placements(&built);
        let outside = outside_bytes(&placements) as isize;
        let nbytes = built.liquid.array.nbytes() as isize;
        let block = built.block.len() as isize;

        // The hydration path: the serialized `Vec` handed over as it is, the
        // way `read_disk_bytes` hands over what the store returned.
        let serialized = built.liquid.to_bytes();
        let (ser_len, ser_cap) = (serialized.len() as isize, serialized.capacity() as isize);
        let base = live() - ser_cap;
        let hydrated = LiquidArray::from_bytes(Bytes::from(serialized));
        let hydrated_real = live() - base;
        let hydrated_charged = hydrated.get_array_memory_size();
        drop(hydrated);

        println!("\n=== {name} ===");
        println!(
            "arrow={} charged={} (nbytes {} + size_of {})  real(transcode)={}  real-charged={}",
            s.arrow,
            s.charged,
            nbytes,
            size_of::<LiquidArray>(),
            s.real,
            s.real - s.charged
        );
        println!(
            "real = serialized block {block} + buffers copied out of it {outside} + node/metadata heap {}",
            s.real - block - outside
        );
        println!(
            "block {block} = buffers still inside it {} + IPC framing/flatbuffers/padding {}",
            nbytes - outside,
            block - (nbytes - outside)
        );
        println!(
            "hydrated from Bytes::from(Vec): charged={hydrated_charged} real={hydrated_real} (vec len {ser_len}, capacity {ser_cap})"
        );
        for p in &placements {
            let place = match (p.inside, p.original_offset) {
                (true, _) => "in serialized block".to_string(),
                (false, Some(offset)) => format!(
                    "SEPARATE ALLOCATION; its bytes are still in the block at offset {offset}, \
                     address % align = {}",
                    (built.block.start + offset) % p.align
                ),
                (false, None) => "SEPARATE ALLOCATION".to_string(),
            };
            println!(
                "  {:<24} {:<18} len={:>7} align={:>2} addr%64={:>2} {}",
                p.encoding,
                p.buffer,
                p.len,
                p.align,
                p.addr % 64,
                place
            );
        }
        println!("{}", built.liquid.array.display_tree());
    }
}

/// Item 6. Transcoding one array allocates scratch nothing is charged for. It
/// is the caller's transient, so it should scale with the input: under 10x.
#[test]
fn audit_transcode_scratch_is_bounded_by_its_input() {
    let subjects: Vec<Fixture> = vec![
        ("avb Dict<u16,Utf8>(200)".into(), Box::new(|| avb_dict_utf8(1))),
        ("avb Utf8 lowcard(16)".into(), Box::new(|| avb_utf8_lowcard(1))),
        ("avb Utf8 highcard url".into(), Box::new(|| avb_utf8_highcard(1))),
        ("avb Timestamp(us) ramp".into(), Box::new(|| avb_timestamp(1))),
        (
            "Int64 rand<1e6 (control)".into(),
            Box::new(|| {
                let mut r = Rng(22);
                Arc::new(Int64Array::from_iter_values(
                    (0..N).map(|_| (r.next() % 1_000_000) as i64),
                )) as ArrayRef
            }),
        ),
        (
            "Dict<u16,Utf8>(200), 64 rows".into(),
            Box::new(|| {
                let values: Vec<String> =
                    (0..64).map(|i| format!("dictionary-value-{:05}", i % 200)).collect();
                let dict: DictionaryArray<UInt16Type> =
                    values.iter().map(|v| v.as_str()).collect();
                Arc::new(dict) as ArrayRef
            }),
        ),
    ];
    println!(
        "\n{:<30}|{:>9}|{:>10}|{:>9}|{:>8}|{:>5}",
        "fixture", "input", "peak", "result", "peak/in", "ok"
    );
    let mut failures = Vec::new();
    for (name, make) in subjects {
        // Warm lazily built statics on a throwaway array.
        drop(LiquidArray::from_arrow_array(&make()).unwrap());
        let source = make();
        let input = source.get_array_memory_size() as isize;
        let base = live();
        thread_reset_peak();
        let liquid = LiquidArray::from_arrow_array(&source).unwrap();
        let peak = thread_peak() - base;
        let result = live() - base;
        drop(liquid);
        let ok = peak < 10 * input;
        println!(
            "{:<30}|{:>9}|{:>10}|{:>9}|{:>8.1}|{:>5}",
            name,
            input,
            peak,
            result,
            peak as f64 / input as f64,
            if ok { "pass" } else { "FAIL" }
        );
        if !ok {
            failures.push(format!(
                "{name}: peak {peak} >= 10 * input {input} ({:.1}x)",
                peak as f64 / input as f64
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "peak heap during from_arrow_array < 10 * input size failed:\n  {}",
        failures.join("\n  ")
    );
}

fn cmp_expr(op: DataFusionOperator, value: ScalarValue, dt: &DataType) -> LiquidExpr {
    let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("c", 0)),
        op,
        Arc::new(Literal::new(value)),
    ));
    LiquidExpr::try_new(expr, dt).unwrap()
}

fn like_expr(pattern: ScalarValue, dt: &DataType) -> LiquidExpr {
    let expr: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
        false,
        false,
        Arc::new(Column::new("c", 0)),
        Arc::new(Literal::new(pattern)),
    ));
    LiquidExpr::try_new(expr, dt).unwrap()
}

fn hex_strings(seed: u64, words: usize) -> Vec<String> {
    let mut r = Rng(seed);
    (0..N)
        .map(|_| (0..words).map(|_| format!("{:016x}", r.next())).collect())
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum PredicateOp {
    Eq,
    NotEq,
    Lt,
    LikePrefix,
    LikeContains,
}

/// Item 1. A predicate is a read: it must not grow the cached array, or must
/// grow its charge by as much. Evaluates `op` once on a fresh FSST-encoded
/// high-cardinality string column, drops the result, and asserts the heap the
/// array still holds grew by under 4 KB or by what its charge grew.
fn assert_predicate_does_not_grow_the_array(op: PredicateOp) {
    type Mk = fn(&[String]) -> (ArrayRef, DataType, fn(String) -> ScalarValue);
    let utf8: Mk = |s| {
        (
            Arc::new(StringArray::from_iter_values(s)) as ArrayRef,
            DataType::Utf8,
            |v| ScalarValue::Utf8(Some(v)),
        )
    };
    let view: Mk = |s| {
        (
            Arc::new(StringViewArray::from_iter_values(s)) as ArrayRef,
            DataType::Utf8View,
            |v| ScalarValue::Utf8View(Some(v)),
        )
    };
    let fixtures: Vec<(&str, usize, Mk)> = vec![
        ("Utf8 hex16", 1, utf8),
        ("Utf8 hex128", 8, utf8),
        ("Utf8View hex128", 8, view),
    ];
    let filter = BooleanBuffer::new_set(N);

    println!(
        "\n{:<16}|{:<13}|{:>8}|{:>10}|{:>10}|{:>10}|{:>7}|{:>10}| encoding tree (dfs)",
        "fixture", "op", "charged", "heap +1st", "heap +2nd", "heap +3rd", "chg +", "freed@drop"
    );
    let mut failures = Vec::new();
    for (fname, words, mk) in &fixtures {
        let strings = hex_strings(77, *words);
        let needle = strings[100].clone();
        let run = |measure: bool| -> Option<String> {
            let before_build = live();
            let (source, dt, scalar) = mk(&strings);
            let liquid = LiquidArray::from_arrow_array(&source).unwrap();
            drop(source);
            let expr = match op {
                PredicateOp::Eq => cmp_expr(DataFusionOperator::Eq, scalar(needle.clone()), &dt),
                PredicateOp::NotEq => {
                    cmp_expr(DataFusionOperator::NotEq, scalar(needle.clone()), &dt)
                }
                PredicateOp::Lt => cmp_expr(DataFusionOperator::Lt, scalar(needle.clone()), &dt),
                PredicateOp::LikePrefix => like_expr(scalar(format!("{}%", &needle[..6])), &dt),
                PredicateOp::LikeContains => {
                    like_expr(scalar(format!("%{}%", &needle[4..10])), &dt)
                }
            };
            let enc = encodings(&liquid);
            let l0 = live();
            let s0 = liquid.get_array_memory_size() as isize;
            let result = liquid.try_eval_predicate(&expr, &filter);
            let hits = result.true_count();
            drop(result);
            let l1 = live();
            let s1 = liquid.get_array_memory_size() as isize;
            drop(liquid.try_eval_predicate(&expr, &filter));
            let l2 = live();
            drop(liquid.try_eval_predicate(&expr, &filter));
            let l3 = live();
            drop(expr);
            drop(liquid);
            let after_drop = live();
            if !measure {
                return None;
            }
            println!(
                "{:<16}|{:<13}|{:>8}|{:>10}|{:>10}|{:>10}|{:>7}|{:>10}| {} (hits={})",
                fname,
                format!("{op:?}"),
                s0,
                l1 - l0,
                l2 - l1,
                l3 - l2,
                s1 - s0,
                if after_drop == before_build {
                    "yes".to_string()
                } else {
                    format!("{}", after_drop - before_build)
                },
                enc,
                hits
            );
            assert!(enc.contains("fsst"), "{fname} is not FSST encoded: {enc}");
            let (growth, charge_growth) = (l1 - l0, s1 - s0);
            (!(growth < 4096 || charge_growth == growth)).then(|| {
                format!(
                    "{fname} {op:?}: live heap grew {growth} bytes after one predicate, \
                     get_array_memory_size() grew {charge_growth} (charged {s0} before)"
                )
            })
        };
        run(false); // warm lazies (kernel registries etc.) on a throwaway entry
        failures.extend(run(true));
    }
    assert!(
        failures.is_empty(),
        "heap growth < 4096 || charge grew by the same failed:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn audit_fsst_eq_predicate_does_not_grow_the_array() {
    assert_predicate_does_not_grow_the_array(PredicateOp::Eq);
}

#[test]
fn audit_fsst_not_eq_predicate_does_not_grow_the_array() {
    assert_predicate_does_not_grow_the_array(PredicateOp::NotEq);
}

/// Control: expected to pass.
#[test]
fn audit_fsst_lt_predicate_does_not_grow_the_array() {
    assert_predicate_does_not_grow_the_array(PredicateOp::Lt);
}

/// Control: expected to pass.
#[test]
fn audit_fsst_like_prefix_predicate_does_not_grow_the_array() {
    assert_predicate_does_not_grow_the_array(PredicateOp::LikePrefix);
}

/// Control: expected to pass.
#[test]
fn audit_fsst_like_contains_predicate_does_not_grow_the_array() {
    assert_predicate_does_not_grow_the_array(PredicateOp::LikeContains);
}

