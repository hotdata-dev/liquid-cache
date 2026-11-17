//! Size audit of a real cache: what the allocator says the process holds
//! while the cache works, against the memory budget it was given.
//!
//! Every `audit_*` test asserts the behaviour a caller sizing the cache
//! relies on, so a failure is a finding, not a broken test. Run:
//!
//! ```text
//! cargo +1.96.0 test -p liquid-cache --test audit_cache_heap -- --nocapture --test-threads=1
//! ```
//!
//! The counters are process wide, so the tests take a lock and run one at a
//! time whatever `--test-threads` says.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};

use arrow::array::{
    Array, ArrayRef, Decimal128Array, DictionaryArray, Float64Array, Int64Array, StringArray,
    StringViewArray, TimestampMicrosecondArray,
};
use arrow::datatypes::UInt16Type;
use liquid_cache::cache::{EntryID, LiquidCache, LiquidCacheBuilder};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

fn sub(n: usize) {
    LIVE.fetch_sub(n, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                add(new_size - layout.size());
            } else {
                sub(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live() -> i64 {
    LIVE.load(Relaxed) as i64
}

fn reset_peak() {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

fn peak() -> i64 {
    PEAK.load(Relaxed) as i64
}

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

const ROWS: usize = 8192;
const ENTRIES: usize = 64;
const BUDGET: usize = 512 * 1024;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Int64,
    Float64,
    Utf8LowCard,
    Utf8HighCard,
    StringView,
    DictUtf8,
    TimestampUs,
    Decimal128,
}

const KINDS: [Kind; 8] = [
    Kind::Int64,
    Kind::Float64,
    Kind::Utf8LowCard,
    Kind::Utf8HighCard,
    Kind::StringView,
    Kind::DictUtf8,
    Kind::TimestampUs,
    Kind::Decimal128,
];

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn make(kind: Kind, i: usize) -> ArrayRef {
    let mut s = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    match kind {
        Kind::Int64 => Arc::new(Int64Array::from_iter_values(
            (0..ROWS).map(|_| (lcg(&mut s) % 1_000_000) as i64),
        )),
        Kind::Float64 => Arc::new(Float64Array::from_iter_values(
            (0..ROWS).map(|_| (lcg(&mut s) % 1_000_000) as f64 / 100.0),
        )),
        Kind::Utf8LowCard => Arc::new(StringArray::from_iter_values(
            (0..ROWS).map(|_| format!("category-value-{:04}", lcg(&mut s) % 16)),
        )),
        Kind::Utf8HighCard => Arc::new(StringArray::from_iter_values((0..ROWS).map(|r| {
            format!("https://example.com/path/{}/{}/item?id={}", i, r, lcg(&mut s))
        }))),
        Kind::StringView => Arc::new(StringViewArray::from_iter_values((0..ROWS).map(|r| {
            format!("https://example.com/view/{}/{}/item?id={}", i, r, lcg(&mut s))
        }))),
        Kind::DictUtf8 => {
            let values: Vec<String> = (0..ROWS)
                .map(|_| format!("dictionary-value-{:05}", lcg(&mut s) % 200))
                .collect();
            let dict: DictionaryArray<UInt16Type> = values.iter().map(|v| v.as_str()).collect();
            Arc::new(dict)
        }
        Kind::TimestampUs => Arc::new(TimestampMicrosecondArray::from_iter_values(
            (0..ROWS).map(|r| 1_700_000_000_000_000 + (r as i64) * 1000 + (lcg(&mut s) % 997) as i64),
        )),
        Kind::Decimal128 => Arc::new(
            Decimal128Array::from_iter_values((0..ROWS).map(|_| (lcg(&mut s) % 10_000_000) as i128))
                .with_precision_and_scale(20, 2)
                .unwrap(),
        ),
    }
}

/// A cache with the default policies (`TranscodeEvict`, `LiquidPolicy`,
/// `AlwaysHydrate`) over a store in `dir`.
async fn build_cache(max_memory: usize, dir: &std::path::Path) -> Arc<LiquidCache> {
    let store = t4::mount(dir.join("cache.t4")).await.expect("mount t4");
    LiquidCacheBuilder::new()
        .with_max_memory_bytes(max_memory)
        .with_max_disk_bytes(usize::MAX / 4)
        .with_store(store)
        .build()
        .await
}

/// Debug builds keep every cache event in an unbounded trace. Drop it, so it
/// is not counted as heap the cache holds.
fn drain(cache: &LiquidCache) {
    drop(cache.consume_event_trace());
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Run one throwaway cache through insert, read, flush and hydrate, so
/// lazily built statics and the runtime's pools exist before any baseline.
async fn warm_up(kind: Kind) {
    let dir = tempfile::tempdir().unwrap();
    let cache = build_cache(BUDGET, dir.path()).await;
    for i in 0..16 {
        let _ = cache.insert(EntryID::from(i), 0, make(kind, i + 1000)).await;
    }
    for i in 0..16 {
        drop(cache.get(&EntryID::from(i), 0).await);
    }
    let _ = cache.flush_all_to_disk().await;
    for i in 0..16 {
        drop(cache.get(&EntryID::from(i), 0).await);
    }
}

/// What a pass over the cache peaked at, above the heap the empty cache held.
#[derive(Debug)]
struct Pass {
    /// Highest live heap during any one operation.
    peak: i64,
    /// Live heap, and the memory budget's charge, at the end of the operation
    /// that peaked.
    live_after_peak_op: i64,
    budget_after_peak_op: usize,
    /// The operation that peaked.
    at: usize,
    /// Largest array an operation handed over (inserts) or got back (reads).
    largest_array: usize,
    failed: usize,
}

struct Scenario {
    cache: Arc<LiquidCache>,
    base: i64,
    _dir: tempfile::TempDir,
}

async fn scenario(kind: Kind) -> Scenario {
    warm_up(kind).await;
    let dir = tempfile::tempdir().unwrap();
    let cache = build_cache(BUDGET, dir.path()).await;
    drain(&cache);
    Scenario {
        base: live(),
        cache,
        _dir: dir,
    }
}

/// Insert `ENTRIES` arrays. The peak is re-armed once each source array is
/// built: the array itself counts, since the cache is handed it, but the
/// scratch that built it does not.
async fn insert_pass(s: &Scenario, kind: Kind) -> Pass {
    let mut pass = Pass {
        peak: 0,
        live_after_peak_op: 0,
        budget_after_peak_op: 0,
        at: 0,
        largest_array: 0,
        failed: 0,
    };
    for i in 0..ENTRIES {
        let array = make(kind, i);
        pass.largest_array = pass.largest_array.max(array.get_array_memory_size());
        drain(&s.cache);
        reset_peak();
        if s.cache.insert(EntryID::from(i), 0, array).await.is_err() {
            pass.failed += 1;
        }
        let peak = peak() - s.base;
        drain(&s.cache);
        if peak > pass.peak {
            pass.peak = peak;
            pass.at = i;
            pass.live_after_peak_op = live() - s.base;
            pass.budget_after_peak_op = s.cache.budget().memory_usage_bytes();
        }
    }
    pass
}

/// Read every entry back and drop it. Entries on disk are hydrated.
async fn read_pass(s: &Scenario) -> Pass {
    let mut pass = Pass {
        peak: 0,
        live_after_peak_op: 0,
        budget_after_peak_op: 0,
        at: 0,
        largest_array: 0,
        failed: 0,
    };
    for i in 0..ENTRIES {
        let id = EntryID::from(i);
        drain(&s.cache);
        reset_peak();
        let read = s.cache.get(&id, 0).await;
        let peak = peak() - s.base;
        match read {
            Some(array) => {
                pass.largest_array = pass.largest_array.max(array.get_array_memory_size());
            }
            None => pass.failed += 1,
        }
        drain(&s.cache);
        if peak > pass.peak {
            pass.peak = peak;
            pass.at = i;
            pass.live_after_peak_op = live() - s.base;
            pass.budget_after_peak_op = s.cache.budget().memory_usage_bytes();
        }
    }
    pass
}

fn tiers(cache: &LiquidCache) -> String {
    let st = cache.stats();
    format!(
        "memArrow/memLiquid/diskLiquid/diskArrow={}/{}/{}/{} budget.mem={} disk={}",
        st.memory_arrow_entries,
        st.memory_liquid_entries,
        st.disk_liquid_entries,
        st.disk_arrow_entries,
        st.memory_usage_bytes,
        st.disk_usage_bytes
    )
}

fn report(what: &str, kind: Kind, pass: &Pass, cache: &LiquidCache) {
    println!(
        "{what} {kind:?}: peak live heap = {} = {:.2} x budget ({BUDGET}); at op {} \
         (after it: live {} vs budget.mem {}); largest array {}; failed {}; end state {}",
        pass.peak,
        pass.peak as f64 / BUDGET as f64,
        pass.at,
        pass.live_after_peak_op,
        pass.budget_after_peak_op,
        pass.largest_array,
        pass.failed,
        tiers(cache),
    );
}

/// Item 5. `max_memory_bytes` is the caller's bound on what the cache holds.
/// Inserting may transcode and spill, and that work needs scratch, but the
/// whole of it should stay within twice the budget.
fn assert_insert_peak_within_twice_the_budget(kind: Kind) {
    let _serial = serial();
    let pass = runtime().block_on(async {
        let s = scenario(kind).await;
        let pass = insert_pass(&s, kind).await;
        report("insert x64", kind, &pass, &s.cache);
        pass
    });
    assert!(
        pass.peak <= 2 * BUDGET as i64,
        "peak live heap during inserts <= 2 * max_memory_bytes failed for {kind:?}: \
         peak {} = {:.2} x {BUDGET}",
        pass.peak,
        pass.peak as f64 / BUDGET as f64
    );
}

/// Item 5, the read pass: reading every entry back (hydrating the spilled
/// ones) should stay within twice the budget too.
fn assert_read_peak_within_twice_the_budget(kind: Kind) {
    let _serial = serial();
    let (pass, hydrate) = runtime().block_on(async {
        let s = scenario(kind).await;
        let inserted = insert_pass(&s, kind).await;
        report("insert x64", kind, &inserted, &s.cache);
        let pass = read_pass(&s).await;
        report("read x64", kind, &pass, &s.cache);
        s.cache.flush_all_to_disk().await.unwrap();
        drain(&s.cache);
        let hydrate = read_pass(&s).await;
        report("flush_all then read x64", kind, &hydrate, &s.cache);
        (pass, hydrate)
    });
    let worst = pass.peak.max(hydrate.peak);
    assert!(
        worst <= 2 * BUDGET as i64,
        "peak live heap during reads <= 2 * max_memory_bytes failed for {kind:?}: \
         read pass {} = {:.2} x, read-after-flush pass {} = {:.2} x {BUDGET} \
         (largest array returned {})",
        pass.peak,
        pass.peak as f64 / BUDGET as f64,
        hydrate.peak,
        hydrate.peak as f64 / BUDGET as f64,
        pass.largest_array.max(hydrate.largest_array),
    );
}

#[test]
fn audit_insert_peak_within_twice_the_budget_utf8() {
    assert_insert_peak_within_twice_the_budget(Kind::Utf8HighCard);
}

#[test]
fn audit_insert_peak_within_twice_the_budget_string_view() {
    assert_insert_peak_within_twice_the_budget(Kind::StringView);
}

#[test]
fn audit_insert_peak_within_twice_the_budget_int64() {
    assert_insert_peak_within_twice_the_budget(Kind::Int64);
}

#[test]
fn audit_read_peak_within_twice_the_budget_utf8() {
    assert_read_peak_within_twice_the_budget(Kind::Utf8HighCard);
}

#[test]
fn audit_read_peak_within_twice_the_budget_int64() {
    assert_read_peak_within_twice_the_budget(Kind::Int64);
}

/// Item 7, a report rather than an assertion. Reading an in-memory Liquid
/// entry decodes it into a new Arrow array the caller owns. The budget
/// charges the Liquid entry only, so the decoded copy is counted nowhere.
/// Whether the cache *should* count memory it handed away is a design
/// question, so this prints the sizes and asserts only what it set up.
#[test]
fn audit_report_read_of_liquid_entry_returns_uncharged_arrow() {
    let _serial = serial();
    println!(
        "\n{:<14}|{:>9}|{:>10}|{:>10}|{:>10}|{:>8}|{:>13}",
        "kind", "charged", "returned", "heap +", "budget +", "ret/chg", "tiers A/L"
    );
    runtime().block_on(async {
        for kind in KINDS {
            warm_up(kind).await;
            let dir = tempfile::tempdir().unwrap();
            let first = make(kind, 0);
            let second = make(kind, 1);
            // Room for one Arrow array and a half: the second insert makes
            // the cache transcode the first to Liquid.
            let budget = first.get_array_memory_size() * 3 / 2;
            let cache = build_cache(budget, dir.path()).await;
            cache.insert(EntryID::from(0), 0, first).await.unwrap();
            cache.insert(EntryID::from(1), 0, second).await.unwrap();
            let st = cache.stats();
            if st.memory_liquid_entries != 1 {
                println!("{:<14}| no in-memory Liquid entry: {}", format!("{kind:?}"), tiers(&cache));
                continue;
            }
            let charged = st.memory_liquid_bytes;
            let id = EntryID::from(0);
            drain(&cache);
            let budget_before = cache.budget().memory_usage_bytes();
            let live_before = live();
            let returned = cache.get(&id, 0).await.expect("present");
            drain(&cache);
            let heap_growth = live() - live_before;
            let budget_growth =
                cache.budget().memory_usage_bytes() as i64 - budget_before as i64;
            let after = cache.stats();
            println!(
                "{:<14}|{:>9}|{:>10}|{:>10}|{:>10}|{:>8.2}|{:>13}",
                format!("{kind:?}"),
                charged,
                returned.get_array_memory_size(),
                heap_growth,
                budget_growth,
                returned.get_array_memory_size() as f64 / charged as f64,
                format!("{}/{}", after.memory_arrow_entries, after.memory_liquid_entries),
            );
        }
    });
}

/// Items 2 and 4 at the cache level, a report: under a small budget every
/// entry left in memory is Liquid, some transcoded there and some hydrated
/// from disk. Prints what those entries hold against what they are charged.
#[test]
fn audit_report_liquid_entries_heap_vs_budget() {
    let _serial = serial();
    println!(
        "\n{:<14}|{:>8}|{:>12}|{:>12}|{:>7}|{:>12}|{:>12}|{:>7}",
        "kind", "entries", "budget.mem", "live heap", "ratio", "budget.mem", "live heap", "ratio"
    );
    runtime().block_on(async {
        for kind in KINDS {
            warm_up(kind).await;
            let dir = tempfile::tempdir().unwrap();
            let cache = build_cache(BUDGET, dir.path()).await;
            drain(&cache);
            let base = live();
            let n = ENTRIES;
            for i in 0..n {
                let _ = cache.insert(EntryID::from(i), 0, make(kind, i)).await;
            }
            // After the inserts, then after everything was spilled and read
            // back.
            for i in 0..n {
                drop(cache.get(&EntryID::from(i), 0).await);
            }
            drain(&cache);
            let (budget_1, live_1, tiers_1) =
                (cache.budget().memory_usage_bytes(), live() - base, tiers(&cache));
            cache.flush_all_to_disk().await.unwrap();
            for i in 0..n {
                drop(cache.get(&EntryID::from(i), 0).await);
            }
            drain(&cache);
            let (budget_2, live_2) = (cache.budget().memory_usage_bytes(), live() - base);
            println!(
                "{:<14}|{:>8}|{:>12}|{:>12}|{:>7.2}|{:>12}|{:>12}|{:>7.2}| {} -> {}",
                format!("{kind:?}"),
                n,
                budget_1,
                live_1,
                live_1 as f64 / budget_1.max(1) as f64,
                budget_2,
                live_2,
                live_2 as f64 / budget_2.max(1) as f64,
                tiers_1,
                tiers(&cache),
            );
        }
    });
}
