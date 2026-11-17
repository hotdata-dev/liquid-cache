//! Measures the gap between the cache's memory budget counter and what the
//! process really holds, using a counting global allocator.
//!
//! Prints a table; asserts nothing. Run with:
//!
//! ```text
//! cargo +1.96.0 test -p liquid-cache --test alloc_vs_budget -- --nocapture --test-threads=1
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

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

const ROWS: usize = 8192;

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
    /// An 8192-row window of a fresh 128K-row buffer, one buffer per entry.
    SlicedWindow,
    /// Every entry is a disjoint 8192-row window of ONE buffer.
    SharedBuffer,
    /// Every entry is the same `Arc`.
    SameArc,
}

const KINDS: [Kind; 11] = [
    Kind::Int64,
    Kind::Float64,
    Kind::Utf8LowCard,
    Kind::Utf8HighCard,
    Kind::StringView,
    Kind::DictUtf8,
    Kind::TimestampUs,
    Kind::Decimal128,
    Kind::SlicedWindow,
    Kind::SharedBuffer,
    Kind::SameArc,
];

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

struct Generator {
    kind: Kind,
    n: usize,
    shared: Option<ArrayRef>,
}

impl Generator {
    fn new(kind: Kind, n: usize) -> Self {
        Self {
            kind,
            n,
            shared: None,
        }
    }

    fn make(&mut self, i: usize) -> ArrayRef {
        let mut s = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        match self.kind {
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
                format!(
                    "https://example.com/path/{}/{}/item?id={}",
                    i,
                    r,
                    lcg(&mut s)
                )
            }))),
            Kind::StringView => {
                Arc::new(StringViewArray::from_iter_values((0..ROWS).map(|r| {
                    format!(
                        "https://example.com/view/{}/{}/item?id={}",
                        i,
                        r,
                        lcg(&mut s)
                    )
                })))
            }
            Kind::DictUtf8 => {
                let values: Vec<String> = (0..ROWS)
                    .map(|_| format!("dictionary-value-{:05}", lcg(&mut s) % 200))
                    .collect();
                let dict: DictionaryArray<UInt16Type> =
                    values.iter().map(|v| v.as_str()).collect();
                Arc::new(dict)
            }
            Kind::TimestampUs => Arc::new(TimestampMicrosecondArray::from_iter_values(
                (0..ROWS).map(|r| 1_700_000_000_000_000 + (r as i64) * 1000 + (lcg(&mut s) % 997) as i64),
            )),
            Kind::Decimal128 => Arc::new(
                Decimal128Array::from_iter_values(
                    (0..ROWS).map(|_| (lcg(&mut s) % 10_000_000) as i128),
                )
                .with_precision_and_scale(20, 2)
                .unwrap(),
            ),
            Kind::SlicedWindow => {
                let big = Int64Array::from_iter_values(
                    (0..ROWS * 16).map(|_| (lcg(&mut s) % 1_000_000) as i64),
                );
                Arc::new(big.slice(ROWS * 3, ROWS))
            }
            Kind::SharedBuffer => {
                let n = self.n;
                let base = self.shared.get_or_insert_with(|| {
                    let mut s = 42u64;
                    Arc::new(Int64Array::from_iter_values(
                        (0..ROWS * n).map(|_| (lcg(&mut s) % 1_000_000) as i64),
                    ))
                });
                base.slice(ROWS * i, ROWS)
            }
            Kind::SameArc => {
                let base = self.shared.get_or_insert_with(|| {
                    let mut s = 42u64;
                    Arc::new(Int64Array::from_iter_values(
                        (0..ROWS).map(|_| (lcg(&mut s) % 1_000_000) as i64),
                    ))
                });
                base.clone()
            }
        }
    }
}

fn header() {
    println!(
        "{:<40} | {:>12} | {:>12} | {:>7} | {:>12} | notes",
        "scenario", "budget.mem", "alloc live d", "ratio", "peak d in op"
    );
    println!("{}", "-".repeat(150));
}

fn ratio(budget: usize, delta: i64) -> String {
    if budget == 0 {
        "n/a".to_string()
    } else {
        format!("{:.2}", delta as f64 / budget as f64)
    }
}

/// Print a row. `ratio` is allocator-live-delta / budget.
fn row(scenario: &str, cache: &LiquidCache, base: i64, peak_delta: i64) {
    // The debug build records every internal event in an unbounded trace;
    // drain it so it does not pollute the measurement.
    drop(cache.consume_event_trace());
    let budget = cache.budget().memory_usage_bytes();
    let delta = live() - base;
    let st = cache.stats();
    println!(
        "{:<40} | {:>12} | {:>12} | {:>7} | {:>12} | idx_sum={} memA/memL/dskL/dskA={}/{}/{}/{} disk={}",
        scenario,
        budget,
        delta,
        ratio(budget, delta),
        peak_delta,
        st.memory_arrow_bytes + st.memory_liquid_bytes,
        st.memory_arrow_entries,
        st.memory_liquid_entries,
        st.disk_liquid_entries,
        st.disk_arrow_entries,
        st.disk_usage_bytes,
    );
}

async fn build_cache(max_memory: usize, dir: &std::path::Path) -> Arc<LiquidCache> {
    let store = t4::mount(dir.join("cache.t4")).await.expect("mount t4");
    LiquidCacheBuilder::new()
        .with_max_memory_bytes(max_memory)
        .with_max_disk_bytes(usize::MAX / 4)
        .with_store(store)
        .build()
        .await
}

/// Insert `n` arrays; returns the peak allocator delta seen during inserts
/// (peak is re-armed after each source array is built, so the source array
/// itself counts but its builder's scratch does not).
async fn insert_all(cache: &Arc<LiquidCache>, generator: &mut Generator, n: usize, base: i64) -> (i64, usize) {
    let mut max_peak = 0i64;
    let mut failures = 0;
    for i in 0..n {
        let array = generator.make(i);
        reset_peak();
        if cache.insert(EntryID::from(i), 0, array).await.is_err() {
            failures += 1;
        }
        max_peak = max_peak.max(peak() - base);
    }
    (max_peak, failures)
}

async fn read_all(cache: &Arc<LiquidCache>, n: usize, base: i64) -> (i64, usize) {
    let mut max_peak = 0i64;
    let mut hits = 0;
    reset_peak();
    for i in 0..n {
        let id = EntryID::from(i);
        if let Some(array) = cache.get(&id, 0).await {
            hits += 1;
            drop(array);
        }
        max_peak = max_peak.max(peak() - base);
    }
    (max_peak, hits)
}

async fn run_kind(kind: Kind, budget_label: &str, max_memory: usize, n: usize) {
    let base0 = live();
    let dir = tempfile::tempdir().unwrap();
    let cache = build_cache(max_memory, dir.path()).await;
    let base = live();
    let tag = format!("{budget_label}/{kind:?}");

    let mut generator = Generator::new(kind, n);
    let (peak_insert, failures) = insert_all(&cache, &mut generator, n, base).await;
    drop(generator);
    row(&format!("{tag} 1.insert x{n}"), &cache, base, peak_insert);
    if failures > 0 {
        println!("    ({failures} inserts returned CacheFull)");
    }

    let (peak_read, hits) = read_all(&cache, n, base).await;
    row(&format!("{tag} 2.read all ({hits} hit)"), &cache, base, peak_read);

    reset_peak();
    let flushed = cache.flush_all_to_disk().await;
    let peak_flush = peak() - base;
    row(
        &format!("{tag} 3.flush_all ok={}", flushed.is_ok()),
        &cache,
        base,
        peak_flush,
    );

    let (peak_read, hits) = read_all(&cache, n, base).await;
    row(&format!("{tag} 4.hydrate all ({hits} hit)"), &cache, base, peak_read);

    reset_peak();
    cache.reset();
    row(&format!("{tag} 5.reset()"), &cache, base, peak() - base);

    drop(cache);
    println!(
        "{:<40} | {:>12} | {:>12} | {:>7} | {:>12} | vs pre-build baseline (empty cache+store cost was {})",
        format!("{tag} 6.cache dropped"),
        "-",
        live() - base0,
        "n/a",
        "-",
        base - base0,
    );
    drop(dir);
}

fn sample(label: &str, iter: usize, cache: &LiquidCache, base: i64) -> (usize, i64) {
    drop(cache.consume_event_trace());
    let budget = cache.budget().memory_usage_bytes();
    let delta = live() - base;
    let st = cache.stats();
    println!(
        "{:<40} | {:>12} | {:>12} | {:>7} | {:>12} | iter={} idx_sum={} memA/memL/dskL/dskA={}/{}/{}/{} disk={}",
        label,
        budget,
        delta,
        ratio(budget, delta),
        "-",
        iter,
        st.memory_arrow_bytes + st.memory_liquid_bytes,
        st.memory_arrow_entries,
        st.memory_liquid_entries,
        st.disk_liquid_entries,
        st.disk_arrow_entries,
        st.disk_usage_bytes,
    );
    (budget, delta)
}

fn drift_summary(label: &str, samples: &[(usize, i64)]) {
    // Skip the first sample: one-time warm-up allocations.
    let s = &samples[1..];
    let monotonic_live = s.windows(2).all(|w| w[1].1 > w[0].1);
    let monotonic_budget = s.windows(2).all(|w| w[1].0 > w[0].0);
    println!(
        "  => {label}: live {} -> {} (d={}, strictly rising={monotonic_live}); budget {} -> {} (d={}, strictly rising={monotonic_budget})",
        s[0].1,
        s[s.len() - 1].1,
        s[s.len() - 1].1 - s[0].1,
        s[0].0,
        s[s.len() - 1].0,
        s[s.len() - 1].0 as i64 - s[0].0 as i64,
    );
}

/// Overwrite the same keys `iters` times.
async fn drift_overwrite(kind: Kind, budget_label: &str, max_memory: usize, iters: usize) {
    let dir = tempfile::tempdir().unwrap();
    let cache = build_cache(max_memory, dir.path()).await;
    let base = live();
    let keys = 16;
    let label = format!("drift-overwrite {budget_label}/{kind:?}");
    let mut samples = Vec::new();
    let mut generator = Generator::new(kind, keys);
    let step = (iters / 10).max(1);
    for it in 0..=iters {
        for k in 0..keys {
            // vary content per iteration
            let array = generator.make(k + it * keys);
            let _ = cache.insert(EntryID::from(k), 0, array).await;
        }
        if it % step == 0 {
            samples.push(sample(&label, it, &cache, base));
        }
    }
    drift_summary(&label, &samples);
}

/// insert -> (evict under pressure | flush_all) -> hydrate by reading, repeated.
async fn drift_cycle(budget_label: &str, max_memory: usize, cycles: usize, flush: bool) {
    let dir = tempfile::tempdir().unwrap();
    let cache = build_cache(max_memory, dir.path()).await;
    let base = live();
    let per_kind = 8;
    let kinds = [Kind::Int64, Kind::Utf8HighCard, Kind::StringView, Kind::Utf8LowCard];
    let label = format!(
        "drift-cycle {budget_label} {}",
        if flush { "ins/flush/read" } else { "ins/read" }
    );
    let mut samples = Vec::new();
    let step = (cycles / 10).max(1);
    for it in 0..=cycles {
        for (ki, kind) in kinds.iter().enumerate() {
            let mut generator = Generator::new(*kind, per_kind);
            for k in 0..per_kind {
                let array = generator.make(k + it * per_kind);
                let _ = cache
                    .insert(EntryID::from(ki * per_kind + k), 0, array)
                    .await;
            }
        }
        if flush {
            let _ = cache.flush_all_to_disk().await;
        }
        for id in 0..kinds.len() * per_kind {
            let id = EntryID::from(id);
            drop(cache.get(&id, 0).await);
        }
        if it % step == 0 {
            samples.push(sample(&label, it, &cache, base));
        }
    }
    drift_summary(&label, &samples);
}

fn guarded(rt: &tokio::runtime::Runtime, what: &str, fut: impl Future<Output = ()>) {
    let started = std::time::Instant::now();
    if std::panic::catch_unwind(AssertUnwindSafe(|| rt.block_on(fut))).is_err() {
        println!("{what}: PANICKED (see stderr)");
    }
    eprintln!("[{what}] took {:?}", started.elapsed());
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
fn alloc_vs_budget() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let n = env_usize("AVB_N", 64);
    let small = env_usize("AVB_SMALL_BUDGET", 512 * 1024);
    let overwrite_iters = env_usize("AVB_OVERWRITE_ITERS", 1000);
    let cycles = env_usize("AVB_CYCLES", 200);
    let large = 4usize << 30;

    // Warm up stdout, the runtime and the blocking pool before any baseline.
    println!("alloc_vs_budget: n={n} rows/array={ROWS} small_budget={small} large_budget={large}");
    guarded(&rt, "warmup", async {
        let dir = tempfile::tempdir().unwrap();
        let cache = build_cache(small, dir.path()).await;
        let mut generator = Generator::new(Kind::Utf8HighCard, 16);
        insert_all(&cache, &mut generator, 16, 0).await;
        read_all(&cache, 16, 0).await;
    });

    println!("\n== Scenario 1/3/5: LARGE budget (nothing evicted) ==");
    header();
    for kind in KINDS {
        guarded(&rt, &format!("large/{kind:?}"), run_kind(kind, "large", large, n));
    }

    println!("\n== Scenario 2/3/5: SMALL budget ({small} bytes; transcode/squeeze/spill) ==");
    header();
    for kind in KINDS {
        guarded(&rt, &format!("small/{kind:?}"), run_kind(kind, "small", small, n));
    }

    println!("\n== Scenario 4: drift ==");
    header();
    for kind in [Kind::Int64, Kind::Utf8HighCard, Kind::StringView] {
        guarded(
            &rt,
            &format!("drift-overwrite large/{kind:?}"),
            drift_overwrite(kind, "large", large, overwrite_iters),
        );
    }
    guarded(
        &rt,
        "drift-overwrite small/Utf8HighCard",
        drift_overwrite(
            Kind::Utf8HighCard,
            "small",
            small,
            env_usize("AVB_SMALL_OVERWRITE_ITERS", overwrite_iters),
        ),
    );
    guarded(&rt, "drift-cycle small", drift_cycle("small", small, cycles, false));
    guarded(&rt, "drift-cycle small flush", drift_cycle("small", small, cycles, true));
    guarded(&rt, "drift-cycle large flush", drift_cycle("large", large, cycles, true));
}

/// Isolates the liquid array from the cache: what one `LiquidArray` really
/// holds versus what `get_array_memory_size()` (the number the budget is
/// charged) says.
#[test]
fn liquid_array_self_size() {
    use liquid_cache::liquid_array::LiquidArray;
    println!("\n== Liquid array alone: reported vs allocator (one 8192-row array) ==");
    println!(
        "{:<28} | {:>10} | {:>10} | {:>6} | {:>10} | {:>10} | {:>6} | {:>9} | {:>10}",
        "kind/path", "reported", "alloc live", "ratio", "peak d", "arrow src", "", "to_bytes", "arrow back"
    );
    for kind in KINDS {
        let mut generator = Generator::new(kind, 4);
        let source = generator.make(1);
        let source_size = source.get_array_memory_size();
        // Warm up lazily initialised session/compressor statics.
        drop(LiquidArray::from_arrow_array(&source));

        let base = live();
        reset_peak();
        let Ok(liquid) = LiquidArray::from_arrow_array(&source) else {
            println!("{:<28} | unsupported by LiquidArray", format!("{kind:?}"));
            continue;
        };
        let held = live() - base;
        let transcode_peak = peak() - base;
        let reported = liquid.get_array_memory_size();
        let serialized = liquid.to_bytes();
        let serialized_len = serialized.len();
        println!(
            "{:<28} | {:>10} | {:>10} | {:>6} | {:>10} | {:>10} | {:>6} | {:>9} | {:>10}",
            format!("{kind:?}/transcode"),
            reported,
            held,
            ratio(reported, held),
            transcode_peak,
            source_size,
            "",
            serialized_len,
            "-",
        );
        drop(liquid);

        // The hydration path: bytes read from disk -> from_bytes.
        let base = live();
        reset_peak();
        let hydrated = LiquidArray::from_bytes(bytes::Bytes::from(serialized));
        // `serialized` moved in, so count it as part of what the array holds.
        let held = live() - base + serialized_len as i64;
        let reported = hydrated.get_array_memory_size();
        let base2 = live();
        reset_peak();
        let back = hydrated.to_arrow_array();
        let back_held = live() - base2;
        let back_reported = back.get_array_memory_size();
        println!(
            "{:<28} | {:>10} | {:>10} | {:>6} | {:>10} | {:>10} | {:>6} | {:>9} | {:>10}",
            format!("{kind:?}/from_bytes"),
            reported,
            held,
            ratio(reported, held),
            peak() - base2,
            "-",
            "",
            serialized_len,
            format!("{back_reported}/{back_held}"),
        );
    }
    println!("(arrow back = arrow get_array_memory_size / allocator bytes of the array decoded from liquid; peak d on from_bytes rows is the to_arrow_array peak)");
}
