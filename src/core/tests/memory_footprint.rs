//! Regression test for liquid-cache#43: the process heap must track the
//! cache's own budget tally for a working set larger than the memory tier.
//! Every allocation in this test binary goes through a counting allocator, so
//! "live" below is exact live heap, not RSS.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, StringViewArray};
use liquid_cache::cache::{
    AlwaysHydrate, CacheEntry, EntryID, LiquidCache, LiquidCacheBuilder, LiquidPolicy,
    TranscodeSqueezeEvict,
};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn on_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            on_alloc(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            on_alloc(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                on_alloc(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}
fn reset_peak() {
    PEAK.store(live(), Ordering::Relaxed);
}
fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}
fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

const WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "server", "request", "latency",
    "status", "payload", "region", "tenant", "shard",
];

/// ~1 KiB rows of semi-compressible text, unique per row, like a log/JSON column.
fn make_entry(seed: u64, rows: usize) -> ArrayRef {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let values: Vec<String> = (0..rows)
        .map(|row| {
            let mut s = format!("row={row:08} seed={seed:04} ");
            while s.len() < 1024 {
                let w = WORDS[(next() % WORDS.len() as u64) as usize];
                s.push_str(w);
                s.push_str(&format!("={:x} ", next() & 0xffff));
            }
            s
        })
        .collect();
    Arc::new(StringViewArray::from_iter_values(values))
}

fn indexed_bytes(cache: &LiquidCache) -> usize {
    let mut sum = 0;
    cache.for_each_entry(|_, e| sum += e.memory_usage_bytes());
    sum
}

/// Sum of `disk_bytes` over every on-disk entry, read straight from the
/// index rather than the budget, so the budget can be checked against it.
fn indexed_disk_bytes(cache: &LiquidCache) -> usize {
    let mut sum = 0;
    cache.for_each_entry(|_, e| {
        sum += match e {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            _ => 0,
        };
    });
    sum
}

fn report(cache: &LiquidCache, label: &str, baseline: usize) {
    let stats = cache.stats();
    let tally = cache.budget().memory_usage_bytes();
    eprintln!(
        "[{label}] live={:.1} MiB peak={:.1} MiB tally={:.1} MiB indexed={:.1} MiB disk={:.1} MiB \
         entries(arrow={} liquid={} squeezed={} disk_liquid={} disk_arrow={})",
        mib(live().saturating_sub(baseline)),
        mib(peak().saturating_sub(baseline)),
        mib(tally),
        mib(indexed_bytes(cache)),
        mib(cache.budget().disk_usage_bytes()),
        stats.memory_arrow_entries,
        stats.memory_liquid_entries,
        stats.memory_squeezed_liquid_entries,
        stats.disk_liquid_entries,
        stats.disk_arrow_entries,
    );
}

#[tokio::test]
async fn heap_footprint_tracks_budget_for_oversized_working_set() {
    const MEMORY_TIER: usize = 32 * 1024 * 1024;
    const ROWS: usize = 2048;
    const ENTRIES: usize = 96; // ~2 MiB arrow each → ~6x the memory tier

    let dir = tempfile::tempdir().unwrap();
    let store = liquid_cache::store::mount(&dir.path().join("cache.t4"))
        .await
        .unwrap();
    let cache = LiquidCacheBuilder::new()
        .with_max_memory_bytes(MEMORY_TIER)
        .with_max_disk_bytes(4 << 30)
        .with_batch_size(8192)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_store(store)
        .build()
        .await;

    let baseline = live();
    reset_peak();

    // Fill: mimics read_parquet_batch_and_fill_cache inserting each decoded
    // column batch as arrow, dropping the caller's copy right after.
    for i in 0..ENTRIES {
        let arr = make_entry(i as u64, ROWS);
        cache.insert(EntryID::from(i), arr).await.unwrap();
    }
    report(&cache, "after fill", baseline);
    let idle_after_fill = live() - baseline;
    let tally_after_fill = cache.budget().memory_usage_bytes();
    assert!(
        idle_after_fill as f64 <= 1.15 * tally_after_fill as f64,
        "live ({:.1} MiB) must track the budget tally ({:.1} MiB) within 1.15x: \
         evicted entries must not outlive the index (congee deferred-drop measured 6.14x)",
        mib(idle_after_fill),
        mib(tally_after_fill),
    );
    // Reset the IO counters here so the write count measured after the read
    // pass below reflects only the warm churn, not the fill's own writes.
    let _ = cache.observer().runtime_snapshot();

    // Warm churn: a second scan over the same working set, every batch read
    // once per pass, the materialized array dropped immediately.
    reset_peak();
    for _pass in 0..2 {
        for i in 0..ENTRIES {
            let arr = cache.get(&EntryID::from(i)).await.unwrap();
            assert_eq!(arr.len(), ROWS);
            drop(arr);
        }
    }
    // Read before `report` (which also drains the counters via `cache.stats()`),
    // so this reflects only the warm churn since the reset after fill above.
    let rt = cache.observer().runtime_snapshot();
    report(&cache, "after reads", baseline);
    let peak_reads = peak() - baseline;
    assert!(
        peak_reads <= 2 * MEMORY_TIER + MEMORY_TIER / 2,
        "peak during reads ({:.1} MiB) must stay within 2.5x the memory tier ({:.1} MiB) \
         (measured 6.09x before the congee fix, 1.71x after)",
        mib(peak_reads),
        mib(MEMORY_TIER),
    );
    assert!(
        (rt.write_io_count as usize) <= ENTRIES / 2,
        "disk writes during the read pass ({}) must stay under half of the entry count ({}): \
         a hydrated entry must not rewrite its disk copy on re-eviction (measured 190 for 96 entries before the fix, 30 after)",
        rt.write_io_count,
        ENTRIES,
    );

    // Flush every remaining memory entry to a disk stub: one with no disk
    // copy yet must write, one that was hydrated must flip for free via the
    // `write_in_memory_batch_to_disk` shortcut. Snapshot stats and IO
    // counters right before, so both checks below are scoped to the flush.
    let stats_before_flush = cache.stats();
    let memory_entries_before_flush = stats_before_flush.memory_arrow_entries
        + stats_before_flush.memory_liquid_entries
        + stats_before_flush.memory_squeezed_liquid_entries;
    cache.flush_all_to_disk().await.unwrap();
    let flush_rt = cache.observer().runtime_snapshot();
    assert!(
        (flush_rt.write_io_count as usize) < memory_entries_before_flush,
        "flush wrote {} times for {} in-memory entries: a hydrated entry's re-eviction must \
         not reserve its disk object twice (drifted to 2.7x before the fix)",
        flush_rt.write_io_count,
        memory_entries_before_flush,
    );

    let budget_disk_bytes = cache.budget().disk_usage_bytes();
    let index_disk_bytes = indexed_disk_bytes(&cache);
    eprintln!(
        "after flush: writes={} for {memory_entries_before_flush} memory entries, \
         disk budget={:.1} MiB indexed_disk_entries={:.1} MiB",
        flush_rt.write_io_count,
        mib(budget_disk_bytes),
        mib(index_disk_bytes),
    );
    assert_eq!(
        budget_disk_bytes, index_disk_bytes,
        "disk budget must equal the indexed on-disk bytes: a hydrated entry's re-eviction \
         must not reserve its disk object twice (drifted to 2.7x before the fix)"
    );

    // What survives once the index is emptied is held by the store, the
    // policy, or the compressor state — not by indexed entries.
    cache.reset();
    report(&cache, "after reset", baseline);
    let idle_after_reset = live() - baseline;
    assert!(
        idle_after_reset <= 1024 * 1024,
        "live after reset ({:.2} MiB) must drop back under 1 MiB",
        mib(idle_after_reset),
    );
}
