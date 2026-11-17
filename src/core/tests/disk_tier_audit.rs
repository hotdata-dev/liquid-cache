//! Measurements of the disk tier: heap held per disk record, heap held per
//! key ever removed from the store, store file length against the disk
//! budget, and what a new cache sees of a store an earlier one filled.
//!
//! Run: `cargo +1.96.0 test -p liquid-cache --test disk_tier_audit -- --nocapture --test-threads=1`

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arrow::array::{ArrayRef, Int32Array};
use liquid_cache::cache::{
    AlwaysHydrate, CacheEntry, EntryID, Evict, LiquidCache, LiquidCacheBuilder, LiquidPolicy,
};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
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
            LIVE.fetch_add(new_size, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn small_array(seed: usize, rows: usize) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values(
        (0..rows).map(|i| (seed.wrapping_mul(31).wrapping_add(i)) as i32),
    ))
}

/// `dsync` is off so the runs finish; it changes neither heap nor layout.
async fn mount(path: &Path) -> t4::Store {
    t4::mount_with_options(
        path,
        t4::MountOptions {
            dsync: false,
            ..Default::default()
        },
    )
    .await
    .unwrap()
}

async fn cache_over(store: &t4::Store, memory: usize, disk: usize) -> Arc<LiquidCache> {
    LiquidCacheBuilder::new()
        .with_max_memory_bytes(memory)
        .with_max_disk_bytes(disk)
        .with_eviction_policy(Box::new(Evict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_store(store.clone())
        .build()
        .await
}

/// (memory entries, disk entries)
fn census(cache: &LiquidCache) -> (usize, usize) {
    let (mut memory, mut disk) = (0, 0);
    cache.for_each_entry(|_, entry| match entry {
        CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => disk += 1,
        _ => memory += 1,
    });
    (memory, disk)
}

/// The disk charge of one spilled array of `rows` rows.
async fn charge_of(rows: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let store = mount(&dir.path().join("probe.t4")).await;
    let cache = cache_over(&store, 1 << 20, 1 << 20).await;
    cache
        .insert(EntryID::from(0), 0, small_array(0, rows))
        .await
        .unwrap();
    cache.flush_all_to_disk().await.unwrap();
    cache.budget().disk_usage_bytes()
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}

#[tokio::test]
async fn claim1_heap_per_disk_record() {
    let n = env_usize("AUDIT_N1", 100_000);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.t4");
    let store = mount(&path).await;
    let cache = cache_over(&store, 64 * 1024, 1 << 40).await;

    // Warm every lazily allocated structure before the baseline.
    cache
        .insert(EntryID::from(usize::MAX - 1), 0, small_array(0, 16))
        .await
        .unwrap();
    cache.flush_all_to_disk().await.unwrap();

    if env_usize("AUDIT_DRAIN", 0) == 1 {
        drop(cache.consume_event_trace());
    }
    let before = live();
    let started = Instant::now();
    let mut insert_errors = 0;
    for i in 0..n {
        if cache
            .insert(EntryID::from(i), 0, small_array(i, 16))
            .await
            .is_err()
        {
            insert_errors += 1;
        }
    }
    cache.flush_all_to_disk().await.unwrap();
    if env_usize("AUDIT_DRAIN", 0) == 1 {
        let trace = cache.consume_event_trace();
        println!("drained trace (dropped before sampling)");
        drop(trace);
    }
    let after = live();

    let (memory, disk) = census(&cache);
    let delta = after.saturating_sub(before);
    let per = delta as f64 / disk.max(1) as f64;
    println!("--- CLAIM 1: heap per disk record ---");
    println!("inserted                    {n}");
    println!("insert errors               {insert_errors}");
    println!("index: memory entries       {memory}");
    println!("index: disk entries         {disk}");
    println!("store.len()                 {}", store.len().await.unwrap());
    println!("live heap before            {before}");
    println!("live heap after             {after}");
    println!("live heap delta             {delta}");
    println!("heap bytes per disk entry   {per:.1}");
    println!(
        "budget.memory_usage_bytes   {}",
        cache.budget().memory_usage_bytes()
    );
    println!(
        "budget.disk_usage_bytes     {}",
        cache.budget().disk_usage_bytes()
    );
    println!("file length                 {}", file_len(&path));
    println!(
        "projected heap @ 10M        {:.2} GiB",
        per * 1e7 / (1u64 << 30) as f64
    );
    println!(
        "projected heap @ 100M       {:.2} GiB",
        per * 1e8 / (1u64 << 30) as f64
    );
    println!("elapsed                     {:?}", started.elapsed());

    assert_eq!(memory, 0, "everything was flushed");
    // The invariant the claim says is broken: the memory budget accounts for
    // the heap the cache holds on behalf of its entries.
    assert!(
        cache.budget().memory_usage_bytes() >= delta / 2,
        "memory budget reports {} bytes while {} disk records hold {} bytes of heap",
        cache.budget().memory_usage_bytes(),
        disk,
        delta
    );
}

#[tokio::test]
async fn claim2_heap_per_key_ever_spilled() {
    let n = env_usize("AUDIT_N2", 200_000);
    let step = n / 10;
    let charge = charge_of(16).await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.t4");
    let store = mount(&path).await;
    let cache = cache_over(&store, 64 * 1024, charge * 1000).await;

    println!("--- CLAIM 2: heap vs distinct keys ever spilled ---");
    println!("disk charge per entry       {charge}");
    println!("disk budget                 {}", charge * 1000);
    println!("inserted | live heap | index mem | index disk | store.len | disk_usage | file len");

    // Fill past the disk budget once so the steady state is the baseline.
    let warm = 5_000;
    let mut insert_errors = 0;
    for i in 0..warm {
        if cache
            .insert(EntryID::from(i), 0, small_array(i, 16))
            .await
            .is_err()
        {
            insert_errors += 1;
        }
    }
    let mut samples = Vec::new();
    let started = Instant::now();
    for i in 0..=n {
        if i % step == 0 {
            if env_usize("AUDIT_DRAIN", 0) == 1 {
                drop(cache.consume_event_trace());
            }
            let (memory, disk) = census(&cache);
            let heap = live();
            println!(
                "{:>8} | {:>9} | {:>9} | {:>10} | {:>9} | {:>10} | {:>8}",
                i,
                heap,
                memory,
                disk,
                store.len().await.unwrap(),
                cache.budget().disk_usage_bytes(),
                file_len(&path),
            );
            samples.push((i, heap, memory + disk));
        }
        if i == n {
            break;
        }
        let key = warm + i;
        if cache
            .insert(EntryID::from(key), 0, small_array(key, 16))
            .await
            .is_err()
        {
            insert_errors += 1;
        }
    }
    let (first, last) = (samples[0], samples[samples.len() - 1]);
    let slope = (last.1 as f64 - first.1 as f64) / (last.0 - first.0) as f64;
    let mid = samples[samples.len() / 2];
    let slope_first = (mid.1 as f64 - first.1 as f64) / (mid.0 - first.0) as f64;
    let slope_second = (last.1 as f64 - mid.1 as f64) / (last.0 - mid.0) as f64;
    println!("insert errors (CacheFull)   {insert_errors}");
    println!("heap growth over run        {}", last.1 as i64 - first.1 as i64);
    println!("slope, whole run            {slope:.1} bytes/key");
    println!("slope, first half           {slope_first:.1} bytes/key");
    println!("slope, second half          {slope_second:.1} bytes/key");
    println!(
        "projected @ 10M keys        {:.2} GiB",
        slope * 1e7 / (1u64 << 30) as f64
    );
    println!(
        "projected @ 100M keys       {:.2} GiB",
        slope * 1e8 / (1u64 << 30) as f64
    );
    println!("elapsed                     {:?}", started.elapsed());

    // The cache's own population is steady, so the heap should be too.
    assert!(
        last.2 <= first.2 + 1000,
        "the index itself grew: {} -> {}",
        first.2,
        last.2
    );
    assert!(
        slope < 8.0,
        "heap grows {slope:.1} bytes per key ever spilled with a steady population of {}",
        last.2
    );
}

/// Control for claim 2 with no cache: the store alone, 1,000 live keys, each
/// new put paired with a remove of the oldest key.
#[tokio::test]
async fn claim2_control_store_alone() {
    let n = env_usize("AUDIT_N2", 200_000);
    let step = n / 10;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.t4");
    let store = mount(&path).await;
    let value = vec![7u8; 528];
    let key = |i: usize| i.to_le_bytes().to_vec();
    let window = 1000;
    for i in 0..window {
        store.put(key(i), value.clone()).await.unwrap();
    }
    println!("--- CLAIM 2 control: t4 store alone, 1,000 live keys ---");
    println!("removed | live heap | store.len | file len");
    let mut samples = Vec::new();
    for i in 0..=n {
        if i % step == 0 {
            let heap = live();
            println!(
                "{:>7} | {:>9} | {:>9} | {:>8}",
                i,
                heap,
                store.len().await.unwrap(),
                file_len(&path)
            );
            samples.push((i, heap));
        }
        if i == n {
            break;
        }
        store.put(key(window + i), value.clone()).await.unwrap();
        assert!(store.remove(&key(i)).await.unwrap());
    }
    let (first, last) = (samples[0], samples[samples.len() - 1]);
    let slope = (last.1 as f64 - first.1 as f64) / (last.0 - first.0) as f64;
    println!("heap growth over run        {}", last.1 as i64 - first.1 as i64);
    println!("slope                       {slope:.1} bytes/removed key");
    assert!(
        slope < 8.0,
        "the store alone grows {slope:.1} heap bytes per removed key at a steady 1,000 live keys"
    );
}

#[tokio::test]
async fn claim3a_file_length_vs_disk_budget() {
    let n = env_usize("AUDIT_N3A", 10_000);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.t4");
    let store = mount(&path).await;
    let cache = cache_over(&store, 1 << 30, 1 << 40).await;
    let empty_len = file_len(&path);
    for i in 0..n {
        cache
            .insert(EntryID::from(i), 0, small_array(i, 1))
            .await
            .unwrap();
    }
    cache.flush_all_to_disk().await.unwrap();
    store.sync().await.unwrap();

    let (memory, disk) = census(&cache);
    let charged = cache.budget().disk_usage_bytes();
    let len = file_len(&path);
    println!("--- CLAIM 3a: file length vs disk budget ---");
    println!("entries on disk             {disk} (memory {memory})");
    println!("store.len()                 {}", store.len().await.unwrap());
    println!("charged per entry           {}", charged / disk.max(1));
    println!("budget.disk_usage_bytes     {charged}");
    println!("file length (empty store)   {empty_len}");
    println!("file length                 {len}");
    println!("file bytes per entry        {:.1}", len as f64 / disk as f64);
    println!("file length / disk_usage    {:.2}", len as f64 / charged as f64);

    assert_eq!(disk, n);
    assert!(
        len as usize <= charged * 2,
        "the store file is {len} bytes while the disk budget is charged {charged}"
    );
}

#[tokio::test]
async fn claim3b_file_length_under_hydrate_spill_churn() {
    let cycles = env_usize("AUDIT_N3B", 5_000);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.t4");
    let store = mount(&path).await;
    let cache = cache_over(&store, 1 << 20, 1 << 20).await;
    let entry = EntryID::from(7);
    let array = small_array(7, 16);
    cache.insert(entry, 0, array.clone()).await.unwrap();
    cache.flush_all_to_disk().await.unwrap();
    let charged = cache.budget().disk_usage_bytes();
    let first_len = file_len(&path);

    println!("--- CLAIM 3b: hydrate/spill churn of one entry ---");
    println!("cycle | file len | disk_usage | store.len | hydrated this cycle");
    println!(
        "{:>5} | {:>8} | {:>10} | {:>9} | -",
        0,
        first_len,
        charged,
        store.len().await.unwrap()
    );
    let mut hydrations = 0;
    let mut usages = vec![charged];
    let mut last_len = first_len;
    for cycle in 1..=cycles {
        let read = cache.get(&entry, 0).await.expect("entry is cached");
        assert_eq!(read.as_ref(), array.as_ref());
        let hydrated = census(&cache) == (1, 0);
        hydrations += usize::from(hydrated);
        cache.flush_all_to_disk().await.unwrap();
        assert_eq!(census(&cache), (0, 1), "flush left the entry in memory");
        if cycle % (cycles / 5).max(1) == 0 {
            store.sync().await.unwrap();
            last_len = file_len(&path);
            let usage = cache.budget().disk_usage_bytes();
            usages.push(usage);
            println!(
                "{:>5} | {:>8} | {:>10} | {:>9} | {}",
                cycle,
                last_len,
                usage,
                store.len().await.unwrap(),
                hydrated
            );
        }
    }
    println!("cycles that hydrated        {hydrations} / {cycles}");
    println!("file growth                 {}", last_len - first_len);
    println!(
        "file growth per cycle       {:.1}",
        (last_len - first_len) as f64 / cycles as f64
    );

    assert!(
        usages.iter().all(|u| *u == charged),
        "disk usage moved: {usages:?}"
    );
    assert!(
        last_len <= first_len + 64 * 4096,
        "one {charged}-byte entry: file grew {first_len} -> {last_len} over {cycles} cycles"
    );
}

#[tokio::test]
async fn claim4_restart_orphans() {
    let n = env_usize("AUDIT_N4", 1_000);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.t4");
    let (charged_before, len_before, objects_before);
    {
        let store = mount(&path).await;
        let cache = cache_over(&store, 1 << 30, 1 << 30).await;
        for i in 0..n {
            cache
                .insert(EntryID::from(i), 0, small_array(i, 16))
                .await
                .unwrap();
        }
        cache.flush_all_to_disk().await.unwrap();
        store.sync().await.unwrap();
        charged_before = cache.budget().disk_usage_bytes();
        objects_before = store.len().await.unwrap();
        len_before = file_len(&path);
        drop(cache);
        drop(store);
    }

    let store = mount(&path).await;
    let cache = cache_over(&store, 1 << 30, 1 << 30).await;
    let objects_after = store.len().await.unwrap();
    let (memory, disk) = census(&cache);
    let hits = {
        let mut hits = 0;
        for i in 0..n {
            hits += usize::from(cache.get(&EntryID::from(i), 0).await.is_some());
        }
        hits
    };
    println!("--- CLAIM 4: restart ---");
    println!("                         | before drop | after re-mount");
    println!(
        "store.len()              | {:>11} | {:>11}",
        objects_before, objects_after
    );
    println!(
        "file length              | {:>11} | {:>11}",
        len_before,
        file_len(&path)
    );
    println!(
        "budget.disk_usage_bytes  | {:>11} | {:>11}",
        charged_before,
        cache.budget().disk_usage_bytes()
    );
    println!("index entries (new cache)  memory {memory}, disk {disk}");
    println!("gets served by new cache   {hits} / {n}");

    // A new cache either adopts what the store holds or the store is empty.
    assert!(
        objects_after == 0 || cache.budget().disk_usage_bytes() >= charged_before,
        "the store holds {objects_after} objects ({} file bytes) that the new cache charges {} for",
        file_len(&path),
        cache.budget().disk_usage_bytes()
    );
}
