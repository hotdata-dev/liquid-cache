//! Size audit of a concurrent spill: what eight victims hold while their
//! writes are in flight, against what the memory budget says.
//!
//! ```text
//! cargo +1.96.0 test -p liquid-cache --lib audit_ -- --nocapture --test-threads=1
//! ```

use std::future::{Future, IntoFuture};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array};

use super::*;
use crate::audit_alloc::thread_live as live;
use crate::cache::{AlwaysHydrate, Evict, LiquidCacheBuilder, LiquidPolicy};

const ROWS: usize = 8192;
const VICTIMS: usize = 8;

fn array(seed: usize) -> ArrayRef {
    let mut state = (seed as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Arc::new(Int64Array::from_iter_values((0..ROWS).map(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as i64
    })))
}

fn poll_once<F: Future + ?Sized>(future: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
    future.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
}

/// Item 8. An insert that does not fit evicts eight victims at once
/// (`with_evict_victims_concurrently(true)`, the `Evict` policy). Every
/// victim's write is parked on its key's lock, which is where a slow store
/// would hold it: after the entry is serialized and copied into the object to
/// put, before the put. The entries are still in memory and still charged.
/// Whatever else the eviction holds there is charged to nothing, and should
/// stay under one and a half victims.
#[tokio::test]
async fn audit_concurrent_spill_scratch_is_bounded_by_one_victim() {
    let victim = array(0).get_array_memory_size();
    let max_memory = VICTIMS * victim + victim / 2;
    let directory = tempfile::tempdir().unwrap();
    let store = t4::mount(directory.path().join("cache.t4")).await.unwrap();
    let cache = LiquidCacheBuilder::new()
        .with_max_memory_bytes(max_memory)
        .with_max_disk_bytes(1 << 30)
        .with_eviction_policy(Box::new(Evict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .with_evict_victims_concurrently(true)
        .with_store(store)
        .build()
        .await;
    for i in 0..VICTIMS {
        cache.insert(EntryID::from(i), 0, array(i)).await.unwrap();
    }
    assert_eq!(cache.stats().memory_arrow_entries, VICTIMS);

    // Park every put: a write takes its key's lock just before putting.
    let mut parked = Vec::new();
    for lock in cache.disk_key_locks.locks.iter() {
        parked.push(lock.lock().await);
    }

    let incoming = array(VICTIMS);
    let mut insert = cache
        .insert(EntryID::from(VICTIMS), 0, incoming)
        .into_future();
    drop(cache.consume_event_trace());
    let budget_before = cache.budget.memory_usage_bytes() as isize;
    let live_before = live();
    assert!(
        poll_once(insert.as_mut()).is_pending(),
        "the insert waits for its victims' puts"
    );
    // Debug builds keep every event in an unbounded trace.
    drop(cache.consume_event_trace());
    let budget_parked = cache.budget.memory_usage_bytes() as isize;
    let live_parked = live();
    let disk_reserved = cache.budget.disk_usage_bytes();
    let still_in_memory = cache.stats().memory_arrow_entries;

    let heap_growth = live_parked - live_before;
    let budget_growth = budget_parked - budget_before;
    let uncharged = heap_growth - budget_growth;
    println!(
        "\none victim = {victim} bytes; max_memory = {max_memory}\n\
         budget.memory_usage: {budget_before} -> {budget_parked} ({budget_growth:+})\n\
         live heap (this thread) during the parked eviction: {heap_growth:+}\n\
         uncharged = {uncharged} bytes = {:.2} victims; disk reserved = {disk_reserved} \
         ({:.2} victims); entries still in memory = {still_in_memory}",
        uncharged as f64 / victim as f64,
        disk_reserved as f64 / victim as f64,
    );

    drop(parked);
    insert.await.unwrap();

    // What one victim's write holds, measured on its own.
    let one = array(0);
    let before = live();
    let serialized = crate::cache::utils::arrow_to_bytes(&one).unwrap();
    let serialized_heap = live() - before;
    println!(
        "one victim's write: serialized Bytes len {} holding {serialized_heap} heap bytes, \
         plus an object Vec of {} (len + trailer)",
        serialized.len(),
        serialized.len() + IDENTITY_TRAILER_LEN,
    );
    drop(serialized);

    assert!(
        uncharged * 2 <= victim as isize * 3,
        "live heap <= budget.memory_usage + 1.5 * one victim failed: with {VICTIMS} victims \
         mid-put the eviction holds {uncharged} uncharged bytes, {:.2}x one victim ({victim})",
        uncharged as f64 / victim as f64
    );
}
