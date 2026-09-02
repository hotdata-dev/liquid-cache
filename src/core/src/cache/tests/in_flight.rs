//! What the memory tier holds outside its index, and what bounds it.
//!
//! The budget used to count an entry only once it had landed in the index, so
//! every intermediate the hydrate -> insert -> squeeze cycle created on the way
//! there was invisible to it. A tier could report itself exactly at its limit
//! while the process holding it had several times that resident.

use crate::cache::{
    CacheEntry, EntryID, LiquidCache, LiquidCacheBuilder, LiquidCompressorStates, LiquidPolicy,
    TranscodeSqueezeEvict, transcode_liquid_inner,
    utils::{create_cache_store, create_test_arrow_array},
};
use crate::sync::Arc;

/// A cache whose memory tier holds exactly `entries` arrays of `rows` rows, so
/// that the next insert has to make room for itself.
async fn cache_holding(
    entries: usize,
    rows: usize,
    squeeze_concurrently: bool,
) -> Arc<LiquidCache> {
    let entry_bytes = create_test_arrow_array(rows).get_array_memory_size();
    LiquidCacheBuilder::new()
        .with_max_memory_bytes(entry_bytes * entries)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_squeeze_victims_concurrently(squeeze_concurrently)
        .build()
        .await
}

/// Sum of what the indexed entries actually hold in memory, which is what the
/// tier's tally is supposed to be tracking.
fn indexed_memory_bytes(cache: &LiquidCache) -> usize {
    let mut total = 0;
    cache.for_each_entry(|_, entry| total += entry.memory_usage_bytes());
    total
}

#[tokio::test]
async fn decoding_a_disk_entry_is_counted_while_it_is_in_flight() {
    let cache = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
    let entry_id = EntryID::from(700usize);
    let array = create_test_arrow_array(4096);
    cache.insert(entry_id, array.clone()).await.unwrap();
    cache.flush_all_to_disk().await.unwrap();

    let at_rest = cache.stats();
    assert_eq!(at_rest.in_flight_memory_bytes, 0);
    assert_eq!(
        at_rest.peak_in_flight_memory_bytes, 0,
        "an insert that fits holds nothing outside the index"
    );

    cache.get(&entry_id).read().await.unwrap();

    let stats = cache.stats();
    assert_eq!(
        stats.in_flight_memory_bytes, 0,
        "every reservation is released by the time the read returns"
    );
    assert!(
        stats.peak_in_flight_memory_bytes >= array.get_array_memory_size(),
        "reading the entry back off disk decodes a full copy of it, so the peak \
         should be at least the entry's size, but it was {} against {}",
        stats.peak_in_flight_memory_bytes,
        array.get_array_memory_size()
    );
}

/// Victims already in liquid form are the ones that pile up: squeezing them
/// writes their backing to disk, and that write is an await, so every victim in
/// a concurrently squeezed group is holding its output while the others run.
/// (An arrow victim transcodes without ever awaiting, so its future runs to
/// completion in a single poll and it never overlaps with a sibling.)
#[tokio::test]
async fn squeezing_victims_concurrently_bounds_what_they_hold_at_once() {
    let rows = 4096;
    let array = create_test_arrow_array(rows);
    let compressor = LiquidCompressorStates::new();
    let liquid = transcode_liquid_inner(&array, &compressor).expect("int64 transcodes");
    let entry_bytes = liquid.get_array_memory_size();

    let victims = 8;
    let cache = LiquidCacheBuilder::new()
        .with_max_memory_bytes(entry_bytes * victims)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_squeeze_victims_concurrently(true)
        .build()
        .await;

    for i in 0..victims {
        cache
            .insert_inner(EntryID::from(i), CacheEntry::memory_liquid(liquid.clone()))
            .await
            .unwrap();
    }
    assert_eq!(
        cache.stats().peak_in_flight_memory_bytes,
        0,
        "the tier is full but nothing has had to make room yet"
    );

    // This insert finds no room, so the policy hands back all eight resident
    // entries and each is squeezed to disk to make space.
    cache
        .insert_inner(
            EntryID::from(victims),
            CacheEntry::memory_liquid(liquid.clone()),
        )
        .await
        .unwrap();

    let peak = cache.stats().peak_in_flight_memory_bytes;
    assert!(
        peak <= 3 * entry_bytes,
        "a squeeze holds at most its input's worth of output, so the pending \
         entry plus one victim bounds this; all eight victims at once would peak \
         near {}, and this run peaked at {peak} against an entry size of \
         {entry_bytes}",
        (victims + 1) * entry_bytes
    );
}

/// The path that ships is the concurrent one (`builders.rs` turns it on
/// everywhere except unit tests), so it gets the same invariants the sequential
/// path has always been held to: nothing is lost, nothing is miscounted, and
/// nothing is left reserved.
#[tokio::test]
async fn squeezing_victims_holds_its_invariants_on_both_paths() {
    // A tier this much larger than a single entry lets several victims share a
    // group, so the concurrent path really does run transcodes together instead
    // of degenerating into the sequential one.
    let rows = 64;
    let array = create_test_arrow_array(rows);
    let entries = 300;

    for concurrently in [false, true] {
        let cache = cache_holding(256, rows, concurrently).await;
        for i in 0..entries {
            cache.insert(EntryID::from(i), array.clone()).await.unwrap();
        }

        for i in 0..entries {
            let read = cache
                .get(&EntryID::from(i))
                .read()
                .await
                .expect("every inserted entry is still readable");
            assert_eq!(
                read.as_ref(),
                array.as_ref(),
                "entry {i} came back changed (squeeze_victims_concurrently={concurrently})"
            );
        }

        let stats = cache.stats();
        assert_eq!(stats.total_entries, entries);
        assert_eq!(
            stats.in_flight_memory_bytes, 0,
            "every reservation taken while making room has been released"
        );
        assert_eq!(
            stats.memory_usage_bytes,
            indexed_memory_bytes(&cache),
            "the tier's tally has drifted from what its entries hold"
        );
        assert!(stats.memory_usage_bytes <= stats.max_memory_bytes);
        assert!(
            stats.peak_in_flight_memory_bytes > 0,
            "making room for this many entries has to hold something outside \
             the index at some point"
        );
    }
}
