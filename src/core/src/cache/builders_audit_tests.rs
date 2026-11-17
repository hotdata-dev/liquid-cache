//! Audit tests for the cache's configuration layer. Each asserts the correct
//! behaviour, so a failure here is a finding, not a broken test.

use arrow::array::{ArrayRef, Int32Array};

use super::*;
use crate::cache::io_context::entry_id_to_key;

const GIB: usize = 1 << 30;

/// Finding 1 (runs everywhere): the builder documents the default as "half
/// of available system memory".
#[test]
fn audit_default_memory_budget_is_half_of_available_memory() {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    let available = sys.available_memory();
    let default = default_max_memory_bytes() as u64;
    println!(
        "audit: default_max_memory_bytes = {default}, sysinfo total = {total} (half = {}), sysinfo available = {available} (half = {})",
        total / 2,
        available / 2
    );
    assert!(
        default <= available / 2,
        "default memory budget {default} bytes is not half of AVAILABLE memory ({} bytes); equals half of TOTAL ({} bytes): {}",
        available / 2,
        total / 2,
        default == total / 2
    );
}

/// Finding 2: a caller who configures 1 GiB of memory and nothing else gets a
/// disk budget derived from the host default, not from 1 GiB.
#[tokio::test]
async fn audit_default_disk_budget_follows_the_configured_memory() {
    let cache = LiquidCacheBuilder::new()
        .with_max_memory_bytes(GIB)
        .build()
        .await;
    let memory = cache.config().max_memory_bytes();
    let disk = cache.config().max_disk_bytes();
    println!(
        "audit: configured max_memory_bytes = {memory}, resulting max_disk_bytes = {disk} ({:.1}x memory), host default memory = {}",
        disk as f64 / memory as f64,
        default_max_memory_bytes()
    );
    assert_eq!(memory, GIB);
    assert!(
        disk <= 10 * GIB,
        "max_disk_bytes = {disk} with max_memory_bytes = {memory}: the disk budget is 10x the DEFAULT memory ({}), not 10x the configured one ({})",
        default_max_memory_bytes().saturating_mul(10),
        10 * GIB
    );
}

/// Finding 3: a cache built without a store makes a temporary directory and
/// must not leave it behind once the cache is gone.
#[tokio::test]
async fn audit_default_cache_temp_dir_is_removed_on_drop() {
    AUDIT_TEMP_DIRS.with(|dirs| dirs.borrow_mut().clear());
    let cache = LiquidCacheBuilder::new().build().await;
    let dirs = AUDIT_TEMP_DIRS.with(|dirs| dirs.borrow().clone());
    assert_eq!(dirs.len(), 1, "build() made one temporary directory");
    let dir = dirs[0].clone();
    assert!(dir.join("liquid_cache.t4").exists());
    drop(cache);
    // Give any background IO teardown a moment.
    for _ in 0..50 {
        if !dir.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let leftover: Vec<_> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| (e.file_name(), e.metadata().map(|m| m.len()).unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    let leaked = dir.exists();
    println!(
        "audit: temp dir {} exists after the cache was dropped = {leaked}, contents = {leftover:?}",
        dir.display()
    );
    // Do not let the test itself add to the leak.
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !leaked,
        "temporary cache directory {} outlives the cache (contents: {leftover:?})",
        dir.display()
    );
}

fn int_array(range: std::ops::Range<i32>) -> ArrayRef {
    std::sync::Arc::new(Int32Array::from_iter_values(range))
}

/// Finding 4: two caches over the same cache file in one process. Either the
/// second mount is refused, or the two share state. What must not happen is
/// two independent indexes over one file.
#[tokio::test]
async fn audit_two_caches_over_one_cache_file_are_refused_or_share_state() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("liquid_cache.t4");
    let store_a = t4::mount(&path).await.unwrap();
    let store_b = match t4::mount(&path).await {
        Ok(store) => store,
        Err(error) => {
            println!("audit: second mount of {} refused: {error}", path.display());
            return;
        }
    };
    println!("audit: second mount of the same file succeeded");
    let build = |store: &t4::Store| {
        LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(1 << 20)
            .with_store(store.clone())
            .build()
    };
    let cache_a = build(&store_a).await;
    let cache_b = build(&store_b).await;
    let id = EntryID::from(7usize);

    let array_a = int_array(0..1000);
    cache_a.insert(id, 0, array_a.clone()).await.unwrap();
    cache_a.flush_all_to_disk().await.unwrap();
    let (a_len_1, b_len_1) = (store_a.len().await.unwrap(), store_b.len().await.unwrap());
    println!(
        "audit: after A spills key 7: store_a.len = {a_len_1}, store_b.len = {b_len_1}, A disk usage = {}, B disk usage = {}, file = {} bytes",
        cache_a.budget().disk_usage_bytes(),
        cache_b.budget().disk_usage_bytes(),
        std::fs::metadata(&path).unwrap().len()
    );
    let raw_a_before = store_a.get(&entry_id_to_key(&id)).await.unwrap();
    let b_sees_a = cache_b.get(&id, 0).await.is_some();
    println!("audit: B can read A's key before writing it: {b_sees_a}");

    let array_b = int_array(5_000_000..5_004_000);
    cache_b.insert(id, 0, array_b.clone()).await.unwrap();
    cache_b.flush_all_to_disk().await.unwrap();
    let (a_len_2, b_len_2) = (store_a.len().await.unwrap(), store_b.len().await.unwrap());
    println!(
        "audit: after B spills the same key: store_a.len = {a_len_2}, store_b.len = {b_len_2}, A disk usage = {}, B disk usage = {}, file = {} bytes",
        cache_a.budget().disk_usage_bytes(),
        cache_b.budget().disk_usage_bytes(),
        std::fs::metadata(&path).unwrap().len()
    );

    // Read the raw objects too, so a damaged payload is reported rather than
    // only panicked on in the decoder.
    let key = entry_id_to_key(&id);
    let raw_a_after = store_a.get(&key).await;
    let raw_b = store_b.get(&key).await;
    let differing = raw_a_after.as_ref().ok().map(|after| {
        after
            .iter()
            .zip(raw_a_before.iter())
            .filter(|(x, y)| x != y)
            .count()
    });
    println!(
        "audit: raw object for key 7 through store_a: {} bytes before B's write, {:?} bytes after, unchanged = {}, differing bytes = {differing:?}; through store_b = {:?} bytes; A's bytes after B's write are a prefix of B's object = {}",
        raw_a_before.len(),
        raw_a_after.as_ref().map(|v| v.len()),
        raw_a_after.as_ref().is_ok_and(|after| *after == raw_a_before),
        raw_b.as_ref().map(|v| v.len()),
        matches!((&raw_a_after, &raw_b), (Ok(a), Ok(b)) if b.starts_with(a)),
    );

    let read_a = tokio::spawn({
        let cache_a = cache_a.clone();
        async move { cache_a.get(&id, 0).await }
    })
    .await;
    let outcome = match &read_a {
        Err(panic) => format!("PANICKED: {panic}"),
        Ok(None) => "MISS (None)".to_string(),
        Ok(Some(array)) if array.as_ref() == array_a.as_ref() => "A's own array".to_string(),
        Ok(Some(array)) if array.as_ref() == array_b.as_ref() => "B's array".to_string(),
        Ok(Some(array)) => format!(
            "NEITHER array: {} rows of {:?}",
            array.len(),
            array.data_type()
        ),
    };
    println!("audit: A re-reads its spilled key after B wrote the same key -> {outcome}");

    let shared = b_len_1 == a_len_1 && b_sees_a;
    let a_intact = matches!(&read_a, Ok(Some(array)) if array.as_ref() == array_a.as_ref());
    assert!(
        shared || a_intact,
        "two caches mounted {} with independent indexes: after A's spill store_a.len = {a_len_1} but store_b.len = {b_len_1}; B saw A's entry = {b_sees_a}; after B's spill store_a.len = {a_len_2}, store_b.len = {b_len_2}; A's read after B's write -> {outcome}",
        path.display()
    );
}

/// Finding 4 (restart): a cache rebuilt over an existing cache file must
/// account for, or clear, the objects already in it.
#[tokio::test]
async fn audit_cache_rebuilt_over_an_existing_file_accounts_for_its_objects() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("liquid_cache.t4");
    {
        let store = t4::mount(&path).await.unwrap();
        let cache = LiquidCacheBuilder::new()
            .with_store(store.clone())
            .build()
            .await;
        for i in 0..8usize {
            cache
                .insert(EntryID::from(i), 0, int_array(0..4096))
                .await
                .unwrap();
        }
        cache.flush_all_to_disk().await.unwrap();
        store.sync().await.unwrap();
        println!(
            "audit: first life: store.len = {}, disk usage = {} bytes",
            store.len().await.unwrap(),
            cache.budget().disk_usage_bytes()
        );
    }
    let store = t4::mount(&path).await.unwrap();
    let cache = LiquidCacheBuilder::new()
        .with_store(store.clone())
        .build()
        .await;
    let objects = store.len().await.unwrap();
    let usage = cache.budget().disk_usage_bytes();
    let mut entries = 0;
    cache.for_each_entry(|_, _| entries += 1);
    println!(
        "audit: second life: store.len = {objects}, cache index entries = {entries}, disk usage = {usage} bytes, file = {} bytes",
        std::fs::metadata(&path).unwrap().len()
    );
    assert!(
        objects == 0 || usage > 0,
        "after a restart the store holds {objects} objects the new cache neither indexes ({entries} entries) nor charges ({usage} bytes of disk usage)"
    );
}
