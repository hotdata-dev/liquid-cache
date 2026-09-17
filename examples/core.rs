use std::sync::Arc;

use arrow::array::UInt64Array;
use liquid_cache::cache::{
    AlwaysHydrate, EntryID, LiquidCacheBuilder, LiquidPolicy, TranscodeSqueezeEvict,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = LiquidCacheBuilder::new()
        .with_max_memory_bytes(1024 * 1024 * 1024) // 1GB
        .with_max_disk_bytes(1024 * 1024 * 1024 * 10) // 10GB
        .with_batch_size(8192)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .with_hydration_policy(Box::new(AlwaysHydrate::new()))
        .build()
        .await;

    let entry_id = EntryID::from(7);
    // Names whose data this is. Entry ids are packed and can alias between
    // sources; the cache compares this and treats a mismatch as a miss, so a
    // caller only ever reads back what it put in.
    let identity = 1;
    let arrow_array = Arc::new(UInt64Array::from_iter_values(0..16));
    storage
        .insert(entry_id, identity, arrow_array.clone())
        .await
        .unwrap();

    // Move data to disk so the read demonstrates async I/O
    storage.flush_all_to_disk().await.unwrap();

    let retrieved = storage.get(&entry_id, identity).await.unwrap();
    assert_eq!(retrieved.as_ref(), arrow_array.as_ref());

    Ok(())
}
