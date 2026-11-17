//! Audit tests for the admin handlers. Each asserts the correct behaviour, so
//! a failure here is a finding, not a broken test.

use std::sync::{Arc, atomic::AtomicU32};

use arrow::array::{ArrayRef, Int32Array};
use axum::extract::State;
use liquid_cache::{
    cache::{AlwaysHydrate, EntryID, TranscodeEvict},
    cache_policies::LiquidPolicy,
};

use super::*;
use crate::{
    LiquidCacheService,
    admin_server::{disk_monitor::DiskMonitor, flamegraph::FlameGraph},
};

async fn app_state(max_memory_bytes: Option<usize>, dir: &std::path::Path) -> Arc<AppState> {
    let service = LiquidCacheService::new(
        LiquidCacheService::context().unwrap(),
        max_memory_bytes,
        Some(dir.to_path_buf()),
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await
    .unwrap();
    Arc::new(AppState {
        liquid_cache: Arc::new(service),
        trace_id: AtomicU32::new(0),
        stats_id: AtomicU32::new(0),
        flamegraph: Arc::new(FlameGraph::new()),
        disk_monitor: Arc::new(DiskMonitor::new()),
    })
}

/// How many objects a fresh mount of the service's cache file finds.
async fn objects_in_cache_file(dir: &std::path::Path) -> usize {
    let path = dir.join("liquid").join("liquid_cache.t4");
    t4::mount(&path).await.unwrap().len().await.unwrap()
}

/// Finding 10: after `/reset_cache` the cache reports no disk usage, so the
/// disk has to actually be empty.
#[tokio::test]
async fn audit_reset_cache_handler_empties_the_disk_it_reports_as_empty() {
    const ENTRIES: usize = 16;
    let dir = tempfile::tempdir().unwrap();
    let state = app_state(Some(1 << 30), dir.path()).await;
    let cache = state.liquid_cache.cache().clone();
    let storage = cache.storage();
    for i in 0..ENTRIES {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..8192));
        storage.insert(EntryID::from(i), 0, array).await.unwrap();
    }
    storage.flush_all_to_disk().await.unwrap();
    let usage_before = cache.disk_usage_bytes();
    let objects_before = objects_in_cache_file(dir.path()).await;
    println!(
        "audit: admin: before reset: disk_usage_bytes = {usage_before}, objects found by a fresh mount = {objects_before}"
    );
    assert!(usage_before > 0, "the flush put the entries on disk");
    assert_eq!(
        objects_before, ENTRIES,
        "a fresh mount of the cache file sees the spilled objects (the probe works)"
    );

    let response = reset_cache_handler(State(state.clone())).await;
    let usage_after = cache.disk_usage_bytes();
    let objects_after = objects_in_cache_file(dir.path()).await;
    let file_bytes = std::fs::metadata(dir.path().join("liquid").join("liquid_cache.t4"))
        .unwrap()
        .len();
    println!(
        "audit: admin: after reset ({:?}): disk_usage_bytes = {usage_after}, objects found by a fresh mount = {objects_after}, cache file = {file_bytes} bytes",
        response.0.message
    );
    assert!(
        usage_after > 0 || objects_after == 0,
        "/reset_cache reports disk_usage_bytes = {usage_after} while the store still holds {objects_after} objects ({file_bytes} byte file)"
    );
}

/// Finding 11: what `/cache_info` tells an operator about an unlimited
/// service. The destructuring is exhaustive, so it also records that the
/// response carries no `max_disk_bytes`.
#[tokio::test]
async fn audit_cache_info_reports_a_real_memory_limit() {
    let dir = tempfile::tempdir().unwrap();
    let state = app_state(None, dir.path()).await;
    let CacheInfo {
        batch_size,
        max_memory_bytes,
        memory_usage_bytes,
        disk_usage_bytes,
    } = get_cache_info_handler(State(state.clone())).await.0;
    println!(
        "audit: admin: /cache_info = {{ batch_size: {batch_size}, max_memory_bytes: {max_memory_bytes}, memory_usage_bytes: {memory_usage_bytes}, disk_usage_bytes: {disk_usage_bytes} }}; no max_disk_bytes field; cache max_disk_bytes = {}",
        state.liquid_cache.cache().max_disk_bytes()
    );
    assert!(
        max_memory_bytes < usize::MAX as u64,
        "/cache_info reports max_memory_bytes = {max_memory_bytes} (usize::MAX) for a service with no limit"
    );
}

/// Finding 11: `/system_info` reports sysinfo's totals. Prints them
/// everywhere; under a memory-limited cgroup on Linux it also asserts the
/// total is the container's, not the host's.
#[tokio::test]
async fn audit_system_info_reports_the_memory_the_server_may_use() {
    let dir = tempfile::tempdir().unwrap();
    let state = app_state(Some(1 << 20), dir.path()).await;
    let info = get_system_info_handler(State(state)).await.0;
    println!(
        "audit: admin: /system_info total_memory_bytes = {}, used_memory_bytes = {}, available_memory_bytes = {}, server_resident_memory_bytes = {}",
        info.total_memory_bytes,
        info.used_memory_bytes,
        info.available_memory_bytes,
        info.server_resident_memory_bytes
    );
    #[cfg(target_os = "linux")]
    {
        let limit = [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]
        .iter()
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .filter_map(|text| text.trim().parse::<u64>().ok())
        .find(|limit| *limit < (1 << 60));
        let Some(limit) = limit else {
            println!("audit: admin: no finite cgroup memory limit; assertion skipped");
            return;
        };
        println!("audit: admin: cgroup memory limit = {limit}");
        assert!(
            info.total_memory_bytes <= limit,
            "/system_info reports total_memory_bytes = {} under a cgroup limit of {limit}",
            info.total_memory_bytes
        );
    }
}
