//! Audit tests for the server's configuration. Each asserts the correct
//! behaviour, so a failure here is a finding, not a broken test.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlanProperties;
use futures::StreamExt;
use liquid_cache::{
    cache::{AlwaysHydrate, TranscodeEvict},
    cache_policies::LiquidPolicy,
};
use uuid::Uuid;

use super::{TEST_FILE, get_physical_plan};
use crate::{ExecutionStats, LiquidCacheService, LiquidCacheServiceInner};

/// Finding 8: a service built without a memory limit must still have finite
/// budgets.
#[tokio::test]
async fn audit_service_default_budgets_are_finite() {
    let dir = tempfile::tempdir().unwrap();
    let service = LiquidCacheService::new(
        LiquidCacheService::context().unwrap(),
        None,
        Some(dir.path().to_path_buf()),
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await
    .unwrap();
    let memory = service.cache().max_memory_bytes();
    let disk = service.cache().max_disk_bytes();
    println!(
        "audit: server: max_memory_bytes = {memory} (usize::MAX = {}), max_disk_bytes = {disk}",
        usize::MAX
    );
    let mut unbounded = Vec::new();
    if memory == usize::MAX {
        unbounded.push("max_memory_bytes");
    }
    if disk == usize::MAX {
        unbounded.push("max_disk_bytes");
    }
    assert!(
        unbounded.is_empty(),
        "LiquidCacheService::new(ctx, None, ..) leaves {unbounded:?} at usize::MAX (memory = {memory}, disk = {disk})"
    );
}

/// Finding 8, second half: a memory limit the caller did give does not bound
/// the disk either.
#[tokio::test]
async fn audit_service_disk_budget_is_finite_when_memory_is_configured() {
    let dir = tempfile::tempdir().unwrap();
    let service = LiquidCacheService::new(
        LiquidCacheService::context().unwrap(),
        Some(1 << 30),
        Some(dir.path().to_path_buf()),
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await
    .unwrap();
    let memory = service.cache().max_memory_bytes();
    let disk = service.cache().max_disk_bytes();
    println!("audit: server: configured max_memory_bytes = {memory}, max_disk_bytes = {disk}");
    assert!(
        disk < usize::MAX,
        "with max_memory_bytes = {memory} the disk budget is still usize::MAX ({disk})"
    );
}

/// Finding 3, server side: a service built without a cache directory makes a
/// temporary one and must not leave it behind.
#[tokio::test]
async fn audit_service_temp_dir_is_removed_on_drop() {
    let service = LiquidCacheService::new(
        LiquidCacheService::context().unwrap(),
        Some(1 << 20),
        None,
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await
    .unwrap();
    let dir = service
        .get_parquet_cache_dir()
        .parent()
        .unwrap()
        .to_path_buf();
    assert!(dir.join("liquid").join("liquid_cache.t4").exists());
    drop(service);
    for _ in 0..50 {
        if !dir.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let leaked = dir.exists();
    println!(
        "audit: server: temp dir {} exists after the service was dropped = {leaked}",
        dir.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !leaked,
        "temporary cache directory {} outlives the service",
        dir.display()
    );
}

async fn service_over_hits(dir: &std::path::Path) -> LiquidCacheServiceInner {
    let ctx = Arc::new(LiquidCacheService::context().unwrap());
    ctx.register_parquet("hits", TEST_FILE, Default::default())
        .await
        .unwrap();
    LiquidCacheServiceInner::new(
        ctx,
        Some(1 << 30),
        dir.to_path_buf(),
        Box::new(LiquidPolicy::new()),
        Box::new(TranscodeEvict),
        Box::new(AlwaysHydrate::new()),
    )
    .await
}

/// Finding 9: a plan whose every partition has been read to the end is
/// finished, and the service must not hold it for ever.
#[tokio::test]
async fn audit_drained_plans_are_pruned() {
    const PLANS: usize = 25;
    let dir = tempfile::tempdir().unwrap();
    let service = service_over_hits(dir.path()).await;
    let sql = r#"select "URL" from hits where "URL" like '%tours%'"#;
    let mut rows = 0;
    let mut streams = 0;
    for _ in 0..PLANS {
        let handle = Uuid::new_v4();
        let plan = get_physical_plan(sql, service.get_ctx()).await;
        service.register_plan(handle, plan, Default::default());
        let partitions = service
            .get_plan(&handle)
            .unwrap()
            .plan
            .output_partitioning()
            .partition_count();
        for partition in 0..partitions {
            let mut stream = service.execute_plan(&handle, partition).await.unwrap();
            while let Some(batch) = stream.next().await {
                rows += batch.unwrap().num_rows();
            }
            drop(stream);
            streams += 1;
        }
    }
    let retained = service.execution_plans_len();
    println!(
        "audit: server: registered {PLANS} plans, drained {streams} streams ({rows} rows), execution_plans still holds {retained}"
    );
    assert_eq!(
        retained, 0,
        "{retained} of {PLANS} fully drained plans are still in execution_plans"
    );
}

/// Finding 9, stats: the stats list has to be bounded somehow.
#[tokio::test]
async fn audit_execution_stats_are_bounded() {
    const PUSHED: usize = 50_000;
    let dir = tempfile::tempdir().unwrap();
    let service = service_over_hits(dir.path()).await;
    let svg = "x".repeat(1024);
    for i in 0..PUSHED {
        service.add_execution_stats(ExecutionStats {
            plan_ids: vec![Uuid::new_v4().to_string()],
            display_name: format!("query {i}"),
            flamegraph_svg: Some(svg.clone()),
            network_traffic_bytes: 0,
            execution_time_ms: 0,
            user_sql: vec!["select 1".to_string()],
        });
    }
    let retained = service.execution_stats_len();
    println!(
        "audit: server: pushed {PUSHED} execution stats (~1 KiB flamegraph each), retained {retained} (~{} MiB of flamegraphs)",
        retained * svg.len() >> 20
    );
    assert!(
        retained < PUSHED,
        "all {retained} of {PUSHED} execution stats are retained: nothing bounds the list"
    );
}
