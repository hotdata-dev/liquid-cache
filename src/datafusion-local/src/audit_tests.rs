//! Audit tests for the local builder. Each asserts the correct behaviour, so
//! a failure here is a finding, not a broken test.

use datafusion::execution::memory_pool::MemoryLimit;
use datafusion::prelude::SessionConfig;
use liquid_cache::cache::default_max_memory_bytes;

use crate::LiquidCacheLocalBuilder;

const GIB: usize = 1 << 30;

/// Finding 2: configuring only the memory budget leaves the disk budget at
/// 10x the host-derived default.
#[tokio::test]
async fn audit_local_default_disk_budget_follows_the_configured_memory() {
    let dir = tempfile::tempdir().unwrap();
    let (_ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(dir.path().to_path_buf())
        .with_max_memory_bytes(GIB)
        .build(SessionConfig::new())
        .await
        .unwrap();
    let memory = cache.max_memory_bytes();
    let disk = cache.max_disk_bytes();
    println!(
        "audit: local: configured max_memory_bytes = {memory}, resulting max_disk_bytes = {disk} ({:.1}x memory), host default memory = {}",
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

/// Finding 5: the context the builder returns must run queries under a
/// bounded DataFusion memory pool. The builder takes a `SessionConfig` only,
/// so a caller has no way to hand it a `RuntimeEnv` carrying one.
#[tokio::test]
async fn audit_local_context_has_a_bounded_memory_pool() {
    let dir = tempfile::tempdir().unwrap();
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(dir.path().to_path_buf())
        .with_max_memory_bytes(GIB)
        .build(SessionConfig::new())
        .await
        .unwrap();
    let pool = ctx.runtime_env().memory_pool.clone();
    let bounded = matches!(pool.memory_limit(), MemoryLimit::Finite(_));
    let limit = match pool.memory_limit() {
        MemoryLimit::Finite(bytes) => format!("Finite({bytes})"),
        MemoryLimit::Infinite => "Infinite".to_string(),
        _ => "Unknown".to_string(),
    };
    println!(
        "audit: local: cache max_memory_bytes = {}, DataFusion memory pool = {pool:?}, memory_limit() = {limit}",
        cache.max_memory_bytes()
    );
    assert!(
        bounded,
        "the SessionContext built by LiquidCacheLocalBuilder runs on an unbounded DataFusion pool: {pool:?}, memory_limit() = {limit}"
    );
}

/// Finding 6: two default builds. Either they share a budget, or what they
/// may use between them fits in the host.
#[tokio::test]
async fn audit_two_default_builds_fit_in_host_memory() {
    let host_total = default_max_memory_bytes() as u128 * 2;
    let (_a_ctx, a) = LiquidCacheLocalBuilder::new()
        .build(SessionConfig::new())
        .await
        .unwrap();
    let (_b_ctx, b) = LiquidCacheLocalBuilder::new()
        .build(SessionConfig::new())
        .await
        .unwrap();
    let shared = std::sync::Arc::ptr_eq(a.storage(), b.storage());
    let sum = a.max_memory_bytes() as u128 + b.max_memory_bytes() as u128;
    let disk_sum = a.max_disk_bytes() as u128 + b.max_disk_bytes() as u128;
    println!(
        "audit: local: two default builds: a.max_memory_bytes = {}, b.max_memory_bytes = {}, sum = {sum}, host total ~= {host_total}, share one cache = {shared}, sum of max_disk_bytes = {disk_sum}",
        a.max_memory_bytes(),
        b.max_memory_bytes()
    );
    assert!(
        shared || sum <= host_total,
        "two default builds hold independent budgets summing to {sum} bytes, more than the host's {host_total}"
    );
}

/// Finding 6, one build further: three default builds may together cache more
/// than the host has.
#[tokio::test]
async fn audit_three_default_builds_fit_in_host_memory() {
    let host_total = default_max_memory_bytes() as u128 * 2;
    let mut caches = Vec::new();
    for _ in 0..3 {
        let (_ctx, cache) = LiquidCacheLocalBuilder::new()
            .build(SessionConfig::new())
            .await
            .unwrap();
        caches.push(cache);
    }
    let shared = caches
        .windows(2)
        .all(|pair| std::sync::Arc::ptr_eq(pair[0].storage(), pair[1].storage()));
    let sum: u128 = caches.iter().map(|c| c.max_memory_bytes() as u128).sum();
    println!(
        "audit: local: three default builds: each max_memory_bytes = {}, sum = {sum}, host total ~= {host_total} ({:.2}x), share one cache = {shared}",
        caches[0].max_memory_bytes(),
        sum as f64 / host_total as f64
    );
    assert!(
        shared || sum <= host_total,
        "three default builds hold independent budgets summing to {sum} bytes, {:.2}x the host's {host_total}",
        sum as f64 / host_total as f64
    );
}

/// Finding 4, local side: two default builders must not mount one cache file
/// with two independent caches.
#[tokio::test]
async fn audit_two_default_builds_do_not_mount_the_same_cache_file() {
    let a = LiquidCacheLocalBuilder::new();
    let b = LiquidCacheLocalBuilder::new();
    let (a_file, b_file) = (
        a.cache_dir.join("liquid_cache.t4"),
        b.cache_dir.join("liquid_cache.t4"),
    );
    let (_a_ctx, a_cache) = a.build(SessionConfig::new()).await.unwrap();
    let b_built = b.build(SessionConfig::new()).await;
    println!(
        "audit: local: default cache file a = {}, b = {}, second build ok = {}",
        a_file.display(),
        b_file.display(),
        b_built.is_ok()
    );
    let Ok((_b_ctx, b_cache)) = b_built else {
        return;
    };
    let shared = std::sync::Arc::ptr_eq(a_cache.storage(), b_cache.storage());
    assert!(
        a_file != b_file || shared,
        "two default builds each mounted {} as an independent cache (shared = {shared})",
        a_file.display()
    );
}

/// Finding 7: records the prefetch default. Passes today; it pins the value
/// the audit reports.
#[test]
fn audit_prefetch_default() {
    let builder = LiquidCacheLocalBuilder::new();
    println!("audit: local: default prefetch = {}", builder.prefetch);
    assert!(builder.prefetch, "prefetch defaults to on");
}
