//! Mounting the on-disk store that backs the cache's disk tier.
//!
//! LiquidCache wants DIRECT I/O. Bypassing the OS page cache is what makes the
//! cache's own byte accounting the whole truth: one copy of a cached page
//! exists, and the cache knows about it. The admission gate
//! ([`crate::cache`] budgets, and `liquid-cache-datafusion`'s footprint gate)
//! is built on that premise.
//!
//! [`t4`] only implements DIRECT I/O on Linux — every other target refuses the
//! option outright rather than silently ignoring it. So off Linux we mount
//! buffered and say so. The cache stays correct: it writes, reads and evicts
//! exactly as before. What it loses is the accounting guarantee, because the
//! kernel now keeps a second copy of every page that the cache does not count.
//! That makes non-Linux fine for development and wrong for measurement.

use std::path::Path;

/// Mount the on-disk store for a LiquidCache instance at `path`, which is the
/// full path to the store file.
///
/// Prefer this over calling [`t4::mount`] directly: it is the one place that
/// decides the store's I/O mode, so the choice cannot drift between the cache
/// builders, the server, benches and tests.
///
/// On Linux this is exactly [`t4::mount`] — `direct_io` and `dsync` both come
/// out `true`, matching [`t4::MountOptions::default`]. See the
/// [module docs](self) for what the buffered fallback costs elsewhere.
pub async fn mount(path: impl AsRef<Path>) -> t4::Result<t4::Store> {
    #[cfg(not(target_os = "linux"))]
    warn_buffered_once();

    // `dsync` stays at t4's default: O_DSYNC is honoured off Linux too, so the
    // fallback keeps the write-durability semantics Linux gets.
    t4::mount_with_options(
        path,
        t4::MountOptions {
            direct_io: cfg!(target_os = "linux"),
            ..Default::default()
        },
    )
    .await
}

#[cfg(not(target_os = "linux"))]
fn warn_buffered_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "mounting the liquid cache store with buffered I/O: t4 implements DIRECT I/O on Linux \
             only. The cache is fully functional, but memory accounting now excludes the kernel \
             page-cache copy of every cached page, so reported usage understates real residency \
             and the admission gate's budget is measured against an incomplete figure. Measure \
             performance on Linux."
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The store must mount, round-trip a value and survive a remount on every
    /// platform. On Linux this covers t4's io_uring backend; elsewhere it is the
    /// only coverage the generic thread-pool backend gets.
    #[tokio::test]
    async fn mount_round_trips_on_this_platform() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liquid_cache.t4");

        let store = mount(&path).await.expect("mount must succeed");
        store.put(b"key".to_vec(), b"hello".to_vec()).await.unwrap();
        assert_eq!(store.get(b"key").await.unwrap(), b"hello");
        store.sync().await.unwrap();
        drop(store);

        let store = mount(&path).await.expect("remount must succeed");
        assert_eq!(
            store.get(b"key").await.unwrap(),
            b"hello",
            "a remounted store must replay what was written"
        );
    }
}
