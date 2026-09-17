//! Allocation of the file ids that name cached data.
//!
//! An id is the part of a cache key that says which file an entry came from.
//! It is narrowed into 16 bits by [`crate::cache::ColumnAccessPath`], so the
//! supply of *distinct* keys is finite while the number of files a process
//! opens is not. Handing ids out from a counter that only ever climbs means a
//! long-lived process eventually reuses a key while its previous owner's data
//! is still cached.
//!
//! So an id is a lease rather than a permanent assignment. It is held by
//! everything that can still compute a key from it — the file handle, the row
//! groups and columns derived from it — and returns to the pool when the last
//! of them is dropped. The live id count is then bounded by what is actually
//! being read, not by everything that has ever been read.
//!
//! Cache *entries* deliberately do not hold a lease. An id can be reused while
//! entries keyed from it are still resident, and those entries are simply
//! unreachable: each records the identity of the file it came from, so the new
//! owner's reads miss and its writes are refused (see
//! `liquid_cache::cache::ArtIndex::get_checked`). The cost is cache space held
//! by data nobody will read until it is evicted; the alternative — releasing
//! ids from inside index removal — would take a process-wide lock underneath a
//! crossbeam-epoch pin, and would deadlock against `reset`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
use std::sync::{Arc, Mutex, Weak};

/// A leased file id. The id returns to its pool when this is dropped.
#[derive(Debug)]
pub(crate) struct FileId {
    id: u64,
    path: String,
    pool: Arc<FileIdPool>,
}

impl FileId {
    /// The id itself, as the cache key and the entry identity use it.
    pub(crate) fn get(&self) -> u64 {
        self.id
    }
}

impl Drop for FileId {
    fn drop(&mut self) {
        self.pool.release(&self.path, self.id);
    }
}

/// Hands out file ids and takes them back.
#[derive(Debug, Default)]
pub(crate) struct FileIdPool {
    inner: Mutex<PoolInner>,
    /// Ids ever allocated that did not fit the key's 16-bit file field. A
    /// non-zero count means keys are aliasing and the cache is refusing to
    /// serve entries across the alias, which is correct but costs hit rate.
    over_key_width: AtomicU64,
}

#[derive(Debug, Default)]
struct PoolInner {
    /// Live leases by path, so concurrent readers of one file share an id.
    /// Entries are weak: the map never keeps a file alive on its own, and a
    /// path is removed when its lease is released.
    leases: AHashMap<String, Weak<FileId>>,
    /// Released ids, reused oldest-first. FIFO rather than LIFO on purpose: a
    /// just-released id is the one whose entries are most likely still
    /// resident, and reusing it last gives them the longest window to be
    /// evicted before anything keys over them.
    free: VecDeque<u64>,
    next: u64,
}

impl FileIdPool {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The lease for `path`, shared with any reader already holding one.
    pub(crate) fn acquire(self: &Arc<Self>, path: &str) -> Arc<FileId> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(existing) = inner.leases.get(path).and_then(Weak::upgrade) {
            return existing;
        }
        let id = match inner.free.pop_front() {
            Some(id) => id,
            None => {
                let id = inner.next;
                inner.next += 1;
                id
            }
        };
        if id > u16::MAX as u64 {
            self.over_key_width.fetch_add(1, Ordering::Relaxed);
        }
        let lease = Arc::new(FileId {
            id,
            path: path.to_string(),
            pool: Arc::clone(self),
        });
        inner
            .leases
            .insert(path.to_string(), Arc::downgrade(&lease));
        lease
    }

    fn release(&self, path: &str, id: u64) {
        let Ok(mut inner) = self.inner.lock() else {
            // A poisoned pool means some other thread panicked holding it.
            // Losing one id is better than panicking again inside a drop.
            return;
        };
        // Only drop the path if it still points at the lease being released.
        // A new lease for the same path may already have replaced it, and
        // removing that one would hand the same file two live ids.
        if inner
            .leases
            .get(path)
            .is_some_and(|weak| weak.strong_count() == 0)
        {
            inner.leases.remove(path);
        }
        inner.free.push_back(id);
    }

    /// Ids currently leased. Bounded by what is being read, which is what
    /// keeps the 16-bit key field from running out.
    pub(crate) fn live_count(&self) -> usize {
        self.inner.lock().map(|i| i.leases.len()).unwrap_or(0)
    }

    /// Ids handed out that do not fit the key's file field. Expected to stay
    /// at zero.
    pub(crate) fn over_key_width(&self) -> u64 {
        self.over_key_width.load(Ordering::Relaxed)
    }

    /// Forget every lease and start ids from zero again.
    ///
    /// Only valid when nothing holds a lease; callers that still do would keep
    /// computing keys from ids this pool is free to hand out again.
    pub(crate) fn reset(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.leases.clear();
            inner.free.clear();
            inner.next = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_readers_of_one_path_share_a_lease() {
        let pool = FileIdPool::new();
        let first = pool.acquire("a.parquet");
        let second = pool.acquire("a.parquet");
        assert_eq!(first.get(), second.get());
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn an_id_returns_to_the_pool_when_its_last_holder_drops() {
        let pool = FileIdPool::new();
        let a = pool.acquire("a.parquet");
        let b = pool.acquire("b.parquet");
        assert_eq!((a.get(), b.get()), (0, 1));
        assert_eq!(pool.live_count(), 2);

        drop(a);
        assert_eq!(pool.live_count(), 1, "the released path is forgotten");

        // Reused rather than climbing to 2: the supply tracks what is being
        // read, which is the whole point.
        let c = pool.acquire("c.parquet");
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn a_second_holder_keeps_the_id_alive() {
        let pool = FileIdPool::new();
        let first = pool.acquire("a.parquet");
        let second = pool.acquire("a.parquet");
        drop(first);
        assert_eq!(pool.live_count(), 1);
        // Bound, not a temporary: an unheld lease is released the moment the
        // expression ends, which would put its id back before the next call.
        let other = pool.acquire("b.parquet");
        assert_eq!(other.get(), 1, "id 0 is still leased");
        drop(second);
        let recycled = pool.acquire("c.parquet");
        assert_eq!(recycled.get(), 0);
    }

    #[test]
    fn released_ids_are_reused_oldest_first() {
        let pool = FileIdPool::new();
        let a = pool.acquire("a.parquet");
        let b = pool.acquire("b.parquet");
        drop(a);
        drop(b);
        // 0 was released first, so it is handed out first — giving b's entries
        // the longer eviction window.
        assert_eq!(pool.acquire("x.parquet").get(), 0);
        assert_eq!(pool.acquire("y.parquet").get(), 1);
    }

    #[test]
    fn ids_beyond_the_key_width_are_counted() {
        let pool = FileIdPool::new();
        {
            let mut inner = pool.inner.lock().unwrap();
            inner.next = u16::MAX as u64;
        }
        let _fits = pool.acquire("fits.parquet");
        assert_eq!(pool.over_key_width(), 0);
        let _over = pool.acquire("over.parquet");
        assert_eq!(pool.over_key_width(), 1);
    }

    /// A path re-registered while its old lease is being dropped must not lose
    /// the new lease's entry in the map — that would give one file two live
    /// ids and split its cache.
    #[test]
    fn releasing_a_stale_lease_leaves_a_newer_one_alone() {
        let pool = FileIdPool::new();
        let first = pool.acquire("a.parquet");
        let first_id = first.get();
        drop(first);
        let second = pool.acquire("a.parquet");
        assert_eq!(second.get(), first_id, "the id came back round");
        assert_eq!(pool.live_count(), 1);
        assert_eq!(
            pool.acquire("a.parquet").get(),
            second.get(),
            "the live lease is still the one the map points at"
        );
    }
}
