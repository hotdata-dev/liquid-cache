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
//! entries keyed from it are still resident, and each entry records the
//! identity of the file it came from, so the new owner's reads miss rather
//! than returning the previous owner's rows.
//!
//! Its writes are not refused, though. A key held by another identity belongs
//! to a file that has already let its id go, so nothing can read that entry
//! any more and the new owner takes the key over
//! (`liquid_cache::cache::ArtIndex::insert`). Refusing instead would leave the
//! key occupied by data nobody can use, and on a cache below its budget
//! nothing evicts it — the new owner would never cache that key again.
//!
//! The alternative, releasing ids from inside index removal, would take a
//! process-wide lock underneath a crossbeam-epoch pin and deadlock against
//! `reset`. Keeping id lifetime and entry lifetime separate is what avoids
//! that, and the identity check is what makes the overlap safe.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
// Through `crate::sync`, not `std::sync`: under the shuttle test feature this
// resolves to shuttle's primitives, which is what lets the model checker
// explore interleavings across `acquire` and `release`. A `std::sync::Mutex`
// is opaque to it, so the pool would be excluded from the very job that is
// meant to cover it.
use crate::sync::{Arc, Mutex, Weak};

/// A leased file id. The id returns to its pool when this is dropped.
///
/// Two numbers, because they answer different questions and only one of them
/// can be recycled:
///
/// * `id` goes into the cache key, whose file field is 16 bits wide. It has to
///   be recycled or a long-lived process runs out.
/// * `identity` names *which file* an entry came from, and is never reused.
///   It cannot be the recycled id: a file that inherits id 0 from a file that
///   has finished would otherwise be indistinguishable from it, and would read
///   the entries it left behind — the exact aliasing the identity exists to
///   catch.
#[derive(Debug)]
pub(crate) struct FileId {
    id: u64,
    identity: u64,
    path: String,
    pool: Arc<FileIdPool>,
}

impl FileId {
    /// The narrow, recycled id the cache key is built from.
    pub(crate) fn get(&self) -> u64 {
        self.id
    }

    /// The wide, never-reused name for this file, recorded alongside every
    /// entry so a recycled key cannot serve one file another's data.
    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }
}

impl Drop for FileId {
    fn drop(&mut self) {
        self.pool.release(&self.path, self.id, self.identity);
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
    ///
    /// Each carries the path that released it and the identity it had. If the
    /// same path comes back it keeps that identity, so its cached entries are
    /// still its own and still readable — a file read twice is a cache hit,
    /// not a collision. Any other path gets a fresh identity, so it cannot
    /// read what the previous holder left behind.
    free: VecDeque<Released>,
    next: u64,
    /// Only ever climbs. A `u64` of these is not a resource worth reclaiming:
    /// at one a microsecond it outlasts the hardware.
    next_identity: u64,
}

#[derive(Debug)]
struct Released {
    id: u64,
    path: String,
    identity: u64,
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
        // Prefer this path's own released record, wherever it sits in the
        // queue. Matching only the front would restore an identity just when
        // release order happens to match acquire order — release order is
        // stream completion order and acquire order is partition open order,
        // so for any scan over more than one file they diverge and every
        // re-read would orphan the entries it cached last time.
        let mine = inner.free.iter().position(|r| r.path == path);
        let (id, reusable_identity) = match mine {
            Some(at) => {
                let released = inner.free.remove(at).expect("index came from the queue");
                (released.id, Some(released.identity))
            }
            None => match inner.free.pop_front() {
                Some(released) => (released.id, None),
                None => {
                    let id = inner.next;
                    inner.next += 1;
                    (id, None)
                }
            },
        };
        if id > u16::MAX as u64 {
            self.over_key_width.fetch_add(1, Ordering::Relaxed);
        }
        let identity = match reusable_identity {
            Some(identity) => identity,
            None => {
                let identity = inner.next_identity;
                inner.next_identity += 1;
                identity
            }
        };
        let lease = Arc::new(FileId {
            id,
            identity,
            path: path.to_string(),
            pool: Arc::clone(self),
        });
        inner
            .leases
            .insert(path.to_string(), Arc::downgrade(&lease));
        lease
    }

    fn release(&self, path: &str, id: u64, self_identity: u64) {
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
        inner.free.push_back(Released {
            id,
            path: path.to_string(),
            identity: self_identity,
        });
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

    /// A file re-opened after other files have come and gone must still find
    /// its own record. Release order is stream completion order and acquire
    /// order is partition open order, so the two rarely line up; matching only
    /// the front of the queue would hand a re-read a fresh identity and orphan
    /// everything it cached before.
    #[test]
    fn a_reopened_path_finds_its_record_anywhere_in_the_queue() {
        let pool = FileIdPool::new();

        let a = pool.acquire("a.parquet");
        let b = pool.acquire("b.parquet");
        let (a_id, a_identity) = (a.get(), a.identity());
        let (b_id, b_identity) = (b.get(), b.identity());

        // Released a-then-b, so b's record sits behind a's.
        drop(a);
        drop(b);

        // Re-open b first: the queue front is a's record, not b's.
        let b_again = pool.acquire("b.parquet");
        assert_eq!(b_again.get(), b_id, "b should get its own id back");
        assert_eq!(
            b_again.identity(),
            b_identity,
            "b should keep its identity, or its cached entries are orphaned"
        );

        let a_again = pool.acquire("a.parquet");
        assert_eq!(a_again.get(), a_id);
        assert_eq!(a_again.identity(), a_identity);
    }

    /// Two leases alive at the same time must never share an id, and never
    /// share an identity. That is the property the whole scheme rests on:
    /// a shared id means two files computing one key, and a shared identity
    /// means the check that catches it cannot tell them apart.
    ///
    /// Run under the model checker because `acquire` and `release` race by
    /// construction — a lease is released from `Drop`, on whatever thread
    /// happened to hold it last.
    fn concurrent_leases_stay_distinct() {
        let pool = FileIdPool::new();
        let mut threads = Vec::new();

        for t in 0..3 {
            let pool = Arc::clone(&pool);
            threads.push(crate::sync::thread::spawn(move || {
                for i in 0..3 {
                    let mine = pool.acquire(&format!("f{t}-{i}.parquet"));

                    // Held at the same time, so they cannot be the same file.
                    let probe = pool.acquire("probe.parquet");
                    assert_ne!(mine.get(), probe.get(), "two live leases shared an id");
                    assert_ne!(
                        mine.identity(),
                        probe.identity(),
                        "two live leases shared an identity"
                    );
                    drop(probe);

                    // The same path always resolves to the same lease.
                    let again = pool.acquire(&format!("f{t}-{i}.parquet"));
                    assert_eq!(mine.get(), again.get());
                    assert_eq!(mine.identity(), again.identity());
                }
            }));
        }

        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn concurrent_leases_stay_distinct_single_threaded() {
        concurrent_leases_stay_distinct();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_concurrent_leases_stay_distinct() {
        let mut runner = shuttle::PortfolioRunner::new(true, Default::default());
        let cores = std::thread::available_parallelism().unwrap().get().min(4);
        for _ in 0..cores {
            runner.add(shuttle::scheduler::PctScheduler::new(10, 1_000));
        }
        runner.run(concurrent_leases_stay_distinct);
    }

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

    /// The two numbers have to move independently. Reusing an id is how the
    /// key space stays bounded; reusing an *identity* for a different file is
    /// how one file reads another's entries. Re-opening the same path must
    /// keep its identity, or every lease boundary silently empties the cache.
    #[test]
    fn identity_follows_the_path_while_the_id_is_recycled() {
        let pool = FileIdPool::new();

        let first = pool.acquire("a.parquet");
        let (a_id, a_identity) = (first.get(), first.identity());
        drop(first);

        // Same file again: same id and the same name, so its cached entries
        // are still its own.
        let reopened = pool.acquire("a.parquet");
        assert_eq!(reopened.get(), a_id);
        assert_eq!(
            reopened.identity(),
            a_identity,
            "re-opening a file must keep its identity, or its cache is dead"
        );
        drop(reopened);

        // A different file inherits the id but must not inherit the name.
        let other = pool.acquire("b.parquet");
        assert_eq!(other.get(), a_id, "the id is recycled");
        assert_ne!(
            other.identity(),
            a_identity,
            "a different file must not be able to read what the last one left"
        );
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
