use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use crate::cache::{cached_batch::CacheEntry, utils::EntryID};
use crate::sync::{Arc, RwLock};

/// The value stored in the ART.
///
/// `CongeeArc` frees a removed or replaced value through crossbeam-epoch's
/// deferred destruction: it clones the `Arc` and drops the clone only when a
/// later pin collects that epoch's garbage — up to 64 objects per thread wait
/// in a thread-local bag, and the global queue drains 8 bags per 128 pins.
/// With multi-megabyte arrays as values, that kept every evicted entry alive
/// for an unbounded, budget-invisible stretch (liquid-cache#43: a tier
/// reporting at its limit while the process held several times that).
///
/// So the tree stores a small slot and the payload is taken out of it the
/// moment the index gives the entry up. The deferred drop then reclaims only
/// an empty shell, and the array dies with the last caller-held reference.
///
/// The slot also records the identity of what it holds. `EntryID` is a packed
/// integer whose fields are narrower than the values they encode, so two
/// distinct sources can compute the same key; the key alone therefore cannot
/// answer "is this the entry I asked for?". `identity` is the caller's
/// unnarrowed name for the data, compared on every read through
/// [`ArtIndex::get_checked`], which turns such aliasing into a miss rather
/// than a wrong answer.
struct Slot {
    identity: u64,
    entry: RwLock<Option<Arc<CacheEntry>>>,
}

impl Slot {
    fn new(identity: u64, entry: CacheEntry) -> Arc<Self> {
        Arc::new(Self {
            identity,
            entry: RwLock::new(Some(Arc::new(entry))),
        })
    }

    fn load(&self) -> Option<Arc<CacheEntry>> {
        self.entry.read().unwrap().clone()
    }

    fn take(&self) -> Option<Arc<CacheEntry>> {
        self.entry.write().unwrap().take()
    }
}

/// Whose data a write carries, and on what terms.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WriteIdentity {
    /// A caller storing its own data. Takes the key over if another identity
    /// holds it: that identity belongs to a source that cannot read this key
    /// any more, so leaving its entry there would cost the key to both.
    Owned(u64),
    /// Maintenance rewriting an entry it read earlier — transcode, squeeze,
    /// hydrate, spill. It carries the identity the entry was read under and
    /// lands only if the key still holds it. Adopting whatever is there
    /// instead would relabel one source's data with another's whenever a
    /// takeover lands between the read and the write, and the new owner would
    /// then read those rows as its own.
    Rewrite(u64),
}

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, Slot>,
    entry_count: AtomicUsize,
    identity_mismatches: AtomicU64,
}

impl Debug for ArtIndex {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArtIndex {
    pub(crate) fn new() -> Self {
        Self {
            art: CongeeArc::new(),
            entry_count: AtomicUsize::new(0),
            identity_mismatches: AtomicU64::new(0),
        }
    }

    /// Look up an entry without checking whose it is.
    ///
    /// This is for maintenance that acts on whatever currently occupies a key —
    /// eviction, squeezing, disk supersession, iteration for stats. A read
    /// serving a caller must use [`Self::get_checked`] instead, so that a key
    /// collision cannot return one caller another's data.
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<CacheEntry>> {
        let guard = self.art.pin();
        // An empty slot means the entry was removed or replaced between the
        // tree lookup and the load. A remove reading as a miss is exactly as
        // if it had won the race outright, but after a replace the key is
        // still present with a new slot, so look the key up once more rather
        // than report a cached entry as absent.
        let slot = self.art.get(*entry_id, &guard)?;
        if let Some(entry) = slot.load() {
            return Some(entry);
        }
        self.art.get(*entry_id, &guard)?.load()
    }

    /// Look up an entry, returning it only if it is the one `identity` names.
    ///
    /// A mismatch reads as a miss, so the caller re-reads from its source and
    /// gets correct data. It is also counted: the tally is expected to stay at
    /// zero, and a non-zero value means two sources are computing the same
    /// `EntryID`.
    pub(crate) fn get_checked(&self, entry_id: &EntryID, identity: u64) -> Option<Arc<CacheEntry>> {
        let guard = self.art.pin();
        let slot = self.art.get(*entry_id, &guard)?;
        if slot.identity != identity {
            self.identity_mismatches.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if let Some(entry) = slot.load() {
            return Some(entry);
        }
        // Re-read as in `get`: a replace leaves the key present under a new
        // slot, which carries its own identity and must be checked again.
        let slot = self.art.get(*entry_id, &guard)?;
        if slot.identity != identity {
            self.identity_mismatches.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        slot.load()
    }

    /// Look up an entry together with the identity recorded against it, for
    /// maintenance that must rewrite it under the identity it observed.
    pub(crate) fn get_with_identity(&self, entry_id: &EntryID) -> Option<(u64, Arc<CacheEntry>)> {
        let guard = self.art.pin();
        let slot = self.art.get(*entry_id, &guard)?;
        let identity = slot.identity;
        if let Some(entry) = slot.load() {
            return Some((identity, entry));
        }
        let slot = self.art.get(*entry_id, &guard)?;
        let identity = slot.identity;
        slot.load().map(|entry| (identity, entry))
    }

    pub(crate) fn is_cached(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.get_checked(entry_id, identity).is_some()
    }

    /// Store `batch` under `entry_id`, returning whether it was stored.
    ///
    /// See [`WriteIdentity`] for the two kinds of write and why they differ.
    pub(crate) fn insert(
        &self,
        entry_id: &EntryID,
        identity: WriteIdentity,
        batch: CacheEntry,
    ) -> bool {
        let guard = self.art.pin();
        let existing_identity = self.art.get(*entry_id, &guard).map(|slot| slot.identity);
        let identity = match (identity, existing_identity) {
            (WriteIdentity::Owned(new), Some(old)) => {
                if new != old {
                    self.identity_mismatches.fetch_add(1, Ordering::Relaxed);
                }
                new
            }
            (WriteIdentity::Owned(new), None) => new,
            // The key still holds what this rewrite was built from.
            (WriteIdentity::Rewrite(expected), Some(old)) if expected == old => expected,
            // It does not: the entry was taken over or removed while this
            // rewrite was in flight, so the payload belongs to a source that
            // no longer owns the key. Drop it.
            (WriteIdentity::Rewrite(_), _) => return false,
        };
        let existing = self
            .art
            .insert(*entry_id, Slot::new(identity, batch), &guard)
            .expect("Insertion failed");
        match existing {
            Some(replaced) => drop(replaced.take()),
            None => {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        true
    }

    pub(crate) fn remove(&self, entry_id: &EntryID) -> Option<Arc<CacheEntry>> {
        let guard = self.art.pin();
        let removed = self.art.remove(*entry_id, &guard)?;
        self.entry_count.fetch_sub(1, Ordering::Relaxed);
        removed.take()
    }

    pub(crate) fn reset(&self) {
        for k in self.art.keys() {
            self.remove(&k);
        }
        self.entry_count.store(0, Ordering::Relaxed);
    }

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, u64, &CacheEntry)) {
        for id in self.art.keys() {
            if let Some((identity, entry)) = self.get_with_identity(&id) {
                f(&id, identity, &entry);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn keys(&self) -> Vec<EntryID> {
        self.art.keys()
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }

    /// How many lookups or inserts found a key held by a different identity.
    ///
    /// Expected to stay at zero. A non-zero value means two sources compute the
    /// same `EntryID`, and every one of them was served correctly only because
    /// the check turned it into a miss.
    pub(crate) fn identity_mismatches(&self) -> u64 {
        self.identity_mismatches.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::cached_batch::CacheEntry;
    use crate::cache::utils::create_test_array;

    use super::*;

    #[test]
    fn test_get_and_is_cached() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.is_cached(&entry_id1, 0));
        assert!(!store.is_cached(&entry_id2, 0));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, WriteIdentity::Owned(0), array1.clone());
        }

        assert!(store.is_cached(&entry_id1, 0));
        assert!(!store.is_cached(&entry_id2, 0));

        // Get should return the cached value
        match store.get(&entry_id1) {
            Some(batch) => match batch.as_ref() {
                CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
                _ => panic!("Expected ArrowMemory batch"),
            },
            None => panic!("Expected ArrowMemory batch"),
        }
    }

    #[test]
    fn test_reset() {
        let store = ArtIndex::new();
        let entry_id: EntryID = EntryID::from(1);
        let array = create_test_array(100);

        store.insert(&entry_id, WriteIdentity::Owned(0), array.clone());

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.is_cached(&entry_id, 0));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.is_cached(&entry_id, 0));
    }

    /// The array behind a removed or replaced entry must die with the last
    /// caller-held reference, not wait for epoch garbage collection.
    #[test]
    fn removed_and_replaced_entries_are_released_immediately() {
        let store = ArtIndex::new();
        let id = EntryID::from(1);

        let first = create_test_array(100);
        let CacheEntry::MemoryArrow(first_array) = &first else {
            unreachable!()
        };
        let weak_first = Arc::downgrade(first_array);
        store.insert(&id, WriteIdentity::Owned(0), first);
        store.insert(&id, WriteIdentity::Owned(0), create_test_array(200));
        assert!(
            weak_first.upgrade().is_none(),
            "replaced entry still alive: held by the index's deferred drop"
        );

        let second = store.get(&id).unwrap();
        let removed = store.remove(&id).unwrap();
        let CacheEntry::MemoryArrow(second_array) = removed.as_ref() else {
            unreachable!()
        };
        let weak_second = Arc::downgrade(second_array);
        drop((second, removed));
        assert!(
            weak_second.upgrade().is_none(),
            "removed entry still alive: held by the index's deferred drop"
        );
        assert_eq!(store.entry_count(), 0);
    }

    /// Two files whose ids narrow to the same `EntryID` must not read each
    /// other's data. Before the identity check this returned the incumbent's
    /// array, which is a wrong answer whenever the two happen to share a type.
    #[test]
    fn an_entry_is_never_served_to_a_different_identity() {
        let store = ArtIndex::new();
        let key: EntryID = EntryID::from(7);

        assert!(store.insert(&key, WriteIdentity::Owned(1), create_test_array(100)));

        // The colliding file asks for the same key and is told nothing is there.
        assert!(store.get_checked(&key, 2).is_none());
        assert!(!store.is_cached(&key, 2));
        assert_eq!(store.identity_mismatches(), 2);

        // The owner still reads its own entry.
        assert!(store.get_checked(&key, 1).is_some());

        // The colliding file takes the key over. It has to: it cannot read
        // what is there, so leaving it would cost the key to both of them.
        assert!(store.insert(&key, WriteIdentity::Owned(2), create_test_array(200)));
        assert!(
            store.get_checked(&key, 1).is_none(),
            "the displaced file reads a miss, never the other file's rows"
        );
        match store.get_checked(&key, 2).unwrap().as_ref() {
            CacheEntry::MemoryArrow(array) => assert_eq!(array.len(), 200),
            other => panic!("expected the new owner's array, found {other}"),
        }
    }

    /// A rewrite is built from an entry read earlier, and the key can be taken
    /// over in between — a squeeze reads, awaits a disk write, then stores. If
    /// the rewrite adopted whatever identity held the key by then, it would
    /// relabel the old file's data as the new owner's, and the new owner would
    /// read those rows as a hit. Carrying the identity it read under makes the
    /// stale write drop instead.
    #[test]
    fn a_rewrite_does_not_land_on_a_key_taken_over_since_it_was_read() {
        let store = ArtIndex::new();
        let key: EntryID = EntryID::from(11);

        // File A caches, and something begins rewriting that entry.
        assert!(store.insert(&key, WriteIdentity::Owned(1), create_test_array(100)));
        let (observed, _read) = store.get_with_identity(&key).unwrap();
        assert_eq!(observed, 1);

        // File B takes the key over while that rewrite is in flight.
        assert!(store.insert(&key, WriteIdentity::Owned(2), create_test_array(200)));

        // The rewrite lands too late and must be dropped, not relabelled.
        assert!(!store.insert(
            &key,
            WriteIdentity::Rewrite(observed),
            create_test_array(100)
        ));

        match store.get_checked(&key, 2).unwrap().as_ref() {
            CacheEntry::MemoryArrow(array) => assert_eq!(
                array.len(),
                200,
                "the new owner must still read its own rows, not the rewrite's"
            ),
            other => panic!("expected the new owner's array, found {other}"),
        }
    }

    /// Maintenance rewrites a key in place and must neither change whose the
    /// entry is nor bring back one that has been removed — a stale reader that
    /// misses goes on to insert what it read, and that write must not land
    /// under a key nobody owns any more.
    #[test]
    fn maintenance_preserves_identity_and_cannot_resurrect_a_removed_key() {
        let store = ArtIndex::new();
        let key: EntryID = EntryID::from(9);

        assert!(store.insert(&key, WriteIdentity::Owned(5), create_test_array(10)));
        assert!(store.insert(&key, WriteIdentity::Rewrite(5), create_test_array(20)));
        assert!(
            store.get_checked(&key, 5).is_some(),
            "rewriting in place kept the identity"
        );

        store.remove(&key);
        assert!(!store.insert(&key, WriteIdentity::Rewrite(5), create_test_array(30)));
        assert!(store.get(&key).is_none());
        assert_eq!(store.entry_count(), 0);
    }
}
