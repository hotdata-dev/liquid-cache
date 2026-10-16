use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::cache::{cached_batch::CacheEntry, utils::EntryID};
use crate::sync::RwLock;

/// What the index holds for a key: the entry, and the identity of the source
/// it was cached for.
///
/// A key does not name its source uniquely: callers pack fields into fixed
/// widths, so two sources can compute the same key. The identity is what tells
/// them apart, so a read under any other identity must treat the slot as
/// absent.
#[derive(Debug)]
pub(crate) struct Slot {
    pub(crate) identity: u64,
    pub(crate) entry: CacheEntry,
}

impl Slot {
    pub(crate) fn belongs_to(&self, identity: u64) -> bool {
        self.identity == identity
    }
}

/// The value stored in the tree: a [`Slot`] behind a cell the index empties
/// the moment it gives the slot up.
///
/// `CongeeArc` frees a removed or replaced value through crossbeam-epoch's
/// deferred destruction: it clones the `Arc` and drops the clone only when a
/// later pin collects that epoch's garbage — up to 64 objects per thread wait
/// in a thread-local bag, and the global queue drains 8 bags per 128 pins.
/// With multi-megabyte arrays as values, that kept every evicted entry alive
/// for an unbounded, budget-invisible stretch: a tier reporting at its limit
/// while the process held several times that.
///
/// Taking the slot out of the cell when the index gives it up means the
/// deferred drop reclaims only an empty shell, and the array dies with the
/// last caller-held reference.
struct Cell {
    slot: RwLock<Option<Arc<Slot>>>,
}

impl Cell {
    fn new(slot: Slot) -> Self {
        Self {
            slot: RwLock::new(Some(Arc::new(slot))),
        }
    }

    fn load(&self) -> Option<Arc<Slot>> {
        self.slot.read().unwrap().clone()
    }

    fn take(&self) -> Option<Arc<Slot>> {
        self.slot.write().unwrap().take()
    }
}

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, Cell>,
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

    /// The slot under `entry_id`, whichever identity holds it.
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
        let guard = self.art.pin();
        // An empty cell means the slot was removed or replaced between the
        // tree lookup and the load. A remove reading as a miss is as if it had
        // won the race outright, but after a replace the key is still present
        // under a new cell, so look it up once more rather than report a
        // cached entry as absent.
        let cell = self.art.get(*entry_id, &guard)?;
        if let Some(slot) = cell.load() {
            return Some(slot);
        }
        self.art.get(*entry_id, &guard)?.load()
    }

    /// The slot under `entry_id`, only if `identity` holds it. A slot held by
    /// another identity is counted in [`Self::identity_mismatches`].
    pub(crate) fn get_for(&self, entry_id: &EntryID, identity: u64) -> Option<Arc<Slot>> {
        let slot = self.get(entry_id)?;
        if !slot.belongs_to(identity) {
            self.identity_mismatches.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(slot)
    }

    /// How many lookups found their key held by another identity.
    pub(crate) fn identity_mismatches(&self) -> u64 {
        self.identity_mismatches.load(Ordering::Relaxed)
    }

    pub(crate) fn contains(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.get_for(entry_id, identity).is_some()
    }

    /// Store `batch` under `entry_id` for `identity`, replacing whatever held
    /// the key, including a slot held by another identity.
    pub(crate) fn insert(&self, entry_id: &EntryID, identity: u64, batch: CacheEntry) {
        let guard = self.art.pin();
        let slot = Slot {
            identity,
            entry: batch,
        };
        let existing = self
            .art
            .insert(*entry_id, Arc::new(Cell::new(slot)), &guard)
            .expect("Insertion failed");
        match existing {
            Some(replaced) => {
                replaced.take();
            }
            None => {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn remove(&self, entry_id: &EntryID) -> Option<Arc<Slot>> {
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

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, &Slot)) {
        for id in self.art.keys() {
            if let Some(slot) = self.get(&id) {
                f(&id, &slot);
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
}

#[cfg(test)]
mod tests {
    use crate::cache::cached_batch::CacheEntry;
    use crate::cache::utils::create_test_array;

    use super::*;

    #[test]
    fn test_get_and_contains() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, 0, array1.clone());
        }

        assert!(store.contains(&entry_id1, 0));
        assert!(!store.contains(&entry_id2, 0));

        // Get should return the cached value
        match store.get_for(&entry_id1, 0) {
            Some(slot) => match &slot.entry {
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

        store.insert(&entry_id, 0, array.clone());

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.contains(&entry_id, 0));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.contains(&entry_id, 0));
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
        store.insert(&id, 0, first);
        store.insert(&id, 0, create_test_array(200));
        assert!(
            weak_first.upgrade().is_none(),
            "replaced entry still alive: held by the index's deferred drop"
        );

        let second = store.get(&id).unwrap();
        let removed = store.remove(&id).unwrap();
        let CacheEntry::MemoryArrow(second_array) = &removed.entry else {
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

    #[test]
    fn a_slot_is_visible_only_to_its_identity() {
        let store = ArtIndex::new();
        let entry_id = EntryID::from(1);
        store.insert(&entry_id, 7, create_test_array(10));

        assert!(store.contains(&entry_id, 7));
        assert!(!store.contains(&entry_id, 8));
        assert!(store.get_for(&entry_id, 8).is_none());
        assert_eq!(store.identity_mismatches(), 2);

        store.insert(&entry_id, 8, create_test_array(20));
        assert!(!store.contains(&entry_id, 7));
        assert!(store.contains(&entry_id, 8));
        assert_eq!(store.entry_count(), 1);
    }
}
