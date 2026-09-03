use congee::CongeeArc;
use std::{
    fmt::{Debug, Formatter},
    sync::atomic::{AtomicUsize, Ordering},
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
struct Slot(RwLock<Option<Arc<CacheEntry>>>);

impl Slot {
    fn new(entry: CacheEntry) -> Arc<Self> {
        Arc::new(Self(RwLock::new(Some(Arc::new(entry)))))
    }

    fn load(&self) -> Option<Arc<CacheEntry>> {
        self.0.read().unwrap().clone()
    }

    fn take(&self) -> Option<Arc<CacheEntry>> {
        self.0.write().unwrap().take()
    }
}

pub(crate) struct ArtIndex {
    art: CongeeArc<EntryID, Slot>,
    entry_count: AtomicUsize,
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
        }
    }

    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<CacheEntry>> {
        let guard = self.art.pin();
        // A slot emptied by a concurrent remove reads as a miss, exactly as if
        // the remove had won the race outright.
        self.art.get(*entry_id, &guard)?.load()
    }

    pub(crate) fn is_cached(&self, entry_id: &EntryID) -> bool {
        self.get(entry_id).is_some()
    }

    pub(crate) fn insert(&self, entry_id: &EntryID, batch: CacheEntry) {
        let guard = self.art.pin();
        let existing = self
            .art
            .insert(*entry_id, Slot::new(batch), &guard)
            .expect("Insertion failed");
        match existing {
            Some(replaced) => drop(replaced.take()),
            None => {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
            }
        }
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

    pub(crate) fn for_each(&self, mut f: impl FnMut(&EntryID, &CacheEntry)) {
        for id in self.art.keys() {
            if let Some(entry) = self.get(&id) {
                f(&id, &entry);
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
    fn test_get_and_is_cached() {
        let store = ArtIndex::new();
        let entry_id1: EntryID = EntryID::from(1);
        let entry_id2: EntryID = EntryID::from(2);
        let array1 = create_test_array(100);

        // Initially, entries should not be cached
        assert!(!store.is_cached(&entry_id1));
        assert!(!store.is_cached(&entry_id2));
        assert!(store.get(&entry_id1).is_none());

        // Insert an entry and verify it's cached
        {
            store.insert(&entry_id1, array1.clone());
        }

        assert!(store.is_cached(&entry_id1));
        assert!(!store.is_cached(&entry_id2));

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

        store.insert(&entry_id, array.clone());

        let entry_id: EntryID = EntryID::from(1);
        assert!(store.is_cached(&entry_id));

        store.reset();
        let entry_id: EntryID = EntryID::from(1);
        assert!(!store.is_cached(&entry_id));
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
        store.insert(&id, first);
        store.insert(&id, create_test_array(200));
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
}
