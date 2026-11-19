//! The files a [`LiquidCacheParquet`](super::LiquidCacheParquet) knows, and the
//! key slot and identity each one is cached under.
//!
//! A cache key holds 16 bits of file id: the file's *slot*. Its *identity* is
//! a 64-bit id handed out once per registration and never reused, and the
//! cache serves an entry only to the identity that cached it. Slots are
//! recycled when a file is forgotten, so registered files keep distinct keys
//! for as long as fewer than 65,536 of them are registered at once.
//!
//! A file is forgotten once nothing can read or write its entries: no handle
//! (file, row group or column) refers to it, and the cache holds no entry
//! for its identity. Until then its registration, and the lineage hints kept
//! under its slot, stay.

use std::sync::Arc;

use ahash::{AHashMap, AHashSet};

use super::ParquetFileIdentity;

/// Registrations at which the first sweep for forgettable files runs, and
/// the fewest registrations between two sweeps.
const MIN_SWEEP_AT: usize = 1024;

/// How many distinct values the cache key's 16-bit file field holds.
const SLOTS: usize = 1 << 16;

/// Where a registered file's entries live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileKey {
    /// The 16-bit file field of the cache key.
    pub(crate) slot: u16,
    /// The id the cache checks an entry against before serving it.
    pub(crate) identity: u64,
}

/// Held by every handle that can read or write a file's entries. The registry
/// holds one more; a file whose lease the registry alone holds has no handle.
#[derive(Debug, Clone, Default)]
pub(crate) struct FileLease(Arc<()>);

impl FileLease {
    fn is_idle(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

#[derive(Debug)]
struct RegisteredFile {
    key: FileKey,
    lease: FileLease,
    /// Bumped on every registration, so a sweep can tell that a handle was
    /// created (and possibly dropped again) while it scanned the cache.
    registrations: u64,
}

/// A file a sweep may forget, as it was when the sweep started.
pub(crate) struct IdleFile {
    identity: ParquetFileIdentity,
    key_identity: u64,
    registrations: u64,
}

#[derive(Debug)]
pub(crate) struct FileRegistry {
    files: AHashMap<ParquetFileIdentity, RegisteredFile>,
    free_slots: Vec<u16>,
    /// The lowest slot never handed out; `1 << 16` once all have been.
    next_slot: u32,
    next_identity: u64,
    over_key_width: u64,
    sweep_at: usize,
}

impl Default for FileRegistry {
    fn default() -> Self {
        Self {
            files: AHashMap::new(),
            free_slots: Vec::new(),
            next_slot: 0,
            next_identity: 0,
            over_key_width: 0,
            sweep_at: MIN_SWEEP_AT,
        }
    }
}

impl FileRegistry {
    /// Register `identity`, or return its existing registration. The second
    /// value is true when the file was not registered before.
    pub(crate) fn register(&mut self, identity: ParquetFileIdentity) -> (FileKey, FileLease, bool) {
        if let Some(file) = self.files.get_mut(&identity) {
            file.registrations += 1;
            return (file.key, file.lease.clone(), false);
        }
        let file_identity = self.next_identity;
        self.next_identity += 1;
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None if self.next_slot <= u16::MAX as u32 => {
                self.next_slot += 1;
                (self.next_slot - 1) as u16
            }
            None => {
                // Every slot is taken by a registered file: share one. The
                // identity check keeps the two files' entries apart.
                self.over_key_width += 1;
                file_identity as u16
            }
        };
        let key = FileKey {
            slot,
            identity: file_identity,
        };
        let lease = FileLease::default();
        self.files.insert(
            identity,
            RegisteredFile {
                key,
                lease: lease.clone(),
                registrations: 0,
            },
        );
        (key, lease, true)
    }

    /// How many files are registered.
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }

    /// Registrations so far that found every slot taken and had to share one.
    pub(crate) fn over_key_width(&self) -> u64 {
        self.over_key_width
    }

    /// Whether the registry has grown enough since the last sweep to sweep
    /// again: once it has doubled, or once every slot could be taken,
    /// whichever comes first, so idle files usually give their slots back
    /// before a new file has to share one. Within [`MIN_SWEEP_AT`]
    /// registrations of every slot being taken, a few may share until the
    /// next sweep.
    pub(crate) fn should_sweep(&self) -> bool {
        self.files.len() >= self.sweep_at
    }

    /// Files no handle refers to.
    pub(crate) fn idle_files(&self) -> Vec<IdleFile> {
        self.files
            .iter()
            .filter(|(_, file)| file.lease.is_idle())
            .map(|(identity, file)| IdleFile {
                identity: identity.clone(),
                key_identity: file.key.identity,
                registrations: file.registrations,
            })
            .collect()
    }

    /// Forget each of `idle` that is still idle, was not registered again
    /// since, and has no entry in the cache (`occupied` is every identity the
    /// cache held an entry for, scanned after `idle` was taken).
    ///
    /// Returns how many files were forgotten and the slots freed by it: those
    /// no registered file uses any more, which are now free to hand out.
    pub(crate) fn forget(
        &mut self,
        idle: &[IdleFile],
        occupied: &AHashSet<u64>,
    ) -> (usize, Vec<u16>) {
        let mut forgotten = 0;
        let mut released = AHashSet::new();
        for candidate in idle {
            let Some(file) = self.files.get(&candidate.identity) else {
                continue;
            };
            if !file.lease.is_idle()
                || file.key.identity != candidate.key_identity
                || file.registrations != candidate.registrations
                || occupied.contains(&file.key.identity)
            {
                continue;
            }
            released.insert(file.key.slot);
            self.files.remove(&candidate.identity);
            forgotten += 1;
        }
        if !released.is_empty() && self.over_key_width > 0 {
            // A shared slot stays taken while any of its files is registered.
            for file in self.files.values() {
                released.remove(&file.key.slot);
            }
        }
        let freed: Vec<u16> = released.into_iter().collect();
        self.free_slots.extend_from_slice(&freed);
        (forgotten, freed)
    }

    /// Schedule the next sweep for when the registry has doubled or could
    /// take every slot, but at least [`MIN_SWEEP_AT`] registrations away.
    pub(crate) fn swept(&mut self) {
        let len = self.files.len();
        self.sweep_at = (len + MIN_SWEEP_AT).max(len.saturating_mul(2).min(SLOTS));
    }

    /// Forget every file. Identities keep counting up, so a handle that
    /// outlives this can never be served a later file's entries.
    pub(crate) fn clear(&mut self) {
        self.files.clear();
        self.free_slots.clear();
        self.next_slot = 0;
        self.sweep_at = MIN_SWEEP_AT;
    }
}

#[cfg(test)]
mod tests {
    use datafusion::execution::object_store::ObjectStoreUrl;

    use super::*;

    fn identity(name: &str) -> ParquetFileIdentity {
        ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), name.to_string())
    }

    #[test]
    fn registering_again_returns_the_same_key() {
        let mut registry = FileRegistry::default();
        let (first, _lease, new) = registry.register(identity("a"));
        assert!(new);
        let (again, _lease2, new) = registry.register(identity("a"));
        assert!(!new);
        assert_eq!(first, again);
        let (other, _lease3, _) = registry.register(identity("b"));
        assert_ne!(other.slot, first.slot);
        assert_ne!(other.identity, first.identity);
    }

    #[test]
    fn forgotten_slot_is_reused_under_a_new_identity() {
        let mut registry = FileRegistry::default();
        let (first, lease, _) = registry.register(identity("a"));
        drop(lease);
        let idle = registry.idle_files();
        let (forgotten, freed) = registry.forget(&idle, &AHashSet::new());
        assert_eq!((forgotten, freed), (1, vec![first.slot]));
        assert_eq!(registry.len(), 0);

        let (second, _lease, new) = registry.register(identity("b"));
        assert!(new);
        assert_eq!(second.slot, first.slot);
        assert_ne!(second.identity, first.identity);
    }

    #[test]
    fn a_file_with_a_handle_entries_or_a_new_registration_is_kept() {
        let mut registry = FileRegistry::default();
        let (_, held, _) = registry.register(identity("held"));
        let (cached, lease, _) = registry.register(identity("cached"));
        drop(lease);
        let (_, lease, _) = registry.register(identity("reopened"));
        drop(lease);

        let idle = registry.idle_files();
        assert_eq!(idle.len(), 2, "only the files without a handle are idle");
        // Reopened and closed again while the sweep scanned the cache.
        drop(registry.register(identity("reopened")).1);
        let occupied = AHashSet::from([cached.identity]);
        let (forgotten, freed) = registry.forget(&idle, &occupied);
        assert_eq!((forgotten, freed.len()), (0, 0));
        assert_eq!(registry.len(), 3);
        drop(held);
    }

    #[test]
    fn a_shared_slot_is_freed_only_when_all_its_files_are_forgotten() {
        let mut registry = FileRegistry {
            next_slot: 1 << 16,
            ..Default::default()
        };
        let (a, lease_a, _) = registry.register(identity("a"));
        let (b, lease_b, _) = registry.register(identity("b"));
        let (c, _lease_c, _) = registry.register(identity("c"));
        assert_eq!(registry.over_key_width(), 3);
        assert_eq!((a.slot, b.slot, c.slot), (0, 1, 2));
        // Force `b` onto `a`'s slot.
        registry.files.get_mut(&identity("b")).unwrap().key.slot = a.slot;

        drop(lease_a);
        let (forgotten, freed) = registry.forget(&registry.idle_files(), &AHashSet::new());
        assert_eq!((forgotten, freed.len()), (1, 0), "b still uses slot 0");
        drop(lease_b);
        let (forgotten, freed) = registry.forget(&registry.idle_files(), &AHashSet::new());
        assert_eq!((forgotten, freed), (1, vec![a.slot]));
    }

    #[test]
    fn a_file_sharing_a_slot_is_forgotten_while_the_other_holds_entries() {
        let mut registry = FileRegistry {
            next_slot: 1 << 16,
            ..Default::default()
        };
        let (a, lease_a, _) = registry.register(identity("a"));
        let (_, lease_b, _) = registry.register(identity("b"));
        registry.files.get_mut(&identity("b")).unwrap().key.slot = a.slot;
        drop((lease_a, lease_b));

        let occupied = AHashSet::from([a.identity]);
        let (forgotten, freed) = registry.forget(&registry.idle_files(), &occupied);
        assert_eq!((forgotten, freed.len()), (1, 0), "b goes, a keeps the slot");
        assert_eq!(registry.len(), 1);
    }

    /// Files that keep cached entries, then a long run of files read once.
    /// Sweeping as `LiquidCacheParquet` does, no file ever has to share a slot.
    #[test]
    fn idle_files_give_their_slots_back_before_slots_run_out() {
        let mut registry = FileRegistry::default();
        let mut occupied = AHashSet::new();
        for i in 0..200_000 {
            let (key, lease, new) = registry.register(identity(&format!("f{i}")));
            drop(lease);
            if i < 40_000 {
                occupied.insert(key.identity);
            }
            if new && registry.should_sweep() {
                let idle = registry.idle_files();
                registry.forget(&idle, &occupied);
                registry.swept();
            }
        }
        assert_eq!(registry.over_key_width(), 0);
        assert!(registry.len() < SLOTS);
    }
}
