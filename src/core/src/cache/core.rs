use arrow::array::cast::AsArray;
use arrow::array::{ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::record_batch::RecordBatch;
use arrow_schema::{Field, Schema};
use bytes::Bytes;
use futures::StreamExt;

use super::{
    budget::BudgetAccounting,
    builders::{EvaluatePredicate, Get, Insert},
    cached_batch::{CacheEntry, CachedBatchType},
    io_context::{EntryMetadata, entry_id_to_key},
    observer::{CacheTracer, InternalEvent, Observer},
    policies::{CachePolicy, HydrationPolicy, HydrationRequest, MaterializedEntry},
    utils::CacheConfig,
};
use crate::cache::policies::{EvictionOutcome, EvictionPolicy};
use crate::cache::utils::arrow_to_bytes;
use crate::cache::{CacheExpression, LiquidExpr, index::ArtIndex, utils::EntryID};
use crate::cache::{CacheFull, CacheStats, EventTrace};
use crate::sync::Arc;

// CacheStats and RuntimeStats moved to stats.rs

/// Cache storage for liquid cache.
///
/// # Identity
///
/// Every entry is cached for an *identity*: a `u64` naming the source the data
/// belongs to. Callers that pack several fields into a fixed-width
/// [`EntryID`] can have two sources compute the same key; the identity is how
/// the cache tells them apart. A lookup under any identity other than the one
/// an entry was cached for is a miss, and an insert replaces whatever held the
/// key. Callers whose keys are already unique can pass any constant.
///
/// Example (async read):
/// ```rust
/// use liquid_cache::cache::{LiquidCacheBuilder, EntryID};
/// use arrow::array::UInt64Array;
/// use std::sync::Arc;
///
/// tokio_test::block_on(async {
/// let storage = LiquidCacheBuilder::new().build().await;
///
/// let entry_id = EntryID::from(0);
/// let arrow_array = Arc::new(UInt64Array::from_iter_values(0..32));
/// // The identity names the source this data belongs to. One source here,
/// // so any constant will do.
/// storage.insert(entry_id, 0, arrow_array.clone()).await;
///
/// // Get the arrow array back asynchronously
/// let retrieved = storage.get(&entry_id, 0).await.unwrap();
/// assert_eq!(retrieved.as_ref(), arrow_array.as_ref());
/// });
/// ```
#[derive(Debug)]
pub struct LiquidCache {
    index: ArtIndex,
    config: CacheConfig,
    budget: BudgetAccounting,
    cache_policy: Box<dyn CachePolicy>,
    hydration_policy: Box<dyn HydrationPolicy>,
    eviction_policy: Box<dyn EvictionPolicy>,
    observer: Arc<Observer>,
    metadata: Arc<dyn EntryMetadata>,
    store: t4::Store,
    disk_key_locks: DiskKeyLocks,
    stray_objects: StrayObjects,
    evict_victims_concurrently: bool,
}

/// Orders the store operations on one key.
///
/// Objects are keyed by entry id alone, so every write of a key lands on the
/// object the previous one left. A write puts the object and then installs the
/// record naming it; a reclaim checks that no record names the object and then
/// deletes it. Unordered, a reclaim can check after a newer write's put and
/// before its install, and delete the object that record is about to name.
/// So a write holds its key's lock from the put through the install, and a
/// reclaim from the check through the delete.
///
/// Keys share a fixed set of locks. Nothing holds one while taking another.
#[derive(Debug)]
struct DiskKeyLocks {
    locks: Box<[futures::lock::Mutex<()>]>,
}

/// A held [`DiskKeyLocks`] lock.
type DiskKeyGuard<'a> = futures::lock::MutexGuard<'a, ()>;

impl DiskKeyLocks {
    const LOCKS: usize = 256;

    fn new() -> Self {
        Self {
            locks: (0..Self::LOCKS)
                .map(|_| futures::lock::Mutex::new(()))
                .collect(),
        }
    }

    async fn lock(&self, entry_id: &EntryID) -> DiskKeyGuard<'_> {
        self.locks[Self::lock_of(entry_id)].lock().await
    }

    fn lock_of(entry_id: &EntryID) -> usize {
        // Callers pack fields into entry ids, so mix the bits rather than let
        // one field pick the lock.
        let mixed = (usize::from(*entry_id) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (mixed >> 56) as usize % Self::LOCKS
    }
}

/// Disk budget held for one stored object, released when dropped.
///
/// A disk record in the index owns its object's charge. Taking the record out
/// hands the charge back as one of these, and an operation that reserves disk
/// holds one until a record takes it over. Every step in between can be
/// awaited, and a caller can drop the future there; the charge still comes
/// back.
#[must_use = "dropping a disk charge releases it"]
struct DiskCharge<'a> {
    budget: &'a BudgetAccounting,
    bytes: usize,
}

impl DiskCharge<'_> {
    /// Leave the charge held, by the record now naming the object.
    fn pass_to_record(self) {
        std::mem::forget(self);
    }
}

impl Drop for DiskCharge<'_> {
    fn drop(&mut self) {
        self.budget.release_disk(self.bytes);
    }
}

/// An object put in the store whose record is not installed yet: its key's
/// lock (see [`DiskKeyLocks`]) and its charge. Installing the record releases
/// the lock and passes the charge to the record; dropping it releases both,
/// and leaves the object for [`StrayObjects`].
struct DiskWrite<'a> {
    key_lock: DiskKeyGuard<'a>,
    charge: DiskCharge<'a>,
    unnamed: UnfinishedObject<'a>,
}

/// Keys whose object an operation was dropped before deleting, or wrote and
/// was dropped before naming.
///
/// The dropped operation gave its charge back (see [`DiskCharge`]), but
/// nothing else would ever find the object. Every spill first deletes these,
/// through [`LiquidCache::delete_unnamed`]: disk only grows by spills, so
/// what they leave is bounded by what was dropped since the last spill.
#[derive(Debug, Default)]
struct StrayObjects {
    keys: crate::sync::Mutex<std::collections::HashSet<EntryID>>,
}

impl StrayObjects {
    fn mark(&self, entry_id: EntryID) {
        self.keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(entry_id);
    }

    fn pop(&self) -> Option<EntryID> {
        let mut keys = self
            .keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry_id = *keys.iter().next()?;
        keys.remove(&entry_id);
        Some(entry_id)
    }
}

/// An object a delete or a write has not settled yet. Dropped before
/// [`Self::settled`], it marks its key in [`StrayObjects`].
struct UnfinishedObject<'a> {
    strays: &'a StrayObjects,
    entry_id: EntryID,
}

impl UnfinishedObject<'_> {
    fn settled(self) {
        std::mem::forget(self);
    }
}

impl Drop for UnfinishedObject<'_> {
    fn drop(&mut self) {
        self.strays.mark(self.entry_id);
    }
}

/// The charge and the object of a disk record that an insert displaced with
/// a memory one.
///
/// The record's object is still in the store and still charged, and nothing
/// names it any more. Deleting it means waiting out any write in flight for
/// the key, which an insert cannot do, so it hands this to its caller for
/// [`LiquidCache::reclaim`]. Dropped unreclaimed, it releases the charge and
/// leaves the object to [`StrayObjects`].
#[must_use = "the displaced object stays on disk until reclaimed"]
struct Orphan<'a>(Option<(DiskCharge<'a>, UnfinishedObject<'a>)>);

/// Outcome of [`LiquidCache::prefetch`].
pub enum PrefetchResult {
    /// A memory-form snapshot of the entry (Arrow or Liquid), ready to hand to a reader.
    Snapshot(Arc<CacheEntry>),
    /// The entry is not in the index, or its disk blob is gone.
    Absent,
}

/// Length of the identity trailer appended to every object written to disk.
const IDENTITY_TRAILER_LEN: usize = std::mem::size_of::<u64>();

/// Builder returned by [`LiquidCache::insert`] for configuring cache writes.
impl LiquidCache {
    /// Return current cache statistics: counts and resource usage.
    pub fn stats(&self) -> CacheStats {
        // Count entries by storage tier and format
        let total_entries = self.index.entry_count();

        let mut memory_arrow_entries = 0usize;
        let mut memory_liquid_entries = 0usize;
        let mut disk_liquid_entries = 0usize;
        let mut disk_arrow_entries = 0usize;

        let mut memory_arrow_bytes = 0usize;
        let mut memory_liquid_bytes = 0usize;

        self.index.for_each(|_, slot| match &slot.entry {
            CacheEntry::MemoryArrow(array) => {
                memory_arrow_entries += 1;
                memory_arrow_bytes += array.get_array_memory_size();
            }
            CacheEntry::MemoryLiquid(array) => {
                memory_liquid_entries += 1;
                memory_liquid_bytes += array.get_array_memory_size();
            }
            CacheEntry::DiskLiquid { .. } => disk_liquid_entries += 1,
            CacheEntry::DiskArrow { .. } => disk_arrow_entries += 1,
        });

        let memory_usage_bytes = self.budget.memory_usage_bytes();
        let disk_usage_bytes = self.budget.disk_usage_bytes();
        let runtime = self.observer.runtime_snapshot();

        CacheStats {
            total_entries,
            memory_arrow_entries,
            memory_liquid_entries,
            disk_liquid_entries,
            disk_arrow_entries,
            memory_arrow_bytes,
            memory_liquid_bytes,
            memory_usage_bytes,
            disk_usage_bytes,
            max_memory_bytes: self.config.max_memory_bytes(),
            max_disk_bytes: self.config.max_disk_bytes(),
            runtime,
        }
    }

    /// Insert a batch into the cache for `identity`.
    pub fn insert<'a>(
        self: &'a Arc<Self>,
        entry_id: EntryID,
        identity: u64,
        batch_to_cache: ArrayRef,
    ) -> Insert<'a> {
        Insert::new(self, entry_id, identity, batch_to_cache)
    }

    /// Create a [`Get`] builder for the provided entry, as cached for `identity`.
    pub fn get<'a>(&'a self, entry_id: &'a EntryID, identity: u64) -> Get<'a> {
        Get::new(self, entry_id, identity)
    }

    /// Create an [`EvaluatePredicate`] builder for evaluating predicates on cached data.
    pub fn eval_predicate<'a>(
        &'a self,
        entry_id: &'a EntryID,
        identity: u64,
        predicate: &'a LiquidExpr,
    ) -> EvaluatePredicate<'a> {
        EvaluatePredicate::new(self, entry_id, identity, predicate)
    }

    /// Prefetch an entry into a memory-form snapshot without recording an access.
    pub async fn prefetch(&self, entry_id: &EntryID, identity: u64) -> PrefetchResult {
        let Some(slot) = self.index.get_for(entry_id, identity) else {
            return PrefetchResult::Absent;
        };
        match &slot.entry {
            entry @ (CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_)) => {
                PrefetchResult::Snapshot(Arc::new(entry.clone()))
            }
            disk @ CacheEntry::DiskArrow { .. } => {
                let Some(array) = self.read_disk_arrow_array(entry_id, identity).await else {
                    return PrefetchResult::Absent;
                };
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    disk,
                    MaterializedEntry::Arrow(&array),
                    None,
                )
                .await;
                PrefetchResult::Snapshot(Arc::new(CacheEntry::memory_arrow(array)))
            }
            disk @ CacheEntry::DiskLiquid { .. } => {
                let Some(array) = self.read_disk_liquid_array(entry_id, identity).await else {
                    return PrefetchResult::Absent;
                };
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    disk,
                    MaterializedEntry::Liquid(&array),
                    None,
                )
                .await;
                PrefetchResult::Snapshot(Arc::new(CacheEntry::memory_liquid(array)))
            }
        }
    }

    /// Try to read a liquid array from the cache.
    /// Returns None if the cached data is not in liquid format.
    pub async fn try_read_liquid(
        &self,
        entry_id: &EntryID,
        identity: u64,
    ) -> Option<crate::liquid_array::LiquidArrayRef> {
        self.observer.on_try_read_liquid();
        self.trace(InternalEvent::TryReadLiquid { entry: *entry_id });
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));

        match &slot.entry {
            CacheEntry::MemoryLiquid(array) => Some(array.clone()),
            entry @ CacheEntry::DiskLiquid { .. } => {
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    None,
                )
                .await;
                Some(liquid)
            }
            CacheEntry::DiskArrow { .. } | CacheEntry::MemoryArrow(_) => None,
        }
    }

    /// Iterate over all entries in the cache, whichever identity holds them.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub fn for_each_entry(&self, mut f: impl FnMut(&EntryID, &CacheEntry)) {
        self.index
            .for_each(|entry_id, slot| f(entry_id, &slot.entry));
    }

    /// Reset the cache.
    pub fn reset(&self) {
        self.index.reset();
        self.budget.reset_usage();
    }

    /// Check whether the cache holds a batch under `entry_id` for `identity`.
    pub fn contains(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.index.contains(entry_id, identity)
    }

    /// How many lookups found their key held by a different identity, and
    /// missed rather than serve it.
    ///
    /// Zero unless two sources compute the same [`EntryID`]. Each such lookup
    /// was served correctly; a rising count means those sources cannot share
    /// the cache.
    pub fn identity_mismatches(&self) -> u64 {
        self.index.identity_mismatches()
    }

    /// Get the config of the cache.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get the budget of the cache.
    pub fn budget(&self) -> &BudgetAccounting {
        &self.budget
    }

    /// Get the tracer of the cache.
    pub fn tracer(&self) -> &CacheTracer {
        self.observer.cache_tracer()
    }

    /// Access the cache observer (runtime stats, debug event trace, and optional cache tracing).
    pub fn observer(&self) -> &Observer {
        &self.observer
    }

    /// Add a lineage expression for an entry.
    pub fn add_lineage(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        self.metadata.add_lineage(entry_id, expression);
    }

    /// Flush all entries to disk.
    pub async fn flush_all_to_disk(&self) -> Result<(), CacheFull> {
        let mut entires = Vec::new();
        self.index.for_each(|entry_id, slot| {
            entires.push((*entry_id, slot.identity, slot.entry.clone()));
        });
        for (entry_id, identity, batch) in entires {
            match &batch {
                CacheEntry::MemoryArrow(array) => {
                    let bytes = arrow_to_bytes(array).expect("failed to convert arrow to bytes");
                    let disk_bytes = bytes.len();
                    match self
                        .write_batch_to_disk(entry_id, identity, &batch, bytes)
                        .await
                    {
                        Ok(write) => {
                            let Orphan(None) = self
                                .try_insert(
                                    entry_id,
                                    identity,
                                    CacheEntry::disk_arrow(array.data_type().clone(), disk_bytes),
                                    Some(write),
                                )
                                .expect("failed to insert disk arrow entry")
                            else {
                                unreachable!("a disk record leaves no orphan");
                            };
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id),
                    }
                }
                CacheEntry::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    let disk_bytes = liquid_bytes.len();
                    match self
                        .write_batch_to_disk(entry_id, identity, &batch, Bytes::from(liquid_bytes))
                        .await
                    {
                        Ok(write) => {
                            let Orphan(None) = self
                                .try_insert(
                                    entry_id,
                                    identity,
                                    CacheEntry::disk_liquid(
                                        liquid_array.original_arrow_data_type(),
                                        disk_bytes,
                                    ),
                                    Some(write),
                                )
                                .expect("failed to insert disk liquid entry")
                            else {
                                unreachable!("a disk record leaves no orphan");
                            };
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id),
                    }
                }
                CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => {
                    // Already on disk, skip
                }
            }
        }
        Ok(())
    }
}

impl LiquidCache {
    /// Returns the batch that replaces `batch`, and the write that put it on
    /// disk, if any, for installing it.
    async fn write_in_memory_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: CacheEntry,
    ) -> Result<(CacheEntry, Option<DiskWrite<'_>>), CacheFull> {
        match &batch {
            batch @ CacheEntry::MemoryArrow(_) => {
                let outcome = self
                    .eviction_policy
                    .evict(batch, self.metadata.lineage(&entry_id).as_deref());
                let EvictionOutcome::Replace {
                    entry: new_batch,
                    bytes_to_write,
                } = outcome
                else {
                    unreachable!("memory Arrow eviction cannot remove entry");
                };
                let write = match bytes_to_write {
                    Some(bytes_to_write) => Some(
                        self.write_batch_to_disk(entry_id, identity, &new_batch, bytes_to_write)
                            .await?,
                    ),
                    None => None,
                };
                Ok((new_batch, write))
            }
            CacheEntry::MemoryLiquid(liquid_array) => {
                let liquid_bytes = Bytes::from(liquid_array.to_bytes());
                let disk_bytes = liquid_bytes.len();
                let write = self
                    .write_batch_to_disk(entry_id, identity, &batch, liquid_bytes)
                    .await?;
                Ok((
                    CacheEntry::disk_liquid(liquid_array.original_arrow_data_type(), disk_bytes),
                    Some(write),
                ))
            }
            CacheEntry::DiskLiquid { .. } | CacheEntry::DiskArrow { .. } => {
                unreachable!("Unexpected batch in write_in_memory_batch_to_disk")
            }
        }
    }

    /// Insert a batch into the cache, it will run cache replacement policy until the batch is inserted.
    pub(crate) async fn insert_inner(
        &self,
        entry_id: EntryID,
        identity: u64,
        mut batch_to_cache: CacheEntry,
    ) -> Result<(), CacheFull> {
        // A write of this entry to disk, made below, for the next install.
        let mut write = None;
        loop {
            let not_inserted =
                match self.try_insert(entry_id, identity, batch_to_cache, write.take()) {
                    Ok(orphan) => {
                        self.reclaim(orphan).await;
                        return Ok(());
                    }
                    Err(not_inserted) => not_inserted,
                };
            self.trace(InternalEvent::InsertFailed {
                entry: entry_id,
                kind: CachedBatchType::from(&not_inserted),
            });

            let victims = self.cache_policy.find_memory_victim(8);
            if victims.is_empty() {
                // no advice, because the cache is already empty
                // this can happen if the entry to be inserted is too large, in that case,
                // we write it to disk
                let (on_disk_batch, written) = self
                    .write_in_memory_batch_to_disk(entry_id, identity, not_inserted)
                    .await?;
                batch_to_cache = on_disk_batch;
                write = written;
                continue;
            }
            self.evict_victims(victims).await?;

            batch_to_cache = not_inserted;
            crate::utils::yield_now_if_shuttle();
        }
    }

    /// Create a new instance of CacheStorage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        eviction_policy: Box<dyn EvictionPolicy>,
        cache_policy: Box<dyn CachePolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        metadata: Arc<dyn EntryMetadata>,
        store: t4::Store,
        evict_victims_concurrently: bool,
    ) -> Self {
        let config = CacheConfig::new(batch_size, max_memory_bytes, max_disk_bytes);
        let observer = Arc::new(Observer::new());
        Self {
            index: ArtIndex::new(),
            budget: BudgetAccounting::new(
                config.max_memory_bytes(),
                config.max_disk_bytes(),
                observer.clone(),
            ),
            config,
            cache_policy,
            hydration_policy,
            eviction_policy,
            observer,
            metadata,
            store,
            disk_key_locks: DiskKeyLocks::new(),
            stray_objects: StrayObjects::default(),
            evict_victims_concurrently,
        }
    }

    /// Install `to_insert` under `entry_id`, if the memory budget allows.
    ///
    /// `write` is the write that put `to_insert`'s object, if this insert's
    /// caller made one; the record takes over its charge, and the lock goes
    /// once the record is installed. If the insert fails, dropping the write
    /// gives both back.
    ///
    /// A disk record this replaces gives up its charge. If `write` put a new
    /// object over the record's, the charge is released. Otherwise the
    /// record's object is left unnamed, and the charge is returned as an
    /// [`Orphan`] for the caller to reclaim.
    fn try_insert<'a>(
        &'a self,
        entry_id: EntryID,
        identity: u64,
        to_insert: CacheEntry,
        write: Option<DiskWrite<'a>>,
    ) -> Result<Orphan<'a>, CacheEntry> {
        debug_assert!(
            write.is_some() || Self::charged_bytes(&to_insert).is_none(),
            "a disk record is installed only by the write that charged it"
        );
        let new_memory_size = to_insert.memory_usage_bytes();
        let reserved = match self.index.get(&entry_id) {
            Some(slot) => self
                .budget
                .try_update_memory_usage(slot.entry.memory_usage_bytes(), new_memory_size),
            // A disk record holds no memory, so it installs however full the
            // memory budget reads. Its write cannot then be dropped, taking
            // its charge, while the caller retries the record.
            None if new_memory_size == 0 => Ok(()),
            None => self.budget.try_reserve_memory(new_memory_size),
        };
        if reserved.is_err() {
            return Err(to_insert);
        }
        let cached_batch_type = CachedBatchType::from(&to_insert);
        let displaced = self
            .index
            .insert(&entry_id, identity, to_insert)
            .and_then(|displaced| self.charge_of(&displaced.entry));
        let orphan = match write {
            Some(DiskWrite {
                key_lock,
                charge,
                unnamed,
            }) => {
                drop(displaced);
                charge.pass_to_record();
                unnamed.settled();
                // Released before the policy hears of the insert, so a policy
                // that reacts by writing this key does not wait on itself.
                drop(key_lock);
                Orphan(None)
            }
            None => Orphan(displaced.map(|charge| (charge, self.unfinished_object(entry_id)))),
        };

        self.trace(InternalEvent::InsertSuccess {
            entry: entry_id,
            kind: cached_batch_type,
        });
        self.cache_policy
            .notify_insert(&entry_id, cached_batch_type);
        Ok(orphan)
    }

    /// The disk budget a record holds: its object's payload and trailer.
    fn charged_bytes(entry: &CacheEntry) -> Option<usize> {
        match entry {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => Some(disk_bytes + IDENTITY_TRAILER_LEN),
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_) => None,
        }
    }

    /// The charge of a record just taken out of the index, if it was on disk.
    fn charge_of(&self, entry: &CacheEntry) -> Option<DiskCharge<'_>> {
        Self::charged_bytes(entry).map(|bytes| DiskCharge {
            budget: &self.budget,
            bytes,
        })
    }

    /// Delete the object an [`Orphan`] left, and release its charge.
    ///
    /// The orphan's charge is released whether or not its object is deleted:
    /// a record that names the object by then had it charged by its own write.
    async fn reclaim(&self, orphan: Orphan<'_>) {
        let Orphan(Some((charge, unfinished))) = orphan else {
            return;
        };
        self.delete_unnamed(unfinished).await;
        drop(charge);
    }

    /// Delete the object `unfinished` stands for, unless a disk record names
    /// it.
    ///
    /// A record installed since the caller decided to delete names the object
    /// now at the key: its write put it there over the one the caller meant.
    /// Dropped before the delete is done, this leaves the key in
    /// [`StrayObjects`], so a later call finishes it.
    async fn delete_unnamed(&self, unfinished: UnfinishedObject<'_>) {
        let entry_id = unfinished.entry_id;
        let _key_lock = self.disk_key_locks.lock(&entry_id).await;
        let named = self
            .index
            .get(&entry_id)
            .is_some_and(|slot| Self::charged_bytes(&slot.entry).is_some());
        if !named {
            self.store
                .remove(&entry_id_to_key(&entry_id))
                .await
                .expect("disk remove failed");
        }
        unfinished.settled();
    }

    /// Delete the objects of [`StrayObjects`] that no record names. Called
    /// holding no key lock, as it takes theirs.
    async fn delete_strays(&self) {
        while let Some(entry_id) = self.stray_objects.pop() {
            self.delete_unnamed(self.unfinished_object(entry_id)).await;
        }
    }

    /// The object at `entry_id`, left to [`StrayObjects`] unless settled.
    fn unfinished_object(&self, entry_id: EntryID) -> UnfinishedObject<'_> {
        UnfinishedObject {
            strays: &self.stray_objects,
            entry_id,
        }
    }

    /// Remove `entry_id` if the index still holds it in memory.
    ///
    /// The flush that calls this read the entry earlier. Another write can
    /// have spilled it since; that record and its object are the other
    /// write's, so they are left alone.
    fn drop_memory_entry(&self, entry_id: EntryID) {
        let Some(removed) = self
            .index
            .remove_if(&entry_id, |slot| Self::charged_bytes(&slot.entry).is_none())
        else {
            return;
        };
        self.budget
            .try_update_memory_usage(removed.entry.memory_usage_bytes(), 0)
            .expect("memory release cannot fail");
        self.cache_policy.notify_remove(&entry_id);
    }

    /// Remove `entry_id` if the index still holds it on disk.
    ///
    /// Callers pick the victim from advice read earlier: the policy's disk
    /// queue, or a slot loaded before the eviction policy ran. A read can have
    /// hydrated the key back into memory since, under the same identity. That
    /// record holds memory, not disk, and its writer files it with the policy,
    /// so it is left alone. The hydrating insert displaced the disk record,
    /// so its reclaim gives that record's disk back, not this call.
    ///
    /// The removal and the delete hold the key's lock, so a write of the key
    /// that follows puts its object after the delete, not before it.
    async fn remove_disk_entry(&self, entry_id: EntryID) {
        let key_lock = self.disk_key_locks.lock(&entry_id).await;
        let Some(removed) = self
            .index
            .remove_if(&entry_id, |slot| Self::charged_bytes(&slot.entry).is_some())
        else {
            return;
        };
        let disk_bytes = match &removed.entry {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            _ => unreachable!("remove_if removed a memory entry"),
        };
        let charge = self.charge_of(&removed.entry);
        let unfinished = self.unfinished_object(entry_id);
        // Told while the key is still held: a spill of the key that follows
        // installs its record and tells the policy only after this. So a
        // custom policy must not wait on a cache operation from
        // `notify_remove`; one that spills a key sharing this lock would wait
        // on itself.
        self.cache_policy.notify_remove(&entry_id);
        self.store
            .remove(&entry_id_to_key(&entry_id))
            .await
            .expect("disk remove failed");
        unfinished.settled();
        drop(key_lock);
        drop(charge);
        self.trace(InternalEvent::DiskEvict {
            entry: entry_id,
            bytes: disk_bytes,
        });
    }

    /// Consume the trace of the cache, for testing only.
    pub fn consume_event_trace(&self) -> EventTrace {
        self.observer.consume_event_trace()
    }

    pub(crate) fn trace(&self, event: InternalEvent) {
        self.observer.record_internal(event);
    }

    /// Get the index of the cache.
    #[cfg(test)]
    pub(crate) fn index(&self) -> &ArtIndex {
        &self.index
    }

    #[fastrace::trace]
    async fn evict_victims(&self, victims: Vec<EntryID>) -> Result<(), CacheFull> {
        self.trace(InternalEvent::EvictionBegin {
            victims: victims.clone(),
        });
        if self.evict_victims_concurrently {
            let results = futures::stream::iter(victims)
                .map(|victim| self.evict_victim_inner(victim))
                .buffer_unordered(usize::MAX)
                .collect::<Vec<_>>()
                .await;
            results.into_iter().collect::<Result<Vec<_>, _>>()?;
        } else {
            for victim in victims {
                self.evict_victim_inner(victim).await?;
            }
        }
        Ok(())
    }

    async fn evict_victim_inner(&self, victim: EntryID) -> Result<(), CacheFull> {
        let Some(slot) = self.index.get(&victim) else {
            return Ok(());
        };
        // The rewrite is made for the identity the victim was read under. If
        // another identity takes the key over meanwhile, the rewrite still
        // carries the old identity, so the new owner reads it as a miss.
        let identity = slot.identity;
        let mut victim_entry = slot.entry.clone();
        self.trace(InternalEvent::EvictionVictim { entry: victim });
        loop {
            let outcome = self
                .eviction_policy
                .evict(&victim_entry, self.metadata.lineage(&victim).as_deref());

            match outcome {
                EvictionOutcome::Replace {
                    entry: new_batch,
                    bytes_to_write,
                } => {
                    let write = match bytes_to_write {
                        Some(bytes_to_write) => Some(
                            self.write_batch_to_disk(victim, identity, &new_batch, bytes_to_write)
                                .await?,
                        ),
                        // The policy left the record on disk as it was. The
                        // key may hold something newer by now, and the object
                        // that record named may be gone, so nothing is
                        // installed.
                        None if Self::charged_bytes(&new_batch).is_some() => break,
                        None => None,
                    };
                    match self.try_insert(victim, identity, new_batch, write) {
                        Ok(orphan) => {
                            self.reclaim(orphan).await;
                            break;
                        }
                        Err(batch) => {
                            victim_entry = batch;
                        }
                    }
                }
                EvictionOutcome::Remove => {
                    self.remove_disk_entry(victim).await;
                    break;
                }
            }
        }
        Ok(())
    }

    async fn maybe_hydrate(
        &self,
        entry_id: &EntryID,
        identity: u64,
        cached: &CacheEntry,
        materialized: MaterializedEntry<'_>,
        expression: Option<&CacheExpression>,
    ) {
        if let Some(new_entry) = self.hydration_policy.hydrate(&HydrationRequest {
            entry_id: *entry_id,
            cached,
            materialized,
            expression,
        }) {
            let cached_type = CachedBatchType::from(cached);
            let new_type = CachedBatchType::from(&new_entry);
            self.trace(InternalEvent::Hydrate {
                entry: *entry_id,
                cached: cached_type,
                new: new_type,
            });
            let _ = self.insert_inner(*entry_id, identity, new_entry).await;
        }
    }

    pub(crate) async fn read_arrow_array(
        &self,
        entry_id: &EntryID,
        identity: u64,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        self.observer.on_get(selection.is_some());
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));
        self.read_entry_inner(entry_id, identity, &slot.entry, selection, expression)
            .await
    }

    /// Read an already-looked-up cache entry, cached for `identity`.
    ///
    /// `entry` is a value the caller already holds, so it is not checked
    /// against `identity`: the caller must have obtained it for that identity,
    /// e.g. from [`Self::prefetch`]. `identity` is checked when the entry's
    /// data has to be read from disk.
    pub async fn read_entry(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        self.observer.on_get(selection.is_some());
        self.read_entry_inner(entry_id, identity, entry, selection, expression)
            .await
    }

    async fn read_entry_inner(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection: Option<&BooleanBuffer>,
        expression: Option<&CacheExpression>,
    ) -> Option<ArrayRef> {
        use arrow::array::BooleanArray;

        self.trace(InternalEvent::Read {
            entry: *entry_id,
            selection: selection.is_some(),
            expr: expression.cloned(),
            cached: CachedBatchType::from(entry),
        });

        match entry {
            CacheEntry::MemoryArrow(array) => match selection {
                Some(selection) => {
                    let selection_array = BooleanArray::new(selection.clone(), None);
                    arrow::compute::filter(array, &selection_array).ok()
                }
                None => Some(array.clone()),
            },
            CacheEntry::MemoryLiquid(array) => match selection {
                Some(selection) => Some(array.filter(selection)),
                None => Some(array.to_arrow_array()),
            },
            CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => {
                self.read_disk_array(entry, entry_id, identity, expression, selection)
                    .await
            }
        }
    }

    async fn read_disk_array(
        &self,
        entry: &CacheEntry,
        entry_id: &EntryID,
        identity: u64,
        expression: Option<&CacheExpression>,
        selection: Option<&BooleanBuffer>,
    ) -> Option<ArrayRef> {
        match entry {
            CacheEntry::DiskArrow { data_type, .. } => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let full_array = self.read_disk_arrow_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Arrow(&full_array),
                    expression,
                )
                .await;
                match selection {
                    Some(selection) => {
                        let selection_array = BooleanArray::new(selection.clone(), None);
                        arrow::compute::filter(&full_array, &selection_array).ok()
                    }
                    None => Some(full_array),
                }
            }
            CacheEntry::DiskLiquid { data_type, .. } => {
                if let Some(selection) = selection
                    && selection.count_set_bits() == 0
                {
                    return Some(arrow::array::new_empty_array(data_type));
                }
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    expression,
                )
                .await;
                match selection {
                    Some(selection) => Some(liquid.filter(selection)),
                    None => Some(liquid.to_arrow_array()),
                }
            }
            _ => unreachable!("Unexpected batch in read_disk_array"),
        }
    }

    /// Write `bytes` to disk for `identity`, followed by the identity itself.
    ///
    /// Objects are keyed by entry id alone, so a write landing after the key
    /// changed hands can leave one identity's bytes where another's entry
    /// points. The trailer is what lets a read notice that and miss. It sits
    /// after the payload so the payload keeps the alignment it is read with.
    /// The disk budget is charged for the trailer as well as the payload.
    ///
    /// Returns the write for [`Self::try_insert`] to install the record
    /// naming the object with.
    async fn write_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: &CacheEntry,
        bytes: Bytes,
    ) -> Result<DiskWrite<'_>, CacheFull> {
        self.delete_strays().await;
        let len = bytes.len();
        loop {
            if self
                .budget
                .try_reserve_disk(len + IDENTITY_TRAILER_LEN)
                .is_ok()
            {
                break;
            }
            let victims = self.cache_policy.find_disk_victim(8);
            if victims.is_empty() {
                return Err(CacheFull);
            }
            for victim in victims {
                self.remove_disk_entry(victim).await;
            }
        }
        let charge = DiskCharge {
            budget: &self.budget,
            bytes: len + IDENTITY_TRAILER_LEN,
        };
        self.trace(InternalEvent::IoWrite {
            entry: entry_id,
            kind: CachedBatchType::from(batch),
            bytes: len,
        });
        let mut object = Vec::with_capacity(len + IDENTITY_TRAILER_LEN);
        object.extend_from_slice(&bytes);
        object.extend_from_slice(&identity.to_le_bytes());
        // Taken only now: removing the victims above takes their keys' locks.
        let key_lock = self.disk_key_locks.lock(&entry_id).await;
        let unnamed = self.unfinished_object(entry_id);
        self.store
            .put(entry_id_to_key(&entry_id), object)
            .await
            .expect("write failed");
        Ok(DiskWrite {
            key_lock,
            charge,
            unnamed,
        })
    }

    /// The payload stored under `entry_id`, if it was written for `identity`.
    async fn read_disk_bytes(&self, entry_id: &EntryID, identity: u64) -> Option<Bytes> {
        let mut object = match self.store.get(&entry_id_to_key(entry_id)).await {
            Ok(object) => object,
            Err(t4::Error::NotFound) => return None,
            Err(error) => panic!("read failed: {error}"),
        };
        let payload_len = object.len().checked_sub(IDENTITY_TRAILER_LEN)?;
        let trailer: [u8; IDENTITY_TRAILER_LEN] = object[payload_len..].try_into().unwrap();
        if u64::from_le_bytes(trailer) != identity {
            return None;
        }
        object.truncate(payload_len);
        Some(Bytes::from(object))
    }

    async fn read_disk_arrow_array(&self, entry_id: &EntryID, identity: u64) -> Option<ArrayRef> {
        let bytes = self.read_disk_bytes(entry_id, identity).await?;
        let bytes_len = bytes.len();
        let cursor = std::io::Cursor::new(bytes);
        let mut reader =
            arrow::ipc::reader::StreamReader::try_new(cursor, None).expect("create reader failed");
        let batch = reader.next().unwrap().expect("read batch failed");
        let array = batch.column(0).clone();
        self.trace(InternalEvent::IoReadArrow {
            entry: *entry_id,
            bytes: bytes_len,
        });
        Some(array)
    }

    async fn read_disk_liquid_array(
        &self,
        entry_id: &EntryID,
        identity: u64,
    ) -> Option<crate::liquid_array::LiquidArrayRef> {
        let bytes = self.read_disk_bytes(entry_id, identity).await?;
        self.trace(InternalEvent::IoReadLiquid {
            entry: *entry_id,
            bytes: bytes.len(),
        });
        Some(Arc::new(crate::liquid_array::LiquidArray::from_bytes(
            bytes,
        )))
    }

    pub(crate) async fn eval_predicate_internal(
        &self,
        entry_id: &EntryID,
        identity: u64,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.observer.on_eval_predicate();
        let slot = self.index.get_for(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(&slot.entry));
        self.eval_predicate_on_entry_inner(
            entry_id,
            identity,
            &slot.entry,
            selection_opt,
            predicate,
        )
        .await
    }

    /// Evaluate a predicate on an already-looked-up cache entry, cached for `identity`.
    ///
    /// As with [`Self::read_entry`], `entry` itself is not checked against
    /// `identity`; only data read from disk is.
    pub async fn eval_predicate_on_entry(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.observer.on_eval_predicate();
        self.eval_predicate_on_entry_inner(entry_id, identity, entry, selection_opt, predicate)
            .await
    }

    async fn eval_predicate_on_entry_inner(
        &self,
        entry_id: &EntryID,
        identity: u64,
        entry: &CacheEntry,
        selection_opt: Option<&BooleanBuffer>,
        predicate: &LiquidExpr,
    ) -> Option<BooleanArray> {
        self.trace(InternalEvent::EvalPredicate {
            entry: *entry_id,
            selection: selection_opt.is_some(),
            cached: CachedBatchType::from(entry),
        });

        match entry {
            CacheEntry::MemoryArrow(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(array, &selection_array)
                    .expect("selection must match array length");
                Some(self.eval_predicate_on_array(filtered, predicate))
            }
            entry @ CacheEntry::DiskArrow { .. } => {
                let array = self.read_disk_arrow_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Arrow(&array),
                    None,
                )
                .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                let selection_array = BooleanArray::new(selection.clone(), None);
                let filtered = arrow::compute::filter(&array, &selection_array)
                    .expect("selection must match array length");
                Some(self.eval_predicate_on_array(filtered, predicate))
            }
            CacheEntry::MemoryLiquid(array) => {
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(array.len()));
                    owned.as_ref().unwrap()
                });
                Some(array.try_eval_predicate(predicate, selection))
            }
            entry @ CacheEntry::DiskLiquid { .. } => {
                let liquid = self.read_disk_liquid_array(entry_id, identity).await?;
                self.maybe_hydrate(
                    entry_id,
                    identity,
                    entry,
                    MaterializedEntry::Liquid(&liquid),
                    None,
                )
                .await;
                let mut owned = None;
                let selection = selection_opt.unwrap_or_else(|| {
                    owned = Some(BooleanBuffer::new_set(liquid.len()));
                    owned.as_ref().unwrap()
                });
                Some(liquid.try_eval_predicate(predicate, selection))
            }
        }
    }

    fn eval_predicate_on_array(&self, array: ArrayRef, predicate: &LiquidExpr) -> BooleanArray {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "liquid_predicate_col",
            array.data_type().clone(),
            true,
        )]));
        let record_batch =
            RecordBatch::try_new(schema, vec![array]).expect("single-column predicate batch");
        let result = predicate
            .physical_expr()
            .evaluate(&record_batch)
            .expect("validated LiquidExpr must evaluate");
        let boolean_array = result
            .into_array(record_batch.num_rows())
            .expect("predicate output must be an array");
        boolean_array.as_boolean().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        AlwaysHydrate, CacheEntry, CachePolicy, LiquidCacheBuilder, LiquidPolicy, TranscodeEvict,
        utils::{arrow_to_bytes, create_cache_store, create_test_array, create_test_arrow_array},
    };
    use crate::sync::thread;
    use arrow::array::{Array, ArrayRef, Int32Array};
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Unified advice type for more concise testing
    #[derive(Debug)]
    struct TestPolicy {
        target_id: Option<EntryID>,
        advice_count: AtomicUsize,
    }

    impl TestPolicy {
        fn new(target_id: Option<EntryID>) -> Self {
            Self {
                target_id,
                advice_count: AtomicUsize::new(0),
            }
        }
    }

    impl CachePolicy for TestPolicy {
        fn find_memory_victim(&self, _cnt: usize) -> Vec<EntryID> {
            self.advice_count.fetch_add(1, Ordering::SeqCst);
            let id_to_use = self.target_id.unwrap();
            vec![id_to_use]
        }
    }

    #[tokio::test]
    async fn test_basic_cache_operations() {
        // Test basic insert, get, and size tracking in one test
        let budget_size = 10 * 1024;
        let store = create_cache_store(budget_size, Box::new(LiquidPolicy::new())).await;

        // 1. Initial budget should be empty
        assert_eq!(store.budget.memory_usage_bytes(), 0);

        // 2. Insert and verify first entry
        let entry_id1: EntryID = EntryID::from(1);
        let array1 = create_test_array(100);
        let size1 = array1.memory_usage_bytes();
        store.insert_inner(entry_id1, 0, array1).await.unwrap();

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.index().get(&entry_id1).unwrap();
        match &retrieved1.entry {
            CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2: EntryID = EntryID::from(2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store.insert_inner(entry_id2, 0, array2).await.unwrap();

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store.insert_inner(entry_id1, 0, array3).await.unwrap();

        assert_eq!(store.budget.memory_usage_bytes(), size3 + size2);
        assert!(store.index().get(&EntryID::from(999)).is_none());
    }

    #[tokio::test]
    async fn test_cache_advice_strategies() {
        // Comprehensive test of all three advice types

        // Create entry IDs we'll use throughout the test
        let entry_id1 = EntryID::from(1);
        let entry_id2 = EntryID::from(2);

        // 1. Test EVICT advice
        {
            let advisor = TestPolicy::new(Some(entry_id1));
            let store = create_cache_store(8000, Box::new(advisor)).await; // Small budget to force advice

            store
                .insert_inner(entry_id1, 0, create_test_array(800))
                .await
                .unwrap();
            match &store.index().get(&entry_id1).unwrap().entry {
                CacheEntry::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store
                .insert_inner(entry_id2, 0, create_test_array(800))
                .await
                .unwrap();
            match &store.index().get(&entry_id1).unwrap().entry {
                CacheEntry::MemoryLiquid(_) => {}
                other => panic!("Expected LiquidMemory after eviction, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_cache_operations() {
        concurrent_cache_operations().await;
    }

    // #[cfg(feature = "shuttle")]
    // #[test]
    // fn shuttle_cache_operations() {
    //     crate::utils::shuttle_test(|| {
    //         block_on(concurrent_cache_operations());
    //     });
    // }

    pub fn block_on<F: Future>(future: F) -> F::Output {
        #[cfg(feature = "shuttle")]
        {
            shuttle::future::block_on(future)
        }
        #[cfg(not(feature = "shuttle"))]
        {
            tokio_test::block_on(future)
        }
    }

    async fn concurrent_cache_operations() {
        let num_threads = 3;
        let ops_per_thread = 50;

        let budget_size = num_threads * ops_per_thread * 100 * 8 / 2;
        let store = create_cache_store(budget_size, Box::new(LiquidPolicy::new())).await;

        let mut handles = vec![];
        for thread_id in 0..num_threads {
            let store = store.clone();
            handles.push(thread::spawn(move || {
                block_on(async {
                    for i in 0..ops_per_thread {
                        let unique_id = thread_id * ops_per_thread + i;
                        let entry_id: EntryID = EntryID::from(unique_id);
                        let array = create_test_arrow_array(100);
                        store.insert(entry_id, 0, array).await.unwrap();
                    }
                });
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // Invariant 1: Every previously inserted entry can be retrieved
        for thread_id in 0..num_threads {
            for i in 0..ops_per_thread {
                let unique_id = thread_id * ops_per_thread + i;
                let entry_id: EntryID = EntryID::from(unique_id);
                assert!(store.index().get(&entry_id).is_some());
            }
        }

        // Invariant 2: Number of entries matches number of insertions
        assert_eq!(store.index().keys().len(), num_threads * ops_per_thread);
    }

    #[tokio::test]
    async fn test_cache_stats_memory_and_disk_usage() {
        // Build a small cache in blocking liquid mode to avoid background tasks
        let storage = LiquidCacheBuilder::new()
            .with_max_memory_bytes(10 * 1024 * 1024)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;

        // Insert two small batches
        let arr1: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let arr2: ArrayRef = Arc::new(Int32Array::from_iter_values(0..128));
        storage
            .insert(EntryID::from(1usize), 0, arr1)
            .await
            .unwrap();
        storage
            .insert(EntryID::from(2usize), 0, arr2)
            .await
            .unwrap();

        // Stats after insert: 2 entries, memory usage > 0, disk usage == 0
        let s = storage.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.memory_usage_bytes > 0);
        assert_eq!(s.disk_usage_bytes, 0);
        assert_eq!(s.max_memory_bytes, 10 * 1024 * 1024);

        // Flush to disk and verify memory usage drops and disk usage increases
        storage.flush_all_to_disk().await.unwrap();
        let s2 = storage.stats();
        assert_eq!(s2.total_entries, 2);
        assert!(s2.disk_usage_bytes > 0);
        // In-memory usage should be reduced after moving to on-disk formats
        assert!(s2.memory_usage_bytes <= s.memory_usage_bytes);
    }

    #[tokio::test]
    async fn hydrate_disk_arrow_on_get_promotes_to_memory() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 0, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::DiskArrow { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::MemoryArrow(_)));
        }
    }

    #[tokio::test]
    async fn an_entry_is_served_only_to_its_identity() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 1, array.clone()).await.unwrap();
        assert!(store.contains(&entry_id, 1));
        assert!(!store.contains(&entry_id, 2));
        assert!(store.get(&entry_id, 2).await.is_none());
        assert!(matches!(
            store.prefetch(&entry_id, 2).await,
            PrefetchResult::Absent
        ));

        store.flush_all_to_disk().await.unwrap();
        assert!(store.get(&entry_id, 2).await.is_none());
        let result = store.get(&entry_id, 1).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());

        let replacement = create_test_arrow_array(16);
        store
            .insert(entry_id, 2, replacement.clone())
            .await
            .unwrap();
        assert!(!store.contains(&entry_id, 1));
        let result = store.get(&entry_id, 2).await.expect("present");
        assert_eq!(result.as_ref(), replacement.as_ref());
    }

    #[tokio::test]
    async fn disk_bytes_written_for_another_identity_are_a_miss() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(321usize);
        let array = create_test_arrow_array(8);

        store.insert(entry_id, 1, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        let disk_entry = store.index().get(&entry_id).unwrap().entry.clone();
        assert!(matches!(disk_entry, CacheEntry::DiskArrow { .. }));

        // A record naming identity 2 over bytes identity 1 wrote: what a
        // rewrite landing after the key changed hands leaves behind.
        store.index().insert(&entry_id, 2, disk_entry);
        assert!(store.get(&entry_id, 2).await.is_none());
    }

    #[tokio::test]
    async fn missing_disk_blob_is_a_cache_miss() {
        let directory = tempfile::tempdir().unwrap();
        let store = t4::mount(directory.path().join("cache.t4")).await.unwrap();
        let cache = LiquidCacheBuilder::new()
            .with_store(store.clone())
            .build()
            .await;
        let id = EntryID::from(320usize);

        cache
            .insert(id, 0, create_test_arrow_array(8))
            .await
            .unwrap();
        cache.flush_all_to_disk().await.unwrap();
        store.remove(&entry_id_to_key(&id)).await.unwrap();

        assert!(cache.get(&id, 0).await.is_none());
    }

    #[tokio::test]
    async fn hydrate_disk_liquid_on_get_promotes_to_memory_liquid() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(322usize);
        let arrow_array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let liquid =
            Arc::new(crate::liquid_array::LiquidArray::from_arrow_array(&arrow_array).unwrap());

        store
            .insert_inner(entry_id, 0, CacheEntry::memory_liquid(liquid.clone()))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::DiskLiquid { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), arrow_array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(&entry.entry, CacheEntry::MemoryLiquid(_)));
        }
    }

    #[tokio::test]
    async fn insert_returns_cache_full_when_memory_and_disk_are_saturated() {
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(0)
            .with_max_disk_bytes(0)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));

        let err = cache.insert(EntryID::from(900usize), 0, array).await;

        assert_eq!(err, Err(CacheFull));
        assert!(!cache.contains(&EntryID::from(900usize), 0));
    }

    #[tokio::test]
    async fn insert_until_disk_full_then_evicts_oldest_disk_entry() {
        let first_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let second_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let first_bytes = arrow_to_bytes(&first_array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let second_bytes = arrow_to_bytes(&second_array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(first_bytes.max(second_bytes))
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;

        let first = EntryID::from(910usize);
        let second = EntryID::from(911usize);
        cache.insert(first, 0, first_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        assert!(cache.contains(&first, 0));

        cache.insert(second, 0, second_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        assert!(!cache.contains(&first, 0));
        assert!(matches!(
            &cache.index().get(&second).unwrap().entry,
            CacheEntry::DiskArrow { .. }
        ));
    }

    #[tokio::test]
    async fn flush_all_to_disk_evicts_when_overflow() {
        let first_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let second_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let disk_bytes = arrow_to_bytes(&first_array).unwrap().len();
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(disk_bytes)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;
        let first = EntryID::from(912usize);
        let second = EntryID::from(913usize);
        cache.insert(first, 0, first_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        cache.insert(second, 0, second_array).await.unwrap();

        cache.flush_all_to_disk().await.unwrap();

        assert!(!cache.contains(&first, 0) || !cache.contains(&second, 0));
    }

    #[tokio::test]
    async fn disk_eviction_releases_budget() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let disk_bytes = arrow_to_bytes(&array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(disk_bytes)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;
        let entry = EntryID::from(914usize);
        cache.insert(entry, 0, array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        let before = cache.stats().disk_usage_bytes;

        cache.remove_disk_entry(entry).await;

        assert_eq!(cache.stats().disk_usage_bytes, before - disk_bytes);
        assert!(!cache.contains(&entry, 0));
    }

    #[tokio::test]
    async fn flush_all_to_disk_drops_entry_on_unrecoverable_overflow() {
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(0)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .build()
            .await;
        let entry_id = EntryID::from(901usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        cache.insert(entry_id, 0, array).await.unwrap();

        let result = cache.flush_all_to_disk().await;

        assert_eq!(result, Ok(()));
        assert!(!cache.contains(&entry_id, 0));
    }

    type DiskVictimHook = Box<dyn FnOnce(&[EntryID]) + Send>;

    /// A [`LiquidPolicy`] that runs a hook once, after it hands out disk
    /// victims and before the cache acts on them.
    struct HookAfterDiskVictims {
        inner: LiquidPolicy,
        hook: Arc<std::sync::Mutex<Option<DiskVictimHook>>>,
    }

    impl std::fmt::Debug for HookAfterDiskVictims {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HookAfterDiskVictims").finish()
        }
    }

    impl CachePolicy for HookAfterDiskVictims {
        fn find_memory_victim(&self, cnt: usize) -> Vec<EntryID> {
            self.inner.find_memory_victim(cnt)
        }

        fn find_disk_victim(&self, cnt: usize) -> Vec<EntryID> {
            let victims = self.inner.find_disk_victim(cnt);
            let hook = self.hook.lock().unwrap().take();
            if let Some(hook) = hook {
                hook(&victims);
            }
            victims
        }

        fn notify_insert(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
            self.inner.notify_insert(entry_id, batch_type);
        }

        fn notify_access(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
            self.inner.notify_access(entry_id, batch_type);
        }

        fn notify_remove(&self, entry_id: &EntryID) {
            self.inner.notify_remove(entry_id);
        }
    }

    /// The memory budget equals what the index's memory entries hold.
    fn assert_memory_budget_matches_index(cache: &LiquidCache) {
        let mut held = 0;
        cache
            .index()
            .for_each(|_, slot| held += slot.entry.memory_usage_bytes());
        assert_eq!(cache.budget.memory_usage_bytes(), held);
    }

    /// A disk victim is picked from the policy's disk queue, and a read can
    /// hydrate that entry back into memory before the eviction removes it.
    /// The eviction must then leave the memory entry alone: it is no longer a
    /// disk entry, and its record is the hydrating read's.
    #[test]
    fn disk_victim_hydrated_before_its_removal_is_left_in_memory() {
        let hydrated = EntryID::from(920usize);
        let flushed = EntryID::from(921usize);
        let hydrated_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let flushed_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let disk_bytes = arrow_to_bytes(&hydrated_array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let hook = Arc::new(std::sync::Mutex::new(None::<DiskVictimHook>));
        let cache = block_on(
            LiquidCacheBuilder::new()
                .with_max_memory_bytes(1 << 20)
                .with_max_disk_bytes(disk_bytes)
                .with_eviction_policy(Box::new(TranscodeEvict))
                .with_cache_policy(Box::new(HookAfterDiskVictims {
                    inner: LiquidPolicy::new(),
                    hook: hook.clone(),
                }))
                .build(),
        );

        block_on(async {
            cache
                .insert(hydrated, 0, hydrated_array.clone())
                .await
                .unwrap();
            cache.flush_all_to_disk().await.unwrap();
            cache.insert(flushed, 0, flushed_array).await.unwrap();
        });
        assert!(matches!(
            &cache.index().get(&hydrated).unwrap().entry,
            CacheEntry::DiskArrow { .. }
        ));

        // Between the advice and the removal, a read hydrates the victim.
        let reader = cache.clone();
        let expected = hydrated_array.clone();
        *hook.lock().unwrap() = Some(Box::new(move |victims: &[EntryID]| {
            assert_eq!(victims, &[hydrated]);
            std::thread::spawn(move || {
                let read = block_on(async { reader.get(&hydrated, 0).await });
                assert_eq!(read.expect("present").as_ref(), expected.as_ref());
            })
            .join()
            .unwrap();
        }));

        // Flushing the second entry needs the first one's disk space.
        assert_eq!(block_on(cache.flush_all_to_disk()), Ok(()));

        assert!(hook.lock().unwrap().is_none(), "the hook never ran");
        assert!(matches!(
            &cache.index().get(&hydrated).unwrap().entry,
            CacheEntry::MemoryArrow(_)
        ));
        let read = block_on(async { cache.get(&hydrated, 0).await });
        assert_eq!(read.expect("present").as_ref(), hydrated_array.as_ref());
        assert_memory_budget_matches_index(&cache);
    }

    /// What one round of [`hydrate_and_spill_cycles_hold_one_disk_copy`] left:
    /// disk charged after the spill, whether the read served the entry, then
    /// disk charged and objects stored after that read hydrated it.
    #[derive(Debug, PartialEq)]
    struct SpillRound {
        charged_after_spill: usize,
        served: bool,
        charged_after_read: usize,
        objects_after_read: usize,
    }

    /// A read hydrates a disk entry back into memory, and the next flush
    /// spills it again. However many rounds that runs, the disk tier holds at
    /// most the one copy the latest spill wrote: a hydrated entry gives its
    /// disk copy back, or each round charges another copy until the cap is
    /// full and spills start failing.
    #[tokio::test]
    async fn hydrate_and_spill_cycles_hold_one_disk_copy() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..1024));
        let one_copy = arrow_to_bytes(&array).unwrap().len() + IDENTITY_TRAILER_LEN;
        let (cache, store, _directory) = cache_over_counted_store(1 << 20, 4 * one_copy).await;
        let entry = EntryID::from(930usize);
        cache.insert(entry, 0, array.clone()).await.unwrap();

        let rounds = 12;
        let mut observed = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            cache.flush_all_to_disk().await.unwrap();
            let charged_after_spill = cache.stats().disk_usage_bytes;
            let read = cache.get(&entry, 0).await;
            if let Some(read) = &read {
                assert_eq!(read.as_ref(), array.as_ref());
            }
            observed.push(SpillRound {
                charged_after_spill,
                served: read.is_some(),
                charged_after_read: cache.stats().disk_usage_bytes,
                objects_after_read: store.len().await.unwrap(),
            });
        }

        let expected: Vec<_> = (0..rounds)
            .map(|_| SpillRound {
                charged_after_spill: one_copy,
                served: true,
                charged_after_read: 0,
                objects_after_read: 0,
            })
            .collect();
        assert_eq!(observed, expected);
    }

    /// A cache that hydrates on every disk read, over a store the test can
    /// count objects in. The directory must outlive the cache.
    async fn cache_over_counted_store(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
    ) -> (Arc<LiquidCache>, t4::Store, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let store = t4::mount(directory.path().join("cache.t4")).await.unwrap();
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(max_memory_bytes)
            .with_max_disk_bytes(max_disk_bytes)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .with_hydration_policy(Box::new(AlwaysHydrate::new()))
            .with_store(store.clone())
            .build()
            .await;
        (cache, store, directory)
    }

    /// The disk budget charges exactly the disk records in the index, and the
    /// store holds exactly the objects they name, each still readable.
    async fn assert_disk_matches_index(cache: &LiquidCache, store: &t4::Store) {
        let mut charged = 0;
        let mut named = Vec::new();
        cache.index().for_each(|entry_id, slot| {
            if let Some(bytes) = LiquidCache::charged_bytes(&slot.entry) {
                charged += bytes;
                named.push((*entry_id, slot.identity));
            }
        });
        assert_eq!(
            cache.budget.disk_usage_bytes(),
            charged,
            "disk charged for anything but the disk records"
        );
        assert_eq!(
            store.len().await.unwrap(),
            named.len(),
            "the store holds objects no disk record names"
        );
        for (entry_id, identity) in named {
            assert!(
                cache.read_disk_bytes(&entry_id, identity).await.is_some(),
                "the disk record for {entry_id:?} names an object that is gone"
            );
        }
    }

    /// Poll `future` once, without a runtime to wake it.
    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        future.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    }

    /// Yield to the runtime until `condition` holds, failing with `what` if it
    /// never does.
    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while !condition() {
            assert!(std::time::Instant::now() < deadline, "{what}");
            tokio::task::yield_now().await;
        }
    }

    /// An Arrow array's disk record, and the bytes a spill of it writes.
    fn arrow_spill(array: &ArrayRef) -> (CacheEntry, Bytes) {
        let bytes = arrow_to_bytes(array).unwrap();
        (
            CacheEntry::disk_arrow(array.data_type().clone(), bytes.len()),
            bytes,
        )
    }

    /// Spill `array` under `entry` the way a flush or an eviction does: put
    /// the object, then install the record naming it.
    async fn spill(cache: &LiquidCache, entry: EntryID, array: &ArrayRef) {
        let (record, bytes) = arrow_spill(array);
        let write = cache
            .write_batch_to_disk(entry, 0, &record, bytes)
            .await
            .unwrap();
        let Ok(Orphan(None)) = cache.try_insert(entry, 0, record, Some(write)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
    }

    /// The disk an Arrow array's record is charged.
    fn arrow_copy(array: &ArrayRef) -> usize {
        arrow_to_bytes(array).unwrap().len() + IDENTITY_TRAILER_LEN
    }

    /// A cache holding `array` under `entry` on disk, spilled by a flush.
    async fn flushed(
        entry: EntryID,
        array: &ArrayRef,
    ) -> (Arc<LiquidCache>, t4::Store, tempfile::TempDir) {
        let (cache, store, directory) = cache_over_counted_store(1 << 20, 1 << 20).await;
        cache.insert(entry, 0, array.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        assert!(matches!(
            cache.index().get(&entry).unwrap().entry,
            CacheEntry::DiskArrow { .. }
        ));
        (cache, store, directory)
    }

    fn is_memory(cache: &LiquidCache, entry: &EntryID) -> bool {
        cache
            .index()
            .get(entry)
            .is_some_and(|slot| LiquidCache::charged_bytes(&slot.entry).is_none())
    }

    /// A flush has spilled an entry, and an eviction that read the entry in
    /// memory before that is spilling it again: its object is put, its record
    /// not installed yet. A read now hydrates the entry, displacing the
    /// flush's record, and reclaims that record's object. The index names no
    /// object at the key at that moment, so a reclaim that went ahead would
    /// delete the object the eviction's record is about to name. The read
    /// must wait for that record, and be woken when it is installed.
    #[tokio::test]
    async fn a_hydrating_read_waits_for_a_spill_in_flight_on_its_key() {
        let entry = EntryID::from(940usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = flushed(entry, &array).await;

        let (respilled, bytes) = arrow_spill(&array);
        let eviction = cache
            .write_batch_to_disk(entry, 0, &respilled, bytes)
            .await
            .unwrap();

        let reader = tokio::spawn({
            let cache = cache.clone();
            let expected = array.clone();
            async move {
                let read = cache.get(&entry, 0).await.expect("present");
                assert_eq!(read.as_ref(), expected.as_ref());
            }
        });
        wait_until("the read never hydrated the entry", || {
            is_memory(&cache, &entry)
        })
        .await;
        assert!(
            !reader.is_finished(),
            "the read reclaimed while a spill of its key was in flight"
        );

        let Ok(Orphan(None)) = cache.try_insert(entry, 0, respilled, Some(eviction)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
        wait_until("the read was never woken", || reader.is_finished()).await;
        reader.await.unwrap();

        assert!(matches!(
            cache.index().get(&entry).unwrap().entry,
            CacheEntry::DiskArrow { .. }
        ));
        assert_disk_matches_index(&cache, &store).await;
    }

    /// The read above, dropped while it waits to reclaim: a query cancelled
    /// mid-read. The displaced record's charge still comes back.
    #[tokio::test]
    async fn a_read_dropped_while_reclaiming_still_releases_the_charge() {
        let entry = EntryID::from(945usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = flushed(entry, &array).await;

        let (respilled, bytes) = arrow_spill(&array);
        let eviction = cache
            .write_batch_to_disk(entry, 0, &respilled, bytes)
            .await
            .unwrap();
        let reader = tokio::spawn({
            let cache = cache.clone();
            async move { cache.get(&entry, 0).await }
        });
        wait_until("the read never hydrated the entry", || {
            is_memory(&cache, &entry)
        })
        .await;
        reader.abort();
        assert!(reader.await.unwrap_err().is_cancelled());

        let Ok(Orphan(None)) = cache.try_insert(entry, 0, respilled, Some(eviction)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));

        // The dropped read left the key to be cleaned up, but the spill's
        // record names the object there now, so the next spill keeps it.
        let other = EntryID::from(9450usize);
        cache.insert(other, 0, array.clone()).await.unwrap();
        spill(&cache, other, &array).await;
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), 2 * arrow_copy(&array));
    }

    /// The read above, but blocked by a spill of another key that shares the
    /// entry's lock, so nothing names the entry's old object when the read is
    /// dropped. The read's charge comes back once, at the drop; the next spill
    /// deletes the object.
    #[tokio::test]
    async fn a_read_dropped_while_reclaiming_leaves_its_object_to_the_next_spill() {
        let entry = EntryID::from(949usize);
        let blocker = (950usize..)
            .map(EntryID::from)
            .find(|id| DiskKeyLocks::lock_of(id) == DiskKeyLocks::lock_of(&entry))
            .unwrap();
        let next = EntryID::from(usize::from(blocker) + 1);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = flushed(entry, &array).await;
        cache.insert(blocker, 0, array.clone()).await.unwrap();
        let (record, bytes) = arrow_spill(&array);
        let blocking = cache
            .write_batch_to_disk(blocker, 0, &record, bytes)
            .await
            .unwrap();

        let reader = tokio::spawn({
            let cache = cache.clone();
            async move { cache.get(&entry, 0).await }
        });
        wait_until("the read never hydrated the entry", || {
            is_memory(&cache, &entry)
        })
        .await;
        reader.abort();
        assert!(reader.await.unwrap_err().is_cancelled());
        assert_eq!(
            cache.budget.disk_usage_bytes(),
            arrow_copy(&array),
            "only the blocking spill's reservation is still charged"
        );

        let Ok(Orphan(None)) = cache.try_insert(blocker, 0, record, Some(blocking)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
        assert_eq!(store.len().await.unwrap(), 2, "the entry's object is left");

        cache.insert(next, 0, array.clone()).await.unwrap();
        spill(&cache, next, &array).await;
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), 2 * arrow_copy(&array));
    }

    /// A spill whose record is never installed, because inserting it failed
    /// or the spill was dropped after its put, leaves its object to the next
    /// spill.
    #[tokio::test]
    async fn a_spill_dropped_after_its_put_leaves_its_object_to_the_next_spill() {
        let (cache, store, _directory) = cache_over_counted_store(1 << 20, 1 << 20).await;
        let entry = EntryID::from(951usize);
        let next = EntryID::from(952usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        cache.insert(entry, 0, array.clone()).await.unwrap();

        let (record, bytes) = arrow_spill(&array);
        let write = cache
            .write_batch_to_disk(entry, 0, &record, bytes)
            .await
            .unwrap();
        drop(write);
        assert_eq!(cache.budget.disk_usage_bytes(), 0);
        assert_eq!(store.len().await.unwrap(), 1, "the put landed");

        cache.insert(next, 0, array.clone()).await.unwrap();
        spill(&cache, next, &array).await;
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    /// A spill reserves its disk and then waits for its key, behind another
    /// spill of the same entry. Dropped there, it gives the reservation back.
    #[tokio::test]
    async fn a_spill_dropped_before_its_record_releases_its_reservation() {
        let (cache, store, _directory) = cache_over_counted_store(1 << 20, 1 << 20).await;
        let entry = EntryID::from(946usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        cache.insert(entry, 0, array.clone()).await.unwrap();

        let (record, bytes) = arrow_spill(&array);
        let first = cache
            .write_batch_to_disk(entry, 0, &record, bytes.clone())
            .await
            .unwrap();
        let mut second = Box::pin(cache.write_batch_to_disk(entry, 0, &record, bytes));
        assert!(poll_once(second.as_mut()).is_pending());
        assert_eq!(
            cache.budget.disk_usage_bytes(),
            2 * arrow_copy(&array),
            "the second spill reserved before waiting for the key"
        );
        drop(second);

        let Ok(Orphan(None)) = cache.try_insert(entry, 0, record, Some(first)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    /// A read hydrates an entry, and a spill of it installs its record before
    /// the read reclaims the displaced one's object. The object at the key is
    /// then the spill's, so it stays; the charge the reclaim carries is the
    /// displaced record's, so it is released.
    #[tokio::test]
    async fn a_reclaim_leaves_the_object_a_later_spill_installed() {
        let entry = EntryID::from(941usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = flushed(entry, &array).await;

        let orphan = cache
            .try_insert(entry, 0, CacheEntry::memory_arrow(array.clone()), None)
            .expect("the hydrated entry fits in memory");
        cache.flush_all_to_disk().await.unwrap();
        cache.reclaim(orphan).await;

        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
        let read = cache
            .get(&entry, 0)
            .await
            .expect("the spill's object is read");
        assert_eq!(read.as_ref(), array.as_ref());
    }

    /// A spill installs its disk record even when the memory budget reads over
    /// its limit, as drift in the memory accounting can leave it. Refused, the
    /// spill would drop its write and with it the charge, and a retry would
    /// then install the record holding no charge.
    #[tokio::test]
    async fn a_spill_installs_its_record_when_the_memory_budget_reads_full() {
        let (cache, store, _directory) = cache_over_counted_store(1 << 20, 1 << 20).await;
        let entry = EntryID::from(957usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        cache
            .budget
            .try_update_memory_usage(1, 0)
            .expect("a release cannot fail");
        assert!(cache.budget.memory_usage_bytes() > cache.config.max_memory_bytes());

        let (record, bytes) = arrow_spill(&array);
        let write = cache
            .write_batch_to_disk(entry, 0, &record, bytes)
            .await
            .unwrap();
        let Ok(Orphan(None)) = cache.try_insert(entry, 0, record, Some(write)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };

        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    /// A flush and an eviction can both read the same memory entry before
    /// either spills it. Both puts land on the key's one object, so the
    /// second record to install gives back the first one's charge.
    #[tokio::test]
    async fn a_second_spill_of_an_entry_releases_the_first_ones_charge() {
        let (cache, store, _directory) = cache_over_counted_store(1 << 20, 1 << 20).await;
        let entry = EntryID::from(942usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        cache.insert(entry, 0, array.clone()).await.unwrap();

        for _ in 0..2 {
            let (record, bytes) = arrow_spill(&array);
            let write = cache
                .write_batch_to_disk(entry, 0, &record, bytes)
                .await
                .unwrap();
            let Ok(Orphan(None)) = cache.try_insert(entry, 0, record, Some(write)) else {
                panic!("a written disk record installs, and leaves no orphan");
            };
        }

        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    type EvictHook = Box<dyn FnOnce() + Send>;

    /// Hands a disk entry back unchanged, writing nothing, after running a
    /// hook once if one is set; otherwise transcodes like [`TranscodeEvict`].
    #[derive(Default)]
    struct KeepDiskEntries {
        hook: Arc<std::sync::Mutex<Option<EvictHook>>>,
    }

    impl std::fmt::Debug for KeepDiskEntries {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("KeepDiskEntries").finish()
        }
    }

    impl EvictionPolicy for KeepDiskEntries {
        fn evict(&self, entry: &CacheEntry, lineage: Option<&CacheExpression>) -> EvictionOutcome {
            match entry {
                CacheEntry::DiskArrow { .. } | CacheEntry::DiskLiquid { .. } => {
                    let hook = self.hook.lock().unwrap().take();
                    if let Some(hook) = hook {
                        hook();
                    }
                    EvictionOutcome::Replace {
                        entry: entry.clone(),
                        bytes_to_write: None,
                    }
                }
                _ => TranscodeEvict.evict(entry, lineage),
            }
        }
    }

    /// A cache whose eviction policy is `policy`, over a counted store.
    async fn cache_keeping_disk_entries(
        policy: KeepDiskEntries,
    ) -> (Arc<LiquidCache>, t4::Store, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let store = t4::mount(directory.path().join("cache.t4")).await.unwrap();
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(1 << 20)
            .with_eviction_policy(Box::new(policy))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .with_hydration_policy(Box::new(AlwaysHydrate::new()))
            .with_store(store.clone())
            .build()
            .await;
        (cache, store, directory)
    }

    /// An eviction policy may hand a disk record back unchanged, writing
    /// nothing. The eviction then leaves the record in place, still naming
    /// its object and holding its charge, so removing it later releases that
    /// charge once.
    #[tokio::test]
    async fn a_disk_record_an_eviction_keeps_keeps_its_charge() {
        let (cache, store, _directory) =
            cache_keeping_disk_entries(KeepDiskEntries::default()).await;
        let entry = EntryID::from(947usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        cache.insert(entry, 0, array.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        cache.evict_victims(vec![entry]).await.unwrap();
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));

        cache.remove_disk_entry(entry).await;
        assert_disk_matches_index(&cache, &store).await;
        assert_eq!(cache.budget.disk_usage_bytes(), 0);
    }

    /// An eviction reads a disk record and asks its policy what to do with
    /// it. Meanwhile a read hydrates the entry, deleting the record's object.
    /// The policy then hands the record back unchanged. That record is stale:
    /// installed, it would replace the hydrated entry with a record naming a
    /// deleted object and holding no charge.
    #[test]
    fn a_stale_disk_record_an_eviction_keeps_is_not_reinstalled() {
        let entry = EntryID::from(948usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let policy = KeepDiskEntries::default();
        let hook = policy.hook.clone();
        let (cache, store, _directory) = block_on(cache_keeping_disk_entries(policy));
        block_on(async {
            cache.insert(entry, 0, array.clone()).await.unwrap();
            cache.flush_all_to_disk().await.unwrap();
        });

        let reader = cache.clone();
        *hook.lock().unwrap() = Some(Box::new(move || {
            std::thread::spawn(move || {
                let read = block_on(async { reader.get(&entry, 0).await });
                assert!(read.is_some());
            })
            .join()
            .unwrap();
        }));
        block_on(cache.evict_victims(vec![entry])).unwrap();

        assert!(hook.lock().unwrap().is_none(), "the hook never ran");
        assert!(is_memory(&cache, &entry), "the stale record was installed");
        block_on(assert_disk_matches_index(&cache, &store));
    }

    /// A [`LiquidPolicy`] whose `notify_remove`, or `notify_insert` once
    /// armed, panics once.
    #[derive(Debug)]
    struct PanicOnce {
        inner: LiquidPolicy,
        on_remove: std::sync::atomic::AtomicBool,
        on_insert: Arc<std::sync::atomic::AtomicBool>,
    }

    impl PanicOnce {
        fn on_remove() -> Self {
            Self {
                inner: LiquidPolicy::new(),
                on_remove: std::sync::atomic::AtomicBool::new(true),
                on_insert: Arc::default(),
            }
        }

        fn on_armed_insert(armed: Arc<std::sync::atomic::AtomicBool>) -> Self {
            Self {
                inner: LiquidPolicy::new(),
                on_remove: std::sync::atomic::AtomicBool::new(false),
                on_insert: armed,
            }
        }
    }

    /// A cache over a counted store whose cache policy is `policy`.
    fn cache_with_policy(policy: PanicOnce) -> (Arc<LiquidCache>, t4::Store, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let store = block_on(t4::mount(directory.path().join("cache.t4"))).unwrap();
        let cache = block_on(
            LiquidCacheBuilder::new()
                .with_max_memory_bytes(1 << 20)
                .with_max_disk_bytes(1 << 20)
                .with_eviction_policy(Box::new(TranscodeEvict))
                .with_cache_policy(Box::new(policy))
                .with_store(store.clone())
                .build(),
        );
        (cache, store, directory)
    }

    impl CachePolicy for PanicOnce {
        fn find_memory_victim(&self, cnt: usize) -> Vec<EntryID> {
            self.inner.find_memory_victim(cnt)
        }

        fn find_disk_victim(&self, cnt: usize) -> Vec<EntryID> {
            self.inner.find_disk_victim(cnt)
        }

        fn notify_insert(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
            if self.on_insert.swap(false, Ordering::SeqCst) {
                panic!("policy panics on insert");
            }
            self.inner.notify_insert(entry_id, batch_type);
        }

        fn notify_access(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
            self.inner.notify_access(entry_id, batch_type);
        }

        fn notify_remove(&self, entry_id: &EntryID) {
            if self.on_remove.swap(false, Ordering::SeqCst) {
                panic!("policy panics on removal");
            }
            self.inner.notify_remove(entry_id);
        }
    }

    /// A disk removal takes the record out of the index, then tells the
    /// policy. If the policy panics there and the caller carries on, the
    /// record's object is left for the next spill to delete.
    #[test]
    fn a_removal_whose_policy_panics_leaves_its_object_to_the_next_spill() {
        let entry = EntryID::from(953usize);
        let next = EntryID::from(954usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = cache_with_policy(PanicOnce::on_remove());
        block_on(async {
            cache.insert(entry, 0, array.clone()).await.unwrap();
            cache.flush_all_to_disk().await.unwrap();
        });

        let removal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on(cache.remove_disk_entry(entry))
        }));
        assert!(removal.is_err(), "the policy never panicked");
        assert!(!cache.contains(&entry, 0));

        block_on(async {
            cache.insert(next, 0, array.clone()).await.unwrap();
            spill(&cache, next, &array).await;
            assert_disk_matches_index(&cache, &store).await;
        });
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    /// An insert puts a memory record over a disk one, then tells the policy.
    /// If the policy panics there and the caller carries on, the displaced
    /// record's charge comes back and its object is left for the next spill.
    #[test]
    fn an_insert_whose_policy_panics_leaves_the_displaced_object_to_the_next_spill() {
        let entry = EntryID::from(955usize);
        let next = EntryID::from(956usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (cache, store, _directory) =
            cache_with_policy(PanicOnce::on_armed_insert(armed.clone()));
        block_on(async {
            cache.insert(entry, 0, array.clone()).await.unwrap();
            cache.flush_all_to_disk().await.unwrap();
        });

        armed.store(true, Ordering::SeqCst);
        let insert = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on(async { cache.insert(entry, 0, array.clone()).await })
        }));
        assert!(insert.is_err(), "the policy never panicked");
        assert!(is_memory(&cache, &entry));
        assert_eq!(cache.budget.disk_usage_bytes(), 0);

        block_on(async {
            cache.insert(next, 0, array.clone()).await.unwrap();
            spill(&cache, next, &array).await;
            assert_disk_matches_index(&cache, &store).await;
        });
        assert_eq!(cache.budget.disk_usage_bytes(), arrow_copy(&array));
    }

    /// A flush that cannot reserve disk for an entry drops it from memory.
    /// While that flush looks for disk victims, an eviction spills the same
    /// entry in a smaller form that fits. That disk record and its object
    /// are the eviction's, and the flush leaves them alone.
    #[test]
    fn a_failed_flush_leaves_a_record_another_writer_spilled() {
        let entry = EntryID::from(943usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let directory = tempfile::tempdir().unwrap();
        let store = block_on(t4::mount(directory.path().join("cache.t4"))).unwrap();
        let hook = Arc::new(std::sync::Mutex::new(None::<DiskVictimHook>));
        let cache = block_on(
            LiquidCacheBuilder::new()
                .with_max_memory_bytes(1 << 20)
                .with_max_disk_bytes(arrow_copy(&array) - 1)
                .with_eviction_policy(Box::new(TranscodeEvict))
                .with_cache_policy(Box::new(HookAfterDiskVictims {
                    inner: LiquidPolicy::new(),
                    hook: hook.clone(),
                }))
                .with_store(store.clone())
                .build(),
        );
        block_on(async { cache.insert(entry, 0, array.clone()).await }).unwrap();

        let evicter = cache.clone();
        *hook.lock().unwrap() = Some(Box::new(move |_: &[EntryID]| {
            std::thread::spawn(move || {
                block_on(async {
                    // Arrow to Liquid in memory, then Liquid to disk.
                    evicter.evict_victims(vec![entry]).await.unwrap();
                    evicter.evict_victims(vec![entry]).await.unwrap();
                })
            })
            .join()
            .unwrap();
        }));

        assert_eq!(block_on(cache.flush_all_to_disk()), Ok(()));

        assert!(hook.lock().unwrap().is_none(), "the hook never ran");
        assert!(matches!(
            cache.index().get(&entry).unwrap().entry,
            CacheEntry::DiskLiquid { .. }
        ));
        block_on(assert_disk_matches_index(&cache, &store));
    }

    /// A flush has spilled an entry, and an eviction that read the same
    /// memory entry earlier is spilling it again: its object is put, its
    /// record not yet installed. Disk pressure now removes the flush's record.
    /// A removal that deleted the object then would delete the eviction's,
    /// and the record the eviction installs next would name nothing. The
    /// removal must wait for that record, and take it with the object.
    #[tokio::test]
    async fn a_disk_removal_waits_for_a_spill_in_flight_on_its_key() {
        let entry = EntryID::from(944usize);
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..256));
        let (cache, store, _directory) = flushed(entry, &array).await;

        let (respilled, bytes) = arrow_spill(&array);
        let eviction = cache
            .write_batch_to_disk(entry, 0, &respilled, bytes)
            .await
            .unwrap();

        let mut removal = std::pin::pin!(cache.remove_disk_entry(entry));
        assert!(
            poll_once(removal.as_mut()).is_pending(),
            "the removal went ahead while a spill of its key was in flight"
        );

        let Ok(Orphan(None)) = cache.try_insert(entry, 0, respilled, Some(eviction)) else {
            panic!("a written disk record installs, and leaves no orphan");
        };
        removal.await;

        assert!(!cache.contains(&entry, 0));
        assert_disk_matches_index(&cache, &store).await;
    }
}
