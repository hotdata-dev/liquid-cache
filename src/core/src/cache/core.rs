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
use crate::cache::{
    CacheExpression, LiquidExpr,
    index::{ArtIndex, WriteIdentity},
    utils::EntryID,
};
use crate::cache::{CacheFull, CacheStats, EventTrace};
use crate::sync::Arc;

// CacheStats and RuntimeStats moved to stats.rs

/// Cache storage for liquid cache.
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
/// // `0` is the file identity: cache keys pack their fields into fixed
/// // widths, so two sources can compute one key, and the identity is what
/// // keeps a read from being served the other's data.
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
    evict_victims_concurrently: bool,
}

/// Outcome of [`LiquidCache::prefetch`].
pub enum PrefetchResult {
    /// A memory-form snapshot of the entry (Arrow or Liquid), ready to hand to a reader.
    Snapshot(Arc<CacheEntry>),
    /// The entry is not in the index, or its disk blob is gone.
    Absent,
}

/// Disk an insert left for its caller to reclaim.
///
/// A store object is addressed by entry id *and* the identity that wrote it, so
/// an object stops being reachable through the index the moment the key changes
/// hands or the write that produced it is dropped. Nothing else will ever delete
/// it or release its bytes: `release_disk` is driven by an index entry, and by
/// then no index entry names it.
#[derive(Default)]
struct DiskResidue {
    /// `(identity, disk_bytes)` of a disk-resident entry this write displaced,
    /// whose object is now unreachable and has to be deleted.
    displaced: Option<(u64, usize)>,
    /// Bytes of a displaced disk entry whose object this write *overwrote in
    /// place*. The object is current and must be kept; only the superseded
    /// entry's reservation is given back, or one object is charged twice.
    superseded: Option<usize>,
    /// The write itself did not land, so whatever the caller already wrote to
    /// the store under its own identity is unreachable too.
    dropped: bool,
}

impl DiskResidue {
    fn dropped() -> Self {
        Self {
            displaced: None,
            superseded: None,
            dropped: true,
        }
    }

    /// A displaced disk entry's object survives this insert unless the insert
    /// put its own bytes over it.
    ///
    /// The store key is `(entry id, identity)`, so only a disk-resident entry
    /// written under the identity that already held the key addresses the very
    /// same object: there the put overwrote it, and reclaiming would delete the
    /// bytes just written. Otherwise the object is left behind — a different
    /// identity addresses a different key, and an entry that lives in memory
    /// wrote nothing at all — and it becomes unreachable the moment the index
    /// stops naming it.
    fn displacing(
        displaced: Option<&(u64, Arc<CacheEntry>)>,
        writer: u64,
        written: CachedBatchType,
    ) -> Self {
        let overwrites_in_place = matches!(
            written,
            CachedBatchType::DiskLiquid | CachedBatchType::DiskArrow
        );
        let disk_bytes = displaced.and_then(|(identity, entry)| match entry.as_ref() {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => Some((*identity, *disk_bytes)),
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_) => None,
        });
        // Same identity and a disk-resident write means this put landed on the
        // very object the displaced entry named: keep the object, give back only
        // its reservation. Anything else leaves an object nothing can reach.
        match disk_bytes {
            Some((identity, bytes)) if overwrites_in_place && identity == writer => Self {
                displaced: None,
                superseded: Some(bytes),
                dropped: false,
            },
            other => Self {
                displaced: other,
                superseded: None,
                dropped: false,
            },
        }
    }
}

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

        self.index.for_each(|_, _, batch| match batch {
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
            identity_mismatches: self.index.identity_mismatches(),
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

    /// Insert a batch into the cache.
    pub fn insert<'a>(
        self: &'a Arc<Self>,
        entry_id: EntryID,
        identity: u64,
        batch_to_cache: ArrayRef,
    ) -> Insert<'a> {
        Insert::new(self, entry_id, identity, batch_to_cache)
    }

    /// Create a [`Get`] builder for the provided entry.
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
        // Checked, not raw: a prefetch hands a snapshot to a caller, so an
        // aliased key must read as absent rather than as someone else's rows.
        let Some(entry) = self.index.get_checked(entry_id, identity) else {
            return PrefetchResult::Absent;
        };
        match entry.as_ref() {
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_) => {
                PrefetchResult::Snapshot(entry)
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
        let batch = self.index.get_checked(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));

        match batch.as_ref() {
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

    /// Iterate over all entries in the cache.
    /// No guarantees are made about the order of the entries.
    /// Isolation level: read-committed
    pub fn for_each_entry(&self, mut f: impl FnMut(&EntryID, u64, &CacheEntry)) {
        self.index.for_each(&mut f);
    }

    /// Reset the cache.
    pub fn reset(&self) {
        self.index.reset();
        self.budget.reset_usage();
    }

    /// Check whether the cache holds this entry *for this identity*.
    ///
    /// A key held by another identity reads as absent: it belongs to a source
    /// that can no longer read it, and serving it here would hand one caller
    /// another's rows.
    pub fn is_cached(&self, entry_id: &EntryID, identity: u64) -> bool {
        self.index.is_cached(entry_id, identity)
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
        self.for_each_entry(|entry_id, identity, batch| {
            entires.push((*entry_id, identity, batch.clone()));
        });
        for (entry_id, flush_identity, batch) in entires {
            match &batch {
                CacheEntry::MemoryArrow(array) => {
                    let bytes = arrow_to_bytes(array).expect("failed to convert arrow to bytes");
                    let disk_bytes = bytes.len();
                    match self
                        .write_batch_to_disk(entry_id, flush_identity, &batch, bytes)
                        .await
                    {
                        Ok(()) => {
                            let residue = self
                                .try_insert(
                                    entry_id,
                                    WriteIdentity::Rewrite(flush_identity),
                                    CacheEntry::disk_arrow(array.data_type().clone(), disk_bytes),
                                )
                                .expect("failed to insert disk arrow entry");
                            self.settle(entry_id, residue, Some((flush_identity, disk_bytes)))
                                .await;
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id, &batch),
                    }
                }
                CacheEntry::MemoryLiquid(liquid_array) => {
                    let liquid_bytes = liquid_array.to_bytes();
                    let disk_bytes = liquid_bytes.len();
                    match self
                        .write_batch_to_disk(
                            entry_id,
                            flush_identity,
                            &batch,
                            Bytes::from(liquid_bytes),
                        )
                        .await
                    {
                        Ok(()) => {
                            let residue = self
                                .try_insert(
                                    entry_id,
                                    WriteIdentity::Rewrite(flush_identity),
                                    CacheEntry::disk_liquid(
                                        liquid_array.original_arrow_data_type(),
                                        disk_bytes,
                                    ),
                                )
                                .expect("failed to insert disk liquid entry");
                            self.settle(entry_id, residue, Some((flush_identity, disk_bytes)))
                                .await;
                        }
                        Err(CacheFull) => self.drop_memory_entry(entry_id, &batch),
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
    /// returns the batch that was written to disk
    async fn write_in_memory_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: CacheEntry,
    ) -> Result<CacheEntry, CacheFull> {
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
                if let Some(bytes_to_write) = bytes_to_write {
                    self.write_batch_to_disk(entry_id, identity, &new_batch, bytes_to_write)
                        .await?;
                }
                Ok(new_batch)
            }
            CacheEntry::MemoryLiquid(liquid_array) => {
                let liquid_bytes = Bytes::from(liquid_array.to_bytes());
                let disk_bytes = liquid_bytes.len();
                self.write_batch_to_disk(entry_id, identity, &batch, liquid_bytes)
                    .await?;
                Ok(CacheEntry::disk_liquid(
                    liquid_array.original_arrow_data_type(),
                    disk_bytes,
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
        identity: WriteIdentity,
        mut batch_to_cache: CacheEntry,
    ) -> Result<(), CacheFull> {
        // Set once this loop spills the entry to disk itself: those bytes are the
        // caller's own write, so a rewrite dropped as stale has to reclaim them.
        let mut wrote = None;
        loop {
            let not_inserted = match self.try_insert(entry_id, identity, batch_to_cache) {
                Ok(residue) => {
                    self.settle(entry_id, residue, wrote).await;
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
                let on_disk_batch = self
                    .write_in_memory_batch_to_disk(entry_id, identity.value(), not_inserted)
                    .await?;
                if let CacheEntry::DiskLiquid { disk_bytes, .. }
                | CacheEntry::DiskArrow { disk_bytes, .. } = &on_disk_batch
                {
                    wrote = Some((identity.value(), *disk_bytes));
                }
                batch_to_cache = on_disk_batch;
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
            evict_victims_concurrently,
        }
    }

    fn try_insert(
        &self,
        entry_id: EntryID,
        identity: WriteIdentity,
        to_insert: CacheEntry,
    ) -> Result<DiskResidue, CacheEntry> {
        let new_memory_size = to_insert.memory_usage_bytes();
        let (cached_batch_type, outcome) = if let Some(entry) = self.index.get(&entry_id) {
            let old_memory_size = entry.memory_usage_bytes();
            if self
                .budget
                .try_update_memory_usage(old_memory_size, new_memory_size)
                .is_err()
            {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            let outcome = self.index.insert(&entry_id, identity, to_insert);
            if !outcome.stored {
                // A rewrite whose key changed hands since it was read. Give the
                // reservation back rather than counting memory for an entry that
                // was never stored, and tell the caller its disk write is now
                // unreachable.
                self.budget
                    .try_update_memory_usage(new_memory_size, old_memory_size)
                    .ok();
                return Ok(DiskResidue::dropped());
            }
            (batch_type, outcome)
        } else {
            if self.budget.try_reserve_memory(new_memory_size).is_err() {
                return Err(to_insert);
            }
            let batch_type = CachedBatchType::from(&to_insert);
            let outcome = self.index.insert(&entry_id, identity, to_insert);
            if !outcome.stored {
                self.budget.try_update_memory_usage(new_memory_size, 0).ok();
                return Ok(DiskResidue::dropped());
            }
            (batch_type, outcome)
        };

        self.trace(InternalEvent::InsertSuccess {
            entry: entry_id,
            kind: cached_batch_type,
        });
        self.cache_policy
            .notify_insert(&entry_id, cached_batch_type);

        Ok(DiskResidue::displacing(
            outcome.displaced.as_ref(),
            identity.value(),
            cached_batch_type,
        ))
    }

    /// Delete a store object nothing can reach any more and give its bytes back.
    ///
    /// Reached on the paths where an object outlives the index entry that named
    /// it: a write dropped as stale after its bytes were already written, an
    /// entry displaced by a write under a different identity, and a disk entry
    /// replaced by a memory one — hydration, or a caller overwriting the value
    /// — which puts nothing in the store and so leaves the old object whole.
    ///
    /// The deletion is conditional because a store key is
    /// `(entry id, identity)` and an identity comes back: the file-id pool
    /// hands a re-opened path its previous record, so a fresh write can occupy
    /// this key between the moment a reclaim was decided and the moment it
    /// runs. `holds_disk_entry` asks the index whether a live record names the
    /// object right now; if one does, that record's write put the bytes that
    /// are there and deleting them would leave it pointing at nothing.
    ///
    /// The check is a read, and the deletion that follows it is an await, so it
    /// narrows the window rather than closing it: a put that has landed but
    /// whose record has not yet been installed is still invisible here. Closing
    /// it needs the store key to name the write, not just the writer.
    async fn reclaim_orphaned_disk(&self, entry_id: EntryID, identity: u64, disk_bytes: usize) {
        if !self.index.holds_disk_entry(&entry_id, identity) {
            match self
                .store
                .remove(&entry_id_to_key(&entry_id, identity))
                .await
            {
                // `false` means the object was already gone, which is fine: the
                // bytes still have to be given back either way.
                Ok(_) | Err(t4::Error::NotFound) => {}
                Err(error) => panic!("orphan remove failed: {error}"),
            }
        }
        // The reservation belonged to the record that is gone, so it comes back
        // whether or not the object did: an object left standing is one a newer
        // record overwrote, and that record reserved its own bytes for it.
        self.budget.release_disk(disk_bytes);
        self.trace(InternalEvent::DiskEvict {
            entry: entry_id,
            bytes: disk_bytes,
        });
    }

    /// Reclaim whatever an insert left unreachable, including the caller's own
    /// write when it was dropped as stale.
    ///
    /// `wrote` is the (identity, bytes) the caller put in the store before the
    /// insert, if any.
    async fn settle(&self, entry_id: EntryID, residue: DiskResidue, wrote: Option<(u64, usize)>) {
        if let Some((identity, bytes)) = residue.displaced {
            self.reclaim_orphaned_disk(entry_id, identity, bytes).await;
        }
        if let Some(bytes) = residue.superseded {
            // The object stays — this write overwrote it — so only the byte
            // count the superseded entry held is returned.
            self.budget.release_disk(bytes);
        }
        if residue.dropped
            && let Some((identity, bytes)) = wrote
        {
            self.reclaim_orphaned_disk(entry_id, identity, bytes).await;
        }
    }

    fn drop_memory_entry(&self, entry_id: EntryID, _expected: &CacheEntry) {
        let Some(removed) = self.index.remove(&entry_id) else {
            return;
        };
        assert!(
            matches!(
                removed.as_ref(),
                CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_)
            ),
            "flush should only drop memory entries"
        );
        self.budget
            .try_update_memory_usage(removed.memory_usage_bytes(), 0)
            .expect("memory release cannot fail");
        self.cache_policy.notify_remove(&entry_id);
    }

    async fn remove_disk_entry(&self, entry_id: EntryID, removed_identity: u64) {
        // Checked: the caller read this entry earlier, and between then and now
        // the key can change hands. Removing the new owner's record would strand
        // its store object while releasing a byte count taken from the record
        // just destroyed.
        let Some(removed) = self.index.remove_checked(&entry_id, removed_identity) else {
            return;
        };
        let disk_bytes = match removed.as_ref() {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            _ => panic!("remove_disk_entry called for non-disk entry"),
        };
        // Through the same reclaim as every other orphan: the record is gone,
        // so the object is unreachable, and the gap before the delete is the
        // gap a re-used identity can write into.
        self.reclaim_orphaned_disk(entry_id, removed_identity, disk_bytes)
            .await;
        self.cache_policy.notify_remove(&entry_id);
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
        // Read the identity alongside the entry: everything this loop writes
        // back is a rewrite of what it just read, and must be dropped rather
        // than relabelled if the key changes hands meanwhile.
        let Some((identity, mut victim_entry)) = self.index.get_with_identity(&victim) else {
            return Ok(());
        };
        self.trace(InternalEvent::EvictionVictim { entry: victim });
        loop {
            let outcome = self.eviction_policy.evict(
                victim_entry.as_ref(),
                self.metadata.lineage(&victim).as_deref(),
            );

            match outcome {
                EvictionOutcome::Replace {
                    entry: new_batch,
                    bytes_to_write,
                } => {
                    // Remember what went to the store: if the rewrite is then
                    // dropped as stale, these bytes are unreachable and have to
                    // be reclaimed here.
                    let mut wrote = None;
                    if let Some(bytes_to_write) = bytes_to_write {
                        let len = bytes_to_write.len();
                        self.write_batch_to_disk(victim, identity, &new_batch, bytes_to_write)
                            .await?;
                        wrote = Some((identity, len));
                    }
                    match self.try_insert(victim, WriteIdentity::Rewrite(identity), new_batch) {
                        Ok(residue) => {
                            self.settle(victim, residue, wrote).await;
                            break;
                        }
                        Err(batch) => {
                            victim_entry = Arc::new(batch);
                        }
                    }
                }
                EvictionOutcome::Remove => {
                    self.remove_disk_entry(victim, identity).await;
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
            let _ = self
                .insert_inner(*entry_id, WriteIdentity::Rewrite(identity), new_entry)
                .await;
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
        let batch = self.index.get_checked(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));
        self.read_entry_inner(entry_id, identity, batch.as_ref(), selection, expression)
            .await
    }

    /// Read an already-looked-up cache entry.
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

    async fn write_batch_to_disk(
        &self,
        entry_id: EntryID,
        identity: u64,
        batch: &CacheEntry,
        bytes: Bytes,
    ) -> Result<(), CacheFull> {
        let len = bytes.len();
        loop {
            if self.budget.try_reserve_disk(len).is_ok() {
                break;
            }
            let victims = self.cache_policy.find_disk_victim(8);
            if victims.is_empty() {
                return Err(CacheFull);
            }
            for victim in victims {
                // Each victim's object is addressed by the identity that wrote
                // it, so look that up rather than assuming this writer's.
                if let Some((victim_identity, _)) = self.index.get_with_identity(&victim) {
                    self.remove_disk_entry(victim, victim_identity).await;
                }
            }
        }
        self.trace(InternalEvent::IoWrite {
            entry: entry_id,
            kind: CachedBatchType::from(batch),
            bytes: len,
        });
        self.store
            .put(entry_id_to_key(&entry_id, identity), bytes.to_vec())
            .await
            .expect("write failed");
        Ok(())
    }

    async fn read_disk_arrow_array(&self, entry_id: &EntryID, identity: u64) -> Option<ArrayRef> {
        let bytes = match self.store.get(&entry_id_to_key(entry_id, identity)).await {
            Ok(bytes) => bytes,
            Err(t4::Error::NotFound) => return None,
            Err(error) => panic!("read failed: {error}"),
        };
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
        let bytes = match self.store.get(&entry_id_to_key(entry_id, identity)).await {
            Ok(bytes) => bytes,
            Err(t4::Error::NotFound) => return None,
            Err(error) => panic!("read failed: {error}"),
        };
        self.trace(InternalEvent::IoReadLiquid {
            entry: *entry_id,
            bytes: bytes.len(),
        });
        Some(Arc::new(crate::liquid_array::LiquidArray::from_bytes(
            Bytes::from(bytes),
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
        let batch = self.index.get_checked(entry_id, identity)?;
        self.cache_policy
            .notify_access(entry_id, CachedBatchType::from(batch.as_ref()));
        self.eval_predicate_on_entry_inner(
            entry_id,
            identity,
            batch.as_ref(),
            selection_opt,
            predicate,
        )
        .await
    }

    /// Evaluate a predicate on an already-looked-up cache entry.
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
        CacheEntry, CachePolicy, LiquidCacheBuilder, LiquidPolicy, TranscodeEvict,
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

    /// Hydrating a disk entry must not charge its bytes to the disk budget
    /// twice.
    ///
    /// The read that materializes a `DiskArrow`/`DiskLiquid` entry replaces it
    /// with a memory entry. The store object under `(entry id, identity)` and
    /// its share of `used_disk_bytes` outlive that replacement, so the next
    /// spill reserves the same byte count again for an object the put simply
    /// overwrites. Over a fixed working set, repeated read/spill rounds make
    /// `disk_usage_bytes` climb without a byte more being written.
    #[tokio::test]
    async fn hydrating_a_disk_entry_does_not_recharge_its_disk_bytes() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(500usize);
        let array = create_test_arrow_array(1024);

        store.insert(entry_id, 0, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        let charged = store.budget.disk_usage_bytes();
        assert!(charged > 0, "flush must have written bytes");

        let mut usage = vec![charged];
        for _ in 0..3 {
            // Reading a disk entry hydrates it back into memory ...
            let read = store.get(&entry_id, 0).await.expect("present");
            assert_eq!(read.as_ref(), array.as_ref());
            assert!(matches!(
                store.index().get(&entry_id).unwrap().as_ref(),
                CacheEntry::MemoryArrow(_)
            ));
            // ... and the next flush spills the very same bytes again.
            store.flush_all_to_disk().await.unwrap();
            usage.push(store.budget.disk_usage_bytes());
        }

        assert_eq!(
            usage,
            vec![charged; 4],
            "one entry of a fixed size occupies the same disk across read/spill rounds"
        );
    }

    /// A spill that overwrites an entry's own disk object in place must release
    /// the copy it superseded.
    ///
    /// Reported by review. The store key is `(entry id, identity)`, so this
    /// write's put landed on the very object the old entry named — deleting it
    /// would destroy the bytes just written. But the old entry's reservation is
    /// still counted, so one object ends up charged twice.
    ///
    /// The two steps are written out rather than reached through `insert`.
    /// Which write displaces the disk entry is a consequence of budget
    /// arithmetic — an Arrow batch is transcoded and kept in memory long
    /// before it is spilled — so an `insert` that lands here today lands
    /// somewhere else after any change to a policy or a size. Driving the two
    /// steps a spill performs, bytes to the store and then the record, pins
    /// the displacement this test is named for.
    #[tokio::test]
    async fn an_in_place_disk_overwrite_releases_the_copy_it_supersedes() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(1usize);

        // Get the key onto disk first, so the next write displaces a disk entry.
        store
            .insert(entry_id, 7, create_test_arrow_array(512))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        let (named_once, charged_once) = charged_disk_bytes_match_the_index(&store);
        assert!(
            named_once > 0,
            "the entry must be on disk for this to test anything"
        );
        assert_eq!(named_once, charged_once, "baseline must be consistent");

        // Same key, same identity, written to disk again over its own object.
        let replacement = create_test_arrow_array(4096);
        let bytes = arrow_to_bytes(&replacement).unwrap();
        let disk_bytes = bytes.len();
        let on_disk = CacheEntry::disk_arrow(replacement.data_type().clone(), disk_bytes);
        store
            .write_batch_to_disk(entry_id, 7, &on_disk, bytes)
            .await
            .unwrap();
        let residue = store
            .try_insert(entry_id, WriteIdentity::Rewrite(7), on_disk)
            .expect("the record swap must land: the key still holds identity 7");
        assert!(
            residue.superseded.is_some(),
            "this is the displacement the test exists to cover"
        );
        store.settle(entry_id, residue, Some((7, disk_bytes))).await;

        let (named, charged) = charged_disk_bytes_match_the_index(&store);
        assert_eq!(
            charged, named,
            "the superseded copy must be released: one object, one reservation"
        );
        assert_eq!(
            store
                .get(&entry_id, 7)
                .await
                .expect("the overwrite must still be readable")
                .as_ref(),
            replacement.as_ref(),
            "settling must keep the object this write put there"
        );
    }

    /// `DiskResidue::displacing` decides whether a displaced entry's store
    /// object survives the write that displaced it, and only one combination
    /// means it did not: a disk-resident write under the identity that already
    /// held the key addresses the very object the old record named, so the put
    /// landed on it. Every other combination leaves an object behind that
    /// nothing can reach. Getting it wrong either deletes bytes just written or
    /// charges one object twice.
    #[test]
    fn only_a_same_identity_disk_write_lands_on_the_object_it_displaces() {
        let on_disk = (
            7u64,
            Arc::new(CacheEntry::disk_arrow(
                arrow::datatypes::DataType::Int64,
                512,
            )),
        );
        let in_memory = (7u64, Arc::new(create_test_array(8)));
        let another = (
            9u64,
            Arc::new(CacheEntry::disk_arrow(
                arrow::datatypes::DataType::Int64,
                512,
            )),
        );

        for written in [CachedBatchType::DiskArrow, CachedBatchType::DiskLiquid] {
            // The key records the identity, not the form, so a liquid write
            // over an Arrow copy lands on the same object as an Arrow one.
            let residue = DiskResidue::displacing(Some(&on_disk), 7, written);
            assert_eq!(
                residue.superseded,
                Some(512),
                "a {written:?} write under identity 7 overwrote 7's own object"
            );
            assert_eq!(residue.displaced, None);
            assert!(!residue.dropped);

            // A different identity is a different store key.
            let residue = DiskResidue::displacing(Some(&another), 7, written);
            assert_eq!(residue.displaced, Some((9, 512)));
            assert_eq!(residue.superseded, None);
        }

        // A memory write puts nothing in the store, so it cannot have
        // overwritten anything.
        let residue = DiskResidue::displacing(Some(&on_disk), 7, CachedBatchType::MemoryArrow);
        assert_eq!(residue.displaced, Some((7, 512)));
        assert_eq!(residue.superseded, None);

        // A displaced memory entry never had an object.
        let residue = DiskResidue::displacing(Some(&in_memory), 7, CachedBatchType::DiskArrow);
        assert_eq!(residue.displaced, None);
        assert_eq!(residue.superseded, None);

        // Nothing displaced at all.
        let residue = DiskResidue::displacing(None, 7, CachedBatchType::DiskArrow);
        assert_eq!(residue.displaced, None);
        assert_eq!(residue.superseded, None);
    }

    /// Every byte counted against the disk budget must belong to an index
    /// entry that is on disk. Anything else can never be released:
    /// `release_disk` is only ever reached from an index entry.
    fn charged_disk_bytes_match_the_index(cache: &LiquidCache) -> (usize, usize) {
        let mut named = 0usize;
        cache.for_each_entry(|_, _, entry| match entry {
            CacheEntry::DiskLiquid { disk_bytes, .. }
            | CacheEntry::DiskArrow { disk_bytes, .. } => named += *disk_bytes,
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_) => {}
        });
        (named, cache.budget.disk_usage_bytes())
    }

    /// Overwriting a disk-resident entry under the identity that already holds
    /// the key must not leave the superseded copy charged.
    ///
    /// The store key is `(entry id, identity)`, so the object the old entry
    /// named is still there — but this insert wrote nothing to the store, so
    /// it did not overwrite it, and the index no longer names it.
    #[tokio::test]
    async fn overwriting_a_disk_entry_releases_the_superseded_copy() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(501usize);

        store
            .insert(entry_id, 5, create_test_arrow_array(1024))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        assert!(
            store.budget.disk_usage_bytes() > 0,
            "flush must have written"
        );

        // Same identity, new value: the index entry becomes a memory one.
        store
            .insert(entry_id, 5, create_test_arrow_array(2048))
            .await
            .unwrap();

        let (named, charged) = charged_disk_bytes_match_the_index(&store);
        assert_eq!(
            charged, named,
            "the superseded copy is still charged but no index entry names it"
        );
    }

    /// An overwrite must never be readable as the value it replaced, and the
    /// form recorded in the index must be the form the store actually holds.
    #[tokio::test]
    async fn an_overwritten_entry_never_reads_back_the_superseded_bytes() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(502usize);
        let first = create_test_arrow_array(1024);
        let second: ArrayRef = Arc::new(arrow::array::Int64Array::from_iter_values(
            (0..2048).map(|v| v + 1_000_000),
        ));

        store.insert(entry_id, 5, first.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();
        assert!(matches!(
            store.index().get(&entry_id).unwrap().as_ref(),
            CacheEntry::DiskArrow { .. }
        ));

        store.insert(entry_id, 5, second.clone()).await.unwrap();
        assert_eq!(
            store.get(&entry_id, 5).await.expect("present").as_ref(),
            second.as_ref(),
            "the overwrite must be what a read returns"
        );

        // Spill again: whatever form the index records has to match the bytes
        // the store now holds, or the next read decodes one as the other.
        store.flush_all_to_disk().await.unwrap();
        assert!(
            matches!(
                store.index().get(&entry_id).unwrap().as_ref(),
                CacheEntry::DiskArrow { .. }
            ),
            "an Arrow flush must be recorded as an Arrow copy"
        );
        assert_eq!(
            store.get(&entry_id, 5).await.expect("present").as_ref(),
            second.as_ref(),
            "reading the spilled copy must not return the superseded value"
        );
    }

    /// A hydrated entry that is then transcoded and spilled must be recorded
    /// as the form it was written in, not the form it was hydrated from.
    #[tokio::test]
    async fn a_hydrated_then_transcoded_entry_is_recorded_as_what_was_written() {
        let store = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(503usize);
        let array = create_test_arrow_array(1024);

        store.insert(entry_id, 5, array.clone()).await.unwrap();
        store.flush_all_to_disk().await.unwrap();

        // Hydrate the Arrow copy back into memory ...
        store.get(&entry_id, 5).await.expect("present");
        // ... transcode it to liquid, then spill it as liquid.
        store
            .evict_victim_inner(entry_id)
            .await
            .expect("transcode must fit");
        assert!(matches!(
            store.index().get(&entry_id).unwrap().as_ref(),
            CacheEntry::MemoryLiquid(_)
        ));
        store.flush_all_to_disk().await.unwrap();
        assert!(
            matches!(
                store.index().get(&entry_id).unwrap().as_ref(),
                CacheEntry::DiskLiquid { .. }
            ),
            "a liquid flush must be recorded as a liquid copy"
        );

        assert_eq!(
            store.get(&entry_id, 5).await.expect("present").as_ref(),
            array.as_ref(),
            "a liquid stub must not be decoded over Arrow IPC bytes"
        );
        let (named, charged) = charged_disk_bytes_match_the_index(&store);
        assert_eq!(
            charged, named,
            "the Arrow copy it was hydrated from is still charged"
        );
    }

    /// A flush that cannot place an entry on disk drops it. Nothing of that
    /// entry may stay charged against the disk budget afterwards.
    #[tokio::test]
    async fn a_flush_that_drops_an_entry_leaves_no_disk_charged_to_it() {
        let array = create_test_arrow_array(1024);
        let one_copy = arrow_to_bytes(&array).unwrap().len();
        let cache = LiquidCacheBuilder::new()
            .with_max_memory_bytes(1 << 20)
            .with_max_disk_bytes(one_copy)
            .with_eviction_policy(Box::new(TranscodeEvict))
            .with_hydration_policy(Box::new(crate::cache::AlwaysHydrate::new()))
            .with_cache_policy(Box::new(LiquidPolicy::new()))
            .build()
            .await;
        let first = EntryID::from(504usize);
        let second = EntryID::from(505usize);

        cache.insert(first, 0, array.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        // Reading it brings it back into memory; the disk tier should now be
        // empty, so the next flush has room for both entries.
        cache.get(&first, 0).await.expect("present");

        cache.insert(second, 0, array.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        let (named, charged) = charged_disk_bytes_match_the_index(&cache);
        assert_eq!(
            charged, named,
            "disk charged to entries the flush dropped can never be released"
        );
    }

    /// A takeover landing *during* a rewrite's `store.put` must leave the new
    /// owner whole: its bytes, its record, and its share of the budget.
    ///
    /// The steps below are what `evict_victim_inner` does — read the entry with
    /// the identity it holds, write it to the store, then swap the record — with
    /// the takeover injected between the read and the write, which is the one
    /// interleaving that ordering cannot be produced by calling it once.
    ///
    /// Three things hold it together, and the first is the decisive one:
    /// `entry_id_to_key` puts the writer's identity in the store key, so the
    /// stale put addresses its own object and can never reach the new owner's;
    /// `WriteIdentity::Rewrite` makes the index refuse the record swap; and
    /// `settle` reclaims the object the refused write had already put there.
    #[tokio::test]
    async fn a_takeover_during_a_rewrites_disk_write_leaves_the_new_owner_whole() {
        let cache = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(600usize);
        let theirs = create_test_arrow_array(1024);
        let ours: ArrayRef = Arc::new(arrow::array::Int64Array::from_iter_values(
            (0..512).map(|v| v + 7_000_000),
        ));

        // Identity 7 caches an entry, and eviction picks it up.
        cache.insert(entry_id, 7, theirs.clone()).await.unwrap();
        let (observed, _read) = cache.index().get_with_identity(&entry_id).unwrap();
        assert_eq!(observed, 7);
        let stale_bytes = arrow_to_bytes(&theirs).unwrap();
        let stale_len = stale_bytes.len();
        let stale_rewrite = CacheEntry::disk_arrow(theirs.data_type().clone(), stale_len);

        // Identity 9 takes the key over and puts its own copy on disk.
        cache.insert(entry_id, 9, ours.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();
        let owner_bytes = match cache.index().get(&entry_id).unwrap().as_ref() {
            CacheEntry::DiskArrow { disk_bytes, .. } => *disk_bytes,
            other => panic!("expected the new owner on disk, found {other}"),
        };

        // Only now does the stale rewrite's write complete ...
        cache
            .write_batch_to_disk(entry_id, observed, &stale_rewrite, stale_bytes)
            .await
            .unwrap();
        // ... and reach the record swap.
        let residue = cache
            .try_insert(entry_id, WriteIdentity::Rewrite(observed), stale_rewrite)
            .expect("a refused rewrite is not a failure to insert");
        cache
            .settle(entry_id, residue, Some((observed, stale_len)))
            .await;

        // The stale writer's object is gone; the new owner's is not.
        assert!(
            matches!(
                cache.store.get(&entry_id_to_key(&entry_id, observed)).await,
                Err(t4::Error::NotFound)
            ),
            "the refused rewrite must take its own write back"
        );
        assert!(
            cache
                .store
                .get(&entry_id_to_key(&entry_id, 9))
                .await
                .is_ok(),
            "the new owner's object must survive a stale writer"
        );
        let (named, charged) = charged_disk_bytes_match_the_index(&cache);
        assert_eq!(charged, named);
        assert_eq!(charged, owner_bytes, "only the new owner's copy is charged");

        // And the new owner's record still names bytes that decode to its rows.
        assert!(matches!(
            cache.index().get(&entry_id).unwrap().as_ref(),
            CacheEntry::DiskArrow { .. }
        ));
        assert_eq!(
            cache.get(&entry_id, 9).await.expect("present").as_ref(),
            ours.as_ref(),
            "the new owner must read its own rows, never the stale writer's"
        );
        assert!(
            cache.get(&entry_id, 7).await.is_none(),
            "the displaced identity reads a miss"
        );
    }

    /// A settlement must not delete an object a *later* write put at the same
    /// store key.
    ///
    /// A store key is `(entry id, identity)` and an identity comes back: the
    /// file-id pool hands a re-opened path its previous record, so the very key
    /// a settlement decided to delete can be occupied again by a fresh write
    /// before the delete runs. `settle` deletes behind an await, which is all
    /// the room that needs. The steps below are what one insert does — take the
    /// key over, then settle what that displaced — with the re-open and its
    /// spill injected in between.
    #[tokio::test]
    async fn a_settlement_does_not_delete_an_object_written_after_it_was_decided() {
        let cache = create_cache_store(1 << 20, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(700usize);
        let first = create_test_arrow_array(1024);
        let takeover: ArrayRef = Arc::new(arrow::array::Int64Array::from_iter_values(
            (0..512).map(|v| v + 3_000_000),
        ));
        let reborn: ArrayRef = Arc::new(arrow::array::Int64Array::from_iter_values(
            (0..256).map(|v| v + 5_000_000),
        ));

        // Identity 7 caches the key and spills it: an object at (E, 7).
        cache.insert(entry_id, 7, first).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        // Identity 9 takes the key over. That displaces 7's disk entry, so this
        // settlement is going to delete (E, 7) ...
        let residue = cache
            .try_insert(
                entry_id,
                WriteIdentity::Owned(9),
                CacheEntry::memory_arrow(takeover),
            )
            .expect("the takeover fits in memory");

        // ... but before it runs, 7's path is re-opened, the pool hands the same
        // identity back, and it caches this key again and spills it. That put
        // lands on the very object the pending delete names.
        cache.insert(entry_id, 7, reborn.clone()).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        // Only now does the settlement run.
        cache.settle(entry_id, residue, None).await;

        assert_eq!(
            cache
                .get(&entry_id, 7)
                .await
                .expect("the re-cached entry must survive a settlement decided before it")
                .as_ref(),
            reborn.as_ref(),
            "the settlement must not reach a write that came after it"
        );
        let (named, charged) = charged_disk_bytes_match_the_index(&cache);
        assert_eq!(charged, named, "one object, one reservation");
    }

    /// A rewrite that loses its key must not leave its disk write behind.
    ///
    /// The bytes were already in the store when the index refused the write, and
    /// the store key carries the identity that wrote them, so nothing reachable
    /// through the index names them afterwards: neither the object nor its share
    /// of `used_disk_bytes` would ever come back.
    #[tokio::test]
    async fn a_dropped_rewrite_reclaims_the_disk_it_already_wrote() {
        let store = create_cache_store(10 * 1024, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(1usize);

        // The owner caches an entry and flushes it, so a disk object exists.
        store
            .insert(entry_id, 7, create_test_arrow_array(64))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        let after_flush = store.budget.disk_usage_bytes();
        assert!(after_flush > 0, "flush must have written bytes");

        // Another identity takes the key over, so the earlier owner's rewrite is
        // now stale. Its disk object is unreachable and must be reclaimed.
        store
            .insert(entry_id, 9, create_test_arrow_array(64))
            .await
            .unwrap();
        assert_eq!(
            store.budget.disk_usage_bytes(),
            0,
            "the displaced owner's disk bytes must be released, not stranded"
        );
        assert!(
            store.get(&entry_id, 9).await.is_some(),
            "the new owner must still read its own entry"
        );
        assert!(
            store.get(&entry_id, 7).await.is_none(),
            "the displaced owner must not read the new owner's rows"
        );
    }

    /// The removal path is identity-checked: a caller that read an entry earlier
    /// must not destroy the record of whoever holds the key now.
    #[tokio::test]
    async fn removing_a_disk_entry_under_a_stale_identity_is_refused() {
        let store = create_cache_store(10 * 1024, Box::new(LiquidPolicy::new())).await;
        let entry_id = EntryID::from(2usize);

        store
            .insert(entry_id, 3, create_test_arrow_array(64))
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();

        // A stale identity tries to evict it. Nothing of the current owner's may
        // be touched.
        store.remove_disk_entry(entry_id, 999).await;
        assert!(
            store.get(&entry_id, 3).await.is_some(),
            "a stale remove must leave the current owner's entry readable"
        );
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
        store
            .insert_inner(entry_id1, WriteIdentity::Owned(0), array1)
            .await
            .unwrap();

        // Verify budget usage and data correctness
        assert_eq!(store.budget.memory_usage_bytes(), size1);
        let retrieved1 = store.index().get(&entry_id1).unwrap();
        match retrieved1.as_ref() {
            CacheEntry::MemoryArrow(arr) => assert_eq!(arr.len(), 100),
            _ => panic!("Expected ArrowMemory"),
        }

        let entry_id2: EntryID = EntryID::from(2);
        let array2 = create_test_array(200);
        let size2 = array2.memory_usage_bytes();
        store
            .insert_inner(entry_id2, WriteIdentity::Owned(0), array2)
            .await
            .unwrap();

        assert_eq!(store.budget.memory_usage_bytes(), size1 + size2);

        let array3 = create_test_array(150);
        let size3 = array3.memory_usage_bytes();
        store
            .insert_inner(entry_id1, WriteIdentity::Owned(0), array3)
            .await
            .unwrap();

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
                .insert_inner(entry_id1, WriteIdentity::Owned(0), create_test_array(800))
                .await
                .unwrap();
            match store.index().get(&entry_id1).unwrap().as_ref() {
                CacheEntry::MemoryArrow(_) => {}
                other => panic!("Expected ArrowMemory, got {other:?}"),
            }

            store
                .insert_inner(entry_id2, WriteIdentity::Owned(0), create_test_array(800))
                .await
                .unwrap();
            match store.index().get(&entry_id1).unwrap().as_ref() {
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
            assert!(matches!(entry.as_ref(), CacheEntry::DiskArrow { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::MemoryArrow(_)));
        }
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
        store.remove(&entry_id_to_key(&id, 0)).await.unwrap();

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
            .insert_inner(
                entry_id,
                WriteIdentity::Owned(0),
                CacheEntry::memory_liquid(liquid.clone()),
            )
            .await
            .unwrap();
        store.flush_all_to_disk().await.unwrap();
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::DiskLiquid { .. }));
        }

        let result = store.get(&entry_id, 0).await.expect("present");
        assert_eq!(result.as_ref(), arrow_array.as_ref());
        {
            let entry = store.index().get(&entry_id).unwrap();
            assert!(matches!(entry.as_ref(), CacheEntry::MemoryLiquid(_)));
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
        assert!(!cache.is_cached(&EntryID::from(900usize), 0));
    }

    #[tokio::test]
    async fn insert_until_disk_full_then_evicts_oldest_disk_entry() {
        let first_array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let second_array: ArrayRef = Arc::new(Int32Array::from_iter_values(16..32));
        let first_bytes = arrow_to_bytes(&first_array).unwrap().len();
        let second_bytes = arrow_to_bytes(&second_array).unwrap().len();
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
        assert!(cache.is_cached(&first, 0));

        cache.insert(second, 0, second_array).await.unwrap();
        cache.flush_all_to_disk().await.unwrap();

        assert!(!cache.is_cached(&first, 0));
        assert!(matches!(
            cache.index().get(&second).unwrap().as_ref(),
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

        assert!(!cache.is_cached(&first, 0) || !cache.is_cached(&second, 0));
    }

    #[tokio::test]
    async fn disk_eviction_releases_budget() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..16));
        let disk_bytes = arrow_to_bytes(&array).unwrap().len();
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

        cache.remove_disk_entry(entry, 0).await;

        assert_eq!(cache.stats().disk_usage_bytes, before - disk_bytes);
        assert!(!cache.is_cached(&entry, 0));
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
        assert!(!cache.is_cached(&entry_id, 0));
    }
}
