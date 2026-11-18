use super::cached_batch::CacheEntry;
use super::core::same_memory_entry;
use super::observer::Observer;
use super::utils::EntryID;
use crate::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use ahash::AHashMap;

/// Snapshots handed out by prefetch, and memory entries removed from the
/// index while one of their snapshots was still alive.
#[derive(Debug, Default)]
struct HeldReleases {
    snapshots: Mutex<AHashMap<EntryID, Vec<Weak<CacheEntry>>>>,
    /// The live snapshots of each removed entry, with the bytes it keeps charged.
    held: Mutex<Vec<(Vec<Weak<CacheEntry>>, usize)>>,
    held_len: AtomicUsize,
}

#[derive(Debug)]
pub struct BudgetAccounting {
    max_memory_bytes: usize,
    max_disk_bytes: usize,
    used_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
    held: HeldReleases,
    /// Whether a release past zero has been logged; later ones are only counted.
    underflow_logged: AtomicBool,
    observer: Arc<Observer>,
}

impl BudgetAccounting {
    pub(super) fn new(
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        observer: Arc<Observer>,
    ) -> Self {
        Self {
            max_memory_bytes,
            max_disk_bytes,
            used_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
            held: HeldReleases::default(),
            underflow_logged: AtomicBool::new(false),
            observer,
        }
    }

    pub(super) fn reset_usage(&self) {
        self.held.snapshots.lock().unwrap().clear();
        self.held.held.lock().unwrap().clear();
        self.held.held_len.store(0, Ordering::Relaxed);
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
    }

    /// Set the memory usage outright, as accounting drift could leave it.
    #[cfg(test)]
    pub(super) fn set_memory_usage(&self, bytes: usize) {
        self.used_memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Record a snapshot of `entry_id` handed to a reader.
    pub(super) fn register_snapshot(&self, entry_id: EntryID, snapshot: &Arc<CacheEntry>) {
        let mut snapshots = self.held.snapshots.lock().unwrap();
        let live = snapshots.entry(entry_id).or_default();
        live.retain(|weak| weak.strong_count() > 0);
        live.push(Arc::downgrade(snapshot));
    }

    /// Keep `removed`'s memory charged while a snapshot of it is alive.
    ///
    /// Called as `removed` leaves the index under `entry_id`. Returns the
    /// bytes kept charged: all of `removed`'s memory, or 0 when no live
    /// snapshot holds its data.
    pub(super) fn hold_if_snapshotted(&self, entry_id: &EntryID, removed: &CacheEntry) -> usize {
        if !matches!(
            removed,
            CacheEntry::MemoryArrow(_) | CacheEntry::MemoryLiquid(_)
        ) {
            return 0;
        }
        let mut snapshots = self.held.snapshots.lock().unwrap();
        let Some(live) = snapshots.get_mut(entry_id) else {
            return 0;
        };
        let mut holding = Vec::new();
        live.retain(|weak| match weak.upgrade() {
            Some(snapshot) if same_memory_entry(&snapshot, removed) => {
                holding.push(weak.clone());
                false
            }
            Some(_) => true,
            None => false,
        });
        if live.is_empty() {
            snapshots.remove(entry_id);
        }
        drop(snapshots);
        if holding.is_empty() {
            return 0;
        }
        let bytes = removed.memory_usage_bytes();
        let mut held = self.held.held.lock().unwrap();
        held.push((holding, bytes));
        self.held.held_len.store(held.len(), Ordering::Relaxed);
        bytes
    }

    /// Release the charge of every held entry whose snapshots were all dropped.
    pub(super) fn sweep_held(&self) {
        if self.held.held_len.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut freed = 0;
        let mut held = self.held.held.lock().unwrap();
        held.retain(|(snapshots, bytes)| {
            let alive = snapshots.iter().any(|weak| weak.strong_count() > 0);
            if !alive {
                freed += bytes;
            }
            alive
        });
        self.held.held_len.store(held.len(), Ordering::Relaxed);
        drop(held);
        if freed > 0 {
            self.release_memory(freed);
        }
    }

    /// Bytes charged for removed entries that snapshots still hold.
    pub fn held_memory_bytes(&self) -> usize {
        self.sweep_held();
        self.held
            .held
            .lock()
            .unwrap()
            .iter()
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// Try to reserve memory in the cache.
    /// Returns ok if the memory was reserved, err if the memory budget is full.
    pub(super) fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
        if reserve(
            &self.used_memory_bytes,
            request_bytes,
            self.max_memory_bytes,
        )
        .is_ok()
        {
            return Ok(());
        }
        // Held entries whose snapshots were dropped since the last sweep may
        // still be charged; release them and try once more.
        if self.held.held_len.load(Ordering::Relaxed) == 0 {
            return Err(());
        }
        self.sweep_held();
        reserve(
            &self.used_memory_bytes,
            request_bytes,
            self.max_memory_bytes,
        )
    }

    /// Charge `bytes` of memory whatever the budget reads: for memory an
    /// entry already holds, which the budget missed.
    pub(super) fn charge_memory(&self, bytes: usize) {
        let _ = update(&self.used_memory_bytes, |used| {
            Some(used.saturating_add(bytes))
        });
    }

    /// Adjust memory usage after transcoding.
    /// Returns ok if the usage was adjusted, err if the memory budget is full when new_size is larger than old_size.
    pub(super) fn try_update_memory_usage(
        &self,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), ()> {
        if old_size < new_size {
            let diff = new_size - old_size;
            self.try_reserve_memory(diff)?;
            Ok(())
        } else {
            self.release_memory(old_size - new_size);
            Ok(())
        }
    }

    /// Charged memory, after releasing held entries whose snapshots were all dropped.
    pub fn memory_usage_bytes(&self) -> usize {
        self.sweep_held();
        self.used_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn disk_usage_bytes(&self) -> usize {
        self.used_disk_bytes.load(Ordering::Relaxed)
    }

    pub(super) fn try_reserve_disk(&self, request_bytes: usize) -> Result<(), ()> {
        let reserved = reserve(&self.used_disk_bytes, request_bytes, self.max_disk_bytes);
        if reserved.is_err() {
            self.observer.on_disk_reservation_failure();
        }
        reserved
    }

    pub(super) fn release_disk(&self, bytes: usize) {
        if let Some(charged) = release(&self.used_disk_bytes, bytes) {
            self.report_underflow("disk", bytes, charged);
        }
    }

    fn release_memory(&self, bytes: usize) {
        if let Some(charged) = release(&self.used_memory_bytes, bytes) {
            self.report_underflow("memory", bytes, charged);
        }
    }

    /// A release past zero is an accounting error elsewhere: a charge given
    /// back twice, or never taken. It is counted every time and logged once,
    /// so a recurring error cannot flood the log. A reset that zeroes usage
    /// while operations are in flight also leads here, when they release.
    fn report_underflow(&self, tier: &str, released: usize, charged: usize) {
        self.observer.on_budget_underflow();
        if !self.underflow_logged.swap(true, Ordering::Relaxed) {
            log::warn!(
                "{tier} budget released {released} bytes with {charged} charged; usage clamped \
                 at zero. Later occurrences are counted in budget_underflows."
            );
        }
    }
}

/// Add `request_bytes` to `used` if the sum stays within `max_bytes`.
fn reserve(used: &AtomicUsize, request_bytes: usize, max_bytes: usize) -> Result<(), ()> {
    update(used, |used| {
        used.checked_add(request_bytes)
            .filter(|&reserved| reserved <= max_bytes)
    })
    .map(|_| ())
    .map_err(|_| ())
}

/// Replace `atomic`'s value with `f` of it, retrying if another thread
/// changes it meanwhile, unless `f` returns `None`. Returns the value `f`
/// was last given.
///
/// What `fetch_update` does; spelled out because newer toolchains deprecate
/// that name for `try_update`, which older ones and shuttle's atomics lack.
fn update(atomic: &AtomicUsize, f: impl Fn(usize) -> Option<usize>) -> Result<usize, usize> {
    let mut current = atomic.load(Ordering::Relaxed);
    loop {
        let new = f(current).ok_or(current)?;
        match atomic.compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(previous) => return Ok(previous),
            Err(actual) => current = actual,
        }
    }
}

/// Take `bytes` off `used`, stopping at zero. Returns the usage it found if
/// that was less than `bytes`.
///
/// A release larger than the usage means a charge was given back twice or
/// never taken. Wrapping would leave the usage near `usize::MAX`, and the tier
/// would refuse every reservation from then on; stopping at zero keeps the
/// error to the size of the mismatch.
fn release(used: &AtomicUsize, bytes: usize) -> Option<usize> {
    match update(used, |used| Some(used.saturating_sub(bytes))) {
        Ok(previous) | Err(previous) => (previous < bytes).then_some(previous),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{Arc, Barrier, thread};

    fn test_budget(max_memory_bytes: usize, max_disk_bytes: usize) -> BudgetAccounting {
        BudgetAccounting::new(max_memory_bytes, max_disk_bytes, Arc::new(Observer::new()))
    }

    #[test]
    fn test_memory_reservation_and_accounting() {
        let config = test_budget(1000, usize::MAX);

        assert_eq!(config.memory_usage_bytes(), 0);

        assert!(config.try_reserve_memory(500).is_ok());
        assert_eq!(config.memory_usage_bytes(), 500);

        assert!(config.try_reserve_memory(300).is_ok());
        assert_eq!(config.memory_usage_bytes(), 800);

        assert!(config.try_reserve_memory(300).is_err());
        assert_eq!(config.memory_usage_bytes(), 800);

        config.reset_usage();
        assert_eq!(config.memory_usage_bytes(), 0);
    }

    #[test]
    fn test_concurrent_memory_operations() {
        test_concurrent_memory_budget();
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_memory_budget_operations() {
        crate::utils::shuttle_test(test_concurrent_memory_budget);
    }

    fn test_concurrent_memory_budget() {
        let num_threads = 3;
        let max_memory = 10000;
        let operations_per_thread = 100;

        let budget = Arc::new(test_budget(max_memory, usize::MAX));
        let barrier = Arc::new(Barrier::new(num_threads));

        let mut thread_handles = vec![];

        for _ in 0..num_threads {
            let budget_clone = budget.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                let mut successful_reservations = Vec::new();

                barrier_clone.wait();

                for i in 0..operations_per_thread {
                    let reserve_size = 10 + (i % 20) * 5; // 10 to 105 bytes
                    if budget_clone.try_reserve_memory(reserve_size).is_ok() {
                        successful_reservations.push(reserve_size);
                    }

                    if i.is_multiple_of(5) && !successful_reservations.is_empty() {
                        let idx = i % successful_reservations.len();
                        let old_size = successful_reservations[idx];
                        let new_size = if i.is_multiple_of(2) {
                            old_size + 5 // Grow
                        } else {
                            old_size.saturating_sub(5) // Shrink
                        };

                        if budget_clone
                            .try_update_memory_usage(old_size, new_size)
                            .is_ok()
                        {
                            successful_reservations[idx] = new_size;
                        }
                    }
                }
                successful_reservations
            });

            thread_handles.push(handle);
        }

        let mut expected_memory_usage = 0;
        for handle in thread_handles {
            let reservations = handle.join().unwrap();
            for size in reservations {
                expected_memory_usage += size;
            }
        }

        assert_eq!(budget.memory_usage_bytes(), expected_memory_usage);
        assert!(budget.memory_usage_bytes() <= max_memory);
    }

    #[test]
    fn disk_reservation_and_release() {
        let budget = test_budget(usize::MAX, 1000);

        assert_eq!(budget.disk_usage_bytes(), 0);
        assert!(budget.try_reserve_disk(400).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 400);
        assert!(budget.try_reserve_disk(600).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 1000);
        assert!(budget.try_reserve_disk(1).is_err());
        assert_eq!(budget.disk_usage_bytes(), 1000);

        budget.release_disk(250);
        assert_eq!(budget.disk_usage_bytes(), 750);
        assert!(budget.try_reserve_disk(250).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 1000);
    }

    /// A release larger than the charge is an accounting error elsewhere, but
    /// it must not wrap the counter to near `usize::MAX`: the tier would then
    /// refuse every reservation for good.
    #[test]
    fn releasing_more_memory_than_charged_leaves_usage_at_zero() {
        let budget = test_budget(1000, usize::MAX);
        budget.try_reserve_memory(100).unwrap();

        budget
            .try_update_memory_usage(50, 0)
            .expect("a release cannot fail");
        assert_eq!(budget.observer.runtime_snapshot().budget_underflows, 0);

        budget
            .try_update_memory_usage(300, 0)
            .expect("a release cannot fail");

        assert_eq!(budget.memory_usage_bytes(), 0);
        assert_eq!(budget.observer.runtime_snapshot().budget_underflows, 1);
        assert!(budget.try_reserve_memory(1000).is_ok());
        assert_eq!(budget.memory_usage_bytes(), 1000);
    }

    #[test]
    fn releasing_more_disk_than_charged_leaves_usage_at_zero() {
        let budget = test_budget(usize::MAX, 1000);
        budget.try_reserve_disk(100).unwrap();

        budget.release_disk(300);

        assert_eq!(budget.disk_usage_bytes(), 0);
        assert_eq!(budget.observer.runtime_snapshot().budget_underflows, 1);
        assert!(budget.try_reserve_disk(1000).is_ok());
        assert_eq!(budget.disk_usage_bytes(), 1000);
    }

    /// A snapshot that keeps a removed entry charged frees that charge once
    /// dropped, and the next reservation the charge would refuse succeeds:
    /// the refusal sweeps held entries and tries again, so nothing has to be
    /// evicted for room that is already free.
    #[test]
    fn a_refused_reservation_sweeps_dropped_snapshots_and_retries() {
        let array: arrow::array::ArrayRef =
            std::sync::Arc::new(arrow::array::Int64Array::from(vec![7; 1024]));
        let entry = CacheEntry::memory_arrow(array);
        let size = entry.memory_usage_bytes();
        let budget = test_budget(2 * size, usize::MAX);
        let entry_id = EntryID::from(0usize);

        // The entry is charged, then a snapshot of it is taken.
        budget.try_reserve_memory(size).unwrap();
        let snapshot = Arc::new(entry.clone());
        budget.register_snapshot(entry_id, &snapshot);

        // It leaves the index while the snapshot is alive: its charge stays.
        assert_eq!(budget.hold_if_snapshotted(&entry_id, &entry), size);
        // Another entry fills the budget.
        budget.try_reserve_memory(size).unwrap();

        drop(snapshot);
        // Fits only once the held charge is released.
        assert!(budget.try_reserve_memory(size).is_ok());
        assert_eq!(budget.memory_usage_bytes(), 2 * size);
    }

    /// Memory an installed entry already holds is charged even past the
    /// limit, and the charge saturates rather than wrapping.
    #[test]
    fn charging_memory_ignores_the_limit_and_does_not_wrap() {
        let budget = test_budget(1000, usize::MAX);
        budget.try_reserve_memory(900).unwrap();

        budget.charge_memory(300);
        assert_eq!(budget.memory_usage_bytes(), 1200);
        assert!(budget.try_reserve_memory(1).is_err());

        budget.charge_memory(usize::MAX);
        assert_eq!(budget.memory_usage_bytes(), usize::MAX);
    }

    /// A reservation whose sum with the usage does not fit a `usize` is
    /// refused, not wrapped into a small usage (or a panic in debug builds).
    #[test]
    fn a_reservation_past_usize_max_is_refused() {
        let budget = test_budget(usize::MAX, usize::MAX);
        budget.try_reserve_memory(usize::MAX - 1).unwrap();
        budget.try_reserve_disk(usize::MAX - 1).unwrap();

        assert!(budget.try_reserve_memory(2).is_err());
        assert!(budget.try_reserve_disk(2).is_err());
        assert!(budget.try_update_memory_usage(0, 2).is_err());

        assert_eq!(budget.memory_usage_bytes(), usize::MAX - 1);
        assert_eq!(budget.disk_usage_bytes(), usize::MAX - 1);
    }
}
