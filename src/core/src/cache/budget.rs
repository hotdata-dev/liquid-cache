use super::observer::Observer;
use crate::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
pub struct BudgetAccounting {
    max_memory_bytes: usize,
    max_disk_bytes: usize,
    used_memory_bytes: AtomicUsize,
    in_flight_memory_bytes: AtomicUsize,
    peak_in_flight_memory_bytes: AtomicUsize,
    used_disk_bytes: AtomicUsize,
    observer: Arc<Observer>,
}

/// Bytes that are materialized in memory but not yet indexed.
///
/// The cache holds one of these for every intermediate the
/// hydrate -> insert -> squeeze cycle creates outside the index: a disk entry
/// being decoded, a squeeze output waiting to be written and inserted, and an
/// entry pending admission while room is made for it. Nothing counted them
/// before, which is why a tier reporting itself at its limit could sit inside a
/// process holding several times that.
///
/// These bytes do not gate admission: they already exist by the time their size
/// is known, so refusing them would free nothing. They are reported, and they
/// bound how many transcodes [`super::core::LiquidCache`] runs at once.
#[derive(Debug)]
#[must_use = "dropping the reservation immediately releases the bytes"]
pub(super) struct InFlightReservation<'a> {
    budget: &'a BudgetAccounting,
    bytes: usize,
}

impl InFlightReservation<'_> {
    /// Change the reserved amount, for when the true size is only known part
    /// way through: a disk read reserves the encoded buffer before decoding and
    /// resizes to the decoded array, a transcode reserves its input size as an
    /// upper bound and resizes to the compressed result.
    pub(super) fn resize(&mut self, bytes: usize) {
        self.budget.release_in_flight(self.bytes);
        self.budget.reserve_in_flight_bytes(bytes);
        self.bytes = bytes;
    }

    /// Stop counting these bytes as in-flight, because the caller is about to
    /// account for them another way: by inserting them into the index.
    pub(super) fn release(self) {
        drop(self);
    }
}

impl Drop for InFlightReservation<'_> {
    fn drop(&mut self) {
        self.budget.release_in_flight(self.bytes);
    }
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
            in_flight_memory_bytes: AtomicUsize::new(0),
            peak_in_flight_memory_bytes: AtomicUsize::new(0),
            used_disk_bytes: AtomicUsize::new(0),
            observer,
        }
    }

    pub(super) fn reset_usage(&self) {
        self.used_memory_bytes.store(0, Ordering::Relaxed);
        self.used_disk_bytes.store(0, Ordering::Relaxed);
        self.peak_in_flight_memory_bytes.store(0, Ordering::Relaxed);
    }

    /// Try to reserve memory in the cache.
    /// Returns ok if the memory was reserved, err if the memory budget is full.
    pub(super) fn try_reserve_memory(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_memory_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_memory_bytes {
            return Err(());
        }

        match self.used_memory_bytes.compare_exchange(
            used,
            used + request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(_) => self.try_reserve_memory(request_bytes),
        }
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
            self.used_memory_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Reserve bytes that are live in memory but not yet in the index.
    ///
    /// Cannot fail: see [`InFlightReservation`].
    pub(super) fn reserve_in_flight(&self, bytes: usize) -> InFlightReservation<'_> {
        self.reserve_in_flight_bytes(bytes);
        InFlightReservation {
            budget: self,
            bytes,
        }
    }

    fn reserve_in_flight_bytes(&self, bytes: usize) {
        let total = self
            .in_flight_memory_bytes
            .fetch_add(bytes, Ordering::Relaxed)
            + bytes;
        self.peak_in_flight_memory_bytes
            .fetch_max(total, Ordering::Relaxed);
    }

    fn release_in_flight(&self, bytes: usize) {
        self.in_flight_memory_bytes
            .fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Bytes held by the index.
    pub fn memory_usage_bytes(&self) -> usize {
        self.used_memory_bytes.load(Ordering::Relaxed)
    }

    /// Bytes materialized in memory right now but not yet in the index.
    ///
    /// Read alongside [`Self::memory_usage_bytes`]: that one reports the end
    /// state of the hydrate -> insert -> squeeze cycle, this one reports what
    /// the cycle is holding on the way there.
    pub fn in_flight_memory_bytes(&self) -> usize {
        self.in_flight_memory_bytes.load(Ordering::Relaxed)
    }

    /// High water mark of [`Self::in_flight_memory_bytes`].
    ///
    /// Transients live and die between two scrapes of a gauge, so this is the
    /// number to look at when a process holds more than its tier reports.
    pub fn peak_in_flight_memory_bytes(&self) -> usize {
        self.peak_in_flight_memory_bytes.load(Ordering::Relaxed)
    }

    pub fn disk_usage_bytes(&self) -> usize {
        self.used_disk_bytes.load(Ordering::Relaxed)
    }

    pub(super) fn try_reserve_disk(&self, request_bytes: usize) -> Result<(), ()> {
        let used = self.used_disk_bytes.load(Ordering::Relaxed);
        if used + request_bytes > self.max_disk_bytes {
            self.observer.on_disk_reservation_failure();
            return Err(());
        }

        match self.used_disk_bytes.compare_exchange(
            used,
            used + request_bytes,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(_) => self.try_reserve_disk(request_bytes),
        }
    }

    pub(super) fn release_disk(&self, bytes: usize) {
        self.used_disk_bytes.fetch_sub(bytes, Ordering::Relaxed);
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
    fn in_flight_reservations_are_reported_and_released() {
        let budget = test_budget(1000, usize::MAX);

        let small = budget.reserve_in_flight(100);
        let mut large = budget.reserve_in_flight(300);
        assert_eq!(budget.in_flight_memory_bytes(), 400);

        // A reservation taken as an upper bound is trued up once the real size
        // is known, and the peak remembers the bound that was held.
        large.resize(50);
        assert_eq!(budget.in_flight_memory_bytes(), 150);
        assert_eq!(budget.peak_in_flight_memory_bytes(), 400);

        // In-flight bytes are reported, not charged: they say what the cache is
        // holding outside the index, and admission does not consult them.
        assert!(budget.try_reserve_memory(1000).is_ok());

        drop(large);
        small.release();
        assert_eq!(budget.in_flight_memory_bytes(), 0);
        assert_eq!(
            budget.peak_in_flight_memory_bytes(),
            400,
            "the high water mark outlives the reservations that set it"
        );
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
}
