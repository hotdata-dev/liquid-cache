//! Test-only counting allocator for the `audit_*` size tests.
//!
//! Two views of the same allocations:
//!
//! - **per thread** (`thread_live`, `thread_peak`): bytes allocated minus bytes
//!   freed *by the calling thread*. Exact for work that stays on one thread,
//!   and immune to other tests running in parallel. Wrong for memory that one
//!   thread allocates and another frees.
//! - **process wide** (`live`, `peak`): only meaningful with
//!   `--test-threads=1`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};

struct Counting;

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    static T_LIVE: Cell<isize> = const { Cell::new(0) };
    static T_PEAK: Cell<isize> = const { Cell::new(0) };
}

fn add(n: usize) {
    let n = n as isize;
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
    let _ = T_LIVE.try_with(|live| {
        let now = live.get() + n;
        live.set(now);
        let _ = T_PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

fn sub(n: usize) {
    let n = n as isize;
    LIVE.fetch_sub(n, Relaxed);
    let _ = T_LIVE.try_with(|live| live.set(live.get() - n));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                add(new_size - layout.size());
            } else {
                sub(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes this thread has allocated and not freed.
pub(crate) fn thread_live() -> isize {
    T_LIVE.with(Cell::get)
}

/// Start a new peak measurement for this thread.
pub(crate) fn thread_reset_peak() {
    T_PEAK.with(|peak| peak.set(thread_live()));
}

/// The highest `thread_live()` since the last `thread_reset_peak()`.
pub(crate) fn thread_peak() -> isize {
    T_PEAK.with(Cell::get)
}

/// Bytes live in the whole process.
pub(crate) fn live() -> isize {
    LIVE.load(Relaxed)
}

/// Start a new process-wide peak measurement.
pub(crate) fn reset_peak() {
    PEAK.store(LIVE.load(Relaxed), Relaxed);
}

/// The highest `live()` since the last `reset_peak()`.
pub(crate) fn peak() -> isize {
    PEAK.load(Relaxed)
}
