//! Test-only allocator that tracks live heap bytes per thread, so a test can
//! see memory a call leaves behind that no `nbytes()`-style accounting reports.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn add(delta: isize) {
    // `try_with`: TLS may already be torn down while a thread exits.
    let _ = LIVE.try_with(|live| live.set(live.get() + delta));
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        add(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            add(new_size as isize - layout.size() as isize);
        }
        new
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// Heap bytes allocated and not yet freed by the current thread.
pub(crate) fn live_bytes() -> isize {
    LIVE.with(Cell::get)
}
