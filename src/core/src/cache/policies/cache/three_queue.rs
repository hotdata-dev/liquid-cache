use std::{
    collections::{HashMap, HashSet},
    ptr::NonNull,
};

use crate::{
    cache::{CachePolicy, EntryID, cached_batch::CachedBatchType},
    sync::Mutex,
};

use super::doubly_linked_list::{DoublyLinkedList, DoublyLinkedNode, drop_boxed_node};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueKind {
    Arrow,
    Liquid,
    Squeezed,
    Disk,
}

#[derive(Debug)]
struct QueueNode {
    entry_id: EntryID,
    queue: QueueKind,
    /// Second-chance (CLOCK) reference bit. Set when the entry is accessed
    /// (a cache hit); cleared when it is passed over during victim search.
    /// A fresh insert starts cold (`false`), so a batch read exactly once —
    /// e.g. every batch of a large one-pass scan — is evicted before any
    /// batch that has been reused, keeping the working set scan-resistant.
    referenced: bool,
}

type NodePtr = NonNull<DoublyLinkedNode<QueueNode>>;

#[derive(Default, Debug)]
struct LiquidQueueInternalState {
    map: HashMap<EntryID, NodePtr>,
    arrow: DoublyLinkedList<QueueNode>,
    liquid: DoublyLinkedList<QueueNode>,
    squeezed: DoublyLinkedList<QueueNode>,
    disk: DoublyLinkedList<QueueNode>,
}

impl LiquidQueueInternalState {
    unsafe fn list_mut(&mut self, queue: QueueKind) -> &mut DoublyLinkedList<QueueNode> {
        match queue {
            QueueKind::Arrow => &mut self.arrow,
            QueueKind::Liquid => &mut self.liquid,
            QueueKind::Squeezed => &mut self.squeezed,
            QueueKind::Disk => &mut self.disk,
        }
    }

    unsafe fn push_back(&mut self, queue: QueueKind, mut node_ptr: NodePtr) {
        unsafe {
            node_ptr.as_mut().data.queue = queue;
            self.list_mut(queue).push_back(node_ptr);
        }
    }

    unsafe fn detach(&mut self, node_ptr: NodePtr) {
        unsafe {
            let queue = node_ptr.as_ref().data.queue;
            self.list_mut(queue).unlink(node_ptr);
        }
    }

    fn upsert_into_queue(&mut self, entry_id: EntryID, target: QueueKind) {
        if let Some(node_ptr) = self.map.get(&entry_id).copied() {
            unsafe {
                self.detach(node_ptr);
                self.push_back(target, node_ptr);
            }
            return;
        }

        let node = DoublyLinkedNode::new(QueueNode {
            entry_id,
            queue: target,
            referenced: false,
        });
        let node_ptr = NonNull::from(Box::leak(node));

        self.map.insert(entry_id, node_ptr);
        unsafe {
            self.push_back(target, node_ptr);
        }
    }

    fn pop_front(&mut self, queue: QueueKind) -> Option<EntryID> {
        let list = match queue {
            QueueKind::Arrow => &mut self.arrow,
            QueueKind::Liquid => &mut self.liquid,
            QueueKind::Squeezed => &mut self.squeezed,
            QueueKind::Disk => &mut self.disk,
        };

        let head_ptr = list.head()?;
        let entry_id = unsafe { head_ptr.as_ref().data.entry_id };
        let node_ptr = self
            .map
            .remove(&entry_id)
            .expect("list head must exist in map");
        unsafe {
            list.unlink(node_ptr);
            drop_boxed_node(node_ptr);
        }
        Some(entry_id)
    }

    fn remove(&mut self, entry_id: &EntryID) -> Option<EntryID> {
        let node_ptr = self.map.remove(entry_id)?;
        let removed = unsafe { node_ptr.as_ref().data.entry_id };
        unsafe {
            self.detach(node_ptr);
            drop_boxed_node(node_ptr);
        }
        Some(removed)
    }

    /// Entry id and reference bit of the queue's front node, if any.
    fn head_info(&self, queue: QueueKind) -> Option<(EntryID, bool)> {
        let list = match queue {
            QueueKind::Arrow => &self.arrow,
            QueueKind::Liquid => &self.liquid,
            QueueKind::Squeezed => &self.squeezed,
            QueueKind::Disk => &self.disk,
        };
        let head_ptr = list.head()?;
        let data = unsafe { &head_ptr.as_ref().data };
        Some((data.entry_id, data.referenced))
    }

    /// Set the reference bit of an entry, if present.
    fn set_referenced(&mut self, entry_id: &EntryID, value: bool) {
        if let Some(node_ptr) = self.map.get(entry_id).copied() {
            unsafe {
                (*node_ptr.as_ptr()).data.referenced = value;
            }
        }
    }

    /// Give the queue's front node a second chance: clear its reference bit and
    /// move it to the back. Caller has already confirmed the queue is non-empty.
    fn second_chance_head(&mut self, queue: QueueKind) {
        let head_ptr = match queue {
            QueueKind::Arrow => self.arrow.head(),
            QueueKind::Liquid => self.liquid.head(),
            QueueKind::Squeezed => self.squeezed.head(),
            QueueKind::Disk => self.disk.head(),
        };
        if let Some(head_ptr) = head_ptr {
            unsafe {
                (*head_ptr.as_ptr()).data.referenced = false;
                self.detach(head_ptr);
                self.push_back(queue, head_ptr);
            }
        }
    }
}

impl Drop for LiquidQueueInternalState {
    fn drop(&mut self) {
        let nodes: Vec<_> = self.map.drain().map(|(_, ptr)| ptr).collect();
        for node_ptr in nodes {
            unsafe {
                match node_ptr.as_ref().data.queue {
                    QueueKind::Arrow => self.arrow.unlink(node_ptr),
                    QueueKind::Liquid => self.liquid.unlink(node_ptr),
                    QueueKind::Squeezed => self.squeezed.unlink(node_ptr),
                    QueueKind::Disk => self.disk.unlink(node_ptr),
                }
                drop_boxed_node(node_ptr);
            }
        }

        unsafe {
            self.arrow.drop_all();
            self.liquid.drop_all();
            self.squeezed.drop_all();
            self.disk.drop_all();
        }
    }
}

/// Cache policy that keeps independent FIFO queues per batch type.
#[derive(Debug, Default)]
pub struct LiquidPolicy {
    inner: Mutex<LiquidQueueInternalState>,
}

impl LiquidPolicy {
    /// Create a new [`LiquidPolicy`].
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LiquidQueueInternalState::default()),
        }
    }
}

// SAFETY: Access to raw pointers is protected by the internal `Mutex`.
unsafe impl Send for LiquidPolicy {}
unsafe impl Sync for LiquidPolicy {}

impl CachePolicy for LiquidPolicy {
    fn notify_insert(&self, entry_id: &EntryID, batch_type: CachedBatchType) {
        let mut inner = self.inner.lock().unwrap();
        let target = match batch_type {
            CachedBatchType::MemoryArrow => QueueKind::Arrow,
            CachedBatchType::MemoryLiquid => QueueKind::Liquid,
            CachedBatchType::MemorySqueezedLiquid => QueueKind::Squeezed,
            CachedBatchType::DiskLiquid | CachedBatchType::DiskArrow => QueueKind::Disk,
        };

        inner.upsert_into_queue(*entry_id, target);
    }

    fn find_memory_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }

        let mut inner = self.inner.lock().unwrap();
        let mut victims = Vec::with_capacity(cnt);

        // Evict decoded (Arrow) first, then Liquid, then Squeezed — cheapest to
        // reconstruct first. Within each queue apply CLOCK/second-chance: a
        // referenced entry is passed over once (bit cleared, moved to the back)
        // before it can be evicted, so a reused batch outlives a read-once one.
        // `chanced` bounds each entry to a single second chance, so the loop
        // always terminates (at most one pass of reprieves, then eviction).
        for queue in [QueueKind::Arrow, QueueKind::Liquid, QueueKind::Squeezed] {
            let mut chanced: HashSet<EntryID> = HashSet::new();
            while victims.len() < cnt {
                let Some((entry_id, referenced)) = inner.head_info(queue) else {
                    break;
                };
                if referenced && chanced.insert(entry_id) {
                    inner.second_chance_head(queue);
                } else {
                    let popped = inner.pop_front(queue);
                    debug_assert_eq!(popped, Some(entry_id));
                    victims.push(entry_id);
                }
            }
            if victims.len() >= cnt {
                break;
            }
        }

        victims
    }

    fn find_disk_victim(&self, cnt: usize) -> Vec<EntryID> {
        if cnt == 0 {
            return vec![];
        }

        let mut inner = self.inner.lock().unwrap();
        let mut victims = Vec::with_capacity(cnt);

        while victims.len() < cnt {
            let Some(entry) = inner.pop_front(QueueKind::Disk) else {
                break;
            };
            victims.push(entry);
        }

        victims
    }

    fn notify_access(&self, entry_id: &EntryID, _batch_type: CachedBatchType) {
        // A cache hit marks the entry as reused, protecting it from the next
        // victim search (see `find_memory_victim`).
        let mut inner = self.inner.lock().unwrap();
        inner.set_referenced(entry_id, true);
    }

    fn notify_remove(&self, entry_id: &EntryID) {
        let mut inner = self.inner.lock().unwrap();
        inner.remove(entry_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::utils::EntryID;

    fn entry(id: usize) -> EntryID {
        id.into()
    }

    #[test]
    fn test_fifo_within_each_queue() {
        let policy = LiquidPolicy::new();

        let arrow_a = entry(1);
        let arrow_b = entry(2);
        let liquid_a = entry(3);
        let liquid_b = entry(4);

        policy.notify_insert(&arrow_a, CachedBatchType::MemoryArrow);
        policy.notify_insert(&arrow_b, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_a, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&liquid_b, CachedBatchType::MemoryLiquid);

        assert_eq!(policy.find_memory_victim(1), vec![arrow_a]);
        assert_eq!(policy.find_memory_victim(2), vec![arrow_b, liquid_a]);
        assert_eq!(policy.find_memory_victim(1), vec![liquid_b]);
    }

    #[test]
    fn test_accessed_entry_gets_second_chance() {
        let policy = LiquidPolicy::new();
        let a = entry(1);
        let b = entry(2);
        let c = entry(3);
        policy.notify_insert(&a, CachedBatchType::MemoryArrow);
        policy.notify_insert(&b, CachedBatchType::MemoryArrow);
        policy.notify_insert(&c, CachedBatchType::MemoryArrow);

        // `a` is reused, so it should be spared on the next victim search even
        // though it was inserted first (plain FIFO would evict it).
        policy.notify_access(&a, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_memory_victim(1), vec![b]);
        assert_eq!(policy.find_memory_victim(1), vec![c]);
        // `a` survived the longest and is evicted last (its bit was cleared when
        // it was passed over, so it gets no further reprieve).
        assert_eq!(policy.find_memory_victim(1), vec![a]);
    }

    #[test]
    fn test_read_once_entries_evicted_before_reused() {
        // Simulates a large one-pass scan (read-once batches) alongside a single
        // reused batch: the scan batches must all be evicted before the reused
        // one, so the scan cannot displace the working set.
        let policy = LiquidPolicy::new();
        let ids: Vec<_> = (1..=5).map(entry).collect();
        for id in &ids {
            policy.notify_insert(id, CachedBatchType::MemoryArrow);
        }
        // ids[2] (entry 3) is reused.
        policy.notify_access(&ids[2], CachedBatchType::MemoryArrow);

        let victims = policy.find_memory_victim(4);
        assert_eq!(victims, vec![ids[0], ids[1], ids[3], ids[4]]);
        // The reused entry is the sole survivor.
        assert_eq!(policy.find_memory_victim(4), vec![ids[2]]);
    }

    #[test]
    fn test_queue_priority_order() {
        let policy = LiquidPolicy::new();

        let arrow_entry = entry(1);
        let liquid_entry = entry(2);
        let hybrid_entry = entry(3);

        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);
        policy.notify_insert(&hybrid_entry, CachedBatchType::MemorySqueezedLiquid);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);

        // Request more victims than available to ensure we only get what exists.
        let victims = policy.find_memory_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry, hybrid_entry]);
    }

    #[test]
    fn test_zero_victim_request_returns_empty() {
        let policy = LiquidPolicy::new();

        policy.notify_insert(&entry(1), CachedBatchType::MemoryArrow);
        assert!(policy.find_memory_victim(0).is_empty());
    }

    #[test]
    fn test_disk_entries_not_evicted() {
        let policy = LiquidPolicy::new();

        let disk_entry = entry(1);
        let arrow_entry = entry(2);
        let liquid_entry = entry(3);

        policy.notify_insert(&disk_entry, CachedBatchType::DiskArrow);
        policy.notify_insert(&arrow_entry, CachedBatchType::MemoryArrow);
        policy.notify_insert(&liquid_entry, CachedBatchType::MemoryLiquid);

        let victims = policy.find_memory_victim(5);
        assert_eq!(victims, vec![arrow_entry, liquid_entry]);

        // Only the disk entry remains and should still not be evicted.
        assert!(policy.find_memory_victim(1).is_empty());
    }

    #[test]
    fn test_disk_victims_and_remove() {
        let policy = LiquidPolicy::new();
        let disk_old = entry(1);
        let disk_new = entry(2);

        policy.notify_insert(&disk_old, CachedBatchType::DiskArrow);
        policy.notify_insert(&disk_new, CachedBatchType::DiskLiquid);

        assert_eq!(policy.find_disk_victim(1), vec![disk_old]);
        policy.notify_remove(&disk_new);
        assert!(policy.find_disk_victim(1).is_empty());
    }

    #[test]
    fn test_reinsert_moves_entry_to_back_of_queue() {
        let policy = LiquidPolicy::new();

        let first = entry(1);
        let second = entry(2);

        policy.notify_insert(&first, CachedBatchType::MemoryArrow);
        policy.notify_insert(&second, CachedBatchType::MemoryArrow);

        // Reinserting should refresh the entry as the newest arrow batch.
        policy.notify_insert(&first, CachedBatchType::MemoryArrow);

        assert_eq!(policy.find_memory_victim(1), vec![second]);
        assert_eq!(policy.find_memory_victim(1), vec![first]);
    }

    #[test]
    fn test_reinsert_handles_cross_queue_move() {
        let policy = LiquidPolicy::new();

        let entry_id = entry(42);

        policy.notify_insert(&entry_id, CachedBatchType::MemoryArrow);
        policy.notify_insert(&entry_id, CachedBatchType::MemoryLiquid);

        let victims = policy.find_memory_victim(2);
        assert_eq!(victims, vec![entry_id]);
    }
}
