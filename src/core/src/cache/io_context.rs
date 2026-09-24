use std::fmt::Debug;

use ahash::AHashMap;

use crate::cache::{CacheExpression, utils::EntryID};
use crate::sync::{Arc, RwLock};

/// Per-entry metadata used by the cache.
///
/// This trait covers only the metadata side of the cache: where to find a
/// batch's compressor and lineage expressions. All actual byte IO goes through the
/// [`t4::Store`] held by the cache itself.
pub trait EntryMetadata: Debug + Send + Sync {
    /// Add a lineage expression for an entry.
    fn add_lineage(&self, _entry_id: &EntryID, _expression: Arc<CacheExpression>) {
        // Do nothing by default
    }

    /// Get the lineage expression for an entry.
    /// If None, the entry will be evicted to disk entirely.
    /// The expression records how the column is used by query plans and may inform
    /// encoding decisions without discarding any values.
    fn lineage(&self, _entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        None
    }
}

/// Convert an [`EntryID`] to a t4 key (8-byte little-endian representation).
/// Convert an [`EntryID`] and the identity that owns it to a t4 key.
///
/// Both halves, not just the entry id. `EntryID` is a packed integer whose
/// fields are narrower than the values they encode, so two sources can compute
/// one id — and a store object addressed by that id alone is *shared*. Scoping
/// only the index entry is not enough: a write in flight for one owner can land
/// after another has taken the key over and installed its own disk entry,
/// overwriting bytes the new owner's index entry agrees are its own. With the
/// identity in the key the two address different objects, so a late write
/// cannot reach the other's bytes at all.
pub(crate) fn entry_id_to_key(entry_id: &EntryID, identity: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&usize::from(*entry_id).to_le_bytes());
    key.extend_from_slice(&identity.to_le_bytes());
    key
}

/// A default implementation of [`EntryMetadata`].
///
/// Lineage expressions are stored in a flat map keyed by [`EntryID`].
#[derive(Debug, Default)]
pub struct DefaultCacheMetadata {
    lineages: RwLock<AHashMap<EntryID, Arc<CacheExpression>>>,
}

impl DefaultCacheMetadata {
    /// Create a new instance of [`DefaultCacheMetadata`].
    pub fn new() -> Self {
        Self {
            lineages: RwLock::new(AHashMap::new()),
        }
    }
}

impl EntryMetadata for DefaultCacheMetadata {
    fn add_lineage(&self, entry_id: &EntryID, expression: Arc<CacheExpression>) {
        let mut guard = self.lineages.write().unwrap();
        guard.insert(*entry_id, expression);
    }

    fn lineage(&self, entry_id: &EntryID) -> Option<Arc<CacheExpression>> {
        let guard = self.lineages.read().unwrap();
        guard.get(entry_id).cloned()
    }
}
