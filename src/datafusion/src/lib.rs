#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod io;
pub mod optimizers;
mod reader;
mod sync;
pub(crate) mod utils;

#[cfg(test)]
pub(crate) mod test_utils {
    //! Shared helpers for this crate's tests.

    /// Mount a t4 store for a test, using the same I/O mode the cache uses in
    /// production on this platform.
    pub(crate) async fn mount_test_store(dir: &std::path::Path) -> t4::Store {
        liquid_cache::store::mount(dir.join("liquid_cache.t4"))
            .await
            .expect("mount t4 test store")
    }
}

pub mod cache;
pub use cache::{LiquidCacheParquet, LiquidCacheParquetRef};
pub use liquid_cache as storage;
pub use liquid_cache_common as common;
pub use reader::variant_udf::{VariantGetUdf, VariantPretty, VariantToJsonUdf};
pub use reader::{FilterCandidateBuilder, LiquidParquetSource, LiquidPredicate, LiquidRowFilter};
pub use utils::{boolean_buffer_and_then, extract_execution_metrics};
