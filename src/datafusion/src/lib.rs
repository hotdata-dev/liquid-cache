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

    /// Mount a t4 store for a test.
    ///
    /// t4's default [`t4::MountOptions`] enable `direct_io`, which only Linux
    /// supports; everywhere else the mount fails outright with
    /// `direct_io not supported on target_os`. Keep it on Linux (production, CI)
    /// and fall back to buffered I/O elsewhere, matching what
    /// `LiquidCacheLocalBuilder` does, so these tests run on macOS dev machines
    /// too.
    pub(crate) async fn mount_test_store(dir: &std::path::Path) -> t4::Store {
        t4::mount_with_options(
            dir.join("liquid_cache.t4"),
            t4::MountOptions {
                direct_io: cfg!(target_os = "linux"),
                ..Default::default()
            },
        )
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
