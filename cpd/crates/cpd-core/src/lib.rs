// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod constraints;
pub mod error;
pub mod missing;
pub mod time_series;

pub use constraints::{
    CachePolicy, Constraints, DegradationStep, ValidatedConstraints, canonicalize_candidates,
    validate_constraints,
};
pub use error::CpdError;
pub use missing::{
    MissingRunStats, MissingSupport, build_missing_mask, check_missing_compatibility,
    compute_missing_run_stats, scan_nans,
};
pub use time_series::{DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

/// Core shared types and traits for cpd-rs.
pub fn crate_name() -> &'static str {
    "cpd-core"
}
