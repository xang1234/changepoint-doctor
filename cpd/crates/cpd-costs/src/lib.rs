// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod model;

pub use cpd_core::MissingSupport;
pub use model::{CachedCost, CostModel};

/// Built-in cost model namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = cpd_core::crate_name();
    "cpd-costs"
}
