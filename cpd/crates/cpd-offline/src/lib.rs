// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod pelt;

pub use pelt::{Pelt, PeltConfig};

/// Offline detector namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = (cpd_core::crate_name(), cpd_costs::crate_name());
    "cpd-offline"
}
