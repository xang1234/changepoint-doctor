// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod error;

pub use error::CpdError;

/// Core shared types and traits for cpd-rs.
pub fn crate_name() -> &'static str {
    "cpd-core"
}
