// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Structured error type for cpd-rs core APIs.
///
/// # Error Philosophy
/// - Error messages are operational and actionable.
/// - Variants are structured for reliable pattern matching.
/// - Expected failures are represented as `CpdError` (not panics).
///
/// # Python Exception Mapping
/// - `InvalidInput` -> `ValueError`
/// - `NumericalIssue` -> `FloatingPointError`
/// - `NotSupported` -> `NotImplementedError`
/// - `ResourceLimit` -> `RuntimeError`
/// - `Cancelled` -> `RuntimeError`
#[derive(thiserror::Error, Debug)]
pub enum CpdError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("numerical issue: {0}")]
    NumericalIssue(String),
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("cancelled")]
    Cancelled,
}

impl CpdError {
    /// Creates a `CpdError::InvalidInput`.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Creates a `CpdError::NumericalIssue`.
    pub fn numerical_issue(msg: impl Into<String>) -> Self {
        Self::NumericalIssue(msg.into())
    }

    /// Creates a `CpdError::NotSupported`.
    pub fn not_supported(msg: impl Into<String>) -> Self {
        Self::NotSupported(msg.into())
    }

    /// Creates a `CpdError::ResourceLimit`.
    pub fn resource_limit(msg: impl Into<String>) -> Self {
        Self::ResourceLimit(msg.into())
    }

    /// Creates a `CpdError::Cancelled`.
    pub fn cancelled() -> Self {
        Self::Cancelled
    }
}

#[cfg(test)]
mod tests {
    use super::CpdError;

    #[test]
    fn helper_constructors_create_expected_variants() {
        match CpdError::invalid_input("series length 0; minimum is 1") {
            CpdError::InvalidInput(msg) => assert_eq!(msg, "series length 0; minimum is 1"),
            _ => panic!("expected InvalidInput"),
        }

        match CpdError::numerical_issue("variance underflow in segment [10, 12)") {
            CpdError::NumericalIssue(msg) => {
                assert_eq!(msg, "variance underflow in segment [10, 12)")
            }
            _ => panic!("expected NumericalIssue"),
        }

        match CpdError::not_supported("f32 + gp feature not enabled") {
            CpdError::NotSupported(msg) => assert_eq!(msg, "f32 + gp feature not enabled"),
            _ => panic!("expected NotSupported"),
        }

        match CpdError::resource_limit("time budget 100ms exceeded at 127ms") {
            CpdError::ResourceLimit(msg) => {
                assert_eq!(msg, "time budget 100ms exceeded at 127ms")
            }
            _ => panic!("expected ResourceLimit"),
        }

        match CpdError::cancelled() {
            CpdError::Cancelled => {}
            _ => panic!("expected Cancelled"),
        }
    }

    #[test]
    fn display_messages_have_required_prefixes() {
        assert!(
            CpdError::invalid_input("series length 0; minimum is 1")
                .to_string()
                .starts_with("invalid input:")
        );
        assert!(
            CpdError::numerical_issue("variance underflow in segment [10, 12)")
                .to_string()
                .starts_with("numerical issue:")
        );
        assert!(
            CpdError::not_supported("f32 + gp feature not enabled")
                .to_string()
                .starts_with("not supported:")
        );
        assert!(
            CpdError::resource_limit("time budget 100ms exceeded at 127ms")
                .to_string()
                .starts_with("resource limit exceeded:")
        );
        assert_eq!(CpdError::cancelled().to_string(), "cancelled");
    }

    #[test]
    fn cpd_error_is_usable_as_std_error_trait_object() {
        let err: Box<dyn std::error::Error> = Box::new(CpdError::cancelled());
        assert_eq!(err.to_string(), "cancelled");
    }
}
