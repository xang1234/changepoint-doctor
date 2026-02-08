// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;
use crate::time_series::{DTypeView, MissingPolicy};

/// Declares how a cost model handles missing values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingSupport {
    /// Cost model cannot handle missing values.
    Reject,
    /// Cost model can correctly skip missing values using a mask.
    MaskAware,
    /// Cost model ignores NaNs but may lose statistical power.
    NaNIgnoreLossy,
}

impl MissingSupport {
    /// Stable, user-facing label for diagnostics and error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reject => "Reject",
            Self::MaskAware => "MaskAware",
            Self::NaNIgnoreLossy => "NaNIgnoreLossy",
        }
    }
}

/// Lightweight missing-data diagnostics produced during execution.
#[derive(Clone, Debug, PartialEq)]
pub struct MissingRunStats {
    pub missing_policy_applied: &'static str,
    pub missing_fraction: f64,
    pub effective_sample_count: usize,
}

/// Returns the stable string representation of a missing-value policy.
pub const fn missing_policy_name(policy: MissingPolicy) -> &'static str {
    match policy {
        MissingPolicy::Error => "Error",
        MissingPolicy::ImputeZero => "ImputeZero",
        MissingPolicy::ImputeLast => "ImputeLast",
        MissingPolicy::Ignore => "Ignore",
    }
}

/// Validates that the requested missing policy is supported by a cost model.
pub fn check_missing_compatibility(
    policy: MissingPolicy,
    support: MissingSupport,
) -> Result<(), CpdError> {
    if matches!(policy, MissingPolicy::Ignore) && !matches!(support, MissingSupport::MaskAware) {
        return Err(CpdError::invalid_input(format!(
            "incompatible missing handling: policy={} requires support=MaskAware, got support={}",
            missing_policy_name(policy),
            support.as_str()
        )));
    }
    Ok(())
}

/// Scans a numeric buffer for NaNs and returns `(count, flattened_positions)`.
pub fn scan_nans(data: DTypeView<'_>) -> (usize, Vec<usize>) {
    let positions = match data {
        DTypeView::F32(values) => values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| value.is_nan().then_some(idx))
            .collect::<Vec<_>>(),
        DTypeView::F64(values) => values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| value.is_nan().then_some(idx))
            .collect::<Vec<_>>(),
    };
    (positions.len(), positions)
}

/// Builds a 0/1 missing mask from NaN values in the provided numeric buffer.
pub fn build_missing_mask(data: DTypeView<'_>) -> Vec<u8> {
    match data {
        DTypeView::F32(values) => values
            .iter()
            .map(|value| u8::from(value.is_nan()))
            .collect::<Vec<_>>(),
        DTypeView::F64(values) => values
            .iter()
            .map(|value| u8::from(value.is_nan()))
            .collect::<Vec<_>>(),
    }
}

/// Computes missing-data run statistics for diagnostics wiring.
pub fn compute_missing_run_stats(
    total_values: usize,
    missing_count: usize,
    policy: MissingPolicy,
) -> MissingRunStats {
    let bounded_missing = missing_count.min(total_values);
    let missing_fraction = if total_values == 0 {
        0.0
    } else {
        bounded_missing as f64 / total_values as f64
    };

    MissingRunStats {
        missing_policy_applied: missing_policy_name(policy),
        missing_fraction,
        effective_sample_count: total_values.saturating_sub(bounded_missing),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MissingSupport, build_missing_mask, check_missing_compatibility, compute_missing_run_stats,
        missing_policy_name, scan_nans,
    };
    use crate::time_series::{DTypeView, MissingPolicy};

    #[test]
    fn compatibility_matrix_matches_spec_for_all_policy_support_pairs() {
        let policies = [
            MissingPolicy::Error,
            MissingPolicy::ImputeZero,
            MissingPolicy::ImputeLast,
            MissingPolicy::Ignore,
        ];
        let supports = [
            MissingSupport::Reject,
            MissingSupport::MaskAware,
            MissingSupport::NaNIgnoreLossy,
        ];

        for policy in policies {
            for support in supports {
                let got_ok = check_missing_compatibility(policy, support).is_ok();
                let expected_ok = !matches!(policy, MissingPolicy::Ignore)
                    || matches!(support, MissingSupport::MaskAware);
                assert_eq!(
                    got_ok,
                    expected_ok,
                    "unexpected compatibility result for policy={} support={}",
                    missing_policy_name(policy),
                    support.as_str()
                );
            }
        }
    }

    #[test]
    fn incompatible_ignore_errors_name_policy_and_support() {
        let err = check_missing_compatibility(MissingPolicy::Ignore, MissingSupport::Reject)
            .expect_err("Ignore+Reject must be incompatible");
        let msg = err.to_string();
        assert!(msg.contains("policy=Ignore"));
        assert!(msg.contains("support=Reject"));

        let err_lossy =
            check_missing_compatibility(MissingPolicy::Ignore, MissingSupport::NaNIgnoreLossy)
                .expect_err("Ignore+NaNIgnoreLossy must be incompatible");
        let msg_lossy = err_lossy.to_string();
        assert!(msg_lossy.contains("policy=Ignore"));
        assert!(msg_lossy.contains("support=NaNIgnoreLossy"));
    }

    #[test]
    fn scan_nans_handles_f32_and_f64_buffers() {
        let f32_values = [1.0_f32, f32::NAN, 3.0_f32, f32::NAN];
        let (f32_count, f32_positions) = scan_nans(DTypeView::F32(&f32_values));
        assert_eq!(f32_count, 2);
        assert_eq!(f32_positions, vec![1, 3]);

        let f64_values = [f64::NAN, 2.0_f64, 3.0_f64, f64::NAN, f64::NAN];
        let (f64_count, f64_positions) = scan_nans(DTypeView::F64(&f64_values));
        assert_eq!(f64_count, 3);
        assert_eq!(f64_positions, vec![0, 3, 4]);
    }

    #[test]
    fn scan_nans_ignores_infinities_and_returns_empty_when_no_nans() {
        let values = [1.0_f64, f64::INFINITY, f64::NEG_INFINITY, -2.0_f64];
        let (count, positions) = scan_nans(DTypeView::F64(&values));
        assert_eq!(count, 0);
        assert!(positions.is_empty());
    }

    #[test]
    fn build_missing_mask_marks_only_nans() {
        let values = [
            1.0_f64,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mask = build_missing_mask(DTypeView::F64(&values));
        assert_eq!(mask, vec![0, 1, 0, 0, 1]);
    }

    #[test]
    fn compute_missing_run_stats_uses_fraction_and_effective_sample_count() {
        let stats = compute_missing_run_stats(8, 3, MissingPolicy::ImputeLast);
        assert_eq!(stats.missing_policy_applied, "ImputeLast");
        assert!((stats.missing_fraction - 0.375).abs() < f64::EPSILON);
        assert_eq!(stats.effective_sample_count, 5);
    }

    #[test]
    fn compute_missing_run_stats_handles_zero_total_and_clamps_missing() {
        let zero = compute_missing_run_stats(0, 5, MissingPolicy::Error);
        assert_eq!(zero.missing_fraction, 0.0);
        assert_eq!(zero.effective_sample_count, 0);

        let clamped = compute_missing_run_stats(3, 10, MissingPolicy::Ignore);
        assert_eq!(clamped.missing_fraction, 1.0);
        assert_eq!(clamped.effective_sample_count, 0);
    }
}
