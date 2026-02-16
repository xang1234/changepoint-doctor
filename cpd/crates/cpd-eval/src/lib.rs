// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CpdError, OfflineChangePointResult, segments_from_breakpoints};

/// Precision/recall/F1 summary for tolerance-based matching.
#[derive(Clone, Debug, PartialEq)]
pub struct F1Metrics {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Aggregated offline metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineMetrics {
    pub f1: F1Metrics,
    pub hausdorff_distance: f64,
    pub rand_index: f64,
    pub annotation_error: f64,
}

/// Computes offline evaluation metrics from detected and true segmentation outputs.
///
/// Returns an error when exactly one side has no change points, because
/// Hausdorff distance and annotation error are undefined for that case.
pub fn offline_metrics(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
    tolerance: usize,
) -> Result<OfflineMetrics, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "offline metrics are undefined when exactly one change-point set is empty",
        ));
    }

    Ok(OfflineMetrics {
        f1: f1_with_tolerance(detected, truth, tolerance)?,
        hausdorff_distance: hausdorff_distance(detected, truth)?,
        rand_index: rand_index(detected, truth)?,
        annotation_error: annotation_error(detected, truth)?,
    })
}

/// Computes precision, recall, and F1 using one-to-one tolerance matching.
///
/// A detected change point is considered a true positive when it can be paired
/// to a true change point within `tolerance`.
pub fn f1_with_tolerance(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
    tolerance: usize,
) -> Result<F1Metrics, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    let true_positives = count_tolerance_matches(detected_cp, truth_cp, tolerance);
    let false_positives = detected_cp.len() - true_positives;
    let false_negatives = truth_cp.len() - true_positives;

    if detected_cp.is_empty() && truth_cp.is_empty() {
        return Ok(F1Metrics {
            true_positives,
            false_positives,
            false_negatives,
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }

    let precision = if detected_cp.is_empty() {
        0.0
    } else {
        true_positives as f64 / detected_cp.len() as f64
    };
    let recall = if truth_cp.is_empty() {
        0.0
    } else {
        true_positives as f64 / truth_cp.len() as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(F1Metrics {
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
    })
}

/// Computes the symmetric Hausdorff distance between detected and true
/// change-point sets.
pub fn hausdorff_distance(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    if detected_cp.is_empty() && truth_cp.is_empty() {
        return Ok(0.0);
    }
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "hausdorff distance is undefined when exactly one change-point set is empty",
        ));
    }

    let d_ab = directed_hausdorff(detected_cp, truth_cp);
    let d_ba = directed_hausdorff(truth_cp, detected_cp);
    Ok(d_ab.max(d_ba) as f64)
}

/// Computes the Rand index between detected and true segmentations.
pub fn rand_index(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    let n = validate_pair(detected, truth)?;
    let total_pairs = choose2(n);
    if total_pairs == 0 {
        return Ok(1.0);
    }

    let truth_segments = segments_from_breakpoints(n, truth.breakpoints.as_slice());
    let detected_segments = segments_from_breakpoints(n, detected.breakpoints.as_slice());

    let same_true = truth_segments
        .iter()
        .map(|(start, end)| choose2(end - start))
        .sum::<u128>();
    let same_detected = detected_segments
        .iter()
        .map(|(start, end)| choose2(end - start))
        .sum::<u128>();
    let same_both = overlapping_same_pairs(truth_segments.as_slice(), detected_segments.as_slice());

    let disagreements = same_true + same_detected - 2 * same_both;
    Ok(1.0 - disagreements as f64 / total_pairs as f64)
}

/// Computes mean absolute distance from each detected change point to the
/// nearest true change point.
pub fn annotation_error(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    if detected_cp.is_empty() {
        return Ok(0.0);
    }
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "annotation error is undefined when true change-point set is empty and detected set is non-empty",
        ));
    }

    let total_distance = detected_cp
        .iter()
        .map(|&point| nearest_distance(point, truth_cp) as u128)
        .sum::<u128>();
    Ok(total_distance as f64 / detected_cp.len() as f64)
}

fn validate_pair(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<usize, CpdError> {
    let detected_n = validate_result(detected, "detected")?;
    let truth_n = validate_result(truth, "true")?;
    if detected_n != truth_n {
        return Err(CpdError::invalid_input(format!(
            "detected and true results must share n; got detected_n={detected_n}, true_n={truth_n}"
        )));
    }
    Ok(truth_n)
}

fn validate_result(result: &OfflineChangePointResult, label: &str) -> Result<usize, CpdError> {
    let inferred_n = result
        .breakpoints
        .last()
        .copied()
        .unwrap_or(result.diagnostics.n);
    result.validate(inferred_n).map_err(|err| {
        CpdError::invalid_input(format!(
            "{label} OfflineChangePointResult is invalid: {err}"
        ))
    })?;
    Ok(inferred_n)
}

fn exactly_one_empty(a: &[usize], b: &[usize]) -> bool {
    a.is_empty() != b.is_empty()
}

fn count_tolerance_matches(detected: &[usize], truth: &[usize], tolerance: usize) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut matches = 0usize;

    while i < detected.len() && j < truth.len() {
        let d = detected[i];
        let t = truth[j];
        if d.abs_diff(t) <= tolerance {
            matches += 1;
            i += 1;
            j += 1;
            continue;
        }
        if d < t {
            i += 1;
        } else {
            j += 1;
        }
    }

    matches
}

fn directed_hausdorff(a: &[usize], b: &[usize]) -> usize {
    a.iter()
        .map(|&point| nearest_distance(point, b))
        .max()
        .unwrap_or(0)
}

fn nearest_distance(point: usize, sorted_points: &[usize]) -> usize {
    let insertion = sorted_points.partition_point(|&candidate| candidate < point);
    let mut best = usize::MAX;
    if insertion < sorted_points.len() {
        best = best.min(point.abs_diff(sorted_points[insertion]));
    }
    if insertion > 0 {
        best = best.min(point.abs_diff(sorted_points[insertion - 1]));
    }
    best
}

fn choose2(value: usize) -> u128 {
    if value < 2 {
        0
    } else {
        let value_u128 = value as u128;
        value_u128 * (value_u128 - 1) / 2
    }
}

fn overlapping_same_pairs(truth: &[(usize, usize)], detected: &[(usize, usize)]) -> u128 {
    let mut truth_idx = 0usize;
    let mut detected_idx = 0usize;
    let mut pairs = 0u128;

    while truth_idx < truth.len() && detected_idx < detected.len() {
        let (truth_start, truth_end) = truth[truth_idx];
        let (detected_start, detected_end) = detected[detected_idx];
        let overlap_start = truth_start.max(detected_start);
        let overlap_end = truth_end.min(detected_end);
        if overlap_end > overlap_start {
            pairs += choose2(overlap_end - overlap_start);
        }

        if truth_end <= detected_end {
            truth_idx += 1;
        }
        if detected_end <= truth_end {
            detected_idx += 1;
        }
    }

    pairs
}

/// Evaluation utilities crate name helper.
pub fn crate_name() -> &'static str {
    "cpd-eval"
}

#[cfg(test)]
mod tests {
    use super::{
        annotation_error, f1_with_tolerance, hausdorff_distance, offline_metrics, rand_index,
    };
    use cpd_core::{Diagnostics, OfflineChangePointResult};
    use std::borrow::Cow;

    fn diagnostics_with_n(n: usize) -> Diagnostics {
        Diagnostics {
            n,
            d: 1,
            algorithm: Cow::Borrowed("test"),
            cost_model: Cow::Borrowed("l2"),
            ..Diagnostics::default()
        }
    }

    fn result(n: usize, breakpoints: &[usize]) -> OfflineChangePointResult {
        OfflineChangePointResult::new(n, breakpoints.to_vec(), diagnostics_with_n(n))
            .expect("test result should be valid")
    }

    fn assert_approx_eq(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= 1e-12,
            "expected {expected}, got {actual} (delta={delta})"
        );
    }

    #[test]
    fn f1_with_tolerance_uses_one_to_one_matching() {
        let detected = result(100, &[18, 22, 48, 85, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let metrics = f1_with_tolerance(&detected, &truth, 3).expect("f1 should compute");

        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 2);
        assert_eq!(metrics.false_negatives, 1);
        assert_approx_eq(metrics.precision, 0.5);
        assert_approx_eq(metrics.recall, 2.0 / 3.0);
        assert_approx_eq(metrics.f1, 4.0 / 7.0);
    }

    #[test]
    fn f1_with_tolerance_returns_perfect_score_for_no_change_case() {
        let detected = result(100, &[100]);
        let truth = result(100, &[100]);

        let metrics = f1_with_tolerance(&detected, &truth, 0).expect("f1 should compute");
        assert_eq!(metrics.true_positives, 0);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_approx_eq(metrics.precision, 1.0);
        assert_approx_eq(metrics.recall, 1.0);
        assert_approx_eq(metrics.f1, 1.0);
    }

    #[test]
    fn hausdorff_distance_matches_hand_computed_value() {
        let detected = result(100, &[18, 52, 77, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let distance = hausdorff_distance(&detected, &truth).expect("hausdorff should compute");
        assert_approx_eq(distance, 3.0);
    }

    #[test]
    fn hausdorff_distance_rejects_one_sided_empty_change_sets() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = hausdorff_distance(&detected, &truth)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("hausdorff distance is undefined"));
    }

    #[test]
    fn rand_index_matches_hand_computed_value() {
        let detected = result(8, &[5, 8]);
        let truth = result(8, &[3, 8]);

        let value = rand_index(&detected, &truth).expect("rand index should compute");
        assert_approx_eq(value, 4.0 / 7.0);
    }

    #[test]
    fn rand_index_returns_one_for_single_sample() {
        let detected = result(1, &[1]);
        let truth = result(1, &[1]);

        let value = rand_index(&detected, &truth).expect("rand index should compute");
        assert_approx_eq(value, 1.0);
    }

    #[test]
    fn annotation_error_matches_hand_computed_mean_distance() {
        let detected = result(100, &[18, 52, 77, 95, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let value = annotation_error(&detected, &truth).expect("annotation error should compute");
        assert_approx_eq(value, 5.5);
    }

    #[test]
    fn annotation_error_returns_zero_when_detected_has_no_change_points() {
        let detected = result(100, &[100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let value = annotation_error(&detected, &truth).expect("annotation error should compute");
        assert_approx_eq(value, 0.0);
    }

    #[test]
    fn annotation_error_rejects_empty_true_change_set_with_detections() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = annotation_error(&detected, &truth)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("annotation error is undefined"));
    }

    #[test]
    fn offline_metrics_returns_all_metric_components() {
        let detected = result(8, &[5, 8]);
        let truth = result(8, &[3, 8]);

        let metrics = offline_metrics(&detected, &truth, 2).expect("metrics should compute");
        assert_eq!(metrics.f1.true_positives, 1);
        assert_approx_eq(metrics.hausdorff_distance, 2.0);
        assert_approx_eq(metrics.rand_index, 4.0 / 7.0);
        assert_approx_eq(metrics.annotation_error, 2.0);
    }

    #[test]
    fn offline_metrics_rejects_one_sided_empty_change_sets() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = offline_metrics(&detected, &truth, 3)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("offline metrics are undefined"));
    }

    #[test]
    fn metrics_reject_results_with_mismatched_n() {
        let detected = result(100, &[20, 100]);
        let truth = result(120, &[20, 120]);

        let err = f1_with_tolerance(&detected, &truth, 0).expect_err("mismatched n should fail");
        assert!(err.to_string().contains("must share n"));
    }
}
