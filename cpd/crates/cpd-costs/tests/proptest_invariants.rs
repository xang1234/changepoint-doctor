// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, MemoryLayout, MissingPolicy, ReproMode, TimeIndex, TimeSeriesView};
use cpd_costs::{CostL2Mean, CostModel, CostNormalMeanVar};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, FileFailurePersistence};

const ABS_TOL: f64 = 1e-7;
const REL_TOL: f64 = 1e-6;
const VAR_FLOOR: f64 = f64::EPSILON * 1e6;

fn make_univariate_view(values: &[f64]) -> TimeSeriesView<'_> {
    TimeSeriesView::from_f64(
        values,
        values.len(),
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("generated test data should always form a valid TimeSeriesView")
}

fn relative_close(actual: f64, expected: f64) -> bool {
    let diff = (actual - expected).abs();
    let scale = 1.0 + expected.abs();
    diff <= ABS_TOL || diff <= REL_TOL * scale
}

fn naive_l2(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let len = segment.len() as f64;
    let sum: f64 = segment.iter().sum();
    let mean = sum / len;
    segment
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum()
}

fn normalize_variance(raw_var: f64) -> f64 {
    if raw_var.is_nan() || raw_var <= VAR_FLOOR {
        VAR_FLOOR
    } else if raw_var == f64::INFINITY {
        f64::MAX
    } else {
        raw_var
    }
}

fn naive_normal(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let len = segment.len() as f64;
    let sum: f64 = segment.iter().sum();
    let sum_sq: f64 = segment.iter().map(|value| value * value).sum();
    let mean = sum / len;
    let raw_var = sum_sq / len - mean * mean;
    let var = normalize_variance(raw_var);
    len * var.ln()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 1024,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct("proptest-regressions/tests/proptest_invariants.txt"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn l2_segment_cost_is_non_negative_deterministic_and_matches_naive(
        values in prop::collection::vec(-1_000.0f64..1_000.0, 8..96),
        start in 0usize..96,
        len in 1usize..48,
    ) {
        let n = values.len();
        prop_assume!(start < n);
        let end = start.saturating_add(len).min(n);
        prop_assume!(start < end);

        let view = make_univariate_view(&values);
        let model = CostL2Mean::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_l2(&values, start, end);

        prop_assert!(actual_first >= -ABS_TOL);
        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn normal_segment_cost_is_deterministic_and_matches_naive(
        values in prop::collection::vec(-1_000.0f64..1_000.0, 8..96),
        start in 0usize..96,
        len in 2usize..48,
    ) {
        let n = values.len();
        prop_assume!(start < n);
        let end = start.saturating_add(len).min(n);
        prop_assume!(start + 1 < end);

        let view = make_univariate_view(&values);
        let model = CostNormalMeanVar::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_normal(&values, start, end);

        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn l2_segment_cost_is_shift_invariant(
        values in prop::collection::vec(-1_000.0f64..1_000.0, 8..96),
        shift in -500.0f64..500.0,
        start in 0usize..96,
        len in 1usize..48,
    ) {
        let n = values.len();
        prop_assume!(start < n);
        let end = start.saturating_add(len).min(n);
        prop_assume!(start < end);

        let shifted: Vec<f64> = values.iter().map(|value| value + shift).collect();

        let base_view = make_univariate_view(&values);
        let shifted_view = make_univariate_view(&shifted);

        let model = CostL2Mean::new(ReproMode::Balanced);
        let base_cache = model
            .precompute(&base_view, &CachePolicy::Full)
            .expect("base precompute should succeed");
        let shifted_cache = model
            .precompute(&shifted_view, &CachePolicy::Full)
            .expect("shifted precompute should succeed");

        let base_cost = model.segment_cost(&base_cache, start, end);
        let shifted_cost = model.segment_cost(&shifted_cache, start, end);

        prop_assert!(relative_close(base_cost, shifted_cost));
    }
}
