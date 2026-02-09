// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, Penalty,
    ReproMode, Stopping, TimeIndex, TimeSeriesView, validate_breakpoints,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{BinSeg, BinSegConfig, Pelt, PeltConfig};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, FileFailurePersistence};

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

fn pelt_l2_breakpoints(
    values: &[f64],
    constraints: &Constraints,
    stopping: Stopping,
) -> Vec<usize> {
    let view = make_univariate_view(values);
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = Pelt::new(
        CostL2Mean::new(ReproMode::Balanced),
        PeltConfig {
            stopping,
            params_per_segment: 2,
            cancel_check_every: 64,
        },
    )
    .expect("detector configuration should be valid");
    detector
        .detect(&view, &ctx)
        .expect("detection should succeed for generated input")
        .breakpoints
}

fn binseg_l2_breakpoints(
    values: &[f64],
    constraints: &Constraints,
    stopping: Stopping,
) -> Vec<usize> {
    let view = make_univariate_view(values);
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = BinSeg::new(
        CostL2Mean::new(ReproMode::Balanced),
        BinSegConfig {
            stopping,
            params_per_segment: 2,
            cancel_check_every: 64,
        },
    )
    .expect("detector configuration should be valid");
    detector
        .detect(&view, &ctx)
        .expect("detection should succeed for generated input")
        .breakpoints
}

fn binseg_normal_breakpoints(
    values: &[f64],
    constraints: &Constraints,
    stopping: Stopping,
) -> Vec<usize> {
    let view = make_univariate_view(values);
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = BinSeg::new(
        CostNormalMeanVar::new(ReproMode::Balanced),
        BinSegConfig {
            stopping,
            params_per_segment: 3,
            cancel_check_every: 64,
        },
    )
    .expect("detector configuration should be valid");
    detector
        .detect(&view, &ctx)
        .expect("detection should succeed for generated input")
        .breakpoints
}

fn assert_breakpoint_invariants(
    breakpoints: &[usize],
    n: usize,
    min_segment_len: usize,
    jump: usize,
    max_change_points: Option<usize>,
) {
    validate_breakpoints(n, breakpoints).expect("breakpoint contract must hold");

    let mut start = 0usize;
    for &end in breakpoints {
        assert!(
            end.saturating_sub(start) >= min_segment_len,
            "segment [{start}, {end}) violates min_segment_len={min_segment_len}"
        );
        start = end;
    }

    for &bp in breakpoints {
        if bp != n {
            assert_eq!(
                bp % jump,
                0,
                "non-terminal breakpoint {bp} must respect jump={jump}"
            );
        }
    }

    if let Some(max_changes) = max_change_points {
        assert!(
            breakpoints.len().saturating_sub(1) <= max_changes,
            "change-point count must not exceed max_change_points"
        );
    }
}

fn three_regime_signal() -> Vec<f64> {
    let mut out = Vec::with_capacity(120);
    out.extend(std::iter::repeat_n(0.0, 40));
    out.extend(std::iter::repeat_n(8.0, 40));
    out.extend(std::iter::repeat_n(-4.0, 40));
    out
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 1024,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct("proptest-regressions/tests/proptest_invariants.txt"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn pelt_and_binseg_outputs_respect_breakpoint_constraints(
        values in prop::collection::vec(-50.0f64..50.0, 32..128),
        min_segment_len in 1usize..8,
        jump in 1usize..6,
        max_change_points in 1usize..10,
    ) {
        let n = values.len();
        prop_assume!(min_segment_len.saturating_mul(2) <= n);

        let constraints = Constraints {
            min_segment_len,
            jump,
            max_change_points: Some(max_change_points),
            ..Constraints::default()
        };
        let stopping = Stopping::Penalized(Penalty::Manual(5.0));

        let pelt_first = pelt_l2_breakpoints(&values, &constraints, stopping.clone());
        let pelt_second = pelt_l2_breakpoints(&values, &constraints, stopping.clone());
        prop_assert_eq!(&pelt_first, &pelt_second);
        assert_breakpoint_invariants(
            &pelt_first,
            n,
            constraints.min_segment_len,
            constraints.jump,
            constraints.max_change_points,
        );

        let binseg_first = binseg_l2_breakpoints(&values, &constraints, stopping.clone());
        let binseg_second = binseg_l2_breakpoints(&values, &constraints, stopping);
        prop_assert_eq!(&binseg_first, &binseg_second);
        assert_breakpoint_invariants(
            &binseg_first,
            n,
            constraints.min_segment_len,
            constraints.jump,
            constraints.max_change_points,
        );
    }

    #[test]
    fn constant_series_with_large_penalty_has_no_spurious_changes(
        value in -20.0f64..20.0,
        n in 16usize..128,
    ) {
        let series = vec![value; n];
        let constraints = Constraints {
            min_segment_len: 2,
            ..Constraints::default()
        };
        let stopping = Stopping::Penalized(Penalty::Manual(1_000_000.0));

        let pelt = pelt_l2_breakpoints(&series, &constraints, stopping.clone());
        let binseg = binseg_l2_breakpoints(&series, &constraints, stopping);

        prop_assert_eq!(pelt, vec![n]);
        prop_assert_eq!(binseg, vec![n]);
    }

    #[test]
    fn binseg_known_k_detection_is_invariant_to_shift_and_scale(
        shift in -100.0f64..100.0,
        scale in 0.2f64..8.0,
    ) {
        let base = three_regime_signal();
        let shifted: Vec<f64> = base.iter().map(|value| value + shift).collect();
        let scaled: Vec<f64> = base.iter().map(|value| value * scale).collect();

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let stopping = Stopping::KnownK(2);

        let binseg_l2_base = binseg_l2_breakpoints(&base, &constraints, stopping.clone());
        let binseg_l2_shifted = binseg_l2_breakpoints(&shifted, &constraints, stopping.clone());
        let binseg_l2_scaled = binseg_l2_breakpoints(&scaled, &constraints, stopping.clone());
        prop_assert_eq!(&binseg_l2_base, &binseg_l2_shifted);
        prop_assert_eq!(&binseg_l2_base, &binseg_l2_scaled);

        let binseg_normal_base = binseg_normal_breakpoints(&base, &constraints, stopping.clone());
        let binseg_normal_shifted =
            binseg_normal_breakpoints(&shifted, &constraints, stopping.clone());
        let binseg_normal_scaled = binseg_normal_breakpoints(&scaled, &constraints, stopping);
        prop_assert_eq!(&binseg_normal_base, &binseg_normal_shifted);
        prop_assert_eq!(&binseg_normal_base, &binseg_normal_scaled);
    }

    #[test]
    fn concatenated_constant_segments_produce_join_breakpoint_for_binseg_known_k(
        left_len in 8usize..64,
        right_len in 8usize..64,
        left_level in -30.0f64..30.0,
        right_level in -30.0f64..30.0,
    ) {
        prop_assume!((left_level - right_level).abs() >= 1.0);

        let mut values = Vec::with_capacity(left_len + right_len);
        values.extend(std::iter::repeat_n(left_level, left_len));
        values.extend(std::iter::repeat_n(right_level, right_len));
        let n = values.len();

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let stopping = Stopping::KnownK(1);

        let binseg = binseg_l2_breakpoints(&values, &constraints, stopping);

        prop_assert_eq!(binseg, vec![left_len, n]);
    }
}
