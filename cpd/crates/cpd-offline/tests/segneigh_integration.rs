// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, Stopping,
    TimeIndex, TimeSeriesView, validate_breakpoints,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{Dynp, DynpConfig, SegNeigh, SegNeighConfig};

fn make_view(values: &[f64], n: usize) -> TimeSeriesView<'_> {
    TimeSeriesView::from_f64(
        values,
        n,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("test view should be valid")
}

fn run_segneigh_l2(values: &[f64], n: usize, constraints: &Constraints, k: usize) -> Vec<usize> {
    let view = make_view(values, n);
    let ctx = ExecutionContext::new(constraints);
    let detector = SegNeigh::new(
        CostL2Mean::default(),
        SegNeighConfig {
            stopping: Stopping::KnownK(k),
            cancel_check_every: 64,
        },
    )
    .expect("SegNeigh config should be valid");
    detector
        .detect(&view, &ctx)
        .expect("SegNeigh detect should succeed")
        .breakpoints
}

fn segment_l2_cost(prefix: &[f64], prefix_sq: &[f64], start: usize, end: usize) -> f64 {
    let length = (end - start) as f64;
    let sum = prefix[end] - prefix[start];
    let sum_sq = prefix_sq[end] - prefix_sq[start];
    let cost = sum_sq - (sum * sum) / length;
    cost.max(0.0)
}

fn breakpoints_respect_min_segment_len(
    breakpoints: &[usize],
    n: usize,
    min_segment_len: usize,
) -> bool {
    let mut prev = 0usize;
    for &bp in breakpoints {
        if bp <= prev || bp > n {
            return false;
        }
        if bp - prev < min_segment_len {
            return false;
        }
        prev = bp;
    }
    n - prev >= min_segment_len
}

fn brute_force_known_k_l2(
    values: &[f64],
    n: usize,
    constraints: &Constraints,
    k: usize,
) -> Vec<usize> {
    assert_eq!(values.len(), n, "fixture length must match n");
    let min_segment_len = constraints.min_segment_len.max(1);
    let jump = constraints.jump.max(1);

    let mut candidates: Vec<usize> = match &constraints.candidate_splits {
        Some(explicit) => explicit.clone(),
        None => (1..n).collect(),
    };
    candidates.retain(|&split| split > 0 && split < n && split % jump == 0);
    candidates.sort_unstable();
    candidates.dedup();

    assert!(
        k <= candidates.len(),
        "fixture does not have enough candidate splits for k={k}"
    );

    let mut prefix = vec![0.0; n + 1];
    let mut prefix_sq = vec![0.0; n + 1];
    for (idx, &value) in values.iter().enumerate() {
        prefix[idx + 1] = prefix[idx] + value;
        prefix_sq[idx + 1] = prefix_sq[idx] + value * value;
    }

    fn search(
        candidates: &[usize],
        prefix: &[f64],
        prefix_sq: &[f64],
        n: usize,
        min_segment_len: usize,
        next_idx: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        best: &mut Option<(f64, Vec<usize>)>,
    ) {
        if remaining == 0 {
            if !breakpoints_respect_min_segment_len(current, n, min_segment_len) {
                return;
            }

            let mut objective = 0.0;
            let mut start = 0usize;
            for &end in current.iter().chain(std::iter::once(&n)) {
                objective += segment_l2_cost(prefix, prefix_sq, start, end);
                start = end;
            }

            match best {
                None => *best = Some((objective, current.clone())),
                Some((best_objective, best_breakpoints)) => {
                    let better = objective < *best_objective - 1e-12
                        || ((objective - *best_objective).abs() <= 1e-12
                            && *current < *best_breakpoints);
                    if better {
                        *best_objective = objective;
                        *best_breakpoints = current.clone();
                    }
                }
            }
            return;
        }

        if candidates.len().saturating_sub(next_idx) < remaining {
            return;
        }

        for candidate_idx in next_idx..=candidates.len() - remaining {
            current.push(candidates[candidate_idx]);
            search(
                candidates,
                prefix,
                prefix_sq,
                n,
                min_segment_len,
                candidate_idx + 1,
                remaining - 1,
                current,
                best,
            );
            current.pop();
        }
    }

    let mut best: Option<(f64, Vec<usize>)> = None;
    search(
        &candidates,
        &prefix,
        &prefix_sq,
        n,
        min_segment_len,
        0,
        k,
        &mut Vec::new(),
        &mut best,
    );
    let (_, mut expected_breakpoints) =
        best.expect("at least one feasible fixed-K combination should exist");
    expected_breakpoints.push(n);
    expected_breakpoints
}

#[test]
fn segneigh_known_k_recovers_three_regime_step_fixture() {
    let values = vec![
        0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -4.0, -4.0, -4.0, -4.0,
    ];
    let constraints = Constraints {
        min_segment_len: 2,
        ..Constraints::default()
    };

    let breakpoints = run_segneigh_l2(&values, values.len(), &constraints, 2);
    assert_eq!(breakpoints, vec![4, 8, 12]);
}

#[test]
fn segneigh_known_k_matches_bruteforce_optima_on_small_fixtures() {
    let fixtures = vec![
        (
            "step_3regimes",
            vec![
                0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -4.0, -4.0, -4.0, -4.0,
            ],
            Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            2usize,
        ),
        (
            "noisy_3regimes",
            vec![1.0, 1.1, 0.9, 8.0, 8.2, 7.8, -3.0, -2.8, -3.2, -3.1],
            Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            2usize,
        ),
        (
            "jump_constrained",
            vec![
                0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -2.0, -2.0, -2.0, -2.0, -2.0,
                -2.0,
            ],
            Constraints {
                min_segment_len: 2,
                jump: 2,
                ..Constraints::default()
            },
            2usize,
        ),
        (
            "explicit_candidates",
            vec![
                0.0, 0.0, 0.0, 6.0, 6.0, 6.0, -3.0, -3.0, -3.0, 2.0, 2.0, 2.0,
            ],
            Constraints {
                min_segment_len: 2,
                candidate_splits: Some(vec![3, 6, 9]),
                ..Constraints::default()
            },
            3usize,
        ),
    ];

    for (name, values, constraints, k) in fixtures {
        let n = values.len();
        let expected = brute_force_known_k_l2(&values, n, &constraints, k);
        let actual = run_segneigh_l2(&values, n, &constraints, k);

        validate_breakpoints(n, &actual).expect("SegNeigh breakpoints should satisfy invariants");
        assert_eq!(
            actual, expected,
            "fixture={name}: SegNeigh should match exhaustive optimum for KnownK({k})"
        );
    }
}

#[test]
fn segneigh_alias_matches_dynp_for_known_k() {
    let values = vec![
        0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0, -2.0, -2.0, -2.0, -2.0,
    ];
    let n = values.len();
    let constraints = Constraints {
        min_segment_len: 2,
        ..Constraints::default()
    };
    let view = make_view(&values, n);
    let ctx = ExecutionContext::new(&constraints);

    let segneigh = SegNeigh::new(
        CostL2Mean::default(),
        SegNeighConfig {
            stopping: Stopping::KnownK(2),
            cancel_check_every: 64,
        },
    )
    .expect("SegNeigh config should be valid");
    let dynp = Dynp::new(
        CostL2Mean::default(),
        DynpConfig {
            stopping: Stopping::KnownK(2),
            cancel_check_every: 64,
        },
    )
    .expect("Dynp config should be valid");

    let segneigh_result = segneigh
        .detect(&view, &ctx)
        .expect("SegNeigh known-k should succeed");
    let dynp_result = dynp
        .detect(&view, &ctx)
        .expect("Dynp known-k should succeed");
    assert_eq!(segneigh_result.breakpoints, dynp_result.breakpoints);
}
