// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;

/// Cache behavior and memory trade-offs for cost-model precomputation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum CachePolicy {
    Full,
    Budgeted {
        max_bytes: usize,
    },
    Approximate {
        max_bytes: usize,
        error_tolerance: f64,
    },
}

/// Ordered fallback steps used when operating under soft budget enforcement.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum DegradationStep {
    IncreaseJump { factor: usize, max_jump: usize },
    DisableUncertaintyBands,
    SwitchCachePolicy(CachePolicy),
}

/// User-facing constraints shared across all detector implementations.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Constraints {
    pub min_segment_len: usize,
    pub max_change_points: Option<usize>,
    pub max_depth: Option<usize>,
    pub candidate_splits: Option<Vec<usize>>,
    pub jump: usize,
    pub time_budget_ms: Option<u64>,
    pub max_cost_evals: Option<usize>,
    pub memory_budget_bytes: Option<usize>,
    pub max_cache_bytes: Option<usize>,
    pub cache_policy: CachePolicy,
    pub degradation_plan: Vec<DegradationStep>,
    pub allow_algorithm_fallback: bool,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            min_segment_len: 2,
            max_change_points: None,
            max_depth: None,
            candidate_splits: None,
            jump: 1,
            time_budget_ms: None,
            max_cost_evals: None,
            memory_budget_bytes: None,
            max_cache_bytes: None,
            cache_policy: CachePolicy::Full,
            degradation_plan: vec![],
            allow_algorithm_fallback: false,
        }
    }
}

/// Prevalidated constraints passed into detector execution paths.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedConstraints {
    pub n: usize,
    pub min_segment_len: usize,
    pub max_change_points: Option<usize>,
    pub max_depth: Option<usize>,
    pub jump: usize,
    pub effective_candidates: Vec<usize>,
    pub time_budget_ms: Option<u64>,
    pub max_cost_evals: Option<usize>,
    pub memory_budget_bytes: Option<usize>,
    pub max_cache_bytes: Option<usize>,
    pub cache_policy: CachePolicy,
    pub degradation_plan: Vec<DegradationStep>,
    pub allow_algorithm_fallback: bool,
}

fn split_respects_min_segment_len(split: usize, n: usize, min_segment_len: usize) -> bool {
    split >= min_segment_len && n.saturating_sub(split) >= min_segment_len
}

fn validate_candidate_splits_sorted_unique_in_range(
    candidate_splits: &[usize],
    n: usize,
) -> Result<(), CpdError> {
    let mut prev: Option<usize> = None;
    for (idx, &split) in candidate_splits.iter().enumerate() {
        if split == 0 || split >= n {
            return Err(CpdError::invalid_input(format!(
                "constraints.candidate_splits[{idx}] must satisfy 0 < split < n; got split={split}, n={n}"
            )));
        }
        if let Some(prev_split) = prev
            && split <= prev_split
        {
            return Err(CpdError::invalid_input(format!(
                "constraints.candidate_splits must be strictly increasing and unique: index {} has {}, previous {}",
                idx, split, prev_split
            )));
        }
        prev = Some(split);
    }
    Ok(())
}

fn validate_cache_policy(cache_policy: &CachePolicy) -> Result<(), CpdError> {
    match cache_policy {
        CachePolicy::Full => Ok(()),
        CachePolicy::Budgeted { max_bytes } => {
            if *max_bytes == 0 {
                return Err(CpdError::invalid_input(
                    "constraints.cache_policy.Budgeted.max_bytes must be > 0; got 0",
                ));
            }
            Ok(())
        }
        CachePolicy::Approximate {
            max_bytes,
            error_tolerance,
        } => {
            if *max_bytes == 0 {
                return Err(CpdError::invalid_input(
                    "constraints.cache_policy.Approximate.max_bytes must be > 0; got 0",
                ));
            }
            if !error_tolerance.is_finite() || *error_tolerance <= 0.0 {
                return Err(CpdError::invalid_input(format!(
                    "constraints.cache_policy.Approximate.error_tolerance must be finite and > 0.0; got {error_tolerance}"
                )));
            }
            Ok(())
        }
    }
}

/// Validates shape-level constraints that do not depend on series length `n`.
pub fn validate_constraints_config(constraints: &Constraints) -> Result<(), CpdError> {
    if constraints.min_segment_len == 0 {
        return Err(CpdError::invalid_input(
            "constraints.min_segment_len must be >= 1; got 0",
        ));
    }
    if constraints.jump == 0 {
        return Err(CpdError::invalid_input(
            "constraints.jump must be >= 1; got 0",
        ));
    }

    if let Some(time_budget_ms) = constraints.time_budget_ms
        && time_budget_ms == 0
    {
        return Err(CpdError::invalid_input(
            "constraints.time_budget_ms must be > 0 when provided; got 0",
        ));
    }
    if let Some(max_cost_evals) = constraints.max_cost_evals
        && max_cost_evals == 0
    {
        return Err(CpdError::invalid_input(
            "constraints.max_cost_evals must be > 0 when provided; got 0",
        ));
    }
    if let Some(memory_budget_bytes) = constraints.memory_budget_bytes
        && memory_budget_bytes == 0
    {
        return Err(CpdError::invalid_input(
            "constraints.memory_budget_bytes must be > 0 when provided; got 0",
        ));
    }
    if let Some(max_cache_bytes) = constraints.max_cache_bytes
        && max_cache_bytes == 0
    {
        return Err(CpdError::invalid_input(
            "constraints.max_cache_bytes must be > 0 when provided; got 0",
        ));
    }

    validate_cache_policy(&constraints.cache_policy)?;

    if let (Some(memory_budget_bytes), Some(max_cache_bytes)) =
        (constraints.memory_budget_bytes, constraints.max_cache_bytes)
        && max_cache_bytes > memory_budget_bytes
    {
        return Err(CpdError::invalid_input(format!(
            "constraints.max_cache_bytes must be <= constraints.memory_budget_bytes; got max_cache_bytes={max_cache_bytes}, memory_budget_bytes={memory_budget_bytes}"
        )));
    }

    if let Some(candidate_splits) = constraints.candidate_splits.as_deref()
        && let Some(max_split) = candidate_splits.iter().copied().max()
    {
        let n_shape = max_split
            .checked_add(1)
            .ok_or_else(|| CpdError::invalid_input("constraints.candidate_splits overflow"))?;
        validate_candidate_splits_sorted_unique_in_range(candidate_splits, n_shape)?;
    }

    Ok(())
}

/// Canonicalizes candidate split positions by applying jump and segment-length rules.
///
/// For valid inputs, output is deterministic, sorted ascending, and unique.
pub fn canonicalize_candidates(constraints: &Constraints, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }

    let jump = constraints.jump.max(1);
    let min_segment_len = constraints.min_segment_len;

    if let Some(candidate_splits) = constraints.candidate_splits.as_deref() {
        candidate_splits
            .iter()
            .copied()
            .filter(|&split| split % jump == 0)
            .filter(|&split| split_respects_min_segment_len(split, n, min_segment_len))
            .collect::<Vec<_>>()
    } else {
        (jump..n)
            .step_by(jump)
            .filter(|&split| split_respects_min_segment_len(split, n, min_segment_len))
            .collect::<Vec<_>>()
    }
}

/// Validates and canonicalizes constraints for a specific series length `n`.
pub fn validate_constraints(
    constraints: &Constraints,
    n: usize,
) -> Result<ValidatedConstraints, CpdError> {
    if n == 0 {
        return Err(CpdError::invalid_input(
            "constraints validation requires n >= 1; got n=0",
        ));
    }
    validate_constraints_config(constraints)?;

    if let Some(candidate_splits) = constraints.candidate_splits.as_deref() {
        validate_candidate_splits_sorted_unique_in_range(candidate_splits, n)?;
    }

    let effective_candidates = canonicalize_candidates(constraints, n);

    Ok(ValidatedConstraints {
        n,
        min_segment_len: constraints.min_segment_len,
        max_change_points: constraints.max_change_points,
        max_depth: constraints.max_depth,
        jump: constraints.jump,
        effective_candidates,
        time_budget_ms: constraints.time_budget_ms,
        max_cost_evals: constraints.max_cost_evals,
        memory_budget_bytes: constraints.memory_budget_bytes,
        max_cache_bytes: constraints.max_cache_bytes,
        cache_policy: constraints.cache_policy.clone(),
        degradation_plan: constraints.degradation_plan.clone(),
        allow_algorithm_fallback: constraints.allow_algorithm_fallback,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CachePolicy, Constraints, DegradationStep, ValidatedConstraints, canonicalize_candidates,
        validate_constraints,
    };

    #[test]
    fn constraints_default_matches_spec() {
        let defaults = Constraints::default();
        assert_eq!(defaults.min_segment_len, 2);
        assert_eq!(defaults.jump, 1);
        assert_eq!(defaults.cache_policy, CachePolicy::Full);
        assert_eq!(defaults.max_change_points, None);
        assert_eq!(defaults.max_depth, None);
        assert_eq!(defaults.candidate_splits, None);
        assert_eq!(defaults.time_budget_ms, None);
        assert_eq!(defaults.max_cost_evals, None);
        assert_eq!(defaults.memory_budget_bytes, None);
        assert_eq!(defaults.max_cache_bytes, None);
        assert_eq!(defaults.degradation_plan, vec![]);
        assert!(!defaults.allow_algorithm_fallback);
    }

    #[test]
    fn validate_constraints_rejects_n_zero() {
        let err = validate_constraints(&Constraints::default(), 0).expect_err("n=0 must fail");
        assert!(err.to_string().contains("n >= 1"));
    }

    #[test]
    fn validate_constraints_rejects_jump_zero() {
        let constraints = Constraints {
            jump: 0,
            ..Constraints::default()
        };
        let err = validate_constraints(&constraints, 10).expect_err("jump=0 must fail");
        assert!(err.to_string().contains("constraints.jump"));
    }

    #[test]
    fn validate_constraints_rejects_min_segment_len_zero() {
        let constraints = Constraints {
            min_segment_len: 0,
            ..Constraints::default()
        };
        let err = validate_constraints(&constraints, 10).expect_err("min_segment_len=0 must fail");
        assert!(err.to_string().contains("constraints.min_segment_len"));
    }

    #[test]
    fn validate_constraints_rejects_zero_optional_budgets() {
        let err_time = validate_constraints(
            &Constraints {
                time_budget_ms: Some(0),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("time_budget_ms=0 must fail");
        assert!(err_time.to_string().contains("constraints.time_budget_ms"));

        let err_evals = validate_constraints(
            &Constraints {
                max_cost_evals: Some(0),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("max_cost_evals=0 must fail");
        assert!(err_evals.to_string().contains("constraints.max_cost_evals"));

        let err_mem = validate_constraints(
            &Constraints {
                memory_budget_bytes: Some(0),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("memory_budget_bytes=0 must fail");
        assert!(
            err_mem
                .to_string()
                .contains("constraints.memory_budget_bytes")
        );

        let err_cache = validate_constraints(
            &Constraints {
                max_cache_bytes: Some(0),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("max_cache_bytes=0 must fail");
        assert!(
            err_cache
                .to_string()
                .contains("constraints.max_cache_bytes")
        );
    }

    #[test]
    fn validate_constraints_rejects_invalid_budgeted_cache_policy() {
        let constraints = Constraints {
            cache_policy: CachePolicy::Budgeted { max_bytes: 0 },
            ..Constraints::default()
        };
        let err = validate_constraints(&constraints, 10).expect_err("budgeted cache must validate");
        assert!(err.to_string().contains("Budgeted.max_bytes"));
    }

    #[test]
    fn validate_constraints_rejects_invalid_approximate_cache_policy() {
        let err_zero_bytes = validate_constraints(
            &Constraints {
                cache_policy: CachePolicy::Approximate {
                    max_bytes: 0,
                    error_tolerance: 0.1,
                },
                ..Constraints::default()
            },
            10,
        )
        .expect_err("approximate max_bytes=0 must fail");
        assert!(err_zero_bytes.to_string().contains("Approximate.max_bytes"));

        let err_zero_tol = validate_constraints(
            &Constraints {
                cache_policy: CachePolicy::Approximate {
                    max_bytes: 1024,
                    error_tolerance: 0.0,
                },
                ..Constraints::default()
            },
            10,
        )
        .expect_err("approximate error_tolerance=0 must fail");
        assert!(err_zero_tol.to_string().contains("error_tolerance"));

        let err_nan_tol = validate_constraints(
            &Constraints {
                cache_policy: CachePolicy::Approximate {
                    max_bytes: 1024,
                    error_tolerance: f64::NAN,
                },
                ..Constraints::default()
            },
            10,
        )
        .expect_err("approximate NaN error_tolerance must fail");
        assert!(err_nan_tol.to_string().contains("error_tolerance"));

        let err_inf_tol = validate_constraints(
            &Constraints {
                cache_policy: CachePolicy::Approximate {
                    max_bytes: 1024,
                    error_tolerance: f64::INFINITY,
                },
                ..Constraints::default()
            },
            10,
        )
        .expect_err("approximate infinite error_tolerance must fail");
        assert!(err_inf_tol.to_string().contains("error_tolerance"));
    }

    #[test]
    fn validate_constraints_rejects_unsorted_candidate_splits() {
        let constraints = Constraints {
            candidate_splits: Some(vec![2, 5, 4]),
            ..Constraints::default()
        };
        let err =
            validate_constraints(&constraints, 10).expect_err("unsorted candidates must fail");
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn validate_constraints_rejects_duplicate_candidate_splits() {
        let constraints = Constraints {
            candidate_splits: Some(vec![2, 4, 4]),
            ..Constraints::default()
        };
        let err =
            validate_constraints(&constraints, 10).expect_err("duplicate candidates must fail");
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn validate_constraints_rejects_out_of_range_candidate_splits() {
        let err_zero = validate_constraints(
            &Constraints {
                candidate_splits: Some(vec![0, 2, 4]),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("split=0 must fail");
        assert!(err_zero.to_string().contains("0 < split < n"));

        let err_equal_n = validate_constraints(
            &Constraints {
                candidate_splits: Some(vec![2, 10]),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("split=n must fail");
        assert!(err_equal_n.to_string().contains("0 < split < n"));

        let err_gt_n = validate_constraints(
            &Constraints {
                candidate_splits: Some(vec![2, 11]),
                ..Constraints::default()
            },
            10,
        )
        .expect_err("split>n must fail");
        assert!(err_gt_n.to_string().contains("0 < split < n"));
    }

    #[test]
    fn validate_constraints_accepts_zero_caps_and_preserves_values() {
        let validated = validate_constraints(
            &Constraints {
                max_change_points: Some(0),
                max_depth: Some(0),
                ..Constraints::default()
            },
            10,
        )
        .expect("zero caps must be valid");

        assert_eq!(validated.max_change_points, Some(0));
        assert_eq!(validated.max_depth, Some(0));
    }

    #[test]
    fn canonicalize_candidates_generates_implicit_candidates_with_bounds() {
        let constraints = Constraints {
            min_segment_len: 2,
            jump: 3,
            ..Constraints::default()
        };
        let got = canonicalize_candidates(&constraints, 12);
        assert_eq!(got, vec![3, 6, 9]);
    }

    #[test]
    fn canonicalize_candidates_applies_intersection_for_explicit_candidates_and_jump() {
        let constraints = Constraints {
            min_segment_len: 1,
            jump: 3,
            candidate_splits: Some(vec![2, 3, 6, 7, 9]),
            ..Constraints::default()
        };
        let got = canonicalize_candidates(&constraints, 12);
        assert_eq!(got, vec![3, 6, 9]);
    }

    #[test]
    fn canonicalize_candidates_filters_left_and_right_min_segment_len_bounds() {
        let constraints = Constraints {
            min_segment_len: 3,
            jump: 1,
            candidate_splits: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            ..Constraints::default()
        };
        let got = canonicalize_candidates(&constraints, 10);
        assert_eq!(got, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn validate_constraints_allows_empty_effective_candidates() {
        let constraints = Constraints {
            min_segment_len: 6,
            jump: 1,
            ..Constraints::default()
        };
        let validated = validate_constraints(&constraints, 10).expect("empty candidates are valid");
        assert!(validated.effective_candidates.is_empty());
    }

    #[test]
    fn validate_constraints_rejects_max_cache_exceeding_memory_budget() {
        let constraints = Constraints {
            memory_budget_bytes: Some(1024),
            max_cache_bytes: Some(2048),
            ..Constraints::default()
        };
        let err = validate_constraints(&constraints, 10)
            .expect_err("max_cache_bytes > memory_budget_bytes must fail");
        assert!(err.to_string().contains("max_cache_bytes"));
        assert!(err.to_string().contains("memory_budget_bytes"));
    }

    #[test]
    fn validated_constraints_effective_candidates_match_canonicalize_candidates() {
        let constraints = Constraints {
            min_segment_len: 2,
            jump: 2,
            candidate_splits: Some(vec![1, 2, 3, 4, 6, 8]),
            time_budget_ms: Some(100),
            max_cost_evals: Some(1000),
            memory_budget_bytes: Some(2048),
            max_cache_bytes: Some(1024),
            cache_policy: CachePolicy::Budgeted { max_bytes: 1024 },
            degradation_plan: vec![
                DegradationStep::DisableUncertaintyBands,
                DegradationStep::IncreaseJump {
                    factor: 2,
                    max_jump: 16,
                },
            ],
            allow_algorithm_fallback: true,
            ..Constraints::default()
        };

        let validated =
            validate_constraints(&constraints, 12).expect("constraints should validate");
        let expected = canonicalize_candidates(&constraints, 12);
        assert_eq!(validated.effective_candidates, expected);

        let expected_validated = ValidatedConstraints {
            n: 12,
            min_segment_len: 2,
            max_change_points: None,
            max_depth: None,
            jump: 2,
            effective_candidates: vec![2, 4, 6, 8],
            time_budget_ms: Some(100),
            max_cost_evals: Some(1000),
            memory_budget_bytes: Some(2048),
            max_cache_bytes: Some(1024),
            cache_policy: CachePolicy::Budgeted { max_bytes: 1024 },
            degradation_plan: vec![
                DegradationStep::DisableUncertaintyBands,
                DegradationStep::IncreaseJump {
                    factor: 2,
                    max_jump: 16,
                },
            ],
            allow_algorithm_fallback: true,
        };
        assert_eq!(validated, expected_validated);
    }
}
