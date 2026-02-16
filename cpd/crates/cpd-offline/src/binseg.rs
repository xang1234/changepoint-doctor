// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value, validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const DEFAULT_PARAMS_PER_SEGMENT: usize = 2;

/// Configuration for [`BinSeg`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BinSegConfig {
    pub stopping: Stopping,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for BinSegConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            params_per_segment: 2,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl BinSegConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;

        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "BinSegConfig.params_per_segment must be >= 1; got 0",
            ));
        }

        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Binary Segmentation offline detector.
#[derive(Debug)]
pub struct BinSeg<C: CostModel> {
    cost_model: C,
    config: BinSegConfig,
}

impl<C: CostModel> BinSeg<C> {
    pub fn new(cost_model: C, config: BinSegConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &BinSegConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: usize,
    end: usize,
    depth: usize,
}

#[derive(Clone, Copy, Debug)]
struct SegmentCandidate {
    segment: Segment,
    split: usize,
    gain: f64,
}

#[derive(Clone, Debug)]
struct ResolvedPenalty {
    beta: f64,
    params_per_segment: usize,
    params_source: &'static str,
}

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

fn resolve_penalty_params<C: CostModel>(
    model: &C,
    penalty: &Penalty,
    configured_params_per_segment: usize,
) -> (usize, &'static str) {
    match penalty {
        Penalty::BIC | Penalty::AIC => {
            if configured_params_per_segment == DEFAULT_PARAMS_PER_SEGMENT {
                (model.penalty_params_per_segment(), "model_default")
            } else {
                (configured_params_per_segment, "config_override")
            }
        }
        Penalty::Manual(_) => (configured_params_per_segment, "config"),
    }
}

fn resolve_penalty_beta<C: CostModel>(
    model: &C,
    penalty: &Penalty,
    n: usize,
    d: usize,
    configured_params_per_segment: usize,
) -> Result<ResolvedPenalty, CpdError> {
    let (params_per_segment, params_source) =
        resolve_penalty_params(model, penalty, configured_params_per_segment);
    if params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "resolved params_per_segment must be >= 1; got 0",
        ));
    }
    let beta = penalty_value(penalty, n, d, params_per_segment)?;
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty must be finite and > 0.0; got beta={beta}"
        )));
    }
    Ok(ResolvedPenalty {
        beta,
        params_per_segment,
        params_source,
    })
}

fn evaluate_segment_cost<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    start: usize,
    end: usize,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<f64, CpdError> {
    checked_counter_increment(&mut runtime.cost_evals, "cost_evals")?;

    match ctx.check_cost_eval_budget(runtime.cost_evals)? {
        BudgetStatus::WithinBudget => {}
        BudgetStatus::ExceededSoftDegrade => {
            runtime.soft_budget_exceeded = true;
        }
    }

    let segment_cost = model.segment_cost(cache, start, end);
    if !segment_cost.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite segment cost at [{start}, {end}): {segment_cost}"
        )));
    }
    Ok(segment_cost)
}

fn check_runtime_controls(
    iteration: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<(), CpdError> {
    if iteration.is_multiple_of(cancel_check_every) {
        ctx.check_cancelled_every(iteration, 1)?;
        match ctx.check_time_budget(started_at)? {
            BudgetStatus::WithinBudget => {}
            BudgetStatus::ExceededSoftDegrade => {
                runtime.soft_budget_exceeded = true;
            }
        }
    }

    Ok(())
}

fn candidate_window(candidates: &[usize], lower: usize, upper: usize) -> Option<(usize, usize)> {
    if lower > upper {
        return None;
    }

    let start_idx = candidates.partition_point(|&split| split < lower);
    let end_idx = candidates.partition_point(|&split| split <= upper);
    if start_idx >= end_idx {
        return None;
    }
    Some((start_idx, end_idx))
}

fn segment_can_split(segment: Segment, validated: &ValidatedConstraints) -> bool {
    if segment.end <= segment.start {
        return false;
    }

    if let Some(max_depth) = validated.max_depth
        && segment.depth >= max_depth
    {
        return false;
    }

    let min_len = validated.min_segment_len;
    segment.end - segment.start >= min_len.saturating_mul(2)
}

#[allow(clippy::too_many_arguments)]
fn best_split_for_segment<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    candidates: &[usize],
    segment: Segment,
    validated: &ValidatedConstraints,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Option<SegmentCandidate>, CpdError> {
    if !segment_can_split(segment, validated) {
        return Ok(None);
    }

    let lower = segment
        .start
        .checked_add(validated.min_segment_len)
        .ok_or_else(|| CpdError::resource_limit("segment lower bound overflow"))?;
    let upper = segment.end.saturating_sub(validated.min_segment_len);

    let Some((start_idx, end_idx)) = candidate_window(candidates, lower, upper) else {
        return Ok(None);
    };

    let full_cost = evaluate_segment_cost(model, cache, segment.start, segment.end, ctx, runtime)?;

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split = usize::MAX;

    for &split in &candidates[start_idx..end_idx] {
        checked_counter_increment(iteration, "iteration")?;
        check_runtime_controls(*iteration, cancel_check_every, ctx, started_at, runtime)?;

        checked_counter_increment(&mut runtime.candidates_considered, "candidates_considered")?;

        let left_cost = evaluate_segment_cost(model, cache, segment.start, split, ctx, runtime)?;
        let right_cost = evaluate_segment_cost(model, cache, split, segment.end, ctx, runtime)?;
        let gain = full_cost - left_cost - right_cost;
        if !gain.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite gain at segment=[{}, {}), split={split}: full_cost={full_cost}, left_cost={left_cost}, right_cost={right_cost}, gain={gain}",
                segment.start, segment.end
            )));
        }

        if gain > best_gain || (gain == best_gain && split < best_split) {
            best_gain = gain;
            best_split = split;
        }
    }

    if best_split == usize::MAX {
        return Ok(None);
    }

    Ok(Some(SegmentCandidate {
        segment,
        split: best_split,
        gain: best_gain,
    }))
}

#[allow(clippy::too_many_arguments)]
fn add_segment_to_frontier<C: CostModel>(
    frontier: &mut Vec<SegmentCandidate>,
    model: &C,
    cache: &C::Cache,
    candidates: &[usize],
    segment: Segment,
    validated: &ValidatedConstraints,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<(), CpdError> {
    if let Some(candidate) = best_split_for_segment(
        model,
        cache,
        candidates,
        segment,
        validated,
        ctx,
        cancel_check_every,
        started_at,
        runtime,
        iteration,
    )? {
        frontier.push(candidate);
    }
    Ok(())
}

fn pick_best_frontier_index(frontier: &[SegmentCandidate]) -> Option<usize> {
    let mut best_idx: Option<usize> = None;

    for (idx, candidate) in frontier.iter().enumerate() {
        if let Some(current_best_idx) = best_idx {
            let current_best = frontier[current_best_idx];
            let better_gain = candidate.gain > current_best.gain;
            let tie_on_gain = candidate.gain == current_best.gain;
            let better_split = candidate.split < current_best.split;
            let tie_on_split = candidate.split == current_best.split;
            let better_start = candidate.segment.start < current_best.segment.start;
            if better_gain || (tie_on_gain && (better_split || (tie_on_split && better_start))) {
                best_idx = Some(idx);
            }
        } else {
            best_idx = Some(idx);
        }
    }

    best_idx
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) -> Result<(), CpdError> {
    match values.binary_search(&value) {
        Ok(_) => Err(CpdError::invalid_input(format!(
            "duplicate split selected at {value}; internal BinSeg state is inconsistent"
        ))),
        Err(idx) => {
            values.insert(idx, value);
            Ok(())
        }
    }
}

fn build_result_breakpoints(n: usize, change_points: Vec<usize>) -> Vec<usize> {
    let mut breakpoints = change_points;
    breakpoints.push(n);
    breakpoints
}

impl<C: CostModel> OfflineDetector for BinSeg<C> {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        self.config.validate()?;

        let validated = validate_constraints(ctx.constraints, x.n)?;

        self.cost_model.validate(x)?;
        check_missing_compatibility(x.missing, self.cost_model.missing_support())?;
        let cache = self.cost_model.precompute(x, &validated.cache_policy)?;

        let started_at = Instant::now();
        let cancel_check_every = self.config.normalized_cancel_check_every();
        let mut runtime = RuntimeStats::default();
        let mut iteration = 0usize;
        let mut notes = vec![];
        let mut warnings = vec![
            "BinSeg may mask closely spaced weaker changes; consider WBS for robust recovery"
                .to_string(),
        ];
        let candidates = &validated.effective_candidates;

        let root = Segment {
            start: 0,
            end: x.n,
            depth: 0,
        };
        let mut frontier = Vec::new();
        add_segment_to_frontier(
            &mut frontier,
            &self.cost_model,
            &cache,
            candidates,
            root,
            &validated,
            ctx,
            cancel_check_every,
            started_at,
            &mut runtime,
            &mut iteration,
        )?;

        let mut accepted_splits = vec![];

        match &self.config.stopping {
            Stopping::KnownK(k) => {
                if let Some(max_change_points) = validated.max_change_points
                    && max_change_points < *k
                {
                    return Err(CpdError::invalid_input(format!(
                        "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
                    )));
                }

                for _round in 0..*k {
                    checked_counter_increment(&mut iteration, "iteration")?;
                    check_runtime_controls(
                        iteration,
                        cancel_check_every,
                        ctx,
                        started_at,
                        &mut runtime,
                    )?;

                    let Some(best_idx) = pick_best_frontier_index(&frontier) else {
                        return Err(CpdError::invalid_input(format!(
                            "KnownK exact solution unreachable: requested k={k}, accepted={} before frontier exhaustion",
                            accepted_splits.len()
                        )));
                    };

                    let best = frontier.swap_remove(best_idx);
                    insert_sorted_unique(&mut accepted_splits, best.split)?;

                    let child_depth = best
                        .segment
                        .depth
                        .checked_add(1)
                        .ok_or_else(|| CpdError::resource_limit("segment depth overflow"))?;
                    let left = Segment {
                        start: best.segment.start,
                        end: best.split,
                        depth: child_depth,
                    };
                    let right = Segment {
                        start: best.split,
                        end: best.segment.end,
                        depth: child_depth,
                    };

                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        candidates,
                        left,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;
                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        candidates,
                        right,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    ctx.report_progress(accepted_splits.len() as f32 / *k as f32);
                }

                notes.push(format!(
                    "stopping=KnownK({k}), accepted_splits={}",
                    accepted_splits.len()
                ));
            }
            Stopping::Penalized(penalty) => {
                let resolved = resolve_penalty_beta(
                    &self.cost_model,
                    penalty,
                    x.n,
                    x.d,
                    self.config.params_per_segment,
                )?;
                let beta = resolved.beta;
                notes.push(format!(
                    "stopping=Penalized({penalty:?}), beta={beta}, params_per_segment={} ({})",
                    resolved.params_per_segment, resolved.params_source
                ));

                let mut processed_frontier_items = 0usize;
                loop {
                    checked_counter_increment(&mut iteration, "iteration")?;
                    check_runtime_controls(
                        iteration,
                        cancel_check_every,
                        ctx,
                        started_at,
                        &mut runtime,
                    )?;

                    if let Some(max_change_points) = validated.max_change_points
                        && accepted_splits.len() >= max_change_points
                    {
                        break;
                    }

                    let Some(best_idx) = pick_best_frontier_index(&frontier) else {
                        break;
                    };

                    let best = frontier[best_idx];
                    if best.gain <= beta {
                        break;
                    }

                    let best = frontier.swap_remove(best_idx);
                    insert_sorted_unique(&mut accepted_splits, best.split)?;
                    checked_counter_increment(
                        &mut processed_frontier_items,
                        "processed_frontier_items",
                    )?;

                    let child_depth = best
                        .segment
                        .depth
                        .checked_add(1)
                        .ok_or_else(|| CpdError::resource_limit("segment depth overflow"))?;
                    let left = Segment {
                        start: best.segment.start,
                        end: best.split,
                        depth: child_depth,
                    };
                    let right = Segment {
                        start: best.split,
                        end: best.segment.end,
                        depth: child_depth,
                    };

                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        candidates,
                        left,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;
                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        candidates,
                        right,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    let progress_denom = (processed_frontier_items + frontier.len() + 1) as f32;
                    if progress_denom.is_finite() && progress_denom > 0.0 {
                        ctx.report_progress(processed_frontier_items as f32 / progress_denom);
                    }
                }
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "BinSeg penalty sweep is deferred for this issue; got PenaltyPath of length {}",
                    path.len()
                )));
            }
        }

        if runtime.soft_budget_exceeded {
            warnings.push(
                "budget exceeded under SoftDegrade mode; run continued without algorithm fallback"
                    .to_string(),
            );
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

        ctx.record_scalar("offline.binseg.cost_evals", runtime.cost_evals as f64);
        ctx.record_scalar(
            "offline.binseg.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar("offline.binseg.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_change_count={}, cost_evals={}, candidates_considered={}",
            accepted_splits.len(),
            runtime.cost_evals,
            runtime.candidates_considered
        ));

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("binseg"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            repro_mode: ctx.repro_mode,
            ..Diagnostics::default()
        };

        let breakpoints = build_result_breakpoints(x.n, accepted_splits);
        OfflineChangePointResult::new(x.n, breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{BinSeg, BinSegConfig};
    use cpd_core::{
        BudgetMode, Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy,
        OfflineDetector, Penalty, ProgressSink, Stopping, TimeIndex, TimeSeriesView,
    };
    use cpd_costs::{CostL2Mean, CostNormalMeanVar};
    use std::thread;
    use std::time::Duration;

    fn make_f64_view<'a>(
        values: &'a [f64],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F64(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn constraints_with_min_segment_len(min_segment_len: usize) -> Constraints {
        Constraints {
            min_segment_len,
            ..Constraints::default()
        }
    }

    fn assert_strictly_increasing(values: &[usize]) {
        for window in values.windows(2) {
            assert!(window[0] < window[1], "not strictly increasing: {values:?}");
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = BinSegConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.params_per_segment, 2);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = BinSeg::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);

        let err = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                params_per_segment: 0,
                ..default_cfg
            },
        )
        .expect_err("params_per_segment=0 must fail");
        assert!(err.to_string().contains("params_per_segment"));
    }

    #[test]
    fn cancel_check_every_zero_is_normalized() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 0,
            },
        )
        .expect("config with zero cadence should normalize");

        let values = vec![0.0, 0.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
    }

    #[test]
    fn known_small_example_one_change_l2() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8]);
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -4.0, -4.0, -4.0, -4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
    }

    #[test]
    fn normal_cost_detects_variance_change() {
        let detector = BinSeg::new(
            CostNormalMeanVar::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 3,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let mut values = Vec::with_capacity(20);
        for _ in 0..5 {
            values.push(-1.0);
            values.push(1.0);
        }
        for _ in 0..5 {
            values.push(-5.0);
            values.push(5.0);
        }

        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![10, 20]);
    }

    #[test]
    fn known_k_exact_unreachable_is_clear_error() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 2,
            candidate_splits: Some(vec![4]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("known-k should be unreachable");
        assert!(
            err.to_string()
                .contains("KnownK exact solution unreachable")
        );
    }

    #[test]
    fn known_k_fails_when_max_change_points_below_k() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("KnownK must reject when max_change_points < k");
        assert!(err.to_string().contains("max_change_points"));
    }

    #[test]
    fn penalty_path_returns_not_supported() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(1.0)]),
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("PenaltyPath should be deferred");
        assert!(matches!(err, cpd_core::CpdError::NotSupported(_)));
    }

    #[test]
    fn tie_breaking_prefers_leftmost_split() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 1,
            candidate_splits: Some(vec![2, 4]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![2, 6]);
    }

    #[test]
    fn repeated_runs_are_deterministic() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            jump: 1,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let first = detector
            .detect(&view, &ctx)
            .expect("first detect should pass");
        let second = detector
            .detect(&view, &ctx)
            .expect("second detect should pass");
        assert_eq!(first.breakpoints, second.breakpoints);
    }

    #[test]
    fn constraints_min_segment_len_and_jump_are_enforced() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 3,
            },
        )
        .expect("config should be valid");

        let n = 24;
        let values: Vec<f64> = (0..n)
            .map(|idx| {
                if idx < 8 {
                    0.0
                } else if idx < 16 {
                    5.0
                } else {
                    10.0
                }
            })
            .collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 4,
            jump: 4,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        for &cp in &result.change_points {
            assert_eq!(cp % 4, 0, "change point must respect jump=4");
        }
        assert_eq!(result.breakpoints.last().copied(), Some(n));
        let mut start = 0usize;
        for &end in &result.breakpoints {
            assert!(
                end - start >= 4,
                "segment [{start}, {end}) violates min_segment_len=4"
            );
            start = end;
        }
    }

    #[test]
    fn explicit_candidate_splits_and_max_change_points_are_enforced() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_change_points: Some(1),
            candidate_splits: Some(vec![4, 8]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        assert!(result.change_points.len() <= 1);
        for cp in result.change_points {
            assert!(cp == 4 || cp == 8);
        }
    }

    #[test]
    fn max_depth_reached_stops_cleanly() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0, -2.0, -2.0, 4.0, 4.0, 4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_depth: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        assert!(result.change_points.len() <= 1);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
    }

    #[test]
    fn constant_series_has_no_changes_with_reasonable_penalty() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![3.0; 64];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![64]);
        assert_eq!(result.change_points, Vec::<usize>::new());
    }

    #[test]
    fn cancellation_mid_run_returns_cancelled() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let n = 5_000;
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(1);
        let cancel = cpd_core::CancelToken::new();
        cancel.cancel();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("cancelled token must stop detect");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn cost_eval_budget_exceeded_hard_fail() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard budget should fail");
        assert!(err.to_string().contains("max_cost_evals"));
    }

    struct SlowProgressSink;

    impl ProgressSink for SlowProgressSink {
        fn on_progress(&self, _fraction: f32) {
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn time_budget_exceeded_hard_fail() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let slow_progress = SlowProgressSink;
        let ctx = ExecutionContext::new(&constraints)
            .with_budget_mode(BudgetMode::HardFail)
            .with_progress_sink(&slow_progress);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard time budget should fail");
        assert!(err.to_string().contains("time_budget_ms"));
    }

    #[test]
    fn diagnostics_include_soft_budget_and_masking_warning() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 5.0, 6.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_cost_evals: Some(2),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        let result = detector
            .detect(&view, &ctx)
            .expect("soft degrade should continue");
        let diagnostics = result.diagnostics;
        assert_eq!(diagnostics.algorithm, "binseg");
        assert_eq!(diagnostics.cost_model, "l2_mean");
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("SoftDegrade"))
        );
        assert!(diagnostics.warnings.iter().any(|w| w.contains("WBS")));
    }

    #[test]
    fn masking_demonstration_has_limited_changes_and_warning() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(40.0)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let mut values = vec![0.0; 40];
        values.extend(std::iter::repeat_n(6.0, 6));
        values.extend(std::iter::repeat_n(0.0, 40));
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should pass");
        assert!(
            result.change_points.len() <= 1,
            "masking demo expects limited detected changes, got {:?}",
            result.change_points
        );
        assert!(
            result
                .diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("WBS"))
        );
    }

    #[test]
    fn large_n_regression_smoke() {
        let detector = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::BIC),
                params_per_segment: 2,
                cancel_check_every: 1024,
            },
        )
        .expect("config should be valid");

        let n = 100_000;
        let mut values = vec![0.0; n];
        for v in values.iter_mut().skip(n / 2) {
            *v = 2.0;
        }
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 50,
            jump: 50,
            max_change_points: Some(8),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector
            .detect(&view, &ctx)
            .expect("large-n smoke should pass");

        assert_eq!(result.breakpoints.last().copied(), Some(n));
        assert_strictly_increasing(&result.breakpoints);
        assert!(result.change_points.len() <= 8);
    }
}
