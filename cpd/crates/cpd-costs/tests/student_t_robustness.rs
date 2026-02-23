// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostModel, CostNormalMeanVar, CostStudentT};

fn make_view<'a>(values: &'a [f64], n: usize, d: usize) -> TimeSeriesView<'a> {
    TimeSeriesView::new(
        DTypeView::F64(values),
        n,
        d,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("test view should be valid")
}

fn best_split<M: CostModel>(
    model: &M,
    cache: &M::Cache,
    n: usize,
    min_segment_len: usize,
) -> usize {
    let mut best_cp = min_segment_len;
    let mut best_objective = f64::INFINITY;
    for cp in min_segment_len..=(n - min_segment_len) {
        let objective = model.segment_cost(cache, 0, cp) + model.segment_cost(cache, cp, n);
        if objective < best_objective {
            best_objective = objective;
            best_cp = cp;
        }
    }
    best_cp
}

fn split_error<M: CostModel>(
    model: &M,
    values: &[f64],
    n: usize,
    d: usize,
    true_change: usize,
    min_segment_len: usize,
) -> usize {
    let view = make_view(values, n, d);
    let cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed");
    let best = best_split(model, &cache, n, min_segment_len);
    best.abs_diff(true_change)
}

fn make_outlier_fixture(n: usize, d: usize, true_change: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let base = if t < true_change { 0.0 } else { 3.0 };
            let trend = 0.02 * (dim as f64 + 1.0) * ((t as f64) * 0.03).sin();
            let mut value = base + trend;
            if (t + dim.saturating_mul(7)) % 37 == 11 {
                let sign = if (t + dim) % 2 == 0 { 1.0 } else { -1.0 };
                value += sign * 20.0;
            }
            values.push(value);
        }
    }
    values
}

fn cauchy_like_noise(t: usize, dim: usize) -> f64 {
    let mut state = (t as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add((dim as u64).wrapping_mul(1442695040888963407))
        .wrapping_add(1);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51afd7ed558ccd);
    let u = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
    let clamped = u.clamp(1e-6, 1.0 - 1e-6);
    let raw = (std::f64::consts::PI * (clamped - 0.5)).tan();
    raw.clamp(-8.0, 8.0)
}

fn make_heavy_tail_fixture(n: usize, d: usize, true_change: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let base = if t < true_change { -1.0 } else { 1.5 };
            let seasonal = 0.05 * ((t as f64) * 0.09 + dim as f64).cos();
            let heavy_tail = 0.35 * cauchy_like_noise(t, dim);
            values.push(base + seasonal + heavy_tail);
        }
    }
    values
}

#[test]
fn student_t_beats_or_matches_normal_on_outlier_fixture() {
    let n = 240;
    let true_change = 120;
    let min_segment_len = 20;
    let student_t = CostStudentT::default();
    let normal = CostNormalMeanVar::default();

    for d in [1_usize, 8] {
        let values = make_outlier_fixture(n, d, true_change);
        let student_error = split_error(&student_t, &values, n, d, true_change, min_segment_len);
        let normal_error = split_error(&normal, &values, n, d, true_change, min_segment_len);
        assert!(
            student_error <= normal_error,
            "expected Student-t split error <= Normal split error for outlier fixture (d={d}): student={student_error}, normal={normal_error}"
        );
    }
}

#[test]
fn student_t_beats_or_matches_normal_on_heavy_tail_fixture() {
    let n = 320;
    let true_change = 160;
    let min_segment_len = 24;
    let student_t = CostStudentT::default();
    let normal = CostNormalMeanVar::default();

    for d in [1_usize, 8] {
        let values = make_heavy_tail_fixture(n, d, true_change);
        let student_error = split_error(&student_t, &values, n, d, true_change, min_segment_len);
        let normal_error = split_error(&normal, &values, n, d, true_change, min_segment_len);
        assert!(
            student_error <= normal_error,
            "expected Student-t split error <= Normal split error for heavy-tail fixture (d={d}): student={student_error}, normal={normal_error}"
        );
    }
}
