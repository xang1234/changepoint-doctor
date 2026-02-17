// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
};

/// L1 absolute-deviation segment cost around the sample median.
///
/// Slow path: this model stores raw values and recomputes the segment median
/// for every query. Per-segment complexity is O(m) expected with selection
/// (`m = end - start`), not O(1) prefix-stat lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostL1Median {
    pub repro_mode: ReproMode,
}

impl CostL1Median {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self { repro_mode }
    }
}

impl Default for CostL1Median {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Raw-value cache used by the L1 median slow path.
#[derive(Clone, Debug, PartialEq)]
pub struct L1MedianCache {
    values: Vec<f64>, // C-contiguous [t * d + dim]
    n: usize,
    d: usize,
}

fn strided_linear_index(
    t: usize,
    dim: usize,
    row_stride: isize,
    col_stride: isize,
    len: usize,
) -> Result<usize, CpdError> {
    let t_isize = isize::try_from(t).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index overflow: t={t} does not fit into isize"
        ))
    })?;
    let dim_isize = isize::try_from(dim).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index overflow: dim={dim} does not fit into isize"
        ))
    })?;

    let index = t_isize
        .checked_mul(row_stride)
        .and_then(|left| {
            dim_isize
                .checked_mul(col_stride)
                .and_then(|right| left.checked_add(right))
        })
        .ok_or_else(|| {
            CpdError::invalid_input(format!(
                "strided index overflow at t={t}, dim={dim}, row_stride={row_stride}, col_stride={col_stride}"
            ))
        })?;

    let index_usize = usize::try_from(index).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index negative at t={t}, dim={dim}: idx={index}"
        ))
    })?;

    if index_usize >= len {
        return Err(CpdError::invalid_input(format!(
            "strided index out of bounds at t={t}, dim={dim}: idx={index_usize}, len={len}"
        )));
    }

    Ok(index_usize)
}

fn read_value(x: &TimeSeriesView<'_>, t: usize, dim: usize) -> Result<f64, CpdError> {
    match (x.values, x.layout) {
        (DTypeView::F32(values), MemoryLayout::CContiguous) => {
            let idx = t
                .checked_mul(x.d)
                .and_then(|base| base.checked_add(dim))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index overflow"))?;
            values
                .get(idx)
                .map(|v| f64::from(*v))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index out of bounds"))
        }
        (DTypeView::F64(values), MemoryLayout::CContiguous) => {
            let idx = t
                .checked_mul(x.d)
                .and_then(|base| base.checked_add(dim))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index overflow"))?;
            values
                .get(idx)
                .copied()
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index out of bounds"))
        }
        (DTypeView::F32(values), MemoryLayout::FContiguous) => {
            let idx = dim
                .checked_mul(x.n)
                .and_then(|base| base.checked_add(t))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index overflow"))?;
            values
                .get(idx)
                .map(|v| f64::from(*v))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index out of bounds"))
        }
        (DTypeView::F64(values), MemoryLayout::FContiguous) => {
            let idx = dim
                .checked_mul(x.n)
                .and_then(|base| base.checked_add(t))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index overflow"))?;
            values
                .get(idx)
                .copied()
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index out of bounds"))
        }
        (
            DTypeView::F32(values),
            MemoryLayout::Strided {
                row_stride,
                col_stride,
            },
        ) => {
            let idx = strided_linear_index(t, dim, row_stride, col_stride, values.len())?;
            Ok(f64::from(values[idx]))
        }
        (
            DTypeView::F64(values),
            MemoryLayout::Strided {
                row_stride,
                col_stride,
            },
        ) => {
            let idx = strided_linear_index(t, dim, row_stride, col_stride, values.len())?;
            Ok(values[idx])
        }
    }
}

fn cache_overflow_err(n: usize, d: usize) -> CpdError {
    CpdError::resource_limit(format!(
        "cache size overflow while planning L1MedianCache for n={n}, d={d}"
    ))
}

fn median_in_place(values: &mut [f64]) -> f64 {
    debug_assert!(!values.is_empty());
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        let (_, pivot, _) = values.select_nth_unstable_by(mid, |left, right| left.total_cmp(right));
        *pivot
    } else {
        let (lower, upper, _) =
            values.select_nth_unstable_by(mid, |left, right| left.total_cmp(right));
        let upper = *upper;
        let lower = lower
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("even-length median should have non-empty lower partition");
        lower + (upper - lower) * 0.5
    }
}

fn abs_dev_sum(values: &[f64], median: f64, repro_mode: ReproMode) -> f64 {
    if matches!(repro_mode, ReproMode::Strict) {
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for &value in values {
            let y = (value - median).abs() - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        sum
    } else {
        values.iter().map(|value| (value - median).abs()).sum()
    }
}

impl CostModel for CostL1Median {
    type Cache = L1MedianCache;

    fn name(&self) -> &'static str {
        "l1_median"
    }

    fn penalty_params_per_segment(&self) -> usize {
        2
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostL1Median requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostL1Median requires d >= 1; got d=0",
            ));
        }

        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostL1Median does not support missing values: effective_missing_count={}",
                x.n_missing()
            )));
        }

        match x.values {
            DTypeView::F32(_) | DTypeView::F64(_) => Ok(()),
        }
    }

    fn missing_support(&self) -> MissingSupport {
        MissingSupport::Reject
    }

    fn precompute(
        &self,
        x: &TimeSeriesView<'_>,
        policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError> {
        let required_bytes = self.worst_case_cache_bytes(x);

        if matches!(policy, CachePolicy::Approximate { .. }) {
            return Err(CpdError::not_supported(
                "CostL1Median does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostL1Median cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let len =
            x.n.checked_mul(x.d)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let mut values = Vec::with_capacity(len);
        for t in 0..x.n {
            for dim in 0..x.d {
                values.push(read_value(x, t, dim)?);
            }
        }

        Ok(L1MedianCache {
            values,
            n: x.n,
            d: x.d,
        })
    }

    fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize {
        let len = match x.n.checked_mul(x.d) {
            Some(v) => v,
            None => return usize::MAX,
        };
        match len.checked_mul(std::mem::size_of::<f64>()) {
            Some(v) => v,
            None => usize::MAX,
        }
    }

    fn supports_approx_cache(&self) -> bool {
        false
    }

    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64 {
        assert!(
            start < end,
            "segment_cost requires start < end; got start={start}, end={end}"
        );
        assert!(
            end <= cache.n,
            "segment_cost end out of bounds: end={end}, n={} ",
            cache.n
        );

        let segment_len = end - start;
        let mut scratch = Vec::<f64>::with_capacity(segment_len);
        let mut total = 0.0;

        for dim in 0..cache.d {
            scratch.clear();
            for t in start..end {
                scratch.push(cache.values[t * cache.d + dim]);
            }

            let median = median_in_place(scratch.as_mut_slice());
            total += abs_dev_sum(scratch.as_slice(), median, self.repro_mode);
        }

        total.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{CostL1Median, L1MedianCache, cache_overflow_err, read_value};
    use crate::model::CostModel;
    use cpd_core::{
        CachePolicy, DTypeView, MemoryLayout, MissingPolicy, MissingSupport, ReproMode, TimeIndex,
        TimeSeriesView,
    };

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual}, |diff|={diff}, tol={tol}"
        );
    }

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

    fn make_f32_view<'a>(
        values: &'a [f32],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F32(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn naive_median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            let lower = sorted[mid - 1];
            let upper = sorted[mid];
            lower + (upper - lower) * 0.5
        }
    }

    fn naive_l1_univariate(values: &[f64], start: usize, end: usize) -> f64 {
        let segment = &values[start..end];
        let median = naive_median(segment);
        segment.iter().map(|v| (v - median).abs()).sum()
    }

    fn synthetic_multivariate_values(n: usize, d: usize) -> Vec<f64> {
        let mut values = vec![0.0; n * d];
        for t in 0..n {
            for dim in 0..d {
                let drift = 0.17 * t as f64;
                let shift = (dim as f64 - 0.5 * d as f64) * 1.3;
                let wobble = ((t + 3 * dim) as f64).sin() * 0.15;
                values[t * d + dim] = drift + shift + wobble;
            }
        }
        values
    }

    fn dim_series(values: &[f64], n: usize, d: usize, dim: usize) -> Vec<f64> {
        (0..n).map(|t| values[t * d + dim]).collect()
    }

    #[test]
    fn read_value_f32_layout_paths() {
        let c_values: [f32; 6] = [0.0, 1.0, 10.0, 11.0, 20.0, 21.0];
        let c_view = make_f32_view(
            &c_values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        assert_eq!(
            read_value(&c_view, 2, 1).expect("c-layout read should work"),
            21.0
        );

        let f_values: [f32; 6] = [0.0, 10.0, 20.0, 1.0, 11.0, 21.0];
        let f_view = make_f32_view(
            &f_values,
            3,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        assert_eq!(
            read_value(&f_view, 2, 1).expect("f-layout read should work"),
            21.0
        );

        let strided_values: [f32; 6] = [0.0, 10.0, 20.0, 1.0, 11.0, 21.0];
        let strided_view = make_f32_view(
            &strided_values,
            3,
            2,
            MemoryLayout::Strided {
                row_stride: 1,
                col_stride: 3,
            },
            MissingPolicy::Error,
        );
        assert_eq!(
            read_value(&strided_view, 2, 1).expect("strided read should work"),
            21.0
        );
    }

    #[test]
    fn cache_overflow_error_message_is_stable() {
        let err = cache_overflow_err(usize::MAX, usize::MAX);
        assert!(
            err.to_string()
                .contains("cache size overflow while planning L1MedianCache")
        );
    }

    #[test]
    fn trait_contract_and_defaults() {
        let model = CostL1Median::default();
        assert_eq!(model.name(), "l1_median");
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert_eq!(model.penalty_params_per_segment(), 2);
        assert!(!model.supports_approx_cache());
    }

    #[test]
    fn validate_rejects_missing_effective_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let mask = [0_u8, 1, 0, 0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            4,
            1,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("ignore mode with explicit mask should construct");
        let err = CostL1Median::default()
            .validate(&view)
            .expect_err("missing-effective values must be rejected");
        assert!(err.to_string().contains("does not support missing values"));
    }

    #[test]
    fn validate_accepts_clean_f32_and_f64_views() {
        let values_f64 = [0.0, 1.0, 2.0, 3.0];
        let view_f64 = make_f64_view(
            &values_f64,
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        CostL1Median::default()
            .validate(&view_f64)
            .expect("clean f64 view should validate");

        let values_f32 = [0.0f32, 1.0, 2.0, 3.0];
        let view_f32 = make_f32_view(
            &values_f32,
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        CostL1Median::default()
            .validate(&view_f32)
            .expect("clean f32 view should validate");
    }

    #[test]
    fn known_answer_univariate_and_constant_segment() {
        let values = vec![0.0, 0.0, 10.0, 10.0, 10.0, -5.0, -5.0, -5.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let model = CostL1Median::default();
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let expected = naive_l1_univariate(&values, 0, values.len());
        let full_cost = model.segment_cost(&cache, 0, values.len());
        assert_close(full_cost, expected, 1e-10);

        let constant_cost = model.segment_cost(&cache, 2, 5);
        assert_close(constant_cost, 0.0, 1e-12);
    }

    #[test]
    fn segment_cost_matches_naive_on_deterministic_queries() {
        let values = synthetic_multivariate_values(48, 3);
        let view = make_f64_view(
            &values,
            48,
            3,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let model = CostL1Median::default();
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        for (start, end) in [(0, 48), (1, 17), (7, 25), (12, 44), (30, 48)] {
            let mut expected = 0.0;
            for dim in 0..3 {
                let series = dim_series(&values, 48, 3, dim);
                expected += naive_l1_univariate(&series, start, end);
            }
            let actual = model.segment_cost(&cache, start, end);
            assert_close(actual, expected, 1e-10);
        }
    }

    #[test]
    fn layout_coverage_c_f_and_strided() {
        let n = 6;
        let d = 2;
        let c_values = synthetic_multivariate_values(n, d);

        let c_view = make_f64_view(
            &c_values,
            n,
            d,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let mut f_values = vec![0.0; n * d];
        for t in 0..n {
            for dim in 0..d {
                f_values[dim * n + t] = c_values[t * d + dim];
            }
        }
        let f_view = make_f64_view(
            &f_values,
            n,
            d,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );

        let mut strided_values = vec![0.0; n * d];
        for t in 0..n {
            for dim in 0..d {
                strided_values[t + dim * n] = c_values[t * d + dim];
            }
        }
        let strided_view = make_f64_view(
            &strided_values,
            n,
            d,
            MemoryLayout::Strided {
                row_stride: 1,
                col_stride: isize::try_from(n).expect("n should fit in isize"),
            },
            MissingPolicy::Error,
        );

        let model = CostL1Median::default();
        let c_cache = model
            .precompute(&c_view, &CachePolicy::Full)
            .expect("c cache should build");
        let f_cache = model
            .precompute(&f_view, &CachePolicy::Full)
            .expect("f cache should build");
        let s_cache = model
            .precompute(&strided_view, &CachePolicy::Full)
            .expect("strided cache should build");

        for (start, end) in [(0, 6), (1, 4), (2, 6)] {
            let c_cost = model.segment_cost(&c_cache, start, end);
            let f_cost = model.segment_cost(&f_cache, start, end);
            let s_cost = model.segment_cost(&s_cache, start, end);
            assert_close(c_cost, f_cost, 1e-10);
            assert_close(c_cost, s_cost, 1e-10);
        }
    }

    #[test]
    fn cache_policy_behavior_budgeted_and_approximate() {
        let values = synthetic_multivariate_values(16, 3);
        let view = make_f64_view(
            &values,
            16,
            3,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostL1Median::default();
        let required = model.worst_case_cache_bytes(&view);

        let under_budget = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required.saturating_sub(1),
                },
            )
            .expect_err("under-budget request should fail");
        assert!(under_budget.to_string().contains("exceeds budget"));

        model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required,
                },
            )
            .expect("exact budget should pass");

        let approx_err = model
            .precompute(
                &view,
                &CachePolicy::Approximate {
                    max_bytes: required,
                    error_tolerance: 0.1,
                },
            )
            .expect_err("approximate policy should be rejected");
        assert!(
            approx_err
                .to_string()
                .contains("does not support CachePolicy::Approximate")
        );
    }

    #[test]
    fn worst_case_cache_bytes_matches_formula() {
        let view = make_f64_view(
            &[0.0; 5 * 7],
            5,
            7,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostL1Median::default();
        let expected = 5 * 7 * std::mem::size_of::<f64>();
        assert_eq!(model.worst_case_cache_bytes(&view), expected);
    }

    #[test]
    fn strict_mode_matches_balanced_for_same_cache_values() {
        let values = synthetic_multivariate_values(64, 2);
        let view = make_f64_view(
            &values,
            64,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let balanced = CostL1Median::new(ReproMode::Balanced);
        let strict = CostL1Median::new(ReproMode::Strict);

        let balanced_cache = balanced
            .precompute(&view, &CachePolicy::Full)
            .expect("balanced cache should succeed");
        let strict_cache = strict
            .precompute(&view, &CachePolicy::Full)
            .expect("strict cache should succeed");

        let balanced_cost = balanced.segment_cost(&balanced_cache, 5, 57);
        let strict_cost = strict.segment_cost(&strict_cache, 5, 57);
        assert_close(balanced_cost, strict_cost, 1e-9);
    }

    #[test]
    fn segment_cost_panics_when_start_ge_end() {
        let cache = L1MedianCache {
            values: vec![0.0, 1.0],
            n: 2,
            d: 1,
        };
        let model = CostL1Median::default();
        let panic_result = std::panic::catch_unwind(|| model.segment_cost(&cache, 1, 1));
        assert!(panic_result.is_err(), "start >= end must panic");
    }

    #[test]
    fn segment_cost_panics_when_end_exceeds_n() {
        let cache = L1MedianCache {
            values: vec![0.0, 1.0],
            n: 2,
            d: 1,
        };
        let model = CostL1Median::default();
        let panic_result = std::panic::catch_unwind(|| model.segment_cost(&cache, 0, 3));
        assert!(panic_result.is_err(), "end > n must panic");
    }
}
