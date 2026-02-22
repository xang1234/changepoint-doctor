// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sum_squares, prefix_sum_squares_kahan, prefix_sums, prefix_sums_kahan,
};

const LOG_PI: f64 = 1.144_729_885_849_400_2;
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_7e-7,
];

/// Segment scale handling for `CostStudentT`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StudentTScaleMode {
    /// Estimate per-segment scale from the sample variance.
    SegmentVariance,
    /// Estimate per-segment scale from variance and apply `(nu - 2) / nu` when
    /// `nu > 2`; falls back to `SegmentVariance` for `nu <= 2`.
    VarianceMatched,
    /// Use a fixed scale across all segments.
    Fixed(f64),
}

impl Default for StudentTScaleMode {
    fn default() -> Self {
        Self::VarianceMatched
    }
}

/// Student-t negative log-likelihood segment cost (experimental).
///
/// Design and numeric invariants:
/// - `nu` (degrees of freedom) is global and must be finite and `> 0`.
/// - Segment location is the sample mean from cached prefix sums.
/// - Scale is controlled by `scale_mode` and clamped by `min_scale`.
/// - `VarianceMatched` applies `(nu - 2) / nu` only when `nu > 2`; this keeps
///   behavior defined for heavy-tail regimes with infinite variance (`nu <= 2`).
/// - Log terms use `log1p` and finite clamping to keep stress fixtures stable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostStudentT {
    pub nu: f64,
    pub scale_mode: StudentTScaleMode,
    pub min_scale: f64,
    pub repro_mode: ReproMode,
}

impl CostStudentT {
    pub const DEFAULT_NU: f64 = 4.0;
    pub const DEFAULT_MIN_SCALE: f64 = 1.0e-8;

    pub const fn new(repro_mode: ReproMode) -> Self {
        Self {
            nu: Self::DEFAULT_NU,
            scale_mode: StudentTScaleMode::VarianceMatched,
            min_scale: Self::DEFAULT_MIN_SCALE,
            repro_mode,
        }
    }

    pub fn with_params(
        nu: f64,
        scale_mode: StudentTScaleMode,
        min_scale: f64,
        repro_mode: ReproMode,
    ) -> Result<Self, CpdError> {
        let model = Self {
            nu,
            scale_mode,
            min_scale,
            repro_mode,
        };
        model.validate_hyperparameters()?;
        Ok(model)
    }

    fn validate_hyperparameters(&self) -> Result<(), CpdError> {
        if !self.nu.is_finite() || self.nu <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "CostStudentT nu must be finite and > 0; got {}",
                self.nu
            )));
        }

        if !self.min_scale.is_finite() || self.min_scale <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "CostStudentT min_scale must be finite and > 0; got {}",
                self.min_scale
            )));
        }

        if let StudentTScaleMode::Fixed(scale) = self.scale_mode
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(CpdError::invalid_input(format!(
                "CostStudentT fixed scale must be finite and > 0; got {}",
                scale
            )));
        }

        Ok(())
    }

    fn min_scale_sq(&self) -> f64 {
        normalize_positive(self.min_scale * self.min_scale, f64::MIN_POSITIVE)
    }
}

impl Default for CostStudentT {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Cache backing `CostStudentT` segment queries.
#[derive(Clone, Debug, PartialEq)]
pub struct StudentTCache {
    prefix_sum: Vec<f64>,
    prefix_sum_sq: Vec<f64>,
    values_by_dim: Vec<f64>,
    n: usize,
    d: usize,
}

impl StudentTCache {
    fn prefix_len_per_dim(&self) -> usize {
        self.n + 1
    }

    fn dim_offset(&self, dim: usize) -> usize {
        dim * self.prefix_len_per_dim()
    }

    fn series_offset(&self, dim: usize) -> usize {
        dim * self.n
    }
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
        "cache size overflow while planning StudentTCache for n={n}, d={d}"
    ))
}

fn normalize_positive(raw: f64, floor: f64) -> f64 {
    if raw.is_nan() || raw <= floor {
        floor
    } else if raw == f64::INFINITY {
        f64::MAX
    } else {
        raw
    }
}

fn normalize_sse(raw_sse: f64) -> f64 {
    if raw_sse.is_nan() || raw_sse <= 0.0 {
        0.0
    } else if raw_sse == f64::INFINITY {
        f64::MAX
    } else {
        raw_sse
    }
}

fn saturating_finite_add(lhs: f64, rhs: f64) -> f64 {
    let sum = lhs + rhs;
    if sum.is_finite() { sum } else { f64::MAX }
}

fn ln_gamma(z: f64) -> f64 {
    debug_assert!(
        z.is_finite() && z > 0.0,
        "ln_gamma requires z > 0 and finite"
    );

    if z < 1e-8 {
        return -z.ln();
    }

    if z < 0.5 {
        let sin_term = (std::f64::consts::PI * z).sin().abs();
        return std::f64::consts::PI.ln() - sin_term.ln() - ln_gamma(1.0 - z);
    }

    let shifted = z - 1.0;
    let mut x = LANCZOS_COEFFICIENTS[0];
    for (idx, coefficient) in LANCZOS_COEFFICIENTS.iter().copied().enumerate().skip(1) {
        x += coefficient / (shifted + idx as f64);
    }

    let t = shifted + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + x.ln()
}

fn resolve_scale_sq(
    scale_mode: StudentTScaleMode,
    nu: f64,
    sse: f64,
    segment_len: f64,
    min_scale_sq: f64,
) -> f64 {
    let segment_var = normalize_positive(sse / segment_len, min_scale_sq);
    match scale_mode {
        StudentTScaleMode::SegmentVariance => segment_var,
        StudentTScaleMode::VarianceMatched => {
            if nu > 2.0 {
                let matched = segment_var * ((nu - 2.0) / nu);
                normalize_positive(matched, min_scale_sq)
            } else {
                segment_var
            }
        }
        StudentTScaleMode::Fixed(scale) => normalize_positive(scale * scale, min_scale_sq),
    }
}

fn log1p_quadratic_term(value: f64, mean: f64, denom: f64) -> f64 {
    let centered = value - mean;
    let squared = normalize_positive(centered * centered, 0.0);
    let mut ratio = squared / denom;
    if !ratio.is_finite() {
        ratio = f64::MAX;
    }
    ratio.ln_1p()
}

fn sum_log1p_terms(values: &[f64], mean: f64, denom: f64, repro_mode: ReproMode) -> f64 {
    if matches!(repro_mode, ReproMode::Strict) {
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for value in values {
            let term = log1p_quadratic_term(*value, mean, denom);
            let y = term - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        if sum.is_finite() { sum } else { f64::MAX }
    } else {
        let mut sum = 0.0;
        for value in values {
            let term = log1p_quadratic_term(*value, mean, denom);
            sum = saturating_finite_add(sum, term);
        }
        sum
    }
}

impl CostModel for CostStudentT {
    type Cache = StudentTCache;

    fn name(&self) -> &'static str {
        "student_t"
    }

    fn penalty_params_per_segment(&self) -> usize {
        3
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        self.validate_hyperparameters()?;

        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostStudentT requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostStudentT requires d >= 1; got d=0",
            ));
        }
        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostStudentT does not support missing values: effective_missing_count={}",
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
        self.validate_hyperparameters()?;

        let required_bytes = self.worst_case_cache_bytes(x);

        if matches!(policy, CachePolicy::Approximate { .. }) {
            return Err(CpdError::not_supported(
                "CostStudentT does not support CachePolicy::Approximate",
            ));
        }
        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }
        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostStudentT cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_values =
            x.n.checked_mul(x.d)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut prefix_sum = Vec::with_capacity(total_prefix_len);
        let mut prefix_sum_sq = Vec::with_capacity(total_prefix_len);
        let mut values_by_dim = Vec::with_capacity(total_values);

        for dim in 0..x.d {
            let mut series = Vec::with_capacity(x.n);
            for t in 0..x.n {
                let value = read_value(x, t, dim)?;
                series.push(value);
                values_by_dim.push(value);
            }

            let dim_prefix_sum = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&series)
            } else {
                prefix_sums(&series)
            };
            let dim_prefix_sum_sq = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sum_squares_kahan(&series)
            } else {
                prefix_sum_squares(&series)
            };

            debug_assert_eq!(dim_prefix_sum.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_sum_sq.len(), prefix_len_per_dim);

            prefix_sum.extend_from_slice(&dim_prefix_sum);
            prefix_sum_sq.extend_from_slice(&dim_prefix_sum_sq);
        }

        Ok(StudentTCache {
            prefix_sum,
            prefix_sum_sq,
            values_by_dim,
            n: x.n,
            d: x.d,
        })
    }

    fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize {
        let prefix_len_per_dim = match x.n.checked_add(1) {
            Some(value) => value,
            None => return usize::MAX,
        };
        let total_prefix_len = match prefix_len_per_dim.checked_mul(x.d) {
            Some(value) => value,
            None => return usize::MAX,
        };
        let total_values = match x.n.checked_mul(x.d) {
            Some(value) => value,
            None => return usize::MAX,
        };

        let prefix_bytes_single = match total_prefix_len.checked_mul(std::mem::size_of::<f64>()) {
            Some(value) => value,
            None => return usize::MAX,
        };
        let prefix_bytes = match prefix_bytes_single.checked_mul(2) {
            Some(value) => value,
            None => return usize::MAX,
        };
        let value_bytes = match total_values.checked_mul(std::mem::size_of::<f64>()) {
            Some(value) => value,
            None => return usize::MAX,
        };

        match prefix_bytes.checked_add(value_bytes) {
            Some(value) => value,
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

        let segment_len = (end - start) as f64;
        let min_scale_sq = self.min_scale_sq();
        let half_nu = 0.5 * self.nu;
        let base_constant =
            ln_gamma(half_nu) - ln_gamma(half_nu + 0.5) + 0.5 * (self.nu.ln() + LOG_PI);
        let tail_weight = 0.5 * (self.nu + 1.0);

        let mut total = 0.0;
        for dim in 0..cache.d {
            let offset = cache.dim_offset(dim);
            let sum = cache.prefix_sum[offset + end] - cache.prefix_sum[offset + start];
            let sum_sq = cache.prefix_sum_sq[offset + end] - cache.prefix_sum_sq[offset + start];
            let mean = sum / segment_len;
            let sse = normalize_sse(sum_sq - (sum * sum) / segment_len);
            let scale_sq =
                resolve_scale_sq(self.scale_mode, self.nu, sse, segment_len, min_scale_sq);

            let log_sigma = 0.5 * scale_sq.ln();
            let denom = normalize_positive(self.nu * scale_sq, f64::MIN_POSITIVE);
            let series_offset = cache.series_offset(dim);
            let values = &cache.values_by_dim[series_offset + start..series_offset + end];
            let log1p_sum = sum_log1p_terms(values, mean, denom, self.repro_mode);
            let dim_cost = segment_len * (base_constant + log_sigma) + tail_weight * log1p_sum;
            let finite_dim_cost = if dim_cost.is_finite() {
                dim_cost
            } else {
                f64::MAX
            };
            total = saturating_finite_add(total, finite_dim_cost);
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CostStudentT, StudentTCache, StudentTScaleMode, cache_overflow_err, ln_gamma,
        normalize_sse, read_value, resolve_scale_sq, strided_linear_index,
    };
    use crate::model::CostModel;
    use cpd_core::{
        CachePolicy, CpdError, DTypeView, MemoryLayout, MissingPolicy, MissingSupport, ReproMode,
        TimeIndex, TimeSeriesView,
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

    fn synthetic_multivariate_values(n: usize, d: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n * d);
        for t in 0..n {
            for dim in 0..d {
                let x = t as f64 + 1.0;
                let y = dim as f64 + 1.0;
                values.push((x * y) + (0.03 * x).sin() + (0.07 * y).cos());
            }
        }
        values
    }

    fn dim_series(values: &[f64], n: usize, d: usize, dim: usize) -> Vec<f64> {
        (0..n).map(|t| values[t * d + dim]).collect()
    }

    fn naive_student_t(
        values: &[f64],
        nu: f64,
        scale_mode: StudentTScaleMode,
        min_scale: f64,
    ) -> f64 {
        let m = values.len() as f64;
        let sum = values.iter().sum::<f64>();
        let sum_sq = values.iter().map(|value| value * value).sum::<f64>();
        let mean = sum / m;
        let sse = normalize_sse(sum_sq - (sum * sum) / m);
        let min_scale_sq = (min_scale * min_scale).max(f64::MIN_POSITIVE);
        let scale_sq = resolve_scale_sq(scale_mode, nu, sse, m, min_scale_sq);
        let denom = (nu * scale_sq).max(f64::MIN_POSITIVE);

        let base = ln_gamma(0.5 * nu) - ln_gamma(0.5 * (nu + 1.0))
            + 0.5 * (nu.ln() + std::f64::consts::PI.ln());
        let tail_weight = 0.5 * (nu + 1.0);
        let log_sigma = 0.5 * scale_sq.ln();
        let log1p_sum = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                let squared = centered * centered;
                let ratio = (squared / denom).min(f64::MAX);
                ratio.ln_1p()
            })
            .sum::<f64>();

        m * (base + log_sigma) + tail_weight * log1p_sum
    }

    #[test]
    fn helper_functions_cover_edge_paths() {
        let err_t = strided_linear_index(usize::MAX, 0, 1, 1, 8).expect_err("t overflow expected");
        assert!(matches!(err_t, CpdError::InvalidInput(_)));
        assert!(err_t.to_string().contains("t="));

        let err_dim =
            strided_linear_index(0, usize::MAX, 1, 1, 8).expect_err("dim overflow expected");
        assert!(matches!(err_dim, CpdError::InvalidInput(_)));
        assert!(err_dim.to_string().contains("dim="));

        let err_neg = strided_linear_index(1, 0, -1, 0, 8).expect_err("negative index expected");
        assert!(matches!(err_neg, CpdError::InvalidInput(_)));
        assert!(err_neg.to_string().contains("negative"));

        let err_oob =
            strided_linear_index(3, 0, 2, 0, 5).expect_err("out-of-bounds index expected");
        assert!(matches!(err_oob, CpdError::InvalidInput(_)));
        assert!(err_oob.to_string().contains("out of bounds"));
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert_close(ln_gamma(1.0), 0.0, 1e-14);
        assert_close(ln_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), 1e-12);
        assert_close(ln_gamma(5.0), 24.0_f64.ln(), 1e-12);
    }

    #[test]
    fn read_value_f32_layout_paths_and_errors() {
        let c_values = [1.5_f32, 10.5, 2.5, 20.5];
        let c_view = make_f32_view(
            &c_values,
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&c_view, 1, 0).expect("C f32 read should succeed"),
            2.5,
            1e-12,
        );
        let c_oob = read_value(&c_view, 1, 2).expect_err("C f32 oob expected");
        assert!(c_oob.to_string().contains("out of bounds"));

        let f_values = [1.5_f32, 2.5, 10.5, 20.5];
        let f_view = make_f32_view(
            &f_values,
            2,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&f_view, 1, 0).expect("F f32 read should succeed"),
            2.5,
            1e-12,
        );
        let f_oob = read_value(&f_view, 1, 2).expect_err("F f32 oob expected");
        assert!(f_oob.to_string().contains("out of bounds"));
    }

    #[test]
    fn cache_overflow_error_message_is_stable() {
        let err = cache_overflow_err(7, 11);
        assert!(matches!(err, CpdError::ResourceLimit(_)));
        assert!(err.to_string().contains("n=7, d=11"));
    }

    #[test]
    fn trait_contract_and_defaults() {
        let model = CostStudentT::default();
        assert_eq!(model.name(), "student_t");
        assert_eq!(model.repro_mode, ReproMode::Balanced);
        assert_eq!(model.nu, CostStudentT::DEFAULT_NU);
        assert_eq!(model.scale_mode, StudentTScaleMode::VarianceMatched);
        assert_eq!(model.min_scale, CostStudentT::DEFAULT_MIN_SCALE);
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
        assert_eq!(model.penalty_params_per_segment(), 3);
    }

    #[test]
    fn with_params_validates_hyperparameters() {
        let nu_err = CostStudentT::with_params(
            0.0,
            StudentTScaleMode::SegmentVariance,
            1e-6,
            ReproMode::Balanced,
        )
        .expect_err("nu <= 0 should fail");
        assert!(matches!(nu_err, CpdError::InvalidInput(_)));

        let scale_err = CostStudentT::with_params(
            4.0,
            StudentTScaleMode::Fixed(0.0),
            1e-6,
            ReproMode::Balanced,
        )
        .expect_err("fixed scale <= 0 should fail");
        assert!(matches!(scale_err, CpdError::InvalidInput(_)));

        let min_scale_err = CostStudentT::with_params(
            4.0,
            StudentTScaleMode::SegmentVariance,
            0.0,
            ReproMode::Balanced,
        )
        .expect_err("min_scale <= 0 should fail");
        assert!(matches!(min_scale_err, CpdError::InvalidInput(_)));
    }

    #[test]
    fn validate_rejects_missing_effective_values() {
        let values = [1.0, f64::NAN, 3.0];
        let view = make_f64_view(
            &values,
            3,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Ignore,
        );
        let err = CostStudentT::default()
            .validate(&view)
            .expect_err("missing values should be rejected");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("missing values"));
    }

    #[test]
    fn known_answer_univariate_matches_naive() {
        let model = CostStudentT::with_params(
            4.0,
            StudentTScaleMode::SegmentVariance,
            1e-8,
            ReproMode::Balanced,
        )
        .expect("parameters should be valid");
        let values = [1.0, 2.0, 3.0, 5.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let actual = model.segment_cost(&cache, 0, values.len());
        let expected = naive_student_t(&values, 4.0, StudentTScaleMode::SegmentVariance, 1e-8);
        assert_close(actual, expected, 1e-12);
    }

    #[test]
    fn variance_matched_falls_back_for_nu_lte_two() {
        let nu = 1.5;
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let matched = CostStudentT::with_params(
            nu,
            StudentTScaleMode::VarianceMatched,
            1e-8,
            ReproMode::Balanced,
        )
        .expect("params should be valid");
        let segment_var = CostStudentT::with_params(
            nu,
            StudentTScaleMode::SegmentVariance,
            1e-8,
            ReproMode::Balanced,
        )
        .expect("params should be valid");

        let matched_cache = matched
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let segment_cache = segment_var
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let matched_cost = matched.segment_cost(&matched_cache, 0, values.len());
        let segment_cost = segment_var.segment_cost(&segment_cache, 0, values.len());
        assert_close(matched_cost, segment_cost, 1e-12);
    }

    #[test]
    fn segment_cost_matches_naive_on_deterministic_queries_and_nu_regimes() {
        let values = [
            0.5, -0.2, 1.0, 1.8, -2.3, 3.1, -1.7, 0.9, 2.2, -0.4, 1.3, -3.0,
        ];
        let queries = [(0, 4), (2, 8), (0, values.len())];
        let modes = [
            StudentTScaleMode::SegmentVariance,
            StudentTScaleMode::VarianceMatched,
            StudentTScaleMode::Fixed(1.25),
        ];

        for nu in [0.5, 2.0, 8.0, 30.0] {
            for scale_mode in modes {
                let model = CostStudentT::with_params(nu, scale_mode, 1e-8, ReproMode::Balanced)
                    .expect("params should be valid for deterministic query comparisons");
                let view = make_f64_view(
                    &values,
                    values.len(),
                    1,
                    MemoryLayout::CContiguous,
                    MissingPolicy::Error,
                );
                let cache = model
                    .precompute(&view, &CachePolicy::Full)
                    .expect("precompute should succeed");

                for (start, end) in queries {
                    let fast = model.segment_cost(&cache, start, end);
                    let naive = naive_student_t(&values[start..end], nu, scale_mode, 1e-8);
                    assert_close(fast, naive, 1e-10);
                }
            }
        }
    }

    #[test]
    fn multivariate_matches_univariate_sum_for_d1_d4_d16() {
        let model = CostStudentT::with_params(
            6.0,
            StudentTScaleMode::VarianceMatched,
            1e-8,
            ReproMode::Balanced,
        )
        .expect("params should be valid");
        let n = 11;
        let start = 2;
        let end = 10;

        for d in [1_usize, 4, 16] {
            let values = synthetic_multivariate_values(n, d);
            let view = make_f64_view(
                &values,
                n,
                d,
                MemoryLayout::CContiguous,
                MissingPolicy::Error,
            );
            let cache = model
                .precompute(&view, &CachePolicy::Full)
                .expect("multivariate precompute should succeed");
            let multivariate = model.segment_cost(&cache, start, end);

            let mut per_dimension_sum = 0.0;
            for dim in 0..d {
                let series = dim_series(&values, n, d, dim);
                let dim_view = make_f64_view(
                    &series,
                    n,
                    1,
                    MemoryLayout::CContiguous,
                    MissingPolicy::Error,
                );
                let dim_cache = model
                    .precompute(&dim_view, &CachePolicy::Full)
                    .expect("univariate precompute should succeed");
                per_dimension_sum += model.segment_cost(&dim_cache, start, end);
            }

            assert_close(multivariate, per_dimension_sum, 1e-9);
        }
    }

    #[test]
    fn stress_segments_remain_finite_across_nu_regimes() {
        let values = [1.0e150, -1.0e150, 2.0e150, -2.0e150, 3.0e150, -3.0e150];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        for nu in [0.5, 2.0, 10.0, 100.0] {
            let model = CostStudentT::with_params(
                nu,
                StudentTScaleMode::VarianceMatched,
                1e-8,
                ReproMode::Balanced,
            )
            .expect("params should be valid");
            let cache = model
                .precompute(&view, &CachePolicy::Full)
                .expect("precompute should succeed");
            let cost = model.segment_cost(&cache, 0, values.len());
            assert!(
                cost.is_finite(),
                "expected finite cost for nu={nu}, got {cost}"
            );
        }
    }

    #[test]
    fn cache_policy_behavior_budgeted_and_approximate() {
        let model = CostStudentT::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let required = model.worst_case_cache_bytes(&view);
        assert!(required > 0);

        model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required,
                },
            )
            .expect("budget equal to required should succeed");

        let budget_err = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required - 1,
                },
            )
            .expect_err("budget below required should fail");
        assert!(matches!(budget_err, CpdError::ResourceLimit(_)));

        let approx_err = model
            .precompute(
                &view,
                &CachePolicy::Approximate {
                    max_bytes: required,
                    error_tolerance: 0.1,
                },
            )
            .expect_err("approximate cache should be unsupported");
        assert!(matches!(approx_err, CpdError::NotSupported(_)));
    }

    #[test]
    fn worst_case_cache_bytes_matches_formula() {
        let model = CostStudentT::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let prefix_len_per_dim = view.n + 1;
        let total_prefix = prefix_len_per_dim * view.d;
        let total_values = view.n * view.d;
        let expected = 2 * total_prefix * std::mem::size_of::<f64>()
            + total_values * std::mem::size_of::<f64>();

        assert_eq!(model.worst_case_cache_bytes(&view), expected);
    }

    #[test]
    fn strict_and_balanced_modes_are_close_on_same_data() {
        let values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let balanced = CostStudentT::with_params(
            5.0,
            StudentTScaleMode::VarianceMatched,
            1e-8,
            ReproMode::Balanced,
        )
        .expect("params should be valid");
        let strict = CostStudentT::with_params(
            5.0,
            StudentTScaleMode::VarianceMatched,
            1e-8,
            ReproMode::Strict,
        )
        .expect("params should be valid");

        let balanced_cache = balanced
            .precompute(&view, &CachePolicy::Full)
            .expect("balanced precompute should succeed");
        let strict_cache = strict
            .precompute(&view, &CachePolicy::Full)
            .expect("strict precompute should succeed");

        let balanced_cost = balanced.segment_cost(&balanced_cache, 1, values.len());
        let strict_cost = strict.segment_cost(&strict_cache, 1, values.len());
        assert!((balanced_cost - strict_cost).abs() <= 1e-9);
    }

    #[test]
    fn segment_cost_panics_when_start_ge_end() {
        let model = CostStudentT::default();
        let cache = StudentTCache {
            prefix_sum: vec![0.0, 1.0],
            prefix_sum_sq: vec![0.0, 1.0],
            values_by_dim: vec![1.0],
            n: 1,
            d: 1,
        };
        let panic = std::panic::catch_unwind(|| model.segment_cost(&cache, 1, 1));
        assert!(panic.is_err());
    }

    #[test]
    fn segment_cost_panics_when_end_exceeds_n() {
        let model = CostStudentT::default();
        let cache = StudentTCache {
            prefix_sum: vec![0.0, 1.0],
            prefix_sum_sq: vec![0.0, 1.0],
            values_by_dim: vec![1.0],
            n: 1,
            d: 1,
        };
        let panic = std::panic::catch_unwind(|| model.segment_cost(&cache, 0, 2));
        assert!(panic.is_err());
    }
}
