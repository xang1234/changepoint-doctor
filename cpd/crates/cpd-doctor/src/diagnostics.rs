// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use cpd_core::{
    CpdError, DTypeView, MemoryLayout, TimeIndex, TimeSeriesView, prefix_sum_squares, prefix_sums,
    stable_mean, stable_variance,
};

const DEFAULT_SUBSAMPLE_THRESHOLD: usize = 100_000;
const DEFAULT_SUBSAMPLE_TARGET_MIN: usize = 10_000;
const DEFAULT_SUBSAMPLE_TARGET_MAX: usize = 50_000;
const DEFAULT_MAX_AUTOCORR_LAG: usize = 128;
const DEFAULT_LAG_K: usize = 8;
const DEFAULT_ROLLING_WINDOW: usize = 128;
const DEFAULT_MIN_VALID_PER_DIM: usize = 32;
const DEFAULT_EPSILON: f64 = 1.0e-12;
const NORMAL_CONSISTENCY: f64 = 1.4826;

#[derive(Clone, Debug, PartialEq)]
pub struct DoctorDiagnosticsConfig {
    pub subsample_threshold: usize,
    pub subsample_target_min: usize,
    pub subsample_target_max: usize,
    pub max_autocorr_lag: usize,
    pub lag_k: usize,
    pub rolling_window: usize,
    pub min_valid_per_dim: usize,
    pub epsilon: f64,
}

impl Default for DoctorDiagnosticsConfig {
    fn default() -> Self {
        Self {
            subsample_threshold: DEFAULT_SUBSAMPLE_THRESHOLD,
            subsample_target_min: DEFAULT_SUBSAMPLE_TARGET_MIN,
            subsample_target_max: DEFAULT_SUBSAMPLE_TARGET_MAX,
            max_autocorr_lag: DEFAULT_MAX_AUTOCORR_LAG,
            lag_k: DEFAULT_LAG_K,
            rolling_window: DEFAULT_ROLLING_WINDOW,
            min_valid_per_dim: DEFAULT_MIN_VALID_PER_DIM,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingPattern {
    None,
    Random,
    Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DominantPeriodHint {
    pub period: usize,
    pub strength: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DimensionDiagnostics {
    pub dimension: usize,
    pub valid_count: usize,
    pub total_count: usize,
    pub missing_fraction: f64,
    pub kurtosis_proxy: Option<f64>,
    pub outlier_rate_iqr: Option<f64>,
    pub mad_to_std_ratio: Option<f64>,
    pub lag1_autocorr: Option<f64>,
    pub lagk_autocorr: Option<f64>,
    pub pacf_lagk_proxy: Option<f64>,
    pub dominant_period: Option<DominantPeriodHint>,
    pub residual_lag1_autocorr: Option<f64>,
    pub rolling_mean_drift: Option<f64>,
    pub rolling_variance_drift: Option<f64>,
    pub regime_change_proxy: Option<f64>,
    pub change_density_score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticsSummary {
    pub valid_dimensions: usize,
    pub nan_rate: f64,
    pub longest_nan_run: usize,
    pub missing_pattern: MissingPattern,
    pub kurtosis_proxy: f64,
    pub outlier_rate_iqr: f64,
    pub mad_to_std_ratio: f64,
    pub lag1_autocorr: f64,
    pub lagk_autocorr: f64,
    pub pacf_lagk_proxy: f64,
    pub residual_lag1_autocorr: f64,
    pub rolling_mean_drift: f64,
    pub rolling_variance_drift: f64,
    pub regime_change_proxy: f64,
    pub change_density_score: f64,
    pub dominant_period_hints: Vec<DominantPeriodHint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticsReport {
    pub n: usize,
    pub d: usize,
    pub sampling_rate_hz: Option<f64>,
    pub used_subsampling: bool,
    pub subsample_stride: usize,
    pub subsample_n: usize,
    pub summary: DiagnosticsSummary,
    pub per_dimension: Vec<DimensionDiagnostics>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
struct ViewAccessor<'a> {
    x: &'a TimeSeriesView<'a>,
    source_len: usize,
}

impl<'a> ViewAccessor<'a> {
    fn new(x: &'a TimeSeriesView<'a>) -> Self {
        let source_len = match x.values {
            DTypeView::F32(values) => values.len(),
            DTypeView::F64(values) => values.len(),
        };
        Self { x, source_len }
    }

    fn value_and_missing(&self, t: usize, j: usize) -> Result<(f64, bool), CpdError> {
        let src = source_index(self.x.layout, self.x.n, self.x.d, t, j)?;
        if src >= self.source_len {
            return Err(CpdError::invalid_input(format!(
                "source index out of bounds for doctor diagnostics input: idx={src}, len={}, t={t}, j={j}, layout={:?}",
                self.source_len, self.x.layout
            )));
        }
        let value = match self.x.values {
            DTypeView::F32(values) => f64::from(values[src]),
            DTypeView::F64(values) => values[src],
        };
        let mask_missing = self
            .x
            .missing_mask
            .map(|mask| mask[src] == 1)
            .unwrap_or(false);
        let missing = mask_missing || value.is_nan();
        Ok((if missing { f64::NAN } else { value }, missing))
    }
}

#[derive(Clone, Copy, Debug)]
struct MissingOverview {
    nan_rate: f64,
    longest_nan_run: usize,
    missing_pattern: MissingPattern,
}

#[derive(Default)]
struct MetricAccumulator {
    kurtosis_proxy: Vec<f64>,
    outlier_rate_iqr: Vec<f64>,
    mad_to_std_ratio: Vec<f64>,
    lag1_autocorr: Vec<f64>,
    lagk_autocorr: Vec<f64>,
    pacf_lagk_proxy: Vec<f64>,
    residual_lag1_autocorr: Vec<f64>,
    rolling_mean_drift: Vec<f64>,
    rolling_variance_drift: Vec<f64>,
    regime_change_proxy: Vec<f64>,
    change_density_score: Vec<f64>,
}

impl MetricAccumulator {
    fn push(&mut self, metrics: &DimensionMetrics) {
        self.kurtosis_proxy.push(metrics.kurtosis_proxy);
        self.outlier_rate_iqr.push(metrics.outlier_rate_iqr);
        self.mad_to_std_ratio.push(metrics.mad_to_std_ratio);
        self.lag1_autocorr.push(metrics.lag1_autocorr);
        self.lagk_autocorr.push(metrics.lagk_autocorr);
        self.pacf_lagk_proxy.push(metrics.pacf_lagk_proxy);
        self.residual_lag1_autocorr
            .push(metrics.residual_lag1_autocorr);
        self.rolling_mean_drift.push(metrics.rolling_mean_drift);
        self.rolling_variance_drift
            .push(metrics.rolling_variance_drift);
        self.regime_change_proxy.push(metrics.regime_change_proxy);
        self.change_density_score.push(metrics.change_density_score);
    }

    fn valid_dimensions(&self) -> usize {
        self.kurtosis_proxy.len()
    }
}

#[derive(Clone, Debug)]
struct DimensionMetrics {
    kurtosis_proxy: f64,
    outlier_rate_iqr: f64,
    mad_to_std_ratio: f64,
    lag1_autocorr: f64,
    lagk_autocorr: f64,
    pacf_lagk_proxy: f64,
    residual_lag1_autocorr: f64,
    rolling_mean_drift: f64,
    rolling_variance_drift: f64,
    regime_change_proxy: f64,
    change_density_score: f64,
    dominant_period: Option<DominantPeriodHint>,
}

pub fn compute_diagnostics(
    x: &TimeSeriesView<'_>,
    cfg: &DoctorDiagnosticsConfig,
) -> Result<DiagnosticsReport, CpdError> {
    validate_diagnostics_config(cfg)?;

    let accessor = ViewAccessor::new(x);
    let missing = compute_missing_overview(x, &accessor)?;
    let sampling_rate_hz = estimate_sampling_rate_hz(x.time, cfg.epsilon);
    let (sample_indices, used_subsampling, subsample_stride) = build_subsample_indices(x.n, cfg);
    let subsample_n = sample_indices.len();

    let mut per_dimension = Vec::with_capacity(x.d);
    let mut warnings = Vec::new();
    let mut accumulator = MetricAccumulator::default();
    let mut period_votes = BTreeMap::<usize, f64>::new();

    for j in 0..x.d {
        let mut series = Vec::with_capacity(subsample_n);
        for &t in &sample_indices {
            let (value, is_missing) = accessor.value_and_missing(t, j)?;
            if is_missing {
                series.push(f64::NAN);
            } else {
                series.push(value);
            }
        }

        let valid_count = series.iter().filter(|v| !v.is_nan()).count();
        let missing_fraction = if subsample_n == 0 {
            0.0
        } else {
            1.0 - (valid_count as f64 / subsample_n as f64)
        };

        if valid_count < cfg.min_valid_per_dim {
            warnings.push(format!(
                "dimension {j} has only {valid_count} valid samples in diagnostics subsample (minimum required: {})",
                cfg.min_valid_per_dim
            ));
            per_dimension.push(DimensionDiagnostics {
                dimension: j,
                valid_count,
                total_count: subsample_n,
                missing_fraction,
                kurtosis_proxy: None,
                outlier_rate_iqr: None,
                mad_to_std_ratio: None,
                lag1_autocorr: None,
                lagk_autocorr: None,
                pacf_lagk_proxy: None,
                dominant_period: None,
                residual_lag1_autocorr: None,
                rolling_mean_drift: None,
                rolling_variance_drift: None,
                regime_change_proxy: None,
                change_density_score: None,
            });
            continue;
        }

        let metrics = compute_dimension_metrics(&series, cfg);
        if let Some(hint) = &metrics.dominant_period {
            *period_votes.entry(hint.period).or_insert(0.0) += hint.strength;
        }
        accumulator.push(&metrics);
        per_dimension.push(DimensionDiagnostics {
            dimension: j,
            valid_count,
            total_count: subsample_n,
            missing_fraction,
            kurtosis_proxy: Some(metrics.kurtosis_proxy),
            outlier_rate_iqr: Some(metrics.outlier_rate_iqr),
            mad_to_std_ratio: Some(metrics.mad_to_std_ratio),
            lag1_autocorr: Some(metrics.lag1_autocorr),
            lagk_autocorr: Some(metrics.lagk_autocorr),
            pacf_lagk_proxy: Some(metrics.pacf_lagk_proxy),
            dominant_period: metrics.dominant_period,
            residual_lag1_autocorr: Some(metrics.residual_lag1_autocorr),
            rolling_mean_drift: Some(metrics.rolling_mean_drift),
            rolling_variance_drift: Some(metrics.rolling_variance_drift),
            regime_change_proxy: Some(metrics.regime_change_proxy),
            change_density_score: Some(metrics.change_density_score),
        });
    }

    if accumulator.valid_dimensions() == 0 {
        return Err(CpdError::invalid_input(format!(
            "doctor diagnostics require at least one dimension with >= {} valid samples",
            cfg.min_valid_per_dim
        )));
    }

    let summary = DiagnosticsSummary {
        valid_dimensions: accumulator.valid_dimensions(),
        nan_rate: missing.nan_rate,
        longest_nan_run: missing.longest_nan_run,
        missing_pattern: missing.missing_pattern,
        kurtosis_proxy: median(&accumulator.kurtosis_proxy),
        outlier_rate_iqr: median(&accumulator.outlier_rate_iqr),
        mad_to_std_ratio: median(&accumulator.mad_to_std_ratio),
        lag1_autocorr: median(&accumulator.lag1_autocorr),
        lagk_autocorr: median(&accumulator.lagk_autocorr),
        pacf_lagk_proxy: median(&accumulator.pacf_lagk_proxy),
        residual_lag1_autocorr: median(&accumulator.residual_lag1_autocorr),
        rolling_mean_drift: median(&accumulator.rolling_mean_drift),
        rolling_variance_drift: median(&accumulator.rolling_variance_drift),
        regime_change_proxy: median(&accumulator.regime_change_proxy),
        change_density_score: median(&accumulator.change_density_score),
        dominant_period_hints: summarize_period_votes(period_votes),
    };

    Ok(DiagnosticsReport {
        n: x.n,
        d: x.d,
        sampling_rate_hz,
        used_subsampling,
        subsample_stride,
        subsample_n,
        summary,
        per_dimension,
        warnings,
    })
}

fn validate_diagnostics_config(cfg: &DoctorDiagnosticsConfig) -> Result<(), CpdError> {
    if cfg.subsample_threshold == 0 {
        return Err(CpdError::invalid_input(
            "DoctorDiagnosticsConfig.subsample_threshold must be >= 1",
        ));
    }
    if cfg.subsample_target_min == 0 {
        return Err(CpdError::invalid_input(
            "DoctorDiagnosticsConfig.subsample_target_min must be >= 1",
        ));
    }
    if cfg.subsample_target_max < cfg.subsample_target_min {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.subsample_target_max must be >= subsample_target_min, got max={} < min={}",
            cfg.subsample_target_max, cfg.subsample_target_min
        )));
    }
    if cfg.max_autocorr_lag < 2 {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.max_autocorr_lag must be >= 2, got {}",
            cfg.max_autocorr_lag
        )));
    }
    if cfg.lag_k == 0 {
        return Err(CpdError::invalid_input(
            "DoctorDiagnosticsConfig.lag_k must be >= 1",
        ));
    }
    if cfg.lag_k > cfg.max_autocorr_lag {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.lag_k must be <= max_autocorr_lag, got lag_k={} > max_autocorr_lag={}",
            cfg.lag_k, cfg.max_autocorr_lag
        )));
    }
    if cfg.rolling_window < 2 {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.rolling_window must be >= 2, got {}",
            cfg.rolling_window
        )));
    }
    if cfg.min_valid_per_dim < 3 {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.min_valid_per_dim must be >= 3, got {}",
            cfg.min_valid_per_dim
        )));
    }
    if !cfg.epsilon.is_finite() || cfg.epsilon <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "DoctorDiagnosticsConfig.epsilon must be finite and > 0, got {}",
            cfg.epsilon
        )));
    }

    Ok(())
}

fn build_subsample_indices(n: usize, cfg: &DoctorDiagnosticsConfig) -> (Vec<usize>, bool, usize) {
    let mut used_subsampling = false;
    let mut stride = 1usize;

    if n > cfg.subsample_threshold {
        used_subsampling = true;
        let base_target = (n / 10).max(1);
        let target = base_target
            .clamp(cfg.subsample_target_min, cfg.subsample_target_max)
            .min(n);
        // Use floor division so the realized sample count stays at/above the target.
        stride = (n / target).max(1);
    }

    let mut indices = Vec::with_capacity(n.div_ceil(stride) + 1);
    let mut t = 0usize;
    while t < n {
        indices.push(t);
        t = t.saturating_add(stride);
    }
    if let Some(&last) = indices.last()
        && last != n - 1
    {
        indices.push(n - 1);
    }

    (indices, used_subsampling, stride)
}

fn estimate_sampling_rate_hz(time: TimeIndex<'_>, epsilon: f64) -> Option<f64> {
    match time {
        TimeIndex::None => None,
        TimeIndex::Uniform { dt_ns, .. } => {
            if dt_ns <= 0 {
                None
            } else {
                let rate = 1.0e9 / dt_ns as f64;
                rate.is_finite().then_some(rate)
            }
        }
        TimeIndex::Explicit(timestamps) => {
            if timestamps.len() < 2 {
                return None;
            }
            let mut diffs = Vec::with_capacity(timestamps.len() - 1);
            for w in timestamps.windows(2) {
                let diff = w[1].checked_sub(w[0]);
                if let Some(delta) = diff
                    && delta > 0
                {
                    diffs.push(delta as f64);
                }
            }
            if diffs.is_empty() {
                return None;
            }
            let dt = median(&diffs).max(epsilon);
            let rate = 1.0e9 / dt;
            rate.is_finite().then_some(rate)
        }
    }
}

fn compute_missing_overview(
    x: &TimeSeriesView<'_>,
    accessor: &ViewAccessor<'_>,
) -> Result<MissingOverview, CpdError> {
    let total_values = x.n.checked_mul(x.d).ok_or_else(|| {
        CpdError::invalid_input("n*d overflow while computing doctor diagnostics")
    })?;

    let nan_rate = if total_values == 0 {
        0.0
    } else {
        x.n_missing() as f64 / total_values as f64
    };

    let mut longest_nan_run = 0usize;
    let mut block_weight = 0usize;
    let mut random_weight = 0usize;

    for j in 0..x.d {
        let mut dim_missing_count = 0usize;
        let mut dim_longest_run = 0usize;
        let mut dim_current_run = 0usize;

        for t in 0..x.n {
            let (_, is_missing) = accessor.value_and_missing(t, j)?;
            if is_missing {
                dim_missing_count += 1;
                dim_current_run += 1;
                dim_longest_run = dim_longest_run.max(dim_current_run);
            } else {
                dim_current_run = 0;
            }
        }

        longest_nan_run = longest_nan_run.max(dim_longest_run);
        if dim_missing_count == 0 {
            continue;
        }

        let block_threshold = 8usize.max(dim_missing_count.div_ceil(4));
        if dim_longest_run >= block_threshold {
            block_weight = block_weight.saturating_add(dim_missing_count);
        } else {
            random_weight = random_weight.saturating_add(dim_missing_count);
        }
    }

    let missing_pattern = if block_weight == 0 && random_weight == 0 {
        MissingPattern::None
    } else if block_weight >= random_weight {
        MissingPattern::Block
    } else {
        MissingPattern::Random
    };

    Ok(MissingOverview {
        nan_rate,
        longest_nan_run,
        missing_pattern,
    })
}

fn compute_dimension_metrics(series: &[f64], cfg: &DoctorDiagnosticsConfig) -> DimensionMetrics {
    let valid = collect_valid(series);
    let mean = stable_mean(&valid);
    let variance = stable_variance(&valid, mean);
    let std = variance.sqrt();

    let kurtosis_proxy = moment_kurtosis_proxy(&valid, mean, variance, cfg.epsilon);
    let outlier_rate_iqr = iqr_outlier_rate(&valid);
    let mad_to_std_ratio = mad_to_std_ratio(&valid, std, cfg.epsilon);
    let lag1_autocorr = autocorr_at_lag(series, 1, cfg.epsilon);
    let lagk_autocorr = autocorr_at_lag(series, cfg.lag_k, cfg.epsilon);
    let pacf_lagk_proxy = pacf_lagk_proxy(&valid, cfg.lag_k, cfg.epsilon);
    let dominant_period = dominant_period_hint(series, cfg.max_autocorr_lag, cfg.epsilon);
    let residuals = detrend_linear(series, cfg.epsilon);
    let residual_lag1_autocorr = autocorr_at_lag(&residuals, 1, cfg.epsilon);
    let (rolling_mean_drift, rolling_variance_drift, regime_change_proxy, change_density_score) =
        rolling_metrics(&valid, cfg.rolling_window, cfg.epsilon);

    DimensionMetrics {
        kurtosis_proxy,
        outlier_rate_iqr,
        mad_to_std_ratio,
        lag1_autocorr,
        lagk_autocorr,
        pacf_lagk_proxy,
        dominant_period,
        residual_lag1_autocorr,
        rolling_mean_drift,
        rolling_variance_drift,
        regime_change_proxy,
        change_density_score,
    }
}

fn collect_valid(series: &[f64]) -> Vec<f64> {
    series.iter().copied().filter(|v| !v.is_nan()).collect()
}

fn moment_kurtosis_proxy(values: &[f64], mean: f64, variance: f64, epsilon: f64) -> f64 {
    if values.len() < 4 || variance <= epsilon {
        return 0.0;
    }
    let m4 = values
        .iter()
        .map(|v| {
            let centered = *v - mean;
            centered * centered * centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    let denom = (variance * variance).max(epsilon);
    let kurtosis = m4 / denom;
    if kurtosis.is_finite() { kurtosis } else { 0.0 }
}

fn iqr_outlier_rate(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    let outliers = sorted.iter().filter(|v| **v < lower || **v > upper).count();
    outliers as f64 / sorted.len() as f64
}

fn mad_to_std_ratio(values: &[f64], std: f64, epsilon: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = quantile_sorted(&sorted, 0.5);

    let mut deviations = sorted.iter().map(|v| (v - med).abs()).collect::<Vec<_>>();
    deviations.sort_by(|a, b| a.total_cmp(b));
    let mad = quantile_sorted(&deviations, 0.5);

    let ratio = (NORMAL_CONSISTENCY * mad) / (std + epsilon);
    if ratio.is_finite() { ratio } else { 0.0 }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&sorted, 0.5)
}

fn autocorr_at_lag(series: &[f64], lag: usize, epsilon: f64) -> f64 {
    if lag == 0 || lag >= series.len() {
        return 0.0;
    }

    let mut count = 0usize;
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for t in lag..series.len() {
        let a = series[t];
        let b = series[t - lag];
        if !a.is_nan() && !b.is_nan() {
            count += 1;
            sum_a += a;
            sum_b += b;
        }
    }
    if count < 3 {
        return 0.0;
    }

    let mean_a = sum_a / count as f64;
    let mean_b = sum_b / count as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for t in lag..series.len() {
        let a = series[t];
        let b = series[t - lag];
        if !a.is_nan() && !b.is_nan() {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
    }

    let denom = (var_a * var_b).sqrt();
    if denom <= epsilon {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

fn pacf_lagk_proxy(values: &[f64], lag_k: usize, epsilon: f64) -> f64 {
    if values.len() <= lag_k + 2 || lag_k == 0 {
        return 0.0;
    }

    let mean = stable_mean(values);
    if !mean.is_finite() {
        return 0.0;
    }

    let centered = values.iter().map(|v| v - mean).collect::<Vec<_>>();
    let n = centered.len();
    let variance = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if variance <= epsilon {
        return 0.0;
    }

    let mut r = vec![0.0; lag_k + 1];
    r[0] = 1.0;
    for lag in 1..=lag_k {
        if lag >= n {
            r[lag] = 0.0;
            continue;
        }
        let mut sum = 0.0;
        for t in lag..n {
            sum += centered[t] * centered[t - lag];
        }
        let cov = sum / (n - lag) as f64;
        r[lag] = (cov / variance).clamp(-1.0, 1.0);
    }

    let mut phi_prev = vec![0.0; lag_k + 1];
    let mut phi_curr = vec![0.0; lag_k + 1];
    phi_prev[1] = r[1];
    let mut sigma2 = (1.0 - r[1] * r[1]).max(epsilon);

    if lag_k == 1 {
        return phi_prev[1].clamp(-1.0, 1.0);
    }

    for m in 2..=lag_k {
        let mut num = r[m];
        for j in 1..m {
            num -= phi_prev[j] * r[m - j];
        }

        let kappa = if sigma2 <= epsilon {
            0.0
        } else {
            (num / sigma2).clamp(-1.0, 1.0)
        };

        phi_curr[m] = kappa;
        for j in 1..m {
            phi_curr[j] = phi_prev[j] - kappa * phi_prev[m - j];
        }

        sigma2 = (sigma2 * (1.0 - kappa * kappa)).max(epsilon);
        for j in 1..=m {
            phi_prev[j] = phi_curr[j];
            phi_curr[j] = 0.0;
        }
    }

    phi_prev[lag_k].clamp(-1.0, 1.0)
}

fn dominant_period_hint(
    series: &[f64],
    max_autocorr_lag: usize,
    epsilon: f64,
) -> Option<DominantPeriodHint> {
    let valid_len = series.iter().filter(|v| !v.is_nan()).count();
    let max_lag = max_autocorr_lag
        .min(valid_len / 4)
        .min(series.len().saturating_sub(1));
    if max_lag < 2 {
        return None;
    }

    let mut strengths = vec![0.0; max_lag + 1];
    for (lag, strength) in strengths.iter_mut().enumerate().take(max_lag + 1).skip(1) {
        // Use positive correlation only so half-period anti-correlation does not masquerade as seasonality.
        *strength = autocorr_at_lag(series, lag, epsilon).max(0.0);
    }

    let mut peaks = Vec::<(usize, f64)>::new();
    for lag in 2..=max_lag {
        let left = strengths[lag - 1];
        let center = strengths[lag];
        let right = if lag == max_lag {
            strengths[lag]
        } else {
            strengths[lag + 1]
        };
        if center > epsilon && center >= left && center >= right {
            peaks.push((lag, center));
        }
    }

    if peaks.is_empty() {
        for (lag, strength) in strengths
            .iter()
            .copied()
            .enumerate()
            .take(max_lag + 1)
            .skip(2)
        {
            if strength > epsilon {
                peaks.push((lag, strength));
            }
        }
    }

    if peaks.is_empty() {
        return None;
    }

    let max_strength = peaks
        .iter()
        .map(|(_, strength)| *strength)
        .fold(0.0_f64, f64::max);
    let strong_cutoff = (max_strength * 0.95).max(epsilon);

    let mut fundamental = peaks
        .iter()
        .copied()
        .filter(|(_, strength)| *strength >= strong_cutoff)
        .min_by_key(|(lag, _)| *lag);

    if fundamental.is_none() {
        fundamental = peaks
            .into_iter()
            .max_by(|(lag_a, strength_a), (lag_b, strength_b)| {
                strength_a
                    .total_cmp(strength_b)
                    .then_with(|| lag_b.cmp(lag_a))
            });
    }

    fundamental.map(|(period, strength)| DominantPeriodHint { period, strength })
}

fn detrend_linear(series: &[f64], epsilon: f64) -> Vec<f64> {
    let samples = series
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (!value.is_nan()).then_some((idx as f64, *value)))
        .collect::<Vec<_>>();

    if samples.len() < 3 {
        return series.to_vec();
    }

    let n = samples.len() as f64;
    let mean_t = samples.iter().map(|(t, _)| *t).sum::<f64>() / n;
    let mean_y = samples.iter().map(|(_, y)| *y).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_t = 0.0;
    for (t, y) in &samples {
        let dt = *t - mean_t;
        cov += dt * (*y - mean_y);
        var_t += dt * dt;
    }

    let slope = if var_t <= epsilon { 0.0 } else { cov / var_t };
    let intercept = mean_y - slope * mean_t;

    series
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if value.is_nan() {
                f64::NAN
            } else {
                *value - (intercept + slope * idx as f64)
            }
        })
        .collect()
}

fn rolling_metrics(values: &[f64], rolling_window: usize, epsilon: f64) -> (f64, f64, f64, f64) {
    if values.len() < 4 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let window = rolling_window.min(values.len()).max(2);
    let num_windows = values.len().saturating_sub(window) + 1;
    if num_windows < 2 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let prefix = prefix_sums(values);
    let prefix_sq = prefix_sum_squares(values);
    let w = window as f64;

    let mut means = Vec::with_capacity(num_windows);
    let mut vars = Vec::with_capacity(num_windows);
    for start in 0..num_windows {
        let sum = prefix[start + window] - prefix[start];
        let sum_sq = prefix_sq[start + window] - prefix_sq[start];
        let mean = sum / w;
        let var = (sum_sq / w - mean * mean).max(0.0);
        means.push(mean);
        vars.push(var);
    }

    let global_mean = stable_mean(values);
    let global_var = stable_variance(values, global_mean).max(0.0);
    let global_std = global_var.sqrt();

    let means_mean = stable_mean(&means);
    let means_var = stable_variance(&means, means_mean).max(0.0);
    let rolling_mean_drift = means_var.sqrt() / (global_std + epsilon);

    let vars_mean = stable_mean(&vars).max(0.0);
    let vars_var = stable_variance(&vars, vars_mean).max(0.0);
    let rolling_variance_drift = vars_var.sqrt() / (vars_mean + epsilon);

    if values.len() < 2 * window {
        return (
            sanitize_metric(rolling_mean_drift),
            sanitize_metric(rolling_variance_drift),
            0.0,
            0.0,
        );
    }

    let mut divergence_scores = Vec::with_capacity(values.len() - 2 * window + 1);
    for start in 0..=(values.len() - 2 * window) {
        let mean_a = means[start];
        let var_a = vars[start];
        let mean_b = means[start + window];
        let var_b = vars[start + window];

        let mean_term = (mean_b - mean_a).abs() / (global_std + epsilon);
        let var_term = (var_b - var_a).abs() / (global_var + epsilon);
        divergence_scores.push(mean_term + var_term);
    }

    let regime_change_proxy = divergence_scores.iter().copied().fold(0.0_f64, f64::max);
    let dense_count = divergence_scores
        .iter()
        .filter(|score| **score > 1.0)
        .count();
    let change_density_score = dense_count as f64 / divergence_scores.len() as f64;

    (
        sanitize_metric(rolling_mean_drift),
        sanitize_metric(rolling_variance_drift),
        sanitize_metric(regime_change_proxy),
        sanitize_metric(change_density_score),
    )
}

fn summarize_period_votes(votes: BTreeMap<usize, f64>) -> Vec<DominantPeriodHint> {
    let mut hints = votes
        .into_iter()
        .map(|(period, strength)| DominantPeriodHint { period, strength })
        .collect::<Vec<_>>();

    hints.sort_by(|a, b| {
        b.strength
            .total_cmp(&a.strength)
            .then_with(|| a.period.cmp(&b.period))
    });
    hints.truncate(3);
    hints
}

fn sanitize_metric(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn source_index(
    layout: MemoryLayout,
    n: usize,
    d: usize,
    t: usize,
    j: usize,
) -> Result<usize, CpdError> {
    match layout {
        MemoryLayout::CContiguous => t
            .checked_mul(d)
            .and_then(|base| base.checked_add(j))
            .ok_or_else(|| {
                CpdError::invalid_input("C-layout index overflow in doctor diagnostics")
            }),
        MemoryLayout::FContiguous => j
            .checked_mul(n)
            .and_then(|base| base.checked_add(t))
            .ok_or_else(|| {
                CpdError::invalid_input("F-layout index overflow in doctor diagnostics")
            }),
        MemoryLayout::Strided {
            row_stride,
            col_stride,
        } => {
            let t_isize = isize::try_from(t).map_err(|_| {
                CpdError::invalid_input(format!(
                    "time index {t} does not fit in isize for doctor diagnostics"
                ))
            })?;
            let j_isize = isize::try_from(j).map_err(|_| {
                CpdError::invalid_input(format!(
                    "dimension index {j} does not fit in isize for doctor diagnostics"
                ))
            })?;
            let idx = t_isize
                .checked_mul(row_stride)
                .and_then(|left| {
                    j_isize
                        .checked_mul(col_stride)
                        .and_then(|right| left.checked_add(right))
                })
                .ok_or_else(|| {
                    CpdError::invalid_input(format!(
                        "strided index overflow in doctor diagnostics at t={t}, j={j}, row_stride={row_stride}, col_stride={col_stride}"
                    ))
                })?;

            usize::try_from(idx).map_err(|_| {
                CpdError::invalid_input(format!(
                    "strided index became negative in doctor diagnostics at t={t}, j={j}: idx={idx}"
                ))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DoctorDiagnosticsConfig, MissingPattern, compute_diagnostics, estimate_sampling_rate_hz,
    };
    use cpd_core::{MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

    fn make_univariate_view(values: &[f64]) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("test view should be valid")
    }

    fn make_univariate_view_with_mask<'a>(values: &'a [f64], mask: &'a [u8]) -> TimeSeriesView<'a> {
        TimeSeriesView::from_f64(
            values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            Some(mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("test masked view should be valid")
    }

    fn pseudo_uniform_noise(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((*state >> 33) as f64) / ((1_u64 << 31) as f64);
        u * 2.0 - 1.0
    }

    #[test]
    fn ar1_series_has_high_lag1_autocorrelation() {
        let mut state = 13_u64;
        let mut values = vec![0.0; 512];
        let phi = 0.85;
        for t in 1..values.len() {
            let noise = 0.3 * pseudo_uniform_noise(&mut state);
            values[t] = phi * values[t - 1] + noise;
        }

        let view = make_univariate_view(&values);
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");

        assert!(report.summary.lag1_autocorr > 0.6);
        assert!(
            report.per_dimension[0]
                .lag1_autocorr
                .expect("lag-1 should exist")
                > 0.6
        );
    }

    #[test]
    fn periodic_signal_yields_period_hint() {
        let period = 12usize;
        let mut state = 17_u64;
        let values = (0..1024)
            .map(|t| {
                let theta = 2.0 * std::f64::consts::PI * t as f64 / period as f64;
                theta.sin() + 0.05 * pseudo_uniform_noise(&mut state)
            })
            .collect::<Vec<_>>();

        let view = make_univariate_view(&values);
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");

        let top = report
            .summary
            .dominant_period_hints
            .first()
            .expect("period hint should exist");
        assert!(
            (top.period as isize - period as isize).abs() <= 1,
            "period hints: {:?}",
            report.summary.dominant_period_hints,
        );
    }

    #[test]
    fn heavy_tail_and_outlier_signal_elevates_robustness_proxies() {
        let mut state = 19_u64;
        let mut values = Vec::with_capacity(640);
        for i in 0..640 {
            let mut v = pseudo_uniform_noise(&mut state);
            if i % 27 == 0 {
                v *= 20.0;
            }
            values.push(v);
        }

        let view = make_univariate_view(&values);
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");

        assert!(report.summary.kurtosis_proxy > 4.0);
        assert!(report.summary.outlier_rate_iqr > 0.01);
    }

    #[test]
    fn variance_shift_signal_has_high_drift_and_regime_proxy() {
        let mut state = 29_u64;
        let mut values = Vec::with_capacity(800);
        for i in 0..800 {
            let base = pseudo_uniform_noise(&mut state);
            let scale = if i < 400 { 1.0 } else { 6.0 };
            values.push(base * scale);
        }

        let view = make_univariate_view(&values);
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");

        assert!(report.summary.rolling_variance_drift > 0.2);
        assert!(report.summary.regime_change_proxy > 1.0);
    }

    #[test]
    fn missing_pattern_detects_random_and_block_modes() {
        let values = vec![1.0; 256];

        let mut random_mask = vec![0_u8; values.len()];
        for idx in (0..values.len()).step_by(11) {
            random_mask[idx] = 1;
        }
        let random_view = make_univariate_view_with_mask(&values, &random_mask);
        let random_report =
            compute_diagnostics(&random_view, &DoctorDiagnosticsConfig::default()).expect("random");
        assert_eq!(
            random_report.summary.missing_pattern,
            MissingPattern::Random
        );

        let mut block_mask = vec![0_u8; values.len()];
        for item in block_mask.iter_mut().take(160).skip(96) {
            *item = 1;
        }
        let block_view = make_univariate_view_with_mask(&values, &block_mask);
        let block_report =
            compute_diagnostics(&block_view, &DoctorDiagnosticsConfig::default()).expect("block");
        assert_eq!(block_report.summary.missing_pattern, MissingPattern::Block);
    }

    #[test]
    fn multivariate_random_missing_aggregates_as_random() {
        let n = 256usize;
        let d = 4usize;
        let values = vec![1.0_f64; n * d];
        let mut mask = vec![0_u8; n * d];

        for t in 0..n {
            for j in 0..d {
                if (t + j * 7) % 23 == 0 {
                    mask[t * d + j] = 1;
                }
            }
        }

        let view = TimeSeriesView::from_f64(
            &values,
            n,
            d,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("masked multivariate view");
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");
        assert_eq!(report.summary.missing_pattern, MissingPattern::Random);
    }

    #[test]
    fn large_series_triggers_subsampling() {
        let n = 120_000;
        let values = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let view = make_univariate_view(&values);
        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");

        assert!(report.used_subsampling);
        assert!(report.subsample_stride > 1);
        assert!(report.subsample_n <= 50_001);
        assert!(report.subsample_n >= 10_000);
    }

    #[test]
    fn near_threshold_subsampling_meets_minimum_target() {
        let cfg = DoctorDiagnosticsConfig::default();
        let n = cfg.subsample_threshold + 1;
        let values = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let view = make_univariate_view(&values);
        let report = compute_diagnostics(&view, &cfg).expect("diagnostics");

        assert!(report.used_subsampling);
        assert!(report.subsample_n >= cfg.subsample_target_min.min(n));
    }

    #[test]
    fn multivariate_report_keeps_dimension_count_and_summary() {
        let n = 360usize;
        let d = 3usize;
        let mut values = Vec::with_capacity(n * d);
        let mut state = 37_u64;

        for t in 0..n {
            let dim0 = 0.9 * (t as f64 / 20.0).sin() + 0.1 * pseudo_uniform_noise(&mut state);
            let dim1 = (2.0 * std::f64::consts::PI * t as f64 / 18.0).sin();
            let scale = if t < n / 2 { 1.0 } else { 3.0 };
            let dim2 = scale * pseudo_uniform_noise(&mut state);
            values.extend([dim0, dim1, dim2]);
        }

        let view = TimeSeriesView::from_f64(
            &values,
            n,
            d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("multivariate view");

        let report =
            compute_diagnostics(&view, &DoctorDiagnosticsConfig::default()).expect("diagnostics");
        assert_eq!(report.per_dimension.len(), d);
        assert_eq!(report.summary.valid_dimensions, d);
        assert!(report.summary.kurtosis_proxy.is_finite());
    }

    #[test]
    fn diagnostics_are_deterministic() {
        let mut state = 41_u64;
        let values = (0..500)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let cfg = DoctorDiagnosticsConfig::default();
        let first = compute_diagnostics(&view, &cfg).expect("first");
        let second = compute_diagnostics(&view, &cfg).expect("second");

        assert_eq!(first, second);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let values = vec![0.0; 128];
        let view = make_univariate_view(&values);

        let mut cfg = DoctorDiagnosticsConfig::default();
        cfg.lag_k = cfg.max_autocorr_lag + 1;
        let err = compute_diagnostics(&view, &cfg).expect_err("invalid lag config should fail");
        assert!(err.to_string().contains("lag_k"));
    }

    #[test]
    fn sampling_rate_is_reported_for_uniform_and_explicit_time() {
        let values = vec![1.0; 128];

        let view_none = TimeSeriesView::from_f64(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("none view");
        let none_report =
            compute_diagnostics(&view_none, &DoctorDiagnosticsConfig::default()).expect("none");
        assert_eq!(none_report.sampling_rate_hz, None);

        let dt_ns = 2_000_000_i64;
        let view_uniform = TimeSeriesView::from_f64(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform { t0_ns: 0, dt_ns },
            MissingPolicy::Ignore,
        )
        .expect("uniform view");
        let uniform_report =
            compute_diagnostics(&view_uniform, &DoctorDiagnosticsConfig::default())
                .expect("uniform");
        let uniform_rate = uniform_report
            .sampling_rate_hz
            .expect("uniform rate should exist");
        assert!((uniform_rate - 500.0).abs() < 1e-9);

        let timestamps = (0..values.len())
            .map(|i| i as i64 * dt_ns)
            .collect::<Vec<_>>();
        let explicit_rate = estimate_sampling_rate_hz(TimeIndex::Explicit(&timestamps), 1e-12)
            .expect("explicit rate should exist");
        assert!((explicit_rate - 500.0).abs() < 1e-9);

        let view_explicit = TimeSeriesView::from_f64(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&timestamps),
            MissingPolicy::Ignore,
        )
        .expect("explicit view");
        let explicit_report =
            compute_diagnostics(&view_explicit, &DoctorDiagnosticsConfig::default())
                .expect("explicit diagnostics");
        let report_rate = explicit_report
            .sampling_rate_hz
            .expect("explicit rate should exist");
        assert!((report_rate - 500.0).abs() < 1e-9);
    }
}
