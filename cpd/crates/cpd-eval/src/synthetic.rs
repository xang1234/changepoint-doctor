// SPDX-License-Identifier: MIT OR Apache-2.0

use cpd_core::CpdError;
use std::f64::consts::PI;

const MIN_POSITIVE_U01: f64 = f64::from_bits(1);

/// Synthetic generator return type `(data, true_change_points)`.
///
/// `true_change_points` excludes the terminal sample index `n`.
pub type SyntheticSeries = (Vec<f64>, Vec<usize>);

/// Piecewise-constant mean generator configuration.
#[derive(Clone, Debug)]
pub struct PiecewiseMeanConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub snr: f64,
    pub noise_std: f64,
    pub seed: u64,
}

/// Piecewise-constant variance generator configuration.
#[derive(Clone, Debug)]
pub struct PiecewiseVarianceConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub mean: f64,
    pub base_std: f64,
    pub std_step: f64,
    pub seed: u64,
}

/// Piecewise-linear trend generator configuration.
#[derive(Clone, Debug)]
pub struct PiecewiseLinearConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub slope_step: f64,
    pub noise_std: f64,
    pub seed: u64,
}

/// AR(1) regime-shift generator configuration.
#[derive(Clone, Debug)]
pub struct Ar1Config {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub phi: f64,
    pub mean_step: f64,
    pub base_std: f64,
    pub std_step: f64,
    pub seed: u64,
}

/// Heavy-tailed generator configuration using Student-t innovations.
#[derive(Clone, Debug)]
pub struct HeavyTailConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub degrees_of_freedom: usize,
    pub mean_step: f64,
    pub scale: f64,
    pub seed: u64,
}

/// Piecewise Poisson count generator configuration.
#[derive(Clone, Debug)]
pub struct CountConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub base_rate: f64,
    pub rate_step: f64,
    pub seed: u64,
}

/// Piecewise Bernoulli generator configuration.
#[derive(Clone, Debug)]
pub struct BinaryConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub base_prob: f64,
    pub prob_step: f64,
    pub seed: u64,
}

/// Multivariate generator with optional cross-dimensional correlation.
#[derive(Clone, Debug)]
pub struct MultivariateConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub snr: f64,
    pub noise_std: f64,
    pub correlation: Option<f64>,
    pub seed: u64,
}

/// Missing-value insertion pattern.
#[derive(Clone, Debug)]
pub enum MissingPattern {
    Random,
    Block { gap_length: usize },
    Periodic { every: usize, width: usize },
}

/// Missing-data generator configuration.
#[derive(Clone, Debug)]
pub struct MissingDataConfig {
    pub n: usize,
    pub d: usize,
    pub n_changes: usize,
    pub snr: f64,
    pub noise_std: f64,
    pub missing_fraction: f64,
    pub pattern: MissingPattern,
    pub seed: u64,
}

/// Returns deterministic, evenly-spaced true change points.
///
/// Returned indices exclude the terminal sample index `n`.
pub fn evenly_spaced_breakpoints(n: usize, n_changes: usize) -> Result<Vec<usize>, CpdError> {
    validate_shape(n, 1, n_changes)?;
    let mut change_points = Vec::with_capacity(n_changes);
    for i in 1..=n_changes {
        change_points.push(i * n / (n_changes + 1));
    }
    Ok(change_points)
}

/// Converts true change points into `OfflineChangePointResult` breakpoint format.
///
/// This appends the terminal index `n` and validates that all change points are
/// strictly increasing and in `[1, n)`.
pub fn to_offline_breakpoints(
    n: usize,
    true_change_points: &[usize],
) -> Result<Vec<usize>, CpdError> {
    if n == 0 {
        return Err(CpdError::invalid_input(
            "n must be >= 1 for breakpoint conversion".to_string(),
        ));
    }
    let mut last = 0usize;
    for (index, &cp) in true_change_points.iter().enumerate() {
        if cp == 0 || cp >= n {
            return Err(CpdError::invalid_input(format!(
                "true_change_points[{index}] must be in [1, {n}); got {cp}"
            )));
        }
        if index > 0 && cp <= last {
            return Err(CpdError::invalid_input(format!(
                "true_change_points must be strictly increasing; true_change_points[{}]={last}, true_change_points[{index}]={cp}",
                index - 1
            )));
        }
        last = cp;
    }
    let mut breakpoints = Vec::with_capacity(true_change_points.len() + 1);
    breakpoints.extend_from_slice(true_change_points);
    breakpoints.push(n);
    Ok(breakpoints)
}

/// Piecewise-constant mean with configurable SNR.
pub fn piecewise_constant_mean(cfg: &PiecewiseMeanConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.noise_std, "noise_std")?;
    validate_non_negative(cfg.snr, "snr")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];
    let amplitude = cfg.snr * cfg.noise_std;

    fill_piecewise(
        cfg.n,
        cfg.d,
        change_points.as_slice(),
        |segment, _t, j| {
            let segment_mean = amplitude * segment as f64 + 0.1 * amplitude * j as f64;
            segment_mean + cfg.noise_std * rng.standard_normal()
        },
        &mut data,
    );

    Ok((data, change_points))
}

/// Piecewise-constant variance (volatility) shifts.
pub fn piecewise_constant_variance(
    cfg: &PiecewiseVarianceConfig,
) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.base_std, "base_std")?;
    validate_non_negative(cfg.std_step, "std_step")?;
    validate_finite(cfg.mean, "mean")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    fill_piecewise(
        cfg.n,
        cfg.d,
        change_points.as_slice(),
        |segment, _t, _j| {
            let std = cfg.base_std + cfg.std_step * segment as f64;
            cfg.mean + std * rng.standard_normal()
        },
        &mut data,
    );

    Ok((data, change_points))
}

/// Piecewise-linear trends with breakpointed slope changes.
pub fn piecewise_linear(cfg: &PiecewiseLinearConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.noise_std, "noise_std")?;
    validate_finite(cfg.slope_step, "slope_step")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    for dim in 0..cfg.d {
        let mut baseline = 0.0;
        for (segment, (start, end)) in segments_from_change_points(cfg.n, change_points.as_slice())
            .into_iter()
            .enumerate()
        {
            let len = end - start;
            let sign = if segment % 2 == 0 { 1.0 } else { -1.0 };
            let slope = sign * cfg.slope_step * (segment as f64 + 1.0);
            for local_t in 0..len {
                let t = start + local_t;
                let value =
                    baseline + slope * local_t as f64 + cfg.noise_std * rng.standard_normal();
                data[t * cfg.d + dim] = value;
            }
            baseline += slope * len as f64;
        }
    }

    Ok((data, change_points))
}

/// AR(1) process with regime shifts in mean and variance.
pub fn ar1_with_changes(cfg: &Ar1Config) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_finite(cfg.phi, "phi")?;
    if cfg.phi.abs() >= 1.0 {
        return Err(CpdError::invalid_input(format!(
            "phi must satisfy |phi| < 1.0; got {}",
            cfg.phi
        )));
    }
    validate_finite(cfg.mean_step, "mean_step")?;
    validate_non_negative(cfg.base_std, "base_std")?;
    validate_non_negative(cfg.std_step, "std_step")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    for dim in 0..cfg.d {
        let mut prev = 0.0;
        for (segment, (start, end)) in segments_from_change_points(cfg.n, change_points.as_slice())
            .into_iter()
            .enumerate()
        {
            let mean = cfg.mean_step * segment as f64 + 0.2 * dim as f64;
            let std = cfg.base_std + cfg.std_step * segment as f64;
            for t in start..end {
                let innovation = std * rng.standard_normal();
                let value = mean + cfg.phi * (prev - mean) + innovation;
                data[t * cfg.d + dim] = value;
                prev = value;
            }
        }
    }

    Ok((data, change_points))
}

/// Heavy-tailed piecewise series using Student-t innovations.
pub fn heavy_tailed(cfg: &HeavyTailConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    if cfg.degrees_of_freedom == 0 {
        return Err(CpdError::invalid_input(
            "degrees_of_freedom must be > 0".to_string(),
        ));
    }
    validate_finite(cfg.mean_step, "mean_step")?;
    validate_non_negative(cfg.scale, "scale")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    fill_piecewise(
        cfg.n,
        cfg.d,
        change_points.as_slice(),
        |segment, _t, j| {
            let mean = cfg.mean_step * segment as f64 + 0.1 * j as f64;
            mean + cfg.scale * rng.student_t(cfg.degrees_of_freedom)
        },
        &mut data,
    );

    Ok((data, change_points))
}

/// Poisson count data with rate changes.
pub fn count_data(cfg: &CountConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.base_rate, "base_rate")?;
    validate_finite(cfg.rate_step, "rate_step")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    fill_piecewise(
        cfg.n,
        cfg.d,
        change_points.as_slice(),
        |segment, _t, _j| {
            let lambda = (cfg.base_rate + cfg.rate_step * segment as f64).max(1e-6);
            rng.poisson(lambda) as f64
        },
        &mut data,
    );

    Ok((data, change_points))
}

/// Bernoulli binary data with probability changes.
pub fn binary_data(cfg: &BinaryConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_finite(cfg.base_prob, "base_prob")?;
    validate_finite(cfg.prob_step, "prob_step")?;

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];

    fill_piecewise(
        cfg.n,
        cfg.d,
        change_points.as_slice(),
        |segment, _t, _j| {
            let prob = (cfg.base_prob + cfg.prob_step * segment as f64).clamp(0.0, 1.0);
            if rng.bernoulli(prob) { 1.0 } else { 0.0 }
        },
        &mut data,
    );

    Ok((data, change_points))
}

/// Multivariate generator with optional cross-dimensional correlation.
pub fn multivariate(cfg: &MultivariateConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.noise_std, "noise_std")?;
    validate_non_negative(cfg.snr, "snr")?;

    let rho = cfg.correlation.unwrap_or(0.0);
    validate_finite(rho, "correlation")?;
    if !(0.0..=1.0).contains(&rho) {
        return Err(CpdError::invalid_input(format!(
            "correlation must be in [0.0, 1.0]; got {rho}"
        )));
    }

    let change_points = evenly_spaced_breakpoints(cfg.n, cfg.n_changes)?;
    let mut rng = DeterministicRng::new(cfg.seed);
    let mut data = vec![0.0; checked_flat_len(cfg.n, cfg.d)?];
    let n_segments = cfg.n_changes + 1;
    let rho_common = rho.sqrt();
    let rho_ind = (1.0 - rho).sqrt();
    let segment_scales = (0..n_segments)
        .map(|segment| 1.0 + cfg.snr * segment as f64)
        .collect::<Vec<_>>();
    let dim_biases = (0..cfg.d)
        .map(|j| (j as f64 - (cfg.d.saturating_sub(1)) as f64 / 2.0) * 0.05 * cfg.noise_std)
        .collect::<Vec<_>>();

    let mut segment = 0usize;
    let mut bp_idx = 0usize;
    for t in 0..cfg.n {
        while bp_idx < change_points.len() && t >= change_points[bp_idx] {
            segment += 1;
            bp_idx += 1;
        }

        let common = rng.standard_normal();
        let segment_scale = segment_scales[segment];
        for j in 0..cfg.d {
            let idiosyncratic = rng.standard_normal();
            let noise = if rho == 0.0 {
                cfg.noise_std * idiosyncratic
            } else {
                cfg.noise_std * (rho_common * common + rho_ind * idiosyncratic)
            };
            data[t * cfg.d + j] = dim_biases[j] + segment_scale * noise;
        }
    }

    Ok((data, change_points))
}

/// Piecewise signal with injected NaN gaps under configurable missing patterns.
pub fn missing_data(cfg: &MissingDataConfig) -> Result<SyntheticSeries, CpdError> {
    validate_shape(cfg.n, cfg.d, cfg.n_changes)?;
    validate_non_negative(cfg.noise_std, "noise_std")?;
    validate_non_negative(cfg.snr, "snr")?;
    validate_finite(cfg.missing_fraction, "missing_fraction")?;
    if !(0.0..=1.0).contains(&cfg.missing_fraction) {
        return Err(CpdError::invalid_input(format!(
            "missing_fraction must be in [0.0, 1.0]; got {}",
            cfg.missing_fraction
        )));
    }

    let mean_cfg = PiecewiseMeanConfig {
        n: cfg.n,
        d: cfg.d,
        n_changes: cfg.n_changes,
        snr: cfg.snr,
        noise_std: cfg.noise_std,
        seed: cfg.seed,
    };
    let (mut data, change_points) = piecewise_constant_mean(&mean_cfg)?;
    let total = checked_flat_len(cfg.n, cfg.d)?;
    let target_missing = ((total as f64) * cfg.missing_fraction).round() as usize;
    if target_missing == 0 {
        return Ok((data, change_points));
    }

    let mut rng = DeterministicRng::new(cfg.seed ^ 0x9e37_79b9_7f4a_7c15);
    match cfg.pattern {
        MissingPattern::Random => fill_random_missing(&mut data, target_missing, &mut rng),
        MissingPattern::Block { gap_length } => {
            if gap_length == 0 {
                return Err(CpdError::invalid_input(
                    "MissingPattern::Block requires gap_length > 0".to_string(),
                ));
            }
            fill_block_missing(
                &mut data,
                cfg.n,
                cfg.d,
                gap_length,
                target_missing,
                &mut rng,
            );
        }
        MissingPattern::Periodic { every, width } => {
            if every == 0 || width == 0 {
                return Err(CpdError::invalid_input(
                    "MissingPattern::Periodic requires every > 0 and width > 0".to_string(),
                ));
            }
            fill_periodic_missing(
                &mut data,
                cfg.n,
                cfg.d,
                every,
                width,
                target_missing,
                &mut rng,
            );
        }
    }

    Ok((data, change_points))
}

fn validate_shape(n: usize, d: usize, n_changes: usize) -> Result<(), CpdError> {
    if n < 2 {
        return Err(CpdError::invalid_input(format!("n must be >= 2; got {n}")));
    }
    if d == 0 {
        return Err(CpdError::invalid_input("d must be >= 1; got 0".to_string()));
    }
    if n_changes >= n {
        return Err(CpdError::invalid_input(format!(
            "n_changes must be < n; got n_changes={n_changes}, n={n}"
        )));
    }
    Ok(())
}

fn validate_non_negative(value: f64, label: &str) -> Result<(), CpdError> {
    validate_finite(value, label)?;
    if value < 0.0 {
        return Err(CpdError::invalid_input(format!(
            "{label} must be >= 0.0; got {value}"
        )));
    }
    Ok(())
}

fn validate_finite(value: f64, label: &str) -> Result<(), CpdError> {
    if !value.is_finite() {
        return Err(CpdError::invalid_input(format!(
            "{label} must be finite; got {value}"
        )));
    }
    Ok(())
}

fn checked_flat_len(n: usize, d: usize) -> Result<usize, CpdError> {
    n.checked_mul(d).ok_or_else(|| {
        CpdError::invalid_input(format!(
            "n*d overflows usize; got n={n}, d={d}. Use smaller dimensions."
        ))
    })
}

fn fill_piecewise<F>(n: usize, d: usize, breakpoints: &[usize], mut value_fn: F, data: &mut [f64])
where
    F: FnMut(usize, usize, usize) -> f64,
{
    let mut segment = 0usize;
    let mut bp_idx = 0usize;
    for t in 0..n {
        while bp_idx < breakpoints.len() && t >= breakpoints[bp_idx] {
            segment += 1;
            bp_idx += 1;
        }
        for j in 0..d {
            data[t * d + j] = value_fn(segment, t, j);
        }
    }
}

fn segments_from_change_points(n: usize, change_points: &[usize]) -> Vec<(usize, usize)> {
    let mut segments = Vec::with_capacity(change_points.len() + 1);
    let mut start = 0usize;
    for &bp in change_points {
        segments.push((start, bp));
        start = bp;
    }
    segments.push((start, n));
    segments
}

fn fill_random_missing(data: &mut [f64], target_missing: usize, rng: &mut DeterministicRng) {
    let total = data.len();
    let mut idx = (0..total).collect::<Vec<_>>();
    let target = target_missing.min(total);
    for i in 0..target {
        let span = total - i;
        let pick = i + (rng.next_u64() as usize % span);
        idx.swap(i, pick);
        data[idx[i]] = f64::NAN;
    }
}

fn fill_block_missing(
    data: &mut [f64],
    n: usize,
    d: usize,
    gap_length: usize,
    target_missing: usize,
    rng: &mut DeterministicRng,
) {
    let target = target_missing.min(data.len());
    let mut placed = count_missing(data);
    let gap = gap_length.min(n);
    let mut attempts = 0usize;
    let max_attempts = target.saturating_mul(16).saturating_add(128);

    while placed < target && attempts < max_attempts {
        let dim = rng.next_u64() as usize % d;
        let max_start = n.saturating_sub(gap);
        let start = if max_start == 0 {
            0
        } else {
            rng.next_u64() as usize % (max_start + 1)
        };
        for offset in 0..gap {
            let index = (start + offset) * d + dim;
            if !data[index].is_nan() {
                data[index] = f64::NAN;
                placed += 1;
                if placed == target {
                    return;
                }
            }
        }
        attempts += 1;
    }

    if placed < target {
        fill_random_missing_remaining(data, target, rng);
    }
}

fn fill_periodic_missing(
    data: &mut [f64],
    n: usize,
    d: usize,
    every: usize,
    width: usize,
    target_missing: usize,
    rng: &mut DeterministicRng,
) {
    let target = target_missing.min(data.len());
    let mut placed = count_missing(data);

    for t0 in (0..n).step_by(every) {
        for local in 0..width {
            let t = t0 + local;
            if t >= n {
                break;
            }
            for dim in 0..d {
                let index = t * d + dim;
                if !data[index].is_nan() {
                    data[index] = f64::NAN;
                    placed += 1;
                    if placed == target {
                        return;
                    }
                }
            }
        }
    }

    if placed < target {
        fill_random_missing_remaining(data, target, rng);
    }
}

fn fill_random_missing_remaining(
    data: &mut [f64],
    target_missing: usize,
    rng: &mut DeterministicRng,
) {
    let target = target_missing.min(data.len());
    let mut placed = count_missing(data);
    while placed < target {
        let index = rng.next_u64() as usize % data.len();
        if !data[index].is_nan() {
            data[index] = f64::NAN;
            placed += 1;
        }
    }
}

fn count_missing(data: &[f64]) -> usize {
    data.iter().filter(|value| value.is_nan()).count()
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
    cached_normal: Option<f64>,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z ^= z >> 30;
        z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.unit_f64().max(MIN_POSITIVE_U01);
        let u2 = self.unit_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    fn student_t(&mut self, degrees_of_freedom: usize) -> f64 {
        let z = self.standard_normal();
        let mut chi_square = 0.0;
        for _ in 0..degrees_of_freedom {
            let draw = self.standard_normal();
            chi_square += draw * draw;
        }
        z / (chi_square / degrees_of_freedom as f64).sqrt()
    }

    fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 {
            return 0;
        }

        if lambda < 30.0 {
            let bound = (-lambda).exp();
            let mut count = 0u64;
            let mut product = 1.0;
            loop {
                count += 1;
                product *= self.unit_f64();
                if product <= bound {
                    return count - 1;
                }
            }
        }

        let normal = self.standard_normal();
        let sample = (lambda + lambda.sqrt() * normal).round().max(0.0);
        sample as u64
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.unit_f64() < probability
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Ar1Config, BinaryConfig, CountConfig, HeavyTailConfig, MissingDataConfig, MissingPattern,
        MultivariateConfig, PiecewiseLinearConfig, PiecewiseMeanConfig, PiecewiseVarianceConfig,
        ar1_with_changes, binary_data, count_data, count_missing, evenly_spaced_breakpoints,
        heavy_tailed, missing_data, multivariate, piecewise_constant_mean,
        piecewise_constant_variance, piecewise_linear, to_offline_breakpoints,
    };

    fn segment_means(data: &[f64], n: usize, d: usize, breakpoints: &[usize]) -> Vec<f64> {
        let mut means = Vec::with_capacity(breakpoints.len() + 1);
        let mut start = 0usize;
        for &end in breakpoints.iter().chain(std::iter::once(&n)) {
            let mut total = 0.0;
            let mut count = 0usize;
            for t in start..end {
                for j in 0..d {
                    total += data[t * d + j];
                    count += 1;
                }
            }
            means.push(total / count as f64);
            start = end;
        }
        means
    }

    fn segment_variances(data: &[f64], n: usize, d: usize, breakpoints: &[usize]) -> Vec<f64> {
        let mut vars = Vec::with_capacity(breakpoints.len() + 1);
        let means = segment_means(data, n, d, breakpoints);
        let mut start = 0usize;
        for (segment, &end) in breakpoints.iter().chain(std::iter::once(&n)).enumerate() {
            let mut total = 0.0;
            let mut count = 0usize;
            for t in start..end {
                for j in 0..d {
                    let delta = data[t * d + j] - means[segment];
                    total += delta * delta;
                    count += 1;
                }
            }
            vars.push(total / count as f64);
            start = end;
        }
        vars
    }

    fn slope_estimate(values: &[f64]) -> f64 {
        let len = values.len();
        if len < 2 {
            return 0.0;
        }
        (values[len - 1] - values[0]) / (len - 1) as f64
    }

    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        let mean_x = x.iter().take(n).sum::<f64>() / n as f64;
        let mean_y = y.iter().take(n).sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        cov / (var_x.sqrt() * var_y.sqrt())
    }

    fn kurtosis(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let mut m2 = 0.0;
        let mut m4 = 0.0;
        for &value in values {
            let delta = value - mean;
            let delta2 = delta * delta;
            m2 += delta2;
            m4 += delta2 * delta2;
        }
        let m2n = m2 / n;
        let m4n = m4 / n;
        m4n / (m2n * m2n)
    }

    fn assert_same_bits(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (left, right) in a.iter().zip(b) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn breakpoints_are_evenly_spaced() {
        let breakpoints = evenly_spaced_breakpoints(120, 2).expect("breakpoints should be valid");
        assert_eq!(breakpoints, vec![40, 80]);
    }

    #[test]
    fn piecewise_mean_is_seeded_and_shifts_segment_means() {
        let cfg = PiecewiseMeanConfig {
            n: 1200,
            d: 2,
            n_changes: 2,
            snr: 4.0,
            noise_std: 1.0,
            seed: 7,
        };

        let (first_data, first_breakpoints) =
            piecewise_constant_mean(&cfg).expect("generation should succeed");
        let (second_data, second_breakpoints) =
            piecewise_constant_mean(&cfg).expect("generation should succeed");

        assert_eq!(first_breakpoints, vec![400, 800]);
        assert_eq!(first_breakpoints, second_breakpoints);
        assert_eq!(first_data, second_data);

        let means = segment_means(&first_data, cfg.n, cfg.d, first_breakpoints.as_slice());
        assert!(means[1] > means[0] + 2.5);
        assert!(means[2] > means[1] + 2.5);
    }

    #[test]
    fn piecewise_variance_shifts_volatility_between_segments() {
        let cfg = PiecewiseVarianceConfig {
            n: 1200,
            d: 1,
            n_changes: 2,
            mean: 0.0,
            base_std: 0.6,
            std_step: 0.8,
            seed: 13,
        };

        let (data, breakpoints) =
            piecewise_constant_variance(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![400, 800]);

        let vars = segment_variances(&data, cfg.n, cfg.d, breakpoints.as_slice());
        assert!(vars[1] > vars[0] * 2.0);
        assert!(vars[2] > vars[1] * 1.5);
    }

    #[test]
    fn piecewise_linear_changes_slope_by_segment() {
        let cfg = PiecewiseLinearConfig {
            n: 1200,
            d: 1,
            n_changes: 2,
            slope_step: 0.03,
            noise_std: 0.1,
            seed: 17,
        };

        let (data, breakpoints) = piecewise_linear(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![400, 800]);

        let s1 = slope_estimate(&data[0..400]);
        let s2 = slope_estimate(&data[400..800]);
        let s3 = slope_estimate(&data[800..1200]);

        assert!(s1 > 0.01);
        assert!(s2 < -0.02);
        assert!(s3 > 0.06);
    }

    #[test]
    fn ar1_generator_produces_positive_lag_correlation() {
        let cfg = Ar1Config {
            n: 3000,
            d: 1,
            n_changes: 2,
            phi: 0.85,
            mean_step: 1.0,
            base_std: 0.5,
            std_step: 0.3,
            seed: 23,
        };

        let (data, breakpoints) = ar1_with_changes(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![1000, 2000]);

        let x = &data[..cfg.n - 1];
        let y = &data[1..cfg.n];
        let lag1 = correlation(x, y);
        assert!(lag1 > 0.65);
    }

    #[test]
    fn heavy_tailed_generator_has_high_kurtosis() {
        let cfg = HeavyTailConfig {
            n: 5000,
            d: 1,
            n_changes: 1,
            degrees_of_freedom: 3,
            mean_step: 0.0,
            scale: 1.0,
            seed: 31,
        };

        let (data, breakpoints) = heavy_tailed(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![2500]);

        let k = kurtosis(data.as_slice());
        assert!(k > 4.5);
    }

    #[test]
    fn count_generator_tracks_piecewise_rates() {
        let cfg = CountConfig {
            n: 3000,
            d: 1,
            n_changes: 2,
            base_rate: 2.0,
            rate_step: 3.0,
            seed: 37,
        };

        let (data, breakpoints) = count_data(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![1000, 2000]);

        let means = segment_means(&data, cfg.n, cfg.d, breakpoints.as_slice());
        assert!(means[0] > 1.6 && means[0] < 2.4);
        assert!(means[1] > 4.4 && means[1] < 5.6);
        assert!(means[2] > 7.2 && means[2] < 8.8);
    }

    #[test]
    fn binary_generator_tracks_piecewise_probabilities() {
        let cfg = BinaryConfig {
            n: 3600,
            d: 1,
            n_changes: 2,
            base_prob: 0.2,
            prob_step: 0.3,
            seed: 41,
        };

        let (data, breakpoints) = binary_data(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![1200, 2400]);

        let means = segment_means(&data, cfg.n, cfg.d, breakpoints.as_slice());
        assert!(means[0] > 0.14 && means[0] < 0.26);
        assert!(means[1] > 0.44 && means[1] < 0.56);
        assert!(means[2] > 0.74 && means[2] < 0.86);
    }

    #[test]
    fn multivariate_generator_supports_correlation() {
        let cfg = MultivariateConfig {
            n: 3000,
            d: 3,
            n_changes: 1,
            snr: 2.0,
            noise_std: 1.0,
            correlation: Some(0.8),
            seed: 43,
        };

        let (data, breakpoints) = multivariate(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints, vec![1500]);

        let mut first_dim = Vec::with_capacity(cfg.n);
        let mut second_dim = Vec::with_capacity(cfg.n);
        for t in 0..cfg.n {
            first_dim.push(data[t * cfg.d]);
            second_dim.push(data[t * cfg.d + 1]);
        }
        let rho = correlation(first_dim.as_slice(), second_dim.as_slice());
        assert!(rho > 0.65);
    }

    #[test]
    fn multivariate_generator_supports_independent_dimensions() {
        let cfg = MultivariateConfig {
            n: 8000,
            d: 3,
            n_changes: 24,
            snr: 2.0,
            noise_std: 1.0,
            correlation: None,
            seed: 97,
        };

        let (data, breakpoints) = multivariate(&cfg).expect("generation should succeed");
        assert_eq!(breakpoints.len(), cfg.n_changes);

        let mut first_dim = Vec::with_capacity(cfg.n);
        let mut second_dim = Vec::with_capacity(cfg.n);
        for t in 0..cfg.n {
            first_dim.push(data[t * cfg.d]);
            second_dim.push(data[t * cfg.d + 1]);
        }
        let rho = correlation(first_dim.as_slice(), second_dim.as_slice());
        assert!(
            rho.abs() < 0.15,
            "expected near-independent dims; rho={rho}"
        );
    }

    #[test]
    fn missing_generator_is_seeded_and_inserts_expected_nan_count() {
        let cfg = MissingDataConfig {
            n: 1000,
            d: 2,
            n_changes: 1,
            snr: 2.0,
            noise_std: 1.0,
            missing_fraction: 0.1,
            pattern: MissingPattern::Random,
            seed: 47,
        };

        let (first_data, first_breakpoints) =
            missing_data(&cfg).expect("generation should succeed");
        let (second_data, second_breakpoints) =
            missing_data(&cfg).expect("generation should succeed");

        assert_eq!(first_breakpoints, vec![500]);
        assert_eq!(first_breakpoints, second_breakpoints);
        assert_same_bits(first_data.as_slice(), second_data.as_slice());

        let expected_missing = (cfg.n * cfg.d) / 10;
        assert_eq!(count_missing(first_data.as_slice()), expected_missing);
    }

    #[test]
    fn missing_block_pattern_creates_contiguous_gaps() {
        let cfg = MissingDataConfig {
            n: 300,
            d: 2,
            n_changes: 0,
            snr: 1.0,
            noise_std: 1.0,
            missing_fraction: 0.2,
            pattern: MissingPattern::Block { gap_length: 8 },
            seed: 53,
        };

        let (data, _breakpoints) = missing_data(&cfg).expect("generation should succeed");

        let mut longest_run = 0usize;
        for dim in 0..cfg.d {
            let mut run = 0usize;
            for t in 0..cfg.n {
                if data[t * cfg.d + dim].is_nan() {
                    run += 1;
                    longest_run = longest_run.max(run);
                } else {
                    run = 0;
                }
            }
        }

        assert!(longest_run >= 8);
    }

    #[test]
    fn missing_periodic_pattern_inserts_regular_gaps() {
        let cfg = MissingDataConfig {
            n: 120,
            d: 1,
            n_changes: 0,
            snr: 1.0,
            noise_std: 0.1,
            missing_fraction: 0.25,
            pattern: MissingPattern::Periodic {
                every: 10,
                width: 2,
            },
            seed: 59,
        };

        let (data, _breakpoints) = missing_data(&cfg).expect("generation should succeed");
        for start in (0..cfg.n).step_by(10) {
            assert!(data[start].is_nan());
        }
    }

    #[test]
    fn to_offline_breakpoints_appends_terminal_index() {
        let breakpoints =
            to_offline_breakpoints(120, &[40, 80]).expect("conversion should succeed");
        assert_eq!(breakpoints, vec![40, 80, 120]);
    }

    #[test]
    fn to_offline_breakpoints_rejects_invalid_change_points() {
        let err = to_offline_breakpoints(10, &[0, 4]).expect_err("0 should be rejected");
        assert!(err.to_string().contains("must be in [1, 10)"));

        let err = to_offline_breakpoints(10, &[4, 4]).expect_err("non-strict order should fail");
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn generators_reject_flatten_length_overflow() {
        let cfg = PiecewiseMeanConfig {
            n: usize::MAX,
            d: 2,
            n_changes: 1,
            snr: 1.0,
            noise_std: 1.0,
            seed: 0,
        };

        let err = piecewise_constant_mean(&cfg).expect_err("overflow should be rejected");
        assert!(err.to_string().contains("n*d overflows usize"));
    }
}
