// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Returns `log(sum(exp(values)))` using a numerically stable max-subtract transform.
///
/// Semantics:
/// - Empty input returns `-inf`.
/// - Inputs containing `+inf` return `+inf`.
/// - Inputs containing `NaN` (without any `+inf`) return `NaN`.
/// - Inputs containing only `-inf` return `-inf`.
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max = f64::NEG_INFINITY;
    let mut has_nan = false;

    for &value in values {
        if value == f64::INFINITY {
            return f64::INFINITY;
        }
        if value.is_nan() {
            has_nan = true;
            continue;
        }
        if value > max {
            max = value;
        }
    }

    if has_nan {
        return f64::NAN;
    }

    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    let mut sum_exp = 0.0;
    for &value in values {
        sum_exp += (value - max).exp();
    }

    max + sum_exp.ln()
}

/// Returns `log(exp(a) + exp(b))` using a numerically stable branch form.
///
/// Semantics:
/// - `(+inf, anything)` returns `+inf`.
/// - `(-inf, -inf)` returns `-inf`.
/// - If either input is `NaN` and no `+inf` short-circuit applies, returns `NaN`.
pub fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }

    let max = a.max(b);
    let min = a.min(b);
    max + (min - max).exp().ln_1p()
}

/// Computes the mean using Welford's online update.
///
/// Empty input returns `NaN`.
pub fn stable_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mut mean = 0.0;
    for (idx, &value) in values.iter().enumerate() {
        let n = (idx + 1) as f64;
        mean += (value - mean) / n;
    }
    mean
}

/// Computes population variance (`/ n`) using a two-pass approach over the provided mean.
///
/// Empty input returns `NaN`.
/// Any negative round-off artifact is clamped to `0.0`.
pub fn stable_variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    // Use compensated accumulation on squared residuals for better stability.
    let mut sum_sq = 0.0;
    let mut c = 0.0;
    for &value in values {
        let diff = value - mean;
        let term = diff * diff;

        let y = term - c;
        let t = sum_sq + y;
        c = (t - sum_sq) - y;
        sum_sq = t;
    }

    let variance = sum_sq / values.len() as f64;
    if variance <= 0.0 { 0.0 } else { variance }
}

/// Computes a compensated sum using Kahan summation.
///
/// Empty input returns `0.0`.
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &value in values {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            c += (sum - t) + value;
        } else {
            c += (value - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Returns prefix sums with length `n + 1` and `prefix[0] = 0.0`.
pub fn prefix_sums(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);

    let mut sum = 0.0;
    for &value in values {
        sum += value;
        prefix.push(sum);
    }

    prefix
}

/// Returns prefix sums of squares with length `n + 1` and `prefix[0] = 0.0`.
pub fn prefix_sum_squares(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);

    let mut sum = 0.0;
    for &value in values {
        sum += value * value;
        prefix.push(sum);
    }

    prefix
}

/// Returns Kahan-compensated prefix sums with length `n + 1` and `prefix[0] = 0.0`.
pub fn prefix_sums_kahan(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);

    let mut sum = 0.0;
    let mut c = 0.0;
    for &value in values {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            c += (sum - t) + value;
        } else {
            c += (value - t) + sum;
        }
        sum = t;
        prefix.push(sum + c);
    }

    prefix
}

/// Returns Kahan-compensated prefix sums of squares with length `n + 1` and `prefix[0] = 0.0`.
pub fn prefix_sum_squares_kahan(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);

    let mut sum = 0.0;
    let mut c = 0.0;
    for &value in values {
        let square = value * value;
        let t = sum + square;
        if sum.abs() >= square.abs() {
            c += (sum - t) + square;
        } else {
            c += (square - t) + sum;
        }
        sum = t;
        prefix.push(sum + c);
    }

    prefix
}

#[cfg(test)]
mod tests {
    use super::{
        kahan_sum, log_add_exp, log_sum_exp, prefix_sum_squares, prefix_sum_squares_kahan,
        prefix_sums, prefix_sums_kahan, stable_mean, stable_variance,
    };

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual}, |diff|={diff}, tol={tol}"
        );
    }

    fn assert_close_rel(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) {
        let diff = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs());
        let limit = abs_tol.max(scale * rel_tol);
        assert!(
            diff <= limit,
            "expected {expected}, got {actual}, |diff|={diff}, limit={limit}"
        );
    }

    #[test]
    fn log_sum_exp_handles_empty_and_all_neg_inf() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn log_sum_exp_handles_single_and_extreme_magnitudes() {
        assert_eq!(log_sum_exp(&[2.5]), 2.5);

        let value = log_sum_exp(&[1000.0, -1000.0]);
        assert_close(value, 1000.0, 1e-12);
    }

    #[test]
    fn log_sum_exp_pos_inf_short_circuits_nan() {
        let value = log_sum_exp(&[f64::NAN, f64::INFINITY, 1.0]);
        assert_eq!(value, f64::INFINITY);
    }

    #[test]
    fn log_sum_exp_nan_propagates_without_pos_inf() {
        assert!(log_sum_exp(&[1.0, f64::NAN]).is_nan());
    }

    #[test]
    fn log_add_exp_is_symmetric_and_matches_log_sum_exp() {
        let cases = [
            (-3.0, -1.0),
            (0.0, 0.0),
            (100.0, -100.0),
            (f64::NEG_INFINITY, -2.0),
            (-2.0, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            (f64::INFINITY, -1.0),
        ];

        for (a, b) in cases {
            let lhs = log_add_exp(a, b);
            let rhs = log_add_exp(b, a);
            if lhs.is_nan() {
                assert!(rhs.is_nan());
            } else {
                assert_eq!(lhs, rhs);
            }

            let pair = log_sum_exp(&[a, b]);
            if pair.is_nan() {
                assert!(lhs.is_nan());
            } else if pair.is_infinite() {
                assert_eq!(lhs, pair);
            } else {
                assert_close(lhs, pair, 1e-12);
            }
        }
    }

    #[test]
    fn log_add_exp_handles_special_values() {
        assert_eq!(
            log_add_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(log_add_exp(f64::INFINITY, -10.0), f64::INFINITY);
        assert!(log_add_exp(f64::NAN, -10.0).is_nan());
        assert!(log_add_exp(-10.0, f64::NAN).is_nan());
    }

    #[test]
    fn stable_stats_known_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let mean = stable_mean(&values);
        let variance = stable_variance(&values, mean);

        assert_close(mean, 2.5, 1e-12);
        assert_close(variance, 1.25, 1e-12);
    }

    #[test]
    fn stable_stats_empty_input() {
        assert!(stable_mean(&[]).is_nan());
        assert!(stable_variance(&[], 0.0).is_nan());
    }

    #[test]
    fn stable_stats_near_constant_large_magnitude() {
        let values = [1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0, 1e12 + 4.0];
        let mean = stable_mean(&values);
        let variance = stable_variance(&values, mean);

        assert_close_rel(mean, 1e12 + 2.5, 1e-9, 1e-15);
        assert_close_rel(variance, 1.25, 1e-9, 1e-12);
        assert!(variance >= 0.0);
    }

    #[test]
    fn stable_variance_is_non_negative() {
        let values = [1.0, 1.0 + 1e-12, 1.0 - 1e-12, 1.0];
        let mean = stable_mean(&values);
        let variance = stable_variance(&values, mean);
        assert!(variance >= 0.0);
    }

    #[test]
    fn prefix_helpers_shape_and_empty_behavior() {
        assert_eq!(prefix_sums(&[]), vec![0.0]);
        assert_eq!(prefix_sum_squares(&[]), vec![0.0]);
        assert_eq!(prefix_sums_kahan(&[]), vec![0.0]);
        assert_eq!(prefix_sum_squares_kahan(&[]), vec![0.0]);

        let values = [1.0, -2.0, 3.0];
        let prefix = prefix_sums(&values);
        let prefix_sq = prefix_sum_squares(&values);
        assert_eq!(prefix.len(), values.len() + 1);
        assert_eq!(prefix_sq.len(), values.len() + 1);
        assert_eq!(prefix[0], 0.0);
        assert_eq!(prefix_sq[0], 0.0);
    }

    #[test]
    fn prefix_helpers_match_segment_identities() {
        let values = [-2.0, 0.5, 1.25, -3.5, 7.0, 4.75];
        let prefix = prefix_sums(&values);
        let prefix_sq = prefix_sum_squares(&values);

        for start in 0..=values.len() {
            for end in start..=values.len() {
                let expected_sum: f64 = values[start..end].iter().sum();
                let actual_sum = prefix[end] - prefix[start];
                assert_close(actual_sum, expected_sum, 1e-12);

                let expected_sq_sum: f64 = values[start..end].iter().map(|x| x * x).sum();
                let actual_sq_sum = prefix_sq[end] - prefix_sq[start];
                assert_close(actual_sq_sum, expected_sq_sum, 1e-12);
            }
        }
    }

    #[test]
    fn kahan_sum_improves_cancellation_accuracy() {
        let values = [1e16, 1.0, -1e16];
        let naive_sum: f64 = values.iter().sum();
        let compensated = kahan_sum(&values);

        assert_eq!(naive_sum, 0.0);
        assert_close(compensated, 1.0, 1e-12);
    }

    #[test]
    fn prefix_kahan_tracks_compensated_total() {
        let values = [1e16, 1.0, -1e16];

        let prefix_default = prefix_sums(&values);
        let prefix_kahan = prefix_sums_kahan(&values);

        let default_last = *prefix_default
            .last()
            .expect("default prefix should be non-empty");
        let kahan_last = *prefix_kahan
            .last()
            .expect("kahan prefix should be non-empty");

        assert_eq!(kahan_last, kahan_sum(&values));
        assert_close(kahan_last, 1.0, 1e-12);
        assert!(default_last != kahan_last);
    }
}
