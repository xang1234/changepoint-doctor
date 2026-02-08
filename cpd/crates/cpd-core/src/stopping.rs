// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;

/// Penalty specification for offline segmentation algorithms.
#[derive(Clone, Debug, PartialEq)]
pub enum Penalty {
    BIC,
    AIC,
    Manual(f64),
}

/// Stopping strategy for offline segmentation algorithms.
#[derive(Clone, Debug, PartialEq)]
pub enum Stopping {
    KnownK(usize),
    Penalized(Penalty),
    PenaltyPath(Vec<Penalty>),
}

/// Validates a penalty value.
pub fn validate_penalty(penalty: &Penalty) -> Result<(), CpdError> {
    match penalty {
        Penalty::BIC | Penalty::AIC => Ok(()),
        Penalty::Manual(p) => {
            if !p.is_finite() || *p <= 0.0 {
                return Err(CpdError::invalid_input(format!(
                    "Penalty::Manual requires a finite value > 0.0; got {p}"
                )));
            }
            Ok(())
        }
    }
}

/// Validates a stopping strategy.
pub fn validate_stopping(stopping: &Stopping) -> Result<(), CpdError> {
    match stopping {
        Stopping::KnownK(k) => {
            if *k == 0 {
                return Err(CpdError::invalid_input(
                    "Stopping::KnownK requires k >= 1; got 0",
                ));
            }
            Ok(())
        }
        Stopping::Penalized(penalty) => validate_penalty(penalty),
        Stopping::PenaltyPath(path) => {
            if path.is_empty() {
                return Err(CpdError::invalid_input(
                    "Stopping::PenaltyPath requires a non-empty penalty list; got length 0",
                ));
            }
            for (idx, penalty) in path.iter().enumerate() {
                validate_penalty(penalty).map_err(|err| {
                    CpdError::invalid_input(format!(
                        "Stopping::PenaltyPath[{idx}] is invalid: {err}"
                    ))
                })?;
            }
            Ok(())
        }
    }
}

/// Computes the effective penalty value for a series/configuration.
///
/// Formulas:
/// - effective_params = d * params_per_segment
/// - BIC = effective_params * ln(n)
/// - AIC = 2 * effective_params
/// - Manual(p) = p
pub fn penalty_value(
    penalty: &Penalty,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<f64, CpdError> {
    validate_penalty(penalty)?;

    if n == 0 {
        return Err(CpdError::invalid_input(
            "penalty_value requires n >= 1; got n=0",
        ));
    }
    if d == 0 {
        return Err(CpdError::invalid_input(
            "penalty_value requires d >= 1; got d=0",
        ));
    }
    if params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "penalty_value requires params_per_segment >= 1; got 0",
        ));
    }

    let effective_params = d.checked_mul(params_per_segment).ok_or_else(|| {
        CpdError::invalid_input(format!(
            "penalty_value overflow: d * params_per_segment exceeds usize (d={d}, params_per_segment={params_per_segment})"
        ))
    })?;
    let effective_params_f64 = effective_params as f64;

    let value = match penalty {
        Penalty::BIC => effective_params_f64 * (n as f64).ln(),
        Penalty::AIC => 2.0 * effective_params_f64,
        Penalty::Manual(p) => *p,
    };

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{Penalty, Stopping, penalty_value, validate_penalty, validate_stopping};

    const EPS: f64 = 1e-12;

    #[test]
    fn validate_penalty_accepts_bic_aic_and_positive_manual() {
        validate_penalty(&Penalty::BIC).expect("BIC should validate");
        validate_penalty(&Penalty::AIC).expect("AIC should validate");
        validate_penalty(&Penalty::Manual(0.1)).expect("positive manual should validate");
    }

    #[test]
    fn validate_penalty_rejects_manual_zero_negative_nan_inf() {
        let zero = validate_penalty(&Penalty::Manual(0.0)).expect_err("zero should fail");
        assert!(zero.to_string().contains("Manual"));

        let negative = validate_penalty(&Penalty::Manual(-1.0)).expect_err("negative should fail");
        assert!(negative.to_string().contains("Manual"));

        let nan = validate_penalty(&Penalty::Manual(f64::NAN)).expect_err("NaN should fail");
        assert!(nan.to_string().contains("Manual"));

        let inf = validate_penalty(&Penalty::Manual(f64::INFINITY)).expect_err("inf should fail");
        assert!(inf.to_string().contains("Manual"));

        let neg_inf =
            validate_penalty(&Penalty::Manual(f64::NEG_INFINITY)).expect_err("neg inf should fail");
        assert!(neg_inf.to_string().contains("Manual"));
    }

    #[test]
    fn validate_stopping_rejects_known_k_zero() {
        let err = validate_stopping(&Stopping::KnownK(0)).expect_err("KnownK(0) should fail");
        assert!(err.to_string().contains("KnownK"));
    }

    #[test]
    fn validate_stopping_accepts_known_k_positive_and_penalized() {
        validate_stopping(&Stopping::KnownK(3)).expect("KnownK(3) should pass");
        validate_stopping(&Stopping::Penalized(Penalty::BIC)).expect("Penalized(BIC) should pass");
        validate_stopping(&Stopping::Penalized(Penalty::Manual(1.5)))
            .expect("Penalized(Manual) should pass");
    }

    #[test]
    fn validate_stopping_rejects_empty_penalty_path() {
        let err = validate_stopping(&Stopping::PenaltyPath(vec![]))
            .expect_err("empty penalty path should fail");
        assert!(err.to_string().contains("non-empty"));
    }

    #[test]
    fn validate_stopping_rejects_penalty_path_with_invalid_manual() {
        let err = validate_stopping(&Stopping::PenaltyPath(vec![
            Penalty::BIC,
            Penalty::Manual(0.0),
        ]))
        .expect_err("invalid manual in path should fail");
        assert!(err.to_string().contains("PenaltyPath[1]"));
    }

    #[test]
    fn penalty_value_bic_computes_expected_values() {
        let bic_1d = penalty_value(&Penalty::BIC, 100, 1, 2).expect("BIC should compute");
        let expected_1d = 2.0 * 100.0_f64.ln();
        assert!((bic_1d - expected_1d).abs() < EPS);

        let bic_3d = penalty_value(&Penalty::BIC, 100, 3, 2).expect("BIC should compute");
        let expected_3d = 6.0 * 100.0_f64.ln();
        assert!((bic_3d - expected_3d).abs() < EPS);
    }

    #[test]
    fn penalty_value_aic_computes_expected_values() {
        let aic_1d = penalty_value(&Penalty::AIC, 100, 1, 2).expect("AIC should compute");
        assert!((aic_1d - 4.0).abs() < EPS);

        let aic_3d = penalty_value(&Penalty::AIC, 100, 3, 2).expect("AIC should compute");
        assert!((aic_3d - 12.0).abs() < EPS);
    }

    #[test]
    fn penalty_value_manual_returns_input() {
        let p = penalty_value(&Penalty::Manual(12.5), 100, 2, 3).expect("manual should compute");
        assert!((p - 12.5).abs() < EPS);
    }

    #[test]
    fn penalty_value_rejects_zero_inputs_and_overflow() {
        let n_err = penalty_value(&Penalty::BIC, 0, 1, 1).expect_err("n=0 should fail");
        assert!(n_err.to_string().contains("n >= 1"));

        let d_err = penalty_value(&Penalty::BIC, 10, 0, 1).expect_err("d=0 should fail");
        assert!(d_err.to_string().contains("d >= 1"));

        let pps_err = penalty_value(&Penalty::BIC, 10, 1, 0).expect_err("params=0 should fail");
        assert!(pps_err.to_string().contains("params_per_segment"));

        let overflow_err =
            penalty_value(&Penalty::AIC, 10, usize::MAX, 2).expect_err("overflow should fail");
        assert!(overflow_err.to_string().contains("overflow"));
    }
}
