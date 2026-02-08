// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::{CpdError, Diagnostics};

/// Per-segment summary statistics for offline results.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentStats {
    pub start: usize,
    pub end: usize,
    pub mean: Option<Vec<f64>>,
    pub variance: Option<Vec<f64>>,
    pub count: usize,
    pub missing_count: usize,
}

/// Structured result returned by offline detectors.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineChangePointResult {
    pub breakpoints: Vec<usize>,
    pub change_points: Vec<usize>,
    pub scores: Option<Vec<f64>>,
    pub segments: Option<Vec<SegmentStats>>,
    pub diagnostics: Diagnostics,
}

fn derive_change_points(n: usize, breakpoints: &[usize]) -> Vec<usize> {
    breakpoints.iter().copied().filter(|&bp| bp < n).collect()
}

/// Validates breakpoint conventions used by offline detectors.
pub fn validate_breakpoints(n: usize, breakpoints: &[usize]) -> Result<(), CpdError> {
    if n == 0 {
        if breakpoints.is_empty() {
            return Ok(());
        }
        return Err(CpdError::invalid_input(format!(
            "breakpoints must be empty when n=0; got breakpoints={breakpoints:?}"
        )));
    }

    if breakpoints.is_empty() {
        return Err(CpdError::invalid_input(format!(
            "breakpoints must be non-empty and include n={n} as the final element"
        )));
    }

    let mut prev: Option<usize> = None;
    for (idx, &bp) in breakpoints.iter().enumerate() {
        if bp == 0 {
            return Err(CpdError::invalid_input(format!(
                "breakpoints[{idx}] must be > 0; got 0"
            )));
        }
        if bp > n {
            return Err(CpdError::invalid_input(format!(
                "breakpoints[{idx}] must be <= n; got breakpoint={bp}, n={n}"
            )));
        }
        if let Some(prev_bp) = prev
            && bp <= prev_bp
        {
            return Err(CpdError::invalid_input(format!(
                "breakpoints must be strictly increasing and unique: breakpoints[{idx}]={bp}, previous={prev_bp}"
            )));
        }
        prev = Some(bp);
    }

    let last = *breakpoints.last().expect("checked non-empty above");
    if last != n {
        return Err(CpdError::invalid_input(format!(
            "breakpoints must include n as the final element: last={last}, n={n}"
        )));
    }

    Ok(())
}

/// Converts validated breakpoints into contiguous `[start, end)` segments.
pub fn segments_from_breakpoints(n: usize, breakpoints: &[usize]) -> Vec<(usize, usize)> {
    debug_assert!(
        validate_breakpoints(n, breakpoints).is_ok(),
        "segments_from_breakpoints expects validated breakpoints"
    );

    if n == 0 {
        return vec![];
    }

    let mut segments = Vec::with_capacity(breakpoints.len());
    let mut start = 0usize;
    for &end in breakpoints {
        segments.push((start, end));
        start = end;
    }
    segments
}

impl OfflineChangePointResult {
    /// Constructs an offline result and derives `change_points` from breakpoints.
    pub fn new(
        n: usize,
        breakpoints: Vec<usize>,
        diagnostics: Diagnostics,
    ) -> Result<Self, CpdError> {
        validate_breakpoints(n, &breakpoints)?;
        Ok(Self {
            change_points: derive_change_points(n, &breakpoints),
            breakpoints,
            scores: None,
            segments: None,
            diagnostics,
        })
    }

    /// Adds optional per-change-point scores after validating shape.
    pub fn with_scores(mut self, scores: Vec<f64>) -> Result<Self, CpdError> {
        if scores.len() != self.change_points.len() {
            return Err(CpdError::invalid_input(format!(
                "scores length must equal change_points length; got scores={}, change_points={}",
                scores.len(),
                self.change_points.len()
            )));
        }
        self.scores = Some(scores);
        Ok(self)
    }

    /// Adds optional per-segment stats after validating shape and consistency.
    pub fn with_segments(mut self, segments: Vec<SegmentStats>) -> Result<Self, CpdError> {
        self.segments = Some(segments);
        let n = self.breakpoints.last().copied().unwrap_or(0);
        self.validate(n)?;
        Ok(self)
    }

    /// Validates cross-field consistency and documented conventions.
    pub fn validate(&self, n: usize) -> Result<(), CpdError> {
        validate_breakpoints(n, &self.breakpoints)?;

        let expected_change_points = derive_change_points(n, &self.breakpoints);
        if self.change_points != expected_change_points {
            return Err(CpdError::invalid_input(format!(
                "change_points must equal breakpoints excluding n; got change_points={:?}, expected={:?}",
                self.change_points, expected_change_points
            )));
        }

        if let Some(scores) = &self.scores
            && scores.len() != self.change_points.len()
        {
            return Err(CpdError::invalid_input(format!(
                "scores length must equal change_points length; got scores={}, change_points={}",
                scores.len(),
                self.change_points.len()
            )));
        }

        if let Some(segments) = &self.segments {
            if segments.len() != self.breakpoints.len() {
                return Err(CpdError::invalid_input(format!(
                    "segments length must equal breakpoints length; got segments={}, breakpoints={}",
                    segments.len(),
                    self.breakpoints.len()
                )));
            }

            let expected_bounds = segments_from_breakpoints(n, &self.breakpoints);
            for (idx, (segment, (expected_start, expected_end))) in segments
                .iter()
                .zip(expected_bounds.iter().copied())
                .enumerate()
            {
                if segment.end < segment.start {
                    return Err(CpdError::invalid_input(format!(
                        "segments[{idx}] has end < start: start={}, end={}",
                        segment.start, segment.end
                    )));
                }

                let expected_count = segment.end - segment.start;
                if segment.count != expected_count {
                    return Err(CpdError::invalid_input(format!(
                        "segments[{idx}] count mismatch: count={}, expected={} from [start={}, end={})",
                        segment.count, expected_count, segment.start, segment.end
                    )));
                }

                if segment.missing_count > segment.count {
                    return Err(CpdError::invalid_input(format!(
                        "segments[{idx}] missing_count must be <= count; got missing_count={}, count={}",
                        segment.missing_count, segment.count
                    )));
                }

                if segment.start != expected_start || segment.end != expected_end {
                    return Err(CpdError::invalid_input(format!(
                        "segments[{idx}] boundaries mismatch: got [start={}, end={}), expected [start={}, end={})",
                        segment.start, segment.end, expected_start, expected_end
                    )));
                }

                if self.diagnostics.d > 0 {
                    if let Some(mean) = &segment.mean
                        && mean.len() != self.diagnostics.d
                    {
                        return Err(CpdError::invalid_input(format!(
                            "segments[{idx}].mean length must equal diagnostics.d; got mean_len={}, diagnostics.d={}",
                            mean.len(),
                            self.diagnostics.d
                        )));
                    }
                    if let Some(variance) = &segment.variance
                        && variance.len() != self.diagnostics.d
                    {
                        return Err(CpdError::invalid_input(format!(
                            "segments[{idx}].variance length must equal diagnostics.d; got variance_len={}, diagnostics.d={}",
                            variance.len(),
                            self.diagnostics.d
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OfflineChangePointResult, SegmentStats, segments_from_breakpoints, validate_breakpoints,
    };
    use crate::Diagnostics;
    use std::borrow::Cow;

    fn diagnostics_with_shape(n: usize, d: usize) -> Diagnostics {
        Diagnostics {
            n,
            d,
            algorithm: Cow::Borrowed("test"),
            cost_model: Cow::Borrowed("l2"),
            ..Diagnostics::default()
        }
    }

    #[test]
    fn validate_breakpoints_accepts_no_change_case_with_n_only() {
        validate_breakpoints(100, &[100]).expect("n-only breakpoints should validate");
    }

    #[test]
    fn validate_breakpoints_rejects_missing_terminal_n() {
        let err = validate_breakpoints(100, &[50]).expect_err("missing n should fail");
        assert!(err.to_string().contains("final element"));
    }

    #[test]
    fn validate_breakpoints_rejects_zero_unsorted_duplicates_and_out_of_range() {
        let zero_err = validate_breakpoints(100, &[0, 100]).expect_err("0 breakpoint should fail");
        assert!(zero_err.to_string().contains("must be > 0"));

        let unsorted_err =
            validate_breakpoints(100, &[60, 50, 100]).expect_err("unsorted should fail");
        assert!(unsorted_err.to_string().contains("strictly increasing"));

        let dup_err = validate_breakpoints(100, &[50, 50, 100]).expect_err("duplicate should fail");
        assert!(dup_err.to_string().contains("strictly increasing"));

        let oob_err = validate_breakpoints(100, &[50, 101]).expect_err("out-of-range should fail");
        assert!(oob_err.to_string().contains("must be <= n"));
    }

    #[test]
    fn segments_from_breakpoints_empty_n_returns_empty() {
        assert_eq!(segments_from_breakpoints(0, &[]), vec![]);
    }

    #[test]
    fn segments_from_breakpoints_single_change() {
        assert_eq!(
            segments_from_breakpoints(100, &[50, 100]),
            vec![(0, 50), (50, 100)]
        );
    }

    #[test]
    fn segments_from_breakpoints_many_changes() {
        assert_eq!(
            segments_from_breakpoints(120, &[25, 50, 75, 120]),
            vec![(0, 25), (25, 50), (50, 75), (75, 120)]
        );
    }

    #[test]
    fn offline_result_new_derives_change_points_correctly() {
        let diagnostics = diagnostics_with_shape(100, 1);
        let result =
            OfflineChangePointResult::new(100, vec![50, 100], diagnostics).expect("new valid");
        assert_eq!(result.breakpoints, vec![50, 100]);
        assert_eq!(result.change_points, vec![50]);
        assert!(result.scores.is_none());
        assert!(result.segments.is_none());
    }

    #[test]
    fn offline_result_with_scores_validates_length() {
        let diagnostics = diagnostics_with_shape(100, 1);
        let result = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_scores(vec![0.9])
            .expect("scores with matching length should pass");
        assert_eq!(result.scores, Some(vec![0.9]));

        let diagnostics = diagnostics_with_shape(100, 1);
        let err = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_scores(vec![0.1, 0.2])
            .expect_err("scores mismatch should fail");
        assert!(err.to_string().contains("scores length"));
    }

    #[test]
    fn offline_result_with_segments_validates_shape_and_contiguity() {
        let diagnostics = diagnostics_with_shape(100, 2);
        let segments = vec![
            SegmentStats {
                start: 0,
                end: 50,
                mean: Some(vec![0.5, 1.0]),
                variance: Some(vec![0.1, 0.2]),
                count: 50,
                missing_count: 2,
            },
            SegmentStats {
                start: 50,
                end: 100,
                mean: Some(vec![1.5, 2.0]),
                variance: Some(vec![0.3, 0.4]),
                count: 50,
                missing_count: 1,
            },
        ];

        let result = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_segments(segments.clone())
            .expect("valid segments should pass");
        assert_eq!(result.segments, Some(segments));

        let diagnostics = diagnostics_with_shape(100, 2);
        let non_contiguous = vec![
            SegmentStats {
                start: 0,
                end: 49,
                mean: Some(vec![0.1, 0.2]),
                variance: Some(vec![0.1, 0.2]),
                count: 49,
                missing_count: 0,
            },
            SegmentStats {
                start: 50,
                end: 100,
                mean: Some(vec![0.1, 0.2]),
                variance: Some(vec![0.1, 0.2]),
                count: 50,
                missing_count: 0,
            },
        ];
        let err = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_segments(non_contiguous)
            .expect_err("non-contiguous boundaries should fail");
        assert!(err.to_string().contains("boundaries mismatch"));

        let diagnostics = diagnostics_with_shape(100, 2);
        let bad_counts = vec![
            SegmentStats {
                start: 0,
                end: 50,
                mean: Some(vec![0.1, 0.2]),
                variance: Some(vec![0.1, 0.2]),
                count: 49,
                missing_count: 0,
            },
            SegmentStats {
                start: 50,
                end: 100,
                mean: Some(vec![0.1, 0.2]),
                variance: Some(vec![0.1, 0.2]),
                count: 50,
                missing_count: 0,
            },
        ];
        let err = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_segments(bad_counts)
            .expect_err("count mismatch should fail");
        assert!(err.to_string().contains("count mismatch"));
    }

    #[test]
    fn offline_result_validate_catches_manual_field_inconsistency() {
        let mut result =
            OfflineChangePointResult::new(100, vec![50, 100], diagnostics_with_shape(100, 1))
                .expect("new valid");
        result.change_points = vec![40];

        let err = result
            .validate(100)
            .expect_err("manual mismatch must be caught");
        assert!(err.to_string().contains("change_points"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn offline_result_serde_roundtrip() {
        let diagnostics = diagnostics_with_shape(100, 2);
        let result = OfflineChangePointResult::new(100, vec![50, 100], diagnostics)
            .expect("new valid")
            .with_scores(vec![0.85])
            .expect("scores valid")
            .with_segments(vec![
                SegmentStats {
                    start: 0,
                    end: 50,
                    mean: Some(vec![1.0, 2.0]),
                    variance: Some(vec![0.5, 0.75]),
                    count: 50,
                    missing_count: 5,
                },
                SegmentStats {
                    start: 50,
                    end: 100,
                    mean: Some(vec![2.0, 3.0]),
                    variance: Some(vec![0.6, 0.8]),
                    count: 50,
                    missing_count: 3,
                },
            ])
            .expect("segments valid");

        let encoded = serde_json::to_string(&result).expect("result should serialize");
        let decoded: OfflineChangePointResult =
            serde_json::from_str(&encoded).expect("result should deserialize");
        assert_eq!(decoded, result);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn segment_stats_serde_roundtrip() {
        let stats = SegmentStats {
            start: 0,
            end: 10,
            mean: Some(vec![1.0]),
            variance: Some(vec![0.5]),
            count: 10,
            missing_count: 1,
        };

        let encoded = serde_json::to_string(&stats).expect("stats should serialize");
        let decoded: SegmentStats =
            serde_json::from_str(&encoded).expect("stats should deserialize");
        assert_eq!(decoded, stats);
    }
}
