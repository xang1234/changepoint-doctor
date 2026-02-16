// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    DIAGNOSTICS_SCHEMA_VERSION, Diagnostics as CoreDiagnostics,
    OfflineChangePointResult as CoreOfflineChangePointResult,
    OnlineStepResult as CoreOnlineStepResult, PruningStats as CorePruningStats, ReproMode,
    SegmentStats as CoreSegmentStats,
};
#[cfg(not(feature = "serde"))]
use pyo3::exceptions::PyNotImplementedError;
#[cfg(feature = "serde")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "serde")]
use pyo3::types::{PyAnyMethods, PyModule};
#[cfg(feature = "serde")]
use std::borrow::Cow;

fn repro_mode_to_str(mode: ReproMode) -> &'static str {
    match mode {
        ReproMode::Strict => "strict",
        ReproMode::Balanced => "balanced",
        ReproMode::Fast => "fast",
    }
}

#[cfg(feature = "serde")]
fn repro_mode_from_str(value: &str) -> ReproMode {
    match value {
        "strict" | "Strict" => ReproMode::Strict,
        "balanced" | "Balanced" => ReproMode::Balanced,
        "fast" | "Fast" => ReproMode::Fast,
        _ => ReproMode::Balanced,
    }
}

fn derive_change_points(n: usize, breakpoints: &[usize]) -> Vec<usize> {
    breakpoints.iter().copied().filter(|&bp| bp < n).collect()
}

#[pyclass(module = "cpd._cpd_rs", name = "PruningStats", frozen)]
#[derive(Clone, Debug)]
pub struct PyPruningStats {
    candidates_considered: usize,
    candidates_pruned: usize,
}

#[pymethods]
impl PyPruningStats {
    #[getter]
    fn candidates_considered(&self) -> usize {
        self.candidates_considered
    }

    #[getter]
    fn candidates_pruned(&self) -> usize {
        self.candidates_pruned
    }

    fn __repr__(&self) -> String {
        format!(
            "PruningStats(candidates_considered={}, candidates_pruned={})",
            self.candidates_considered, self.candidates_pruned
        )
    }
}

impl From<CorePruningStats> for PyPruningStats {
    fn from(stats: CorePruningStats) -> Self {
        Self {
            candidates_considered: stats.candidates_considered,
            candidates_pruned: stats.candidates_pruned,
        }
    }
}

impl PyPruningStats {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> CorePruningStats {
        CorePruningStats {
            candidates_considered: self.candidates_considered,
            candidates_pruned: self.candidates_pruned,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "SegmentStats", frozen)]
#[derive(Clone, Debug)]
pub struct PySegmentStats {
    start: usize,
    end: usize,
    mean: Option<Vec<f64>>,
    variance: Option<Vec<f64>>,
    count: usize,
    missing_count: usize,
}

#[pymethods]
impl PySegmentStats {
    #[getter]
    fn start(&self) -> usize {
        self.start
    }

    #[getter]
    fn end(&self) -> usize {
        self.end
    }

    #[getter]
    fn mean(&self) -> Option<Vec<f64>> {
        self.mean.clone()
    }

    #[getter]
    fn variance(&self) -> Option<Vec<f64>> {
        self.variance.clone()
    }

    #[getter]
    fn count(&self) -> usize {
        self.count
    }

    #[getter]
    fn missing_count(&self) -> usize {
        self.missing_count
    }

    fn __repr__(&self) -> String {
        format!(
            "SegmentStats(start={}, end={}, count={}, missing_count={})",
            self.start, self.end, self.count, self.missing_count
        )
    }
}

impl From<CoreSegmentStats> for PySegmentStats {
    fn from(stats: CoreSegmentStats) -> Self {
        Self {
            start: stats.start,
            end: stats.end,
            mean: stats.mean,
            variance: stats.variance,
            count: stats.count,
            missing_count: stats.missing_count,
        }
    }
}

impl PySegmentStats {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> CoreSegmentStats {
        CoreSegmentStats {
            start: self.start,
            end: self.end,
            mean: self.mean.clone(),
            variance: self.variance.clone(),
            count: self.count,
            missing_count: self.missing_count,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "Diagnostics", frozen)]
#[derive(Clone, Debug)]
pub struct PyDiagnostics {
    n: usize,
    d: usize,
    schema_version: u32,
    engine_version: Option<String>,
    runtime_ms: Option<u64>,
    notes: Vec<String>,
    warnings: Vec<String>,
    algorithm: String,
    cost_model: String,
    seed: Option<u64>,
    repro_mode: String,
    thread_count: Option<usize>,
    blas_backend: Option<String>,
    cpu_features: Option<Vec<String>>,
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    params_json_text: Option<String>,
    pruning_stats: Option<PyPruningStats>,
    missing_policy_applied: Option<String>,
    missing_fraction: Option<f64>,
    effective_sample_count: Option<usize>,
}

#[pymethods]
impl PyDiagnostics {
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn d(&self) -> usize {
        self.d
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[getter]
    fn engine_version(&self) -> Option<String> {
        self.engine_version.clone()
    }

    #[getter]
    fn runtime_ms(&self) -> Option<u64> {
        self.runtime_ms
    }

    #[getter]
    fn notes(&self) -> Vec<String> {
        self.notes.clone()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    #[getter]
    fn algorithm(&self) -> String {
        self.algorithm.clone()
    }

    #[getter]
    fn cost_model(&self) -> String {
        self.cost_model.clone()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.seed
    }

    #[getter]
    fn repro_mode(&self) -> String {
        self.repro_mode.clone()
    }

    #[getter]
    fn thread_count(&self) -> Option<usize> {
        self.thread_count
    }

    #[getter]
    fn blas_backend(&self) -> Option<String> {
        self.blas_backend.clone()
    }

    #[getter]
    fn cpu_features(&self) -> Option<Vec<String>> {
        self.cpu_features.clone()
    }

    #[getter]
    fn params_json<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        #[cfg(feature = "serde")]
        if let Some(serialized) = &self.params_json_text {
            let json = PyModule::import(py, "json")?;
            let value = json.call_method1("loads", (serialized,))?;
            return Ok(value.into_py(py));
        }

        Ok(py.None())
    }

    #[getter]
    fn pruning_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyPruningStats>>> {
        self.pruning_stats
            .clone()
            .map(|stats| Py::new(py, stats))
            .transpose()
    }

    #[getter]
    fn missing_policy_applied(&self) -> Option<String> {
        self.missing_policy_applied.clone()
    }

    #[getter]
    fn missing_fraction(&self) -> Option<f64> {
        self.missing_fraction
    }

    #[getter]
    fn effective_sample_count(&self) -> Option<usize> {
        self.effective_sample_count
    }

    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(n={}, d={}, algorithm='{}', cost_model='{}', repro_mode='{}')",
            self.n, self.d, self.algorithm, self.cost_model, self.repro_mode
        )
    }
}

impl From<CoreDiagnostics> for PyDiagnostics {
    fn from(diagnostics: CoreDiagnostics) -> Self {
        #[cfg(feature = "serde")]
        let params_json_text = diagnostics
            .params_json
            .and_then(|value| serde_json::to_string(&value).ok());
        #[cfg(not(feature = "serde"))]
        let params_json_text = None;

        let normalized_schema_version = if diagnostics.schema_version == 0 {
            DIAGNOSTICS_SCHEMA_VERSION
        } else {
            diagnostics.schema_version
        };

        let normalized_engine_version = diagnostics
            .engine_version
            .or_else(|| Some(env!("CARGO_PKG_VERSION").to_string()));

        Self {
            n: diagnostics.n,
            d: diagnostics.d,
            schema_version: normalized_schema_version,
            engine_version: normalized_engine_version,
            runtime_ms: diagnostics.runtime_ms,
            notes: diagnostics.notes,
            warnings: diagnostics.warnings,
            algorithm: diagnostics.algorithm.into_owned(),
            cost_model: diagnostics.cost_model.into_owned(),
            seed: diagnostics.seed,
            repro_mode: repro_mode_to_str(diagnostics.repro_mode).to_string(),
            thread_count: diagnostics.thread_count,
            blas_backend: diagnostics.blas_backend,
            cpu_features: diagnostics.cpu_features,
            params_json_text,
            pruning_stats: diagnostics.pruning_stats.map(Into::into),
            missing_policy_applied: diagnostics.missing_policy_applied,
            missing_fraction: diagnostics.missing_fraction,
            effective_sample_count: diagnostics.effective_sample_count,
        }
    }
}

impl PyDiagnostics {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> PyResult<CoreDiagnostics> {
        let params_json = match &self.params_json_text {
            Some(serialized) => Some(serde_json::from_str(serialized).map_err(|err| {
                PyValueError::new_err(format!("failed to parse diagnostics.params_json: {err}"))
            })?),
            None => None,
        };

        Ok(CoreDiagnostics {
            n: self.n,
            d: self.d,
            schema_version: self.schema_version,
            engine_version: self.engine_version.clone(),
            runtime_ms: self.runtime_ms,
            notes: self.notes.clone(),
            warnings: self.warnings.clone(),
            algorithm: Cow::Owned(self.algorithm.clone()),
            cost_model: Cow::Owned(self.cost_model.clone()),
            seed: self.seed,
            repro_mode: repro_mode_from_str(&self.repro_mode),
            thread_count: self.thread_count,
            blas_backend: self.blas_backend.clone(),
            cpu_features: self.cpu_features.clone(),
            params_json,
            pruning_stats: self.pruning_stats.as_ref().map(PyPruningStats::to_core),
            missing_policy_applied: self.missing_policy_applied.clone(),
            missing_fraction: self.missing_fraction,
            effective_sample_count: self.effective_sample_count,
        })
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "OnlineStepResult", frozen)]
#[derive(Clone, Debug)]
pub struct PyOnlineStepResult {
    t: usize,
    p_change: f64,
    alert: bool,
    alert_reason: Option<String>,
    run_length_mode: usize,
    run_length_mean: f64,
    processing_latency_us: Option<u64>,
}

#[pymethods]
impl PyOnlineStepResult {
    #[getter]
    fn t(&self) -> usize {
        self.t
    }

    #[getter]
    fn p_change(&self) -> f64 {
        self.p_change
    }

    #[getter]
    fn alert(&self) -> bool {
        self.alert
    }

    #[getter]
    fn alert_reason(&self) -> Option<String> {
        self.alert_reason.clone()
    }

    #[getter]
    fn run_length_mode(&self) -> usize {
        self.run_length_mode
    }

    #[getter]
    fn run_length_mean(&self) -> f64 {
        self.run_length_mean
    }

    #[getter]
    fn processing_latency_us(&self) -> Option<u64> {
        self.processing_latency_us
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineStepResult(t={}, p_change={:.6}, alert={}, run_length_mode={}, run_length_mean={:.3})",
            self.t, self.p_change, self.alert, self.run_length_mode, self.run_length_mean
        )
    }
}

impl From<CoreOnlineStepResult> for PyOnlineStepResult {
    fn from(step: CoreOnlineStepResult) -> Self {
        Self {
            t: step.t,
            p_change: step.p_change,
            alert: step.alert,
            alert_reason: step.alert_reason,
            run_length_mode: step.run_length_mode,
            run_length_mean: step.run_length_mean,
            processing_latency_us: step.processing_latency_us,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "OfflineChangePointResult", frozen)]
#[derive(Clone, Debug)]
pub struct PyOfflineChangePointResult {
    breakpoints: Vec<usize>,
    change_points: Vec<usize>,
    scores: Option<Vec<f64>>,
    segments: Option<Vec<PySegmentStats>>,
    diagnostics: PyDiagnostics,
}

#[pymethods]
impl PyOfflineChangePointResult {
    #[getter]
    fn breakpoints(&self) -> Vec<usize> {
        self.breakpoints.clone()
    }

    #[getter]
    fn change_points(&self) -> Vec<usize> {
        self.change_points.clone()
    }

    #[getter]
    fn scores(&self) -> Option<Vec<f64>> {
        self.scores.clone()
    }

    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Py<PySegmentStats>>>> {
        self.segments
            .clone()
            .map(|segments| {
                segments
                    .into_iter()
                    .map(|segment| Py::new(py, segment))
                    .collect()
            })
            .transpose()
    }

    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDiagnostics>> {
        Py::new(py, self.diagnostics.clone())
    }

    #[cfg(feature = "serde")]
    fn to_json(&self) -> PyResult<String> {
        let core = self.to_core()?;
        serde_json::to_string(&core)
            .map_err(|err| PyValueError::new_err(format!("failed to serialize result: {err}")))
    }

    #[cfg(not(feature = "serde"))]
    fn to_json(&self) -> PyResult<String> {
        Err(PyNotImplementedError::new_err(
            "to_json() requires cpd-python built with serde feature",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "OfflineChangePointResult(breakpoints={:?}, change_points={:?}, scores_len={}, segments_len={})",
            self.breakpoints,
            self.change_points,
            self.scores.as_ref().map_or(0, Vec::len),
            self.segments.as_ref().map_or(0, Vec::len)
        )
    }
}

impl From<CoreOfflineChangePointResult> for PyOfflineChangePointResult {
    fn from(result: CoreOfflineChangePointResult) -> Self {
        let diagnostics = PyDiagnostics::from(result.diagnostics);
        let n = if diagnostics.n == 0 {
            result.breakpoints.last().copied().unwrap_or(0)
        } else {
            diagnostics.n
        };
        let change_points = derive_change_points(n, &result.breakpoints);

        Self {
            breakpoints: result.breakpoints,
            change_points,
            scores: result.scores,
            segments: result
                .segments
                .map(|segments| segments.into_iter().map(Into::into).collect()),
            diagnostics,
        }
    }
}

impl PyOfflineChangePointResult {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> PyResult<CoreOfflineChangePointResult> {
        Ok(CoreOfflineChangePointResult {
            breakpoints: self.breakpoints.clone(),
            change_points: self.change_points.clone(),
            scores: self.scores.clone(),
            segments: self
                .segments
                .as_ref()
                .map(|segments| segments.iter().map(PySegmentStats::to_core).collect()),
            diagnostics: self.diagnostics.to_core()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PyDiagnostics, PyOfflineChangePointResult, PyOnlineStepResult, PyPruningStats,
        PySegmentStats, repro_mode_to_str,
    };
    use cpd_core::{
        DIAGNOSTICS_SCHEMA_VERSION, Diagnostics as CoreDiagnostics,
        OfflineChangePointResult as CoreOfflineChangePointResult,
        OnlineStepResult as CoreOnlineStepResult, PruningStats as CorePruningStats, ReproMode,
        SegmentStats as CoreSegmentStats,
    };
    #[cfg(not(feature = "serde"))]
    use pyo3::exceptions::PyNotImplementedError;
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyList};
    #[cfg(feature = "serde")]
    use serde_json::Value;
    use std::borrow::Cow;
    use std::sync::Once;

    #[cfg(feature = "serde")]
    const OFFLINE_RESULT_FIXTURE_JSON: &str =
        include_str!("../tests/fixtures/offline_result_v1.json");

    fn with_python<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::prepare_freethreaded_python);
        Python::with_gil(f)
    }

    fn sample_core_result() -> CoreOfflineChangePointResult {
        let diagnostics = CoreDiagnostics {
            n: 100,
            d: 2,
            schema_version: 0,
            engine_version: None,
            runtime_ms: Some(123),
            notes: vec!["run complete".to_string()],
            warnings: vec!["none".to_string()],
            algorithm: Cow::Owned("pelt".to_string()),
            cost_model: Cow::Owned("l2_mean".to_string()),
            seed: Some(7),
            repro_mode: ReproMode::Balanced,
            thread_count: Some(4),
            blas_backend: Some("openblas".to_string()),
            cpu_features: Some(vec!["avx2".to_string(), "fma".to_string()]),
            #[cfg(feature = "serde")]
            params_json: Some(serde_json::json!({
                "jump": 5,
                "min_segment_len": 10
            })),
            pruning_stats: Some(CorePruningStats {
                candidates_considered: 150,
                candidates_pruned: 120,
            }),
            missing_policy_applied: Some("Ignore".to_string()),
            missing_fraction: Some(0.1),
            effective_sample_count: Some(90),
        };

        CoreOfflineChangePointResult {
            breakpoints: vec![40, 100],
            change_points: vec![3, 7, 9],
            scores: Some(vec![0.75]),
            segments: Some(vec![
                CoreSegmentStats {
                    start: 0,
                    end: 40,
                    mean: Some(vec![1.0, 2.0]),
                    variance: Some(vec![0.1, 0.2]),
                    count: 40,
                    missing_count: 2,
                },
                CoreSegmentStats {
                    start: 40,
                    end: 100,
                    mean: Some(vec![3.0, 4.0]),
                    variance: Some(vec![0.3, 0.4]),
                    count: 60,
                    missing_count: 3,
                },
            ]),
            diagnostics,
        }
    }

    fn sample_online_step() -> CoreOnlineStepResult {
        CoreOnlineStepResult {
            t: 12,
            p_change: 0.42,
            alert: true,
            alert_reason: Some("threshold crossed".to_string()),
            run_length_mode: 3,
            run_length_mean: 2.75,
            processing_latency_us: Some(120),
        }
    }

    #[test]
    fn breakpoints_drive_change_points_derivation() {
        let py_result = PyOfflineChangePointResult::from(sample_core_result());
        assert_eq!(py_result.breakpoints, vec![40, 100]);
        assert_eq!(py_result.change_points, vec![40]);
    }

    #[test]
    fn diagnostics_normalizes_schema_and_engine_version() {
        let py_result = PyOfflineChangePointResult::from(sample_core_result());
        assert_eq!(
            py_result.diagnostics.schema_version,
            DIAGNOSTICS_SCHEMA_VERSION
        );
        assert_eq!(
            py_result.diagnostics.engine_version,
            Some(env!("CARGO_PKG_VERSION").to_string())
        );
        assert_eq!(
            py_result.diagnostics.repro_mode,
            repro_mode_to_str(ReproMode::Balanced)
        );
    }

    #[test]
    fn python_properties_are_accessible_and_typed() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_any = py_result.bind(py);

            let breakpoints: Vec<usize> = result_any
                .getattr("breakpoints")
                .expect("breakpoints should be exported")
                .extract()
                .expect("breakpoints should be list[int]");
            let change_points: Vec<usize> = result_any
                .getattr("change_points")
                .expect("change_points should be exported")
                .extract()
                .expect("change_points should be list[int]");
            let scores: Option<Vec<f64>> = result_any
                .getattr("scores")
                .expect("scores should be exported")
                .extract()
                .expect("scores should be optional list[float]");

            assert_eq!(breakpoints, vec![40, 100]);
            assert_eq!(change_points, vec![40]);
            assert_eq!(scores, Some(vec![0.75]));

            let diagnostics = result_any
                .getattr("diagnostics")
                .expect("diagnostics should be exported");
            let algorithm: String = diagnostics
                .getattr("algorithm")
                .expect("algorithm should be exported")
                .extract()
                .expect("algorithm should be a string");
            let missing_fraction: Option<f64> = diagnostics
                .getattr("missing_fraction")
                .expect("missing_fraction should be exported")
                .extract()
                .expect("missing_fraction should be optional float");
            assert_eq!(algorithm, "pelt");
            assert_eq!(missing_fraction, Some(0.1));

            let pruning = diagnostics
                .getattr("pruning_stats")
                .expect("pruning_stats should be exported");
            assert!(!pruning.is_none());
            let considered: usize = pruning
                .getattr("candidates_considered")
                .expect("candidates_considered should be exported")
                .extract()
                .expect("candidates_considered should be int");
            assert_eq!(considered, 150);

            let segments_any = result_any
                .getattr("segments")
                .expect("segments should be exported");
            let segments = segments_any
                .downcast::<PyList>()
                .expect("segments should be list[SegmentStats]");
            assert_eq!(segments.len(), 2);
            let first = segments.get_item(0).expect("first segment should exist");
            let first_count: usize = first
                .getattr("count")
                .expect("segment count should be exported")
                .extract()
                .expect("segment count should be int");
            assert_eq!(first_count, 40);

            let params_json = diagnostics
                .getattr("params_json")
                .expect("params_json should be exported");
            #[cfg(feature = "serde")]
            {
                let as_repr: String = params_json
                    .repr()
                    .expect("repr should succeed")
                    .extract()
                    .expect("repr should be string");
                assert!(as_repr.contains("min_segment_len"));
            }
            #[cfg(not(feature = "serde"))]
            {
                assert!(params_json.is_none());
            }
        });
    }

    #[test]
    fn repr_is_human_readable() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_repr: String = py_result
                .bind(py)
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(result_repr.contains("OfflineChangePointResult"));
            assert!(result_repr.contains("breakpoints"));

            let diagnostics = py_result
                .bind(py)
                .getattr("diagnostics")
                .expect("diagnostics should be exported");
            let diagnostics_repr: String = diagnostics
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(diagnostics_repr.contains("Diagnostics"));

            let pruning = diagnostics
                .getattr("pruning_stats")
                .expect("pruning_stats should be exported");
            let pruning_repr: String = pruning
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(pruning_repr.contains("PruningStats"));

            let segments_any = py_result
                .bind(py)
                .getattr("segments")
                .expect("segments should be exported");
            let segments = segments_any
                .downcast::<PyList>()
                .expect("segments should be list[SegmentStats]");
            let segment_repr: String = segments
                .get_item(0)
                .expect("segment should exist")
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(segment_repr.contains("SegmentStats"));
        });
    }

    #[test]
    fn online_step_result_properties_and_repr_are_stable() {
        with_python(|py| {
            let py_step = Py::new(py, PyOnlineStepResult::from(sample_online_step()))
                .expect("online step should be constructible");
            let step_any = py_step.bind(py);

            let t: usize = step_any
                .getattr("t")
                .expect("t should be exported")
                .extract()
                .expect("t should be int");
            let p_change: f64 = step_any
                .getattr("p_change")
                .expect("p_change should be exported")
                .extract()
                .expect("p_change should be float");
            let alert: bool = step_any
                .getattr("alert")
                .expect("alert should be exported")
                .extract()
                .expect("alert should be bool");
            let reason: Option<String> = step_any
                .getattr("alert_reason")
                .expect("alert_reason should be exported")
                .extract()
                .expect("alert_reason should be optional string");
            let run_length_mode: usize = step_any
                .getattr("run_length_mode")
                .expect("run_length_mode should be exported")
                .extract()
                .expect("run_length_mode should be int");
            let run_length_mean: f64 = step_any
                .getattr("run_length_mean")
                .expect("run_length_mean should be exported")
                .extract()
                .expect("run_length_mean should be float");
            let processing_latency_us: Option<u64> = step_any
                .getattr("processing_latency_us")
                .expect("processing_latency_us should be exported")
                .extract()
                .expect("processing_latency_us should be optional int");

            assert_eq!(t, 12);
            assert!((p_change - 0.42).abs() < f64::EPSILON);
            assert!(alert);
            assert_eq!(reason.as_deref(), Some("threshold crossed"));
            assert_eq!(run_length_mode, 3);
            assert!((run_length_mean - 2.75).abs() < f64::EPSILON);
            assert_eq!(processing_latency_us, Some(120));

            let repr: String = step_any
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(repr.contains("OnlineStepResult"));
            assert!(repr.contains("p_change"));
        });
    }

    #[test]
    #[cfg(not(feature = "serde"))]
    fn to_json_requires_serde_feature() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let err = py_result
                .bind(py)
                .call_method0("to_json")
                .expect_err("to_json should fail without serde feature");
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
            assert!(
                err.to_string()
                    .contains("to_json() requires cpd-python built with serde feature")
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn to_json_roundtrip_preserves_core_payload() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let json_payload: String = py_result
                .bind(py)
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return string");

            let decoded: CoreOfflineChangePointResult =
                serde_json::from_str(&json_payload).expect("json should deserialize");

            assert_eq!(decoded.breakpoints, vec![40, 100]);
            assert_eq!(decoded.change_points, vec![40]);
            assert_eq!(decoded.scores, Some(vec![0.75]));
            assert_eq!(
                decoded.diagnostics.schema_version,
                DIAGNOSTICS_SCHEMA_VERSION
            );
            assert_eq!(
                decoded.diagnostics.engine_version,
                Some(env!("CARGO_PKG_VERSION").to_string())
            );
            assert_eq!(decoded.diagnostics.algorithm, "pelt");
            assert_eq!(decoded.diagnostics.cost_model, "l2_mean");
            assert!(decoded.diagnostics.pruning_stats.is_some());
            assert_eq!(
                decoded.diagnostics.params_json,
                Some(serde_json::json!({
                    "jump": 5,
                    "min_segment_len": 10
                }))
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_fixture_deserializes_and_validates() {
        let decoded: CoreOfflineChangePointResult =
            serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON)
                .expect("fixture should deserialize as core result");
        decoded
            .validate(decoded.diagnostics.n)
            .expect("fixture should validate");
        assert_eq!(decoded.breakpoints, vec![40, 100]);
        assert_eq!(decoded.change_points, vec![40]);
        assert_eq!(
            decoded.diagnostics.schema_version,
            DIAGNOSTICS_SCHEMA_VERSION
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn to_json_matches_versioned_fixture_shape() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let json_payload: String = py_result
                .bind(py)
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return string");

            let generated: Value =
                serde_json::from_str(&json_payload).expect("generated JSON should parse");
            let mut fixture: Value = serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON)
                .expect("fixture JSON should parse");

            // Keep fixture stable across version bumps while asserting full shape equality.
            if let Some(diagnostics) = fixture
                .get_mut("diagnostics")
                .and_then(serde_json::Value::as_object_mut)
            {
                diagnostics.insert(
                    "engine_version".to_string(),
                    Value::String(env!("CARGO_PKG_VERSION").to_string()),
                );
            }

            assert_eq!(generated, fixture);
        });
    }

    #[test]
    fn py_wrappers_convert_from_core_types() {
        let stats = PyPruningStats::from(CorePruningStats {
            candidates_considered: 10,
            candidates_pruned: 9,
        });
        assert_eq!(stats.candidates_considered, 10);
        assert_eq!(stats.candidates_pruned, 9);

        let segment = PySegmentStats::from(CoreSegmentStats {
            start: 0,
            end: 5,
            mean: None,
            variance: None,
            count: 5,
            missing_count: 0,
        });
        assert_eq!(segment.start, 0);
        assert_eq!(segment.end, 5);

        let diagnostics = PyDiagnostics::from(CoreDiagnostics {
            n: 5,
            d: 1,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some("x.y.z".to_string()),
            runtime_ms: Some(1),
            notes: vec![],
            warnings: vec![],
            algorithm: Cow::Borrowed("binseg"),
            cost_model: Cow::Borrowed("normal_mean_var"),
            seed: None,
            repro_mode: ReproMode::Fast,
            thread_count: None,
            blas_backend: None,
            cpu_features: None,
            #[cfg(feature = "serde")]
            params_json: None,
            pruning_stats: None,
            missing_policy_applied: None,
            missing_fraction: None,
            effective_sample_count: None,
        });
        assert_eq!(diagnostics.algorithm, "binseg");
        assert_eq!(diagnostics.repro_mode, "fast");
    }
}
