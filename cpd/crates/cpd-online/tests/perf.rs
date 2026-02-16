// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{Constraints, CpdError, ExecutionContext, OnlineDetector};
use cpd_online::{
    BocpdConfig, BocpdDetector, ConstantHazard, CusumConfig, CusumDetector, HazardSpec,
    LateDataPolicy, ObservationModel, PageHinkleyConfig, PageHinkleyDetector,
};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_RUN_LENGTH: usize = 2_000;
const LOG_PROB_THRESHOLD: f64 = -20.0;
const WARMUP_STEPS: usize = 2_500;
const MEASURE_STEPS: usize = 12_000;
const SLO_P99_UPDATE_US: u64 = 75;
const SLO_UPDATES_PER_SEC: f64 = 150_000.0;
const MIN_P95_RUN_LENGTH_MODE: usize = 64;
const SIGNAL_REGIME_LEN: usize = 96;
const BASELINE_WARMUP_STEPS: usize = 20_000;
const BASELINE_MEASURE_STEPS: usize = 500_000;
const BASELINE_SLO_UPDATES_PER_SEC: f64 = 10_000_000.0;

fn ctx() -> ExecutionContext<'static> {
    static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
    let constraints = CONSTRAINTS.get_or_init(Constraints::default);
    ExecutionContext::new(constraints)
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn percentile(sorted_values: &[u64], q: f64) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }

    let q = q.clamp(0.0, 1.0);
    let max_index = sorted_values.len() - 1;
    let rank = (q * (max_index as f64)).ceil() as usize;
    sorted_values[rank.min(max_index)]
}

fn percentile_usize(sorted_values: &[usize], q: f64) -> usize {
    if sorted_values.is_empty() {
        return 0;
    }

    let q = q.clamp(0.0, 1.0);
    let max_index = sorted_values.len() - 1;
    let rank = (q * (max_index as f64)).ceil() as usize;
    sorted_values[rank.min(max_index)]
}

fn perf_signal(step: usize) -> f64 {
    let regime = (step / SIGNAL_REGIME_LEN) % 3;
    let baseline = match regime {
        0 => 0.0,
        1 => 3.5,
        _ => -1.8,
    };
    let wobble = ((step as f64) * 0.03).sin() * 0.2;
    baseline + wobble
}

fn make_signal(total_steps: usize) -> Vec<f64> {
    (0..total_steps).map(perf_signal).collect()
}

fn emit_bocpd_metrics_if_requested(
    updates_per_sec: f64,
    p99_update_us: u64,
    p95_run_length_mode: usize,
    enforce: bool,
) -> Result<(), CpdError> {
    let Some(path) = std::env::var("CPD_ONLINE_PERF_METRICS_OUT").ok() else {
        return Ok(());
    };

    let payload = format!(
        "{{\n  \"scenario\": \"bocpd_gaussian_d1_max_run_length_2000\",\n  \"max_run_length\": {MAX_RUN_LENGTH},\n  \"warmup_steps\": {WARMUP_STEPS},\n  \"measure_steps\": {MEASURE_STEPS},\n  \"updates_per_sec\": {updates_per_sec},\n  \"p99_update_us\": {p99_update_us},\n  \"p95_run_length_mode\": {p95_run_length_mode},\n  \"min_p95_run_length_mode\": {MIN_P95_RUN_LENGTH_MODE},\n  \"slo_p99_update_us\": {SLO_P99_UPDATE_US},\n  \"slo_updates_per_sec\": {SLO_UPDATES_PER_SEC},\n  \"enforce\": {enforce}\n}}\n",
    );

    std::fs::write(path, payload).map_err(|err| {
        CpdError::resource_limit(format!("failed writing perf metrics artifact: {err}"))
    })
}

fn emit_baseline_metrics_if_requested(
    env_var: &str,
    scenario: &str,
    updates_per_sec: f64,
    max_update_us: u64,
    enforce: bool,
) -> Result<(), CpdError> {
    let Some(path) = std::env::var(env_var).ok() else {
        return Ok(());
    };

    let payload = format!(
        "{{\n  \"scenario\": \"{scenario}\",\n  \"warmup_steps\": {BASELINE_WARMUP_STEPS},\n  \"measure_steps\": {BASELINE_MEASURE_STEPS},\n  \"updates_per_sec\": {updates_per_sec},\n  \"max_update_us\": {max_update_us},\n  \"slo_updates_per_sec\": {BASELINE_SLO_UPDATES_PER_SEC},\n  \"enforce\": {enforce}\n}}\n",
    );

    std::fs::write(path, payload).map_err(|err| {
        CpdError::resource_limit(format!(
            "failed writing baseline perf metrics artifact: {err}"
        ))
    })
}

fn run_baseline_perf_contract<D: OnlineDetector>(
    detector: &mut D,
    signal: &[f64],
    metrics_env: &str,
    scenario: &str,
    enforce: bool,
) -> Result<(f64, u64), CpdError> {
    if signal.len() < BASELINE_WARMUP_STEPS + BASELINE_MEASURE_STEPS {
        return Err(CpdError::invalid_input(format!(
            "baseline perf signal too short: len={}, required={}",
            signal.len(),
            BASELINE_WARMUP_STEPS + BASELINE_MEASURE_STEPS
        )));
    }

    let exec_ctx = ctx();
    for step in 0..BASELINE_WARMUP_STEPS {
        detector.update(&[signal[step]], None, &exec_ctx)?;
    }

    let mut max_update_us = 0_u64;
    let started_at = Instant::now();
    for step in 0..BASELINE_MEASURE_STEPS {
        let result = detector.update(&[signal[BASELINE_WARMUP_STEPS + step]], None, &exec_ctx)?;
        let latency_us = result.processing_latency_us.ok_or_else(|| {
            CpdError::invalid_input(format!(
                "{scenario} detector did not report processing_latency_us"
            ))
        })?;
        max_update_us = max_update_us.max(latency_us);
    }
    let elapsed = started_at.elapsed();
    let updates_per_sec = (BASELINE_MEASURE_STEPS as f64) / elapsed.as_secs_f64().max(1e-9);

    emit_baseline_metrics_if_requested(
        metrics_env,
        scenario,
        updates_per_sec,
        max_update_us,
        enforce,
    )?;

    Ok((updates_per_sec, max_update_us))
}

#[test]
fn bocpd_gaussian_perf_contract() {
    let enforce = parse_env_bool("CPD_ONLINE_PERF_ENFORCE");
    let signal = make_signal(WARMUP_STEPS + MEASURE_STEPS);
    let exec_ctx = ctx();
    let mut detector = BocpdDetector::new(BocpdConfig {
        hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 200.0).expect("valid hazard")),
        observation: ObservationModel::default(),
        max_run_length: MAX_RUN_LENGTH,
        log_prob_threshold: Some(LOG_PROB_THRESHOLD),
        alert_threshold: 0.5,
        late_data_policy: LateDataPolicy::Reject,
    })
    .expect("BOCPD config should be valid");

    for step in 0..WARMUP_STEPS {
        detector
            .update(&[signal[step]], None, &exec_ctx)
            .expect("warmup update should succeed");
    }

    let mut latencies_us: Vec<u64> = Vec::with_capacity(MEASURE_STEPS);
    let mut run_length_modes: Vec<usize> = Vec::with_capacity(MEASURE_STEPS);
    let started_at = Instant::now();
    for step in 0..MEASURE_STEPS {
        let result = detector
            .update(&[signal[WARMUP_STEPS + step]], None, &exec_ctx)
            .expect("measurement update should succeed");
        let latency_us = result
            .processing_latency_us
            .expect("BOCPD should report processing latency");
        latencies_us.push(latency_us);
        run_length_modes.push(result.run_length_mode);
    }
    let elapsed = started_at.elapsed();

    latencies_us.sort_unstable();
    run_length_modes.sort_unstable();
    let p99_update_us = percentile(&latencies_us, 0.99);
    let p95_run_length_mode = percentile_usize(&run_length_modes, 0.95);
    let updates_per_sec = (MEASURE_STEPS as f64) / elapsed.as_secs_f64().max(1e-9);

    println!(
        "BOCPD perf: steps={} elapsed_s={:.6} updates_per_sec={:.2} p99_update_us={} p95_run_length_mode={} max_run_length={} enforce={}",
        MEASURE_STEPS,
        elapsed.as_secs_f64(),
        updates_per_sec,
        p99_update_us,
        p95_run_length_mode,
        MAX_RUN_LENGTH,
        enforce
    );

    emit_bocpd_metrics_if_requested(updates_per_sec, p99_update_us, p95_run_length_mode, enforce)
        .expect("metrics artifact emission should succeed");

    if enforce {
        assert!(
            p99_update_us <= SLO_P99_UPDATE_US,
            "p99 update latency SLO failed: observed={}us, threshold={}us",
            p99_update_us,
            SLO_P99_UPDATE_US
        );
        assert!(
            updates_per_sec >= SLO_UPDATES_PER_SEC,
            "throughput SLO failed: observed={updates_per_sec:.2} updates/sec, threshold={} updates/sec",
            SLO_UPDATES_PER_SEC
        );
        assert!(
            p95_run_length_mode >= MIN_P95_RUN_LENGTH_MODE,
            "run-length occupancy guard failed: observed p95_run_length_mode={}, minimum={}",
            p95_run_length_mode,
            MIN_P95_RUN_LENGTH_MODE
        );
    } else {
        assert!(updates_per_sec.is_finite() && updates_per_sec > 0.0);
        assert!(!latencies_us.is_empty());
    }
}

#[test]
fn cusum_perf_contract() {
    let enforce = parse_env_bool("CPD_ONLINE_PERF_ENFORCE");
    let signal = make_signal(BASELINE_WARMUP_STEPS + BASELINE_MEASURE_STEPS);
    let mut detector = CusumDetector::new(CusumConfig {
        drift: 0.02,
        threshold: 1_000_000.0,
        target_mean: 0.0,
        late_data_policy: LateDataPolicy::Reject,
    })
    .expect("CUSUM config should be valid");

    let (updates_per_sec, max_update_us) = run_baseline_perf_contract(
        &mut detector,
        &signal,
        "CPD_ONLINE_CUSUM_PERF_METRICS_OUT",
        "cusum_upward_d1_update",
        enforce,
    )
    .expect("CUSUM perf run should succeed");

    println!(
        "CUSUM update perf: steps={} updates_per_sec={:.2} max_update_us={} enforce={}",
        BASELINE_MEASURE_STEPS, updates_per_sec, max_update_us, enforce
    );

    if enforce {
        assert!(
            updates_per_sec >= BASELINE_SLO_UPDATES_PER_SEC,
            "CUSUM throughput SLO failed: observed={updates_per_sec:.2} updates/sec, threshold={} updates/sec",
            BASELINE_SLO_UPDATES_PER_SEC
        );
        assert!(
            max_update_us > 0,
            "CUSUM latency metric should remain meaningful (>0us); observed={max_update_us}"
        );
    } else {
        assert!(updates_per_sec.is_finite() && updates_per_sec > 0.0);
    }
}

#[test]
fn page_hinkley_perf_contract() {
    let enforce = parse_env_bool("CPD_ONLINE_PERF_ENFORCE");
    let signal = make_signal(BASELINE_WARMUP_STEPS + BASELINE_MEASURE_STEPS);
    let mut detector = PageHinkleyDetector::new(PageHinkleyConfig {
        delta: 0.02,
        threshold: 1_000_000.0,
        initial_mean: 0.0,
        late_data_policy: LateDataPolicy::Reject,
    })
    .expect("Page-Hinkley config should be valid");

    let (updates_per_sec, max_update_us) = run_baseline_perf_contract(
        &mut detector,
        &signal,
        "CPD_ONLINE_PAGE_HINKLEY_PERF_METRICS_OUT",
        "page_hinkley_d1_update",
        enforce,
    )
    .expect("Page-Hinkley perf run should succeed");

    println!(
        "Page-Hinkley update perf: steps={} updates_per_sec={:.2} max_update_us={} enforce={}",
        BASELINE_MEASURE_STEPS, updates_per_sec, max_update_us, enforce
    );

    if enforce {
        assert!(
            updates_per_sec >= BASELINE_SLO_UPDATES_PER_SEC,
            "Page-Hinkley throughput SLO failed: observed={updates_per_sec:.2} updates/sec, threshold={} updates/sec",
            BASELINE_SLO_UPDATES_PER_SEC
        );
        assert!(
            max_update_us > 0,
            "Page-Hinkley latency metric should remain meaningful (>0us); observed={max_update_us}"
        );
    } else {
        assert!(updates_per_sec.is_finite() && updates_per_sec > 0.0);
    }
}
