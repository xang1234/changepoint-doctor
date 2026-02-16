import json
import os
import time
from pathlib import Path
from statistics import fmean

import numpy as np

import cpd

PERF_BATCH_SIZES = (1, 8, 16, 64, 4096)
WARMUP_ROUNDS = 4
MEASURE_ROUNDS = 12
PERF_ENFORCE_ENV = "CPD_PY_STREAMING_PERF_ENFORCE"
PERF_REPORT_ENV = "CPD_PY_STREAMING_PERF_REPORT_OUT"


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _step_signal(n: int) -> np.ndarray:
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 4.0
    return values


def _measure_pair(batch_size: int) -> dict:
    values = _step_signal(batch_size)
    single_ms: list[float] = []
    batch_ms: list[float] = []

    for idx in range(WARMUP_ROUNDS + MEASURE_ROUNDS):
        detector = cpd.Cusum(threshold=8.0, alert_policy={"threshold": 0.95})
        t0 = time.perf_counter()
        for x_t in values:
            detector.update(float(x_t))
        t1 = time.perf_counter()

        detector = cpd.Cusum(threshold=8.0, alert_policy={"threshold": 0.95})
        t2 = time.perf_counter()
        detector.update_many(values)
        t3 = time.perf_counter()

        if idx >= WARMUP_ROUNDS:
            single_ms.append((t1 - t0) * 1_000.0)
            batch_ms.append((t3 - t2) * 1_000.0)

    single_mean_ms = fmean(single_ms)
    batch_mean_ms = fmean(batch_ms)
    speedup = single_mean_ms / batch_mean_ms
    return {
        "batch_size": batch_size,
        "single_mean_ms": single_mean_ms,
        "batch_mean_ms": batch_mean_ms,
        "speedup_vs_single": speedup,
    }


def test_update_many_perf_contract_snapshot() -> None:
    enforce = _env_flag(PERF_ENFORCE_ENV)
    rows = [_measure_pair(size) for size in PERF_BATCH_SIZES]
    by_size = {int(row["batch_size"]): row for row in rows}
    report = {
        "scenario": "python_streaming_update_vs_update_many",
        "detector": "Cusum",
        "warmup_rounds": WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
        "results": rows,
        "enforce": enforce,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    report_out = os.getenv(PERF_REPORT_ENV)
    if report_out:
        Path(report_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # Always-on checks: broad sanity for crossover behavior.
    assert by_size[1]["speedup_vs_single"] < 0.9
    assert by_size[8]["speedup_vs_single"] < 1.05
    assert by_size[16]["speedup_vs_single"] > 1.0
    assert by_size[64]["speedup_vs_single"] > 1.1
    assert by_size[4096]["speedup_vs_single"] > 1.1

    # Optional stricter gates for local perf validation runs.
    if enforce:
        assert by_size[1]["speedup_vs_single"] < 0.6
        assert by_size[8]["speedup_vs_single"] < 1.0
        assert by_size[16]["speedup_vs_single"] > 1.08
        assert by_size[64]["speedup_vs_single"] > 1.2
        assert by_size[4096]["speedup_vs_single"] > 1.2
