import threading
import time

import numpy as np
import pytest

import cpd


def _step_signal() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(80, dtype=np.float64),
            np.full(80, 6.0, dtype=np.float64),
        ]
    )


def _compare_steps(lhs: cpd.OnlineStepResult, rhs: cpd.OnlineStepResult) -> None:
    assert lhs.t == rhs.t
    assert lhs.alert == rhs.alert
    assert lhs.alert_reason == rhs.alert_reason
    assert lhs.run_length_mode == rhs.run_length_mode
    assert lhs.p_change == pytest.approx(rhs.p_change)
    assert lhs.run_length_mean == pytest.approx(rhs.run_length_mean)


STREAMING_FACTORIES = [
    (
        "bocpd",
        lambda: cpd.Bocpd(
            model="gaussian_nig",
            hazard=1.0 / 200.0,
            max_run_length=512,
            alert_policy={"threshold": 0.35, "cooldown": 5, "min_run_length": 10},
        ),
    ),
    (
        "cusum",
        lambda: cpd.Cusum(
            drift=0.0,
            threshold=6.0,
            target_mean=0.0,
            alert_policy={"threshold": 0.95, "cooldown": 5, "min_run_length": 10},
        ),
    ),
    (
        "page_hinkley",
        lambda: cpd.PageHinkley(
            delta=0.0,
            threshold=6.0,
            initial_mean=0.0,
            alert_policy={"threshold": 0.95, "cooldown": 5, "min_run_length": 10},
        ),
    ),
]


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in STREAMING_FACTORIES],
    ids=[name for name, _ in STREAMING_FACTORIES],
)
def test_streaming_step_function_alerts_near_change_point(factory) -> None:
    values = _step_signal()
    detector = factory()

    first_alert = None
    for idx, value in enumerate(values):
        step = detector.update(float(value))
        if step.alert:
            first_alert = idx
            break

    assert first_alert is not None
    assert 80 <= first_alert <= 120


def test_bocpd_checkpoint_restore_matches_continuation() -> None:
    values = _step_signal()
    split = 95
    kwargs = dict(
        model="gaussian_nig",
        hazard=1.0 / 200.0,
        max_run_length=512,
        alert_policy={"threshold": 0.35, "cooldown": 5, "min_run_length": 10},
    )

    baseline = cpd.Bocpd(**kwargs)
    baseline_steps = [baseline.update(float(x)) for x in values]

    first = cpd.Bocpd(**kwargs)
    prefix_steps = [first.update(float(x)) for x in values[:split]]
    state = first.save_state()

    restored = cpd.Bocpd(**kwargs)
    restored.load_state(state)
    tail_steps = [restored.update(float(x)) for x in values[split:]]
    restored_steps = prefix_steps + tail_steps

    assert len(restored_steps) == len(baseline_steps)
    for lhs, rhs in zip(restored_steps, baseline_steps):
        _compare_steps(lhs, rhs)


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in STREAMING_FACTORIES],
    ids=[name for name, _ in STREAMING_FACTORIES],
)
@pytest.mark.parametrize("checkpoint_format", ["bytes", "json"])
def test_checkpoint_restore_matches_continuation_for_bytes_and_json(
    factory, checkpoint_format: str
) -> None:
    values = _step_signal()
    split = 95

    baseline = factory()
    baseline_steps = [baseline.update(float(x)) for x in values]

    first = factory()
    prefix_steps = [first.update(float(x)) for x in values[:split]]
    state = first.save_state(format=checkpoint_format)

    restored = factory()
    if checkpoint_format == "bytes":
        restored.load_state(state)
    else:
        restored.load_state(state, format="json")
    tail_steps = [restored.update(float(x)) for x in values[split:]]
    restored_steps = prefix_steps + tail_steps

    assert len(restored_steps) == len(baseline_steps)
    for lhs, rhs in zip(restored_steps, baseline_steps):
        _compare_steps(lhs, rhs)


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in STREAMING_FACTORIES],
    ids=[name for name, _ in STREAMING_FACTORIES],
)
def test_checkpoint_restore_matches_continuation_for_file_path(factory, tmp_path) -> None:
    values = _step_signal()
    split = 95
    path = tmp_path / "checkpoint.bin"

    baseline = factory()
    baseline_steps = [baseline.update(float(x)) for x in values]

    first = factory()
    prefix_steps = [first.update(float(x)) for x in values[:split]]
    save_out = first.save_state(path=path)
    assert save_out is None

    restored = factory()
    restored.load_state(path=path)
    tail_steps = [restored.update(float(x)) for x in values[split:]]
    restored_steps = prefix_steps + tail_steps

    assert len(restored_steps) == len(baseline_steps)
    for lhs, rhs in zip(restored_steps, baseline_steps):
        _compare_steps(lhs, rhs)


@pytest.mark.parametrize(
    "detector_id,factory",
    STREAMING_FACTORIES,
    ids=[name for name, _ in STREAMING_FACTORIES],
)
def test_checkpoint_json_metadata_matches_rust_envelope(detector_id: str, factory) -> None:
    detector = factory()
    for value in [0.0, 0.0, 1.0]:
        detector.update(value)

    state = detector.save_state(format="json")
    assert state["detector_id"] == detector_id
    assert isinstance(state["state_schema_version"], int)
    assert isinstance(state["engine_fingerprint"], str)
    assert isinstance(state["created_at_ns"], int)
    assert state["payload_codec"] == "json"
    assert isinstance(state["payload_crc32"], int)
    assert isinstance(state["payload"], list)
    assert all(isinstance(item, int) for item in state["payload"])


def test_checkpoint_load_rejects_crc_mismatch_with_clear_error() -> None:
    detector = cpd.Cusum()
    for value in [0.0, 0.0, 1.0]:
        detector.update(value)

    state = detector.save_state(format="json")
    state["payload"][0] ^= 1

    restored = cpd.Cusum()
    with pytest.raises(ValueError, match="crc32 mismatch"):
        restored.load_state(state, format="json")


def test_checkpoint_load_rejects_unsupported_schema_version_with_clear_error() -> None:
    detector = cpd.Cusum()
    for value in [0.0, 0.0, 1.0]:
        detector.update(value)

    state = detector.save_state(format="json")
    state["state_schema_version"] = 99

    restored = cpd.Cusum()
    with pytest.raises(ValueError, match="state_schema_version=99"):
        restored.load_state(state, format="json")


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in STREAMING_FACTORIES],
    ids=[name for name, _ in STREAMING_FACTORIES],
)
def test_update_many_matches_single_step_results(factory) -> None:
    values = _step_signal()

    single = factory()
    single_steps = [single.update(float(x)) for x in values]

    batch = factory()
    batch_steps = batch.update_many(values)

    assert len(batch_steps) == len(single_steps)
    for lhs, rhs in zip(batch_steps, single_steps):
        _compare_steps(lhs, rhs)


def test_update_accepts_event_time_and_rejects_late_data_by_default() -> None:
    detector = cpd.Bocpd(
        model="gaussian_nig",
        hazard=1.0 / 200.0,
        max_run_length=256,
        alert_policy={"threshold": 0.4, "cooldown": 3},
    )

    first = detector.update(0.0, t_ns=1_000)
    assert first.t == 0

    with pytest.raises(ValueError, match="late event rejected"):
        detector.update(1.0, t_ns=999)


def test_bocpd_load_state_rejects_incompatible_model_immediately() -> None:
    source = cpd.Bocpd(model="gaussian_nig", hazard=1.0 / 200.0, max_run_length=128)
    for value in [0.0, 0.0, 1.0, 1.0]:
        source.update(value)
    state = source.save_state()

    target = cpd.Bocpd(model="bernoulli_beta", hazard=1.0 / 200.0, max_run_length=128)
    with pytest.raises(ValueError, match="incompatible Bocpd state"):
        target.load_state(state)


def test_streaming_config_rejects_unknown_hazard_and_late_data_keys() -> None:
    with pytest.raises(ValueError, match="unsupported hazard key 'extra'"):
        cpd.Bocpd(hazard={"kind": "constant", "p_change": 0.01, "extra": 1})

    with pytest.raises(ValueError, match="unsupported late_data_policy key 'extra'"):
        cpd.Bocpd(
            late_data_policy={
                "kind": "buffer_within_window",
                "max_delay_ns": 10,
                "max_buffer_items": 2,
                "on_overflow": "drop_newest",
                "extra": 1,
            }
        )


def test_update_many_releases_gil() -> None:
    n = 500_000
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 5.0
    detector = cpd.Cusum(threshold=8.0, alert_policy={"threshold": 0.95})

    state = {"running": True, "in_call": False, "ticks": 0}

    def worker() -> None:
        while state["running"]:
            if state["in_call"]:
                state["ticks"] += 1
            time.sleep(0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    time.sleep(0.01)

    try:
        state["in_call"] = True
        _ = detector.update_many(values)
    finally:
        state["in_call"] = False
        state["running"] = False
        thread.join(timeout=2.0)

    assert state["ticks"] > 0
