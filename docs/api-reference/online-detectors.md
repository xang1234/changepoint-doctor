# Online Detectors

Streaming change-point detectors that process one observation at a time.

## `Bocpd`

Bayesian Online Changepoint Detection.

### Constructor

```python
cpd.Bocpd(
    model: str = "gaussian_nig",
    hazard: float | dict[str, Any] | None = None,
    max_run_length: int = 2000,
    alert_policy: dict[str, Any] | None = None,
    late_data_policy: str | dict[str, Any] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gaussian_nig"` | Conjugate observation model family |
| `hazard` | `float \| dict \| None` | `None` | Constant hazard rate (1/expected_run_length) or custom config |
| `max_run_length` | `int` | `2000` | Truncation limit for the run-length posterior |
| `alert_policy` | `dict \| None` | `None` | Alert firing rules (see below) |
| `late_data_policy` | `str \| dict \| None` | `None` | Late-data handling strategy |

**Alert policy shape:**

```python
{
    "threshold": 0.35,       # p_change must exceed this to fire
    "cooldown": 5,           # minimum steps between consecutive alerts
    "min_run_length": 10,    # suppress alerts during initial transient
}
```

### Methods

#### `update(x_t, t_ns=None) -> OnlineStepResult`

Process a single scalar observation.

- **x_t**: `float` -- Observation value
- **t_ns**: `int | None` -- Optional nanosecond timestamp
- **Returns**: {class}`OnlineStepResult`

#### `update_many(x_batch) -> list[OnlineStepResult]`

Process a batch of observations. Uses GIL-release optimization for batches >= 16 elements.

- **x_batch**: array-like of `float64`
- **Returns**: List of {class}`OnlineStepResult`

#### `reset() -> None`

Clear internal state and start fresh.

#### `save_state(*, format="bytes", path=None) -> bytes | dict | None`

Checkpoint the detector state.

- **format**: `str` -- `"bytes"` (compact binary) or `"dict"` (Python dict)
- **path**: `str | Path | None` -- If provided, writes state to this file path and returns `None`
- **Returns**: State in the requested format, or `None` if `path` is provided

#### `load_state(state=None, *, format=None, path=None) -> None`

Restore detector state from a checkpoint.

- **state**: `bytes | dict | _BocpdState | None` -- State object to load
- **format**: `str | None` -- Required when loading from `state` bytes/dict
- **path**: `str | Path | None` -- Load from file instead of `state` parameter

---

## `Cusum`

Cumulative Sum detector for mean-shift monitoring.

### Constructor

```python
cpd.Cusum(
    drift: float = 0.0,
    threshold: float = 8.0,
    target_mean: float = 0.0,
    alert_policy: dict[str, Any] | None = None,
    late_data_policy: str | dict[str, Any] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drift` | `float` | `0.0` | Allowance parameter (slack before accumulation) |
| `threshold` | `float` | `8.0` | Decision threshold for cumulative sum |
| `target_mean` | `float` | `0.0` | Expected mean under null hypothesis |
| `alert_policy` | `dict \| None` | `None` | Alert policy (same shape as BOCPD) |
| `late_data_policy` | `str \| dict \| None` | `None` | Late-data handling |

### Methods

Same interface as `Bocpd`: `update()`, `update_many()`, `reset()`, `save_state()`, `load_state()`.

---

## `PageHinkley`

Page-Hinkley drift/change monitor.

### Constructor

```python
cpd.PageHinkley(
    delta: float = 0.01,
    threshold: float = 8.0,
    initial_mean: float = 0.0,
    alert_policy: dict[str, Any] | None = None,
    late_data_policy: str | dict[str, Any] | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `delta` | `float` | `0.01` | Magnitude tolerance parameter |
| `threshold` | `float` | `8.0` | Decision threshold |
| `initial_mean` | `float` | `0.0` | Initial mean estimate |
| `alert_policy` | `dict \| None` | `None` | Alert policy (same shape as BOCPD) |
| `late_data_policy` | `str \| dict \| None` | `None` | Late-data handling |

### Methods

Same interface as `Bocpd`: `update()`, `update_many()`, `reset()`, `save_state()`, `load_state()`.
