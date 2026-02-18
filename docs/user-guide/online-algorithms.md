# Online Algorithms

Online (streaming) algorithms process one observation at a time and emit a per-step result indicating the probability of a change point. `changepoint-doctor` provides three online detectors.

## BOCPD (Bayesian Online Changepoint Detection)

BOCPD {cite}`adams2007` maintains a posterior distribution over run lengths (time since the last change point). At each step it computes the probability of a change given the observation and the predictive distribution of the current regime.

**When to use:** You need probabilistic change likelihood in streaming data; conjugate Bayesian inference gives calibrated uncertainty estimates.

```python
import cpd

bocpd = cpd.Bocpd(
    model="gaussian_nig",
    hazard=1.0 / 200.0,
    max_run_length=512,
    alert_policy={"threshold": 0.35, "cooldown": 5, "min_run_length": 10},
)

for x_t in stream:
    step = bocpd.update(x_t)
    if step.alert:
        print(f"Change at t={step.t}, p={step.p_change:.3f}")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gaussian_nig"` | Observation model (conjugate prior family) |
| `hazard` | `float \| dict \| None` | `None` | Prior change rate (1/expected_run_length), or a dict for custom hazard |
| `max_run_length` | `int` | `2000` | Truncation limit for the run-length distribution |
| `alert_policy` | `dict \| None` | `None` | Alert thresholds and cooldown configuration |
| `late_data_policy` | `str \| dict \| None` | `None` | How to handle late-arriving data |

### Hazard configuration

The hazard parameter controls the prior belief about how often changes occur:

- **Float:** Constant hazard rate, e.g. `hazard=1/200` means "expect a change every ~200 steps"
- **Dict:** Custom hazard configuration for non-constant hazard functions

### Alert policy

The alert policy determines when `step.alert` fires:

```python
alert_policy = {
    "threshold": 0.35,       # p_change must exceed this
    "cooldown": 5,           # minimum steps between alerts
    "min_run_length": 10,    # suppress alerts during initial transient
}
```

## CUSUM (Cumulative Sum)

CUSUM {cite}`page1954` tracks cumulative deviations from a target mean. When the cumulative sum exceeds a threshold, an alert is triggered.

**When to use:** Lightweight mean-shift monitoring with well-understood statistical properties.

```python
import cpd

cusum = cpd.Cusum(
    drift=0.0,
    threshold=8.0,
    target_mean=0.0,
    alert_policy={"threshold": 0.5, "cooldown": 5},
)

steps = cusum.update_many(data)
alerts = [s for s in steps if s.alert]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drift` | `float` | `0.0` | Allowance parameter (slack before accumulation) |
| `threshold` | `float` | `8.0` | Decision threshold for the cumulative sum |
| `target_mean` | `float` | `0.0` | Expected mean under no-change hypothesis |
| `alert_policy` | `dict \| None` | `None` | Alert thresholds and cooldown |
| `late_data_policy` | `str \| dict \| None` | `None` | Late-data handling policy |

## Page-Hinkley

The Page-Hinkley test is a sequential analysis technique for drift/change monitoring, closely related to CUSUM but with a different accumulation formula.

**When to use:** Low-overhead drift monitoring; slightly different sensitivity profile than CUSUM.

```python
import cpd

ph = cpd.PageHinkley(
    delta=0.01,
    threshold=8.0,
    initial_mean=0.0,
    alert_policy={"threshold": 0.5, "cooldown": 5},
)

for x_t in stream:
    step = ph.update(x_t)
    if step.alert:
        print(f"Drift detected at t={step.t}")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `delta` | `float` | `0.01` | Magnitude tolerance parameter |
| `threshold` | `float` | `8.0` | Decision threshold |
| `initial_mean` | `float` | `0.0` | Initial mean estimate |
| `alert_policy` | `dict \| None` | `None` | Alert thresholds and cooldown |
| `late_data_policy` | `str \| dict \| None` | `None` | Late-data handling policy |

---

## Common patterns

### `update()` vs `update_many()` performance

`update_many()` processes a batch of observations and uses a size-aware GIL strategy:
- Workloads with < 16 scalar work items keep the GIL (lower overhead for tiny batches)
- Workloads with >= 16 scalar work items release the GIL for throughput

| Batch size | `update()` median ms | `update_many()` median ms | Speedup |
|---:|---:|---:|---:|
| 1 | 0.004 | 0.010 | 0.36x |
| 16 | 0.036 | 0.031 | 1.15x |
| 64 | 0.131 | 0.089 | 1.47x |
| 4096 | 7.822 | 4.462 | 1.75x |

**Guidance:** Use `update()` for single-observation real-time processing. Use `update_many()` for batch replay or when processing buffers of observations.

### Checkpoint and restore

All online detectors support stateful checkpoint/restore for fault tolerance:

```python
# Save state
state_bytes = bocpd.save_state(format="bytes")

# ... later, after restart ...
new_bocpd = cpd.Bocpd(model="gaussian_nig", hazard=1.0 / 200.0)
new_bocpd.load_state(state_bytes, format="bytes")
```

Supported formats:
- `"bytes"`: compact binary format (default)
- `"dict"`: Python dict for inspection
- File path: save/load directly to disk via the `path` parameter

### Late data policy

Configure how detectors handle observations arriving after a gap:

```python
bocpd = cpd.Bocpd(
    model="gaussian_nig",
    late_data_policy="ignore",  # or a dict with custom config
)
```

### Reset

All online detectors support `reset()` to clear internal state and start fresh:

```python
bocpd.reset()  # Clears run-length distribution, ready for new stream
```
