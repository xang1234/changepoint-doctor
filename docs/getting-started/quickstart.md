# Quickstart

Get from zero to detected change points in 5 minutes.

> Install/import naming: install with `python -m pip install changepoint-doctor`, then import with `import cpd` in Python. Optional compatibility alias: `import changepoint_doctor as cpd`.

## Offline detection

Detect change points in a batch signal using PELT:

```python
import numpy as np
import cpd

# Create a 3-segment signal with mean shifts
rng = np.random.default_rng(7)
base = np.concatenate([
    np.zeros(40, dtype=np.float64),
    np.full(40, 6.0, dtype=np.float64),
    np.full(40, -3.0, dtype=np.float64),
])
x = base + rng.normal(0.0, 0.6, size=base.shape[0])

# Detect with PELT
result = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
print(result.breakpoints)  # [40, 80, 120]
```

All three high-level offline detectors follow the same `fit/predict` pattern:

```python
pelt_result = cpd.Pelt(model="l2").fit(x).predict(n_bkps=2)
binseg_result = cpd.Binseg(model="l2").fit(x).predict(n_bkps=2)
fpop_result = cpd.Fpop(min_segment_len=2).fit(x).predict(n_bkps=2)
```

## Online (streaming) detection

Monitor a live stream for change points using BOCPD:

```python
import numpy as np
import cpd

rng = np.random.default_rng(11)
stream = np.concatenate([
    rng.normal(0.0, 0.35, 120),
    rng.normal(2.0, 0.35, 120),
]).astype(np.float64)

bocpd = cpd.Bocpd(
    model="gaussian_nig",
    hazard=1.0 / 200.0,
    max_run_length=512,
    alert_policy={"threshold": 0.35, "cooldown": 5, "min_run_length": 10},
)

steps = bocpd.update_many(stream)
first_alert = next((i for i, step in enumerate(steps) if step.alert), None)
print("First BOCPD alert index:", first_alert)
```

## Low-level `detect_offline()`

For full control over detector, cost model, constraints, stopping, and preprocessing:

```python
import numpy as np
import cpd

x = np.concatenate([
    np.zeros(50, dtype=np.float64),
    np.full(50, 5.0, dtype=np.float64),
    np.full(50, -2.0, dtype=np.float64),
])

result = cpd.detect_offline(
    x,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 2, "jump": 1},
    stopping={"n_bkps": 2},
    repro_mode="balanced",
)

print("Breakpoints:", result.breakpoints)
print("Algorithm:", result.diagnostics.algorithm)
print("Cost model:", result.diagnostics.cost_model)
```

## Doctor recommendations

Use the doctor CLI to get pipeline recommendations, then execute in Python:

```bash
cpd doctor --input /path/to/signal.csv --objective balanced --min-confidence 0.2 --output doctor.json
```

Execute a recommended pipeline:

```python
import cpd

# Pipeline dict from doctor output
pipeline = {
    "detector": {"kind": "pelt"},
    "cost": "l2",
    "stopping": {"pen": "bic"},
    "constraints": {"min_segment_len": 2},
}

result = cpd.detect_offline(x, pipeline=pipeline)
print(result.breakpoints)
```

## Serialize and plot results

```python
# Serialize to JSON and restore
payload = result.to_json()
restored = cpd.OfflineChangePointResult.from_json(payload)
assert restored.breakpoints == result.breakpoints

# Plot (requires matplotlib)
try:
    fig = result.plot(x, title="Detected breakpoints")
    fig.savefig("breakpoints.png", dpi=150, bbox_inches="tight")
except ImportError:
    print("Install changepoint-doctor[plot] to enable plotting.")
```

## Next steps

- {doc}`../user-guide/offline-algorithms` -- all 9 offline algorithms with examples
- {doc}`../user-guide/online-algorithms` -- BOCPD, CUSUM, Page-Hinkley
- {doc}`../user-guide/doctor` -- the recommendation engine in depth
- {doc}`../api-reference/index` -- complete Python API reference
