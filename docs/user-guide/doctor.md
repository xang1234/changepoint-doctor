# Doctor Recommendation Engine

The **doctor** analyzes your signal's statistical properties and recommends ranked detector/cost/stopping pipelines with calibrated confidence scores.

## What the doctor does

1. **Diagnose** -- Computes signal statistics (distribution shape, autocorrelation, seasonality, missing data patterns, dimensionality)
2. **Classify** -- Maps the signal to one or more calibration families
3. **Recommend** -- Generates ranked pipeline recommendations scored by confidence and objective fit
4. **Execute** -- Recommendations can be directly executed via `detect_offline(pipeline=...)`

## CLI workflow

```bash
cpd doctor \
    --input /path/to/signal.csv \
    --objective balanced \
    --min-confidence 0.2 \
    --output doctor.json
```

The output is a JSON file with ranked recommendations, each containing:
- Pipeline specification (detector, cost, stopping, constraints, preprocessing)
- Confidence score and confidence interval
- Resource estimates
- Explanation and warnings
- Objective fit scores

## Python integration

Execute a doctor recommendation directly:

```python
import cpd
import json

# Load doctor output
with open("doctor.json") as f:
    recommendations = json.load(f)

# Use the top recommendation's pipeline
pipeline = recommendations[0]["pipeline"]
result = cpd.detect_offline(x, pipeline=pipeline)
print(result.breakpoints)
```

## Objectives

The objective parameter controls the tradeoff between speed, accuracy, and robustness in pipeline ranking:

| Objective | Description |
|---|---|
| `Balanced` | Default. Balances accuracy, speed, and generality |
| `Speed` | Favors fast algorithms (PELT, CUSUM) with simpler cost models |
| `Accuracy` | Favors algorithms with stronger optimality guarantees (FPOP, SegNeigh) |
| `Robustness` | Favors non-parametric or masking-resistant approaches (WBS, Rank cost) |

## Calibration families

The doctor classifies signals into families for calibration-aware scoring:

| Family | Characteristics |
|---|---|
| `Gaussian` | Near-normal distribution, light tails |
| `HeavyTailed` | Excess kurtosis, outlier-prone |
| `Autocorrelated` | Significant temporal dependence |
| `Seasonal` | Periodic patterns detected |
| `Multivariate` | d > 1 dimensions |
| `Binary` | Values near 0 or 1 (within tolerance) |
| `Count` | Non-negative integer-valued data |

## Confidence formula

Each recommendation includes a calibrated confidence score:

```
confidence = clamp(
    (intercept + slope * heuristic_confidence) * (1 - ood_penalty),
    0.01,
    0.99
)
```

Where:
- `intercept` and `slope` are per-family calibration parameters
- `heuristic_confidence` is the raw score from pipeline-data compatibility analysis
- `ood_penalty = clamp(1 - exp(-0.90 * diagnostic_divergence), 0.0, 0.80)` penalizes out-of-distribution signals
- Final confidence is clamped to [0.01, 0.99]

## Preprocessing recommendations

The doctor also recommends preprocessing based on signal diagnostics:

| Signal property | Recommended preprocessing |
|---|---|
| Linear or polynomial trend | `detrend` |
| Seasonal pattern detected | `deseasonalize` |
| High outlier rate | `winsorize` |
| Scale instability across segments | `robust_scale` |

## Worked example

Consider a seasonal signal with a trend and a change in mean at index 500:

```python
import numpy as np
import cpd

# Seasonal + trend + change point
t = np.arange(1000, dtype=np.float64)
seasonal = 2.0 * np.sin(2 * np.pi * t / 50)
trend = 0.005 * t
shift = np.where(t >= 500, 3.0, 0.0)
noise = np.random.default_rng(42).normal(0, 0.5, 1000)
signal = seasonal + trend + shift + noise

# Doctor would recommend preprocessing + PELT
# After running doctor CLI or using the recommendation:
result = cpd.detect_offline(
    signal,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 10},
    stopping={"pen": "bic"},
    preprocess={
        "detrend": {"method": "linear"},
        "deseasonalize": {"method": "stl_like", "period": 50},
    },
)

print("Change points:", result.change_points)
# Expected: change point near index 500
```

## Multivariate awareness

- **Offline:** Doctor emits multivariate-specific guidance for cost model selection (diagonal vs full covariance tradeoffs)
- **Online:** Doctor rejects multivariate inputs (d > 1) with a clear guidance error, as online detectors currently support only univariate data
