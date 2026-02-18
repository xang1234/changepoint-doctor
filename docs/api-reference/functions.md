# Functions

## `detect_offline()`

Low-level offline detection with full control over all parameters.

### Signature

```python
cpd.detect_offline(
    values: Any,
    *,
    pipeline: dict[str, Any] | None = None,
    detector: str = "pelt",
    cost: str = "l2",
    constraints: dict[str, Any] | None = None,
    stopping: dict[str, Any] | None = None,
    preprocess: dict[str, Any] | None = None,
    repro_mode: str = "balanced",
    return_diagnostics: bool = True,
) -> OfflineChangePointResult
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `array-like` | (required) | Input signal. float32 or float64. Shape `(n,)` or `(n, d)` |
| `pipeline` | `dict \| None` | `None` | Complete pipeline spec (overrides individual params when provided) |
| `detector` | `str` | `"pelt"` | Detector algorithm |
| `cost` | `str` | `"l2"` | Cost model |
| `constraints` | `dict \| None` | `None` | Constraint configuration |
| `stopping` | `dict \| None` | `None` | Stopping configuration |
| `preprocess` | `dict \| None` | `None` | Preprocessing pipeline |
| `repro_mode` | `str` | `"balanced"` | Reproducibility mode |
| `return_diagnostics` | `bool` | `True` | Include diagnostics in the result |

### Accepted detector strings

| String | Algorithm | Notes |
|---|---|---|
| `"pelt"` | PELT | Default. Supports all cost models |
| `"binseg"` | Binary Segmentation | Supports all cost models |
| `"fpop"` | FPOP | L2 cost only |
| `"segneigh"` | SegNeigh (exact DP) | Best with `n_bkps` stopping |
| `"dynp"` | Alias for SegNeigh | |

### Accepted cost strings

| String | Cost model | Notes |
|---|---|---|
| `"l2"` | CostL2Mean | Default |
| `"l1_median"` | CostL1Median | |
| `"normal"` | CostNormalMeanVar | Diagonal covariance |
| `"normal_full_cov"` | CostNormalFullCov | Full covariance |
| `"nig"` | CostNIGMarginal | Pipeline-only |

### Constraints dict schema

```python
constraints = {
    "min_segment_len": 2,       # int: minimum segment length
    "jump": 1,                  # int: candidate stride
    "max_change_points": 10,    # int | None: hard cap
}
```

### Stopping dict schema

```python
# Known number of change points
stopping = {"n_bkps": 3}

# Manual penalty
stopping = {"pen": 10.0}

# BIC/AIC
stopping = {"pen": "bic"}
stopping = {"pen": "aic"}

# Penalty path (sweep multiple penalties)
stopping = {"PenaltyPath": [1.0, 5.0, 10.0, 50.0]}
```

### Pipeline mode

When `pipeline` is provided, it overrides `detector`, `cost`, `constraints`, and `stopping`. The pipeline dict accepts both simplified Python format and the full Rust `PipelineSpec` serde format:

```python
# Simplified format
pipeline = {
    "detector": {"kind": "pelt"},
    "cost": "l2",
    "stopping": {"pen": "bic"},
    "constraints": {"min_segment_len": 2},
}

# Rust serde format
pipeline = {
    "detector": {"Offline": {"Pelt": {"stopping": {"Penalized": {"Bic": {}}}}}},
    "cost": "L2",
    "constraints": {"min_segment_len": 2},
}
```

### Example

```python
import numpy as np
import cpd

x = np.concatenate([np.zeros(50), np.full(50, 5.0)])

# Simple usage
result = cpd.detect_offline(x, detector="pelt", cost="l2", stopping={"n_bkps": 1})

# With preprocessing
result = cpd.detect_offline(
    x,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 2},
    stopping={"pen": "bic"},
    preprocess={"detrend": {"method": "linear"}},
    repro_mode="strict",
)

# Using doctor pipeline
result = cpd.detect_offline(x, pipeline=doctor_recommendation["pipeline"])
```

---

## `smoke_detect()`

Quick smoke-test detection for validation and CI pipelines.

### Signature

```python
cpd.smoke_detect(values: Sequence[float]) -> list[int]
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `values` | `Sequence[float]` | Input signal as a sequence of floats |

### Returns

`list[int]` -- Detected breakpoint indices.

### Example

```python
breakpoints = cpd.smoke_detect([0.0] * 50 + [5.0] * 50)
```

:::{note}
`smoke_detect()` uses a fixed internal configuration and is intended for quick validation, not production analysis.
:::
