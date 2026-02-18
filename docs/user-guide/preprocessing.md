# Preprocessing

The `preprocess` parameter in `detect_offline()` applies signal transformations before change-point detection. Preprocessing is validated strictly: unsupported keys or invalid method/parameter combinations raise `ValueError`.

## Pipeline stages

Preprocessing stages are applied in this fixed order:

1. **Detrend** -- Remove linear or polynomial trends
2. **Deseasonalize** -- Remove seasonal patterns
3. **Winsorize** -- Clip extreme values
4. **Robust scale** -- Normalize by robust statistics (MAD)

## Full configuration example

```python
result = cpd.detect_offline(
    x,
    detector="pelt",
    cost="l2",
    constraints={"min_segment_len": 2, "jump": 1},
    stopping={"n_bkps": 2},
    preprocess={
        "detrend": {"method": "linear"},
        "deseasonalize": {"method": "differencing", "period": 12},
        "winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95},
        "robust_scale": {"mad_epsilon": 1e-9, "normal_consistency": 1.4826},
    },
)
```

## Stage details

### Detrend

Removes trends from the signal before detection.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `method` | `str` | Yes | `"linear"` or `"polynomial"` |
| `degree` | `int` | For polynomial | Polynomial degree (required when `method="polynomial"`) |

```python
# Linear detrending
preprocess = {"detrend": {"method": "linear"}}

# Polynomial detrending (degree 2)
preprocess = {"detrend": {"method": "polynomial", "degree": 2}}
```

### Deseasonalize

Removes periodic patterns from the signal.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `method` | `str` | Yes | `"differencing"` or `"stl_like"` |
| `period` | `int` | Yes | Seasonal period (>= 1 for differencing, >= 2 for stl_like) |

```python
# Differencing (simple lag subtraction)
preprocess = {"deseasonalize": {"method": "differencing", "period": 12}}

# STL-like decomposition
preprocess = {"deseasonalize": {"method": "stl_like", "period": 12}}
```

### Winsorize

Clips extreme values at specified quantiles. Reduces the influence of outliers.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lower_quantile` | `float` | `0.01` | Lower clipping quantile |
| `upper_quantile` | `float` | `0.99` | Upper clipping quantile |

```python
# Default winsorization (1st and 99th percentiles)
preprocess = {"winsorize": {}}

# Custom quantiles
preprocess = {"winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95}}
```

### Robust scale

Normalizes the signal using median and MAD (Median Absolute Deviation), providing outlier-robust standardization.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mad_epsilon` | `float` | `1e-9` | Floor for MAD to avoid division by zero |
| `normal_consistency` | `float` | `1.4826` | Consistency constant for normal distribution |

```python
# Default robust scaling
preprocess = {"robust_scale": {}}
```

## Validation rules

- Unknown preprocessing stage keys raise `ValueError`
- `detrend.method` must be `"linear"` or `"polynomial"`
- `deseasonalize.method` must be `"differencing"` (period >= 1) or `"stl_like"` (period >= 2)
- All parameter values must be finite and within valid ranges

:::{note}
Preprocessing requires the `preprocess` feature flag when building from source. PyPI wheels include this by default.
:::
