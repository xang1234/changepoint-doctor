# Offline Detectors

High-level Python classes for batch change-point detection. All follow the `fit/predict` pattern.

## `Pelt`

PELT (Pruned Exact Linear Time) detector.

### Constructor

```python
cpd.Pelt(
    model: str = "l2",
    min_segment_len: int = 2,
    jump: int = 1,
    max_change_points: int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"l2"` | Cost model. Accepts `"l2"`, `"normal"`, `"normal_full_cov"` |
| `min_segment_len` | `int` | `2` | Minimum number of samples in each segment |
| `jump` | `int` | `1` | Stride over candidate split points. Increase to trade precision for speed |
| `max_change_points` | `int \| None` | `None` | Hard upper bound on the number of change points returned |

### Methods

#### `fit(values) -> Pelt`

Precompute cost cache on the input signal.

- **values**: array-like of `float32` or `float64`. Shape `(n,)` for univariate, `(n, d)` for multivariate.
- **Returns**: `self` (for method chaining)
- **Raises**: `TypeError` if input dtype is not float32/float64; `RuntimeError` if input contains NaN

#### `predict(*, pen=None, n_bkps=None) -> OfflineChangePointResult`

Run PELT detection and return results.

- **pen**: `float | str | None` -- Penalty value. A float for manual penalty, `"bic"` for BIC, `"aic"` for AIC
- **n_bkps**: `int | None` -- Exact number of change points to detect
- **Returns**: {class}`OfflineChangePointResult`
- **Raises**: `RuntimeError` if `fit()` was not called first

:::{note}
Exactly one of `pen` or `n_bkps` must be provided.
:::

### Example

```python
import numpy as np
import cpd

x = np.concatenate([np.zeros(50), np.full(50, 5.0), np.full(50, -2.0)])
result = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(n_bkps=2)
print(result.breakpoints)  # [50, 100, 150]
```

---

## `Binseg`

Binary Segmentation detector.

### Constructor

```python
cpd.Binseg(
    model: str = "l2",
    min_segment_len: int = 2,
    jump: int = 1,
    max_change_points: int | None = None,
    max_depth: int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"l2"` | Cost model. Accepts `"l2"`, `"normal"`, `"normal_full_cov"` |
| `min_segment_len` | `int` | `2` | Minimum segment length |
| `jump` | `int` | `1` | Candidate stride |
| `max_change_points` | `int \| None` | `None` | Hard cap on detected change points |
| `max_depth` | `int \| None` | `None` | Maximum recursion depth for the binary split tree |

### Methods

#### `fit(values) -> Binseg`

Same interface as `Pelt.fit()`.

#### `predict(*, pen=None, n_bkps=None) -> OfflineChangePointResult`

Same interface as `Pelt.predict()`.

### Example

```python
result = cpd.Binseg(model="l2").fit(x).predict(n_bkps=3)
```

:::{warning}
BinSeg may miss closely spaced changes (masking). Check diagnostics warnings. If masking is indicated, prefer WBS via `detect_offline(pipeline=...)`.
:::

---

## `Fpop`

Functional Pruning Optimal Partitioning detector. **L2 cost only.**

### Constructor

```python
cpd.Fpop(
    min_segment_len: int = 2,
    jump: int = 1,
    max_change_points: int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_segment_len` | `int` | `2` | Minimum segment length |
| `jump` | `int` | `1` | Candidate stride |
| `max_change_points` | `int \| None` | `None` | Hard cap |

:::{note}
FPOP does not accept a `model` parameter because it is restricted to L2 cost.
:::

### Methods

#### `fit(values) -> Fpop`

Same interface as `Pelt.fit()`.

#### `predict(*, pen=None, n_bkps=None) -> OfflineChangePointResult`

Same interface as `Pelt.predict()`.

### Example

```python
result = cpd.Fpop(min_segment_len=2).fit(x).predict(pen="bic")
```
