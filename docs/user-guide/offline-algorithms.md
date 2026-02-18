# Offline Algorithms

Offline (batch) algorithms process the entire signal at once and return a set of detected change points. `changepoint-doctor` provides 9 offline algorithms spanning exact, pruned, greedy, and non-parametric approaches.

## PELT (Pruned Exact Linear Time)

PELT {cite}`killick2012` is the recommended default for most use cases. It uses dynamic programming with a pruning rule that achieves O(n) complexity when the number of change points grows linearly with n.

**When to use:** Strong default when n is large and the number of change points is unknown.

**Complexity:** O(n) expected with pruning; O(n^2) worst case.

```python
import cpd

result = cpd.Pelt(model="l2", min_segment_len=2).fit(x).predict(pen="bic")
print(result.breakpoints)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"l2"` | Cost model: `"l2"`, `"normal"`, `"normal_full_cov"` |
| `min_segment_len` | `int` | `2` | Minimum segment length |
| `jump` | `int` | `1` | Candidate stride (increase to trade accuracy for speed) |
| `max_change_points` | `int \| None` | `None` | Hard cap on detected change points |

## BinSeg (Binary Segmentation)

Binary Segmentation {cite}`scott1974` is a greedy top-down approach that recursively splits the signal at the point of maximum cost reduction.

**When to use:** Fast approximate segmentation; good when speed matters more than optimality.

**Complexity:** O(n log n) for k change points.

**Masking risk:** BinSeg can miss closely spaced changes (masking). If diagnostics indicate masking risk, prefer WBS.

```python
result = cpd.Binseg(model="l2", min_segment_len=2).fit(x).predict(n_bkps=3)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"l2"` | Cost model |
| `min_segment_len` | `int` | `2` | Minimum segment length |
| `jump` | `int` | `1` | Candidate stride |
| `max_change_points` | `int \| None` | `None` | Hard cap |
| `max_depth` | `int \| None` | `None` | Maximum recursion depth |

## FPOP (Functional Pruning Optimal Partitioning)

FPOP {cite}`maidstone2017` uses functional pruning for optimal partitioning with L2 cost. It provides strong pruning behavior and exact solutions.

**When to use:** You want optimal partitioning with strong pruning; L2 cost only.

**Complexity:** O(n) expected.

```python
result = cpd.Fpop(min_segment_len=2).fit(x).predict(pen="bic")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_segment_len` | `int` | `2` | Minimum segment length |
| `jump` | `int` | `1` | Candidate stride |
| `max_change_points` | `int \| None` | `None` | Hard cap |

:::{note}
FPOP is restricted to L2 cost. Specifying a different cost model will raise an error.
:::

## SegNeigh / DynP (Exact Dynamic Programming)

Exact DP segmentation for a known number of change points k. Finds the globally optimal k-segmentation.

**When to use:** You know the exact number of change points and want guaranteed optimality.

**Complexity:** O(k * m^2) time, O(k * m) memory, where m is the effective candidate count.

```python
result = cpd.detect_offline(
    x,
    detector="segneigh",  # "dynp" is an alias
    cost="l2",
    constraints={"min_segment_len": 2},
    stopping={"n_bkps": 3},
)
```

**Sizing guide:**
- Use when k is known and m is modest
- Increase `jump` and/or `min_segment_len` to reduce m when runtime is too high
- Prefer PELT/FPOP when k is unknown or n is very large

## WBS (Wild Binary Segmentation)

WBS {cite}`fryzlewicz2014` draws random intervals and applies CUSUM-like tests within each, providing robustness against the masking effect that plagues standard BinSeg.

**When to use:** Closely spaced changes where BinSeg might mask weaker signals.

**Complexity:** O(M * n) where M is the number of random intervals.

```python
result = cpd.detect_offline(
    x,
    pipeline={
        "detector": {"kind": "wbs"},
        "cost": "l2",
        "stopping": {"pen": "bic"},
        "constraints": {"min_segment_len": 2},
    },
)
```

:::{note}
WBS is available via `detect_offline(pipeline=...)` but not yet as a high-level Python class.
:::

## BottomUp

Bottom-up segmentation starts with many short segments and iteratively merges adjacent segments with the lowest merge cost.

**When to use:** Complementary to BinSeg; can detect changes that top-down methods miss.

```python
result = cpd.detect_offline(
    x,
    pipeline={
        "detector": {"kind": "bottomup"},
        "cost": "l2",
        "stopping": {"pen": "bic"},
        "constraints": {"min_segment_len": 2},
    },
)
```

## SlidingWindow

Computes a local discrepancy score by sliding a window across the signal and comparing left/right sub-windows.

**When to use:** Quick visual scan of where changes might be; useful as a feature for downstream models.

```python
result = cpd.detect_offline(
    x,
    pipeline={
        "detector": {"kind": "sliding_window"},
        "cost": "l2",
        "stopping": {"pen": "bic"},
        "constraints": {"min_segment_len": 2},
    },
)
```

## KernelCpd (experimental)

Kernel-based change-point detection {cite}`harchaoui2007` uses RKHS embeddings for non-parametric detection. It can detect distributional changes beyond mean/variance shifts.

**When to use:** Non-parametric detection where the type of distributional change is unknown.

```python
result = cpd.detect_offline(
    x,
    pipeline={
        "detector": {"kind": "kernel_cpd"},
        "cost": "kernel",
        "stopping": {"n_bkps": 2},
        "constraints": {"min_segment_len": 5},
    },
)
```

:::{warning}
KernelCpd is experimental. Requires the `kernel` feature flag.
:::

## GpCpd / ArgpCpd (experimental)

Gaussian process change-point detection {cite}`saatci2010` uses GP marginal likelihood for change detection, modeling the signal as piecewise-stationary GP segments.

**When to use:** Smooth signals where GP models are appropriate and you need change detection that respects temporal correlation.

:::{warning}
GpCpd and ArgpCpd are experimental. Requires the `gp` feature flag.
:::

---

## Stopping criteria

All offline detectors support multiple stopping strategies:

| Stopping | Python usage | When to use |
|---|---|---|
| Known k (`n_bkps`) | `predict(n_bkps=3)` | You know the expected number of changes |
| Manual penalty | `predict(pen=10.0)` | Tight operational control over sensitivity |
| BIC {cite}`schwarz1978` | `predict(pen="bic")` | Good default for automatic model selection |
| AIC {cite}`akaike1974` | `predict(pen="aic")` | Less conservative than BIC; recovers weaker changes |
| PenaltyPath | `stopping={"PenaltyPath": [...]}` | Sweep multiple penalties in one pass |

### Penalty selection guidance

| Scenario | Recommended stopping |
|---|---|
| Unknown number of changes, moderate noise | `pen="bic"` |
| Unknown number, want to find weaker changes | `pen="aic"` |
| Known number of changes | `n_bkps=k` |
| Operational threshold tuning | `pen=<float>` (lower = more sensitive) |
| Exploratory analysis | `PenaltyPath` with a range of values |

BIC/AIC complexity terms are cost-model-aware:
- `l2`: `params_per_segment=2` (mean + residual variance proxy)
- `normal`: `params_per_segment=3` (mean + variance + residual term)
- `normal_full_cov`: model-aware complexity `1 + d + d(d+1)/2`
