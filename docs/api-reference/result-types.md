# Result Types

## `OfflineChangePointResult`

Returned by all offline detectors. Contains detected breakpoints, optional scores and segment statistics, and rich diagnostics.

### Properties

| Property | Type | Description |
|---|---|---|
| `breakpoints` | `list[int]` | Breakpoint indices (strictly increasing, final element = n) |
| `change_points` | `list[int]` | Change point indices (breakpoints excluding the terminal n) |
| `scores` | `list[float] \| None` | Per-change-point scores (length = len(change_points), if available) |
| `segments` | `list[SegmentStats] \| None` | Per-segment statistics (length = len(breakpoints)) |
| `diagnostics` | `Diagnostics` | Rich diagnostics metadata |

### Methods

#### `to_json() -> str`

Serialize the result to a JSON string following the versioned contract.

- Current writer version: schema marker `1`
- Unknown fields are preserved during round-trip

#### `from_json(payload: str) -> OfflineChangePointResult` (static)

Deserialize a result from a JSON string.

- Accepts schema markers in the supported window (currently `1..=2`)
- **Raises**: `ValueError` on unsupported schema version or structural validation failure

#### `plot(values=None, *, ax=None, title=None, breakpoint_color="crimson", breakpoint_style="--", line_width=1.5, show_legend=True) -> Any`

Plot the signal with detected breakpoints overlaid.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | `array \| None` | `None` | Signal values. Required if `segments` is `None` |
| `ax` | `matplotlib.Axes \| None` | `None` | Axes to plot on. Creates new figure if `None` |
| `title` | `str \| None` | `None` | Plot title |
| `breakpoint_color` | `str` | `"crimson"` | Color for breakpoint lines |
| `breakpoint_style` | `str` | `"--"` | Line style for breakpoints |
| `line_width` | `float` | `1.5` | Line width |
| `show_legend` | `bool` | `True` | Whether to show legend |

- **Returns**: matplotlib Figure
- **Raises**: `ImportError` if matplotlib is not installed

:::{note}
`plot(ax=...)` is supported only for univariate data (`diagnostics.d == 1`).
:::

### Example

```python
result = cpd.Pelt(model="l2").fit(x).predict(n_bkps=2)

# Inspect results
print(result.breakpoints)    # [50, 100, 150]
print(result.change_points)  # [50, 100]

# Serialize round-trip
payload = result.to_json()
restored = cpd.OfflineChangePointResult.from_json(payload)
assert restored.breakpoints == result.breakpoints

# Plot
fig = result.plot(x, title="Detected breakpoints")
```

---

## `OnlineStepResult`

Emitted by online detectors for each processed observation.

### Properties

| Property | Type | Description |
|---|---|---|
| `t` | `int` | Time index (0-based observation count) |
| `p_change` | `float` | Posterior probability of a change at this step |
| `alert` | `bool` | Whether the alert policy fired |
| `alert_reason` | `str \| None` | Reason the alert fired (e.g. `"threshold"`) |
| `run_length_mode` | `int` | Mode of the run-length posterior |
| `run_length_mean` | `float` | Mean of the run-length posterior |
| `processing_latency_us` | `int \| None` | Processing latency in microseconds (if measured) |

### Example

```python
bocpd = cpd.Bocpd(model="gaussian_nig", hazard=1/200)
step = bocpd.update(3.5)

print(step.t)          # 0
print(step.p_change)   # 0.005 (low for first observation)
print(step.alert)      # False
```
