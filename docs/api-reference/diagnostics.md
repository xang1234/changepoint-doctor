# Diagnostics

Rich metadata returned with every offline detection result.

## `Diagnostics`

Accessible via `result.diagnostics`.

### Properties

| Property | Type | Description |
|---|---|---|
| `n` | `int` | Number of observations in the input |
| `d` | `int` | Number of dimensions (1 for univariate) |
| `schema_version` | `int` | JSON schema version marker |
| `engine_version` | `str \| None` | Rust engine version string |
| `runtime_ms` | `int \| None` | Detection runtime in milliseconds |
| `notes` | `list[str]` | Informational notes (e.g., penalty path results) |
| `warnings` | `list[str]` | Warning messages (e.g., masking risk) |
| `algorithm` | `str` | Algorithm name (e.g., `"pelt"`, `"binseg"`) |
| `cost_model` | `str` | Cost model name (e.g., `"l2"`, `"normal"`) |
| `seed` | `int \| None` | RNG seed (for WBS and randomized methods) |
| `repro_mode` | `str` | Reproducibility mode (`"strict"`, `"balanced"`, `"fast"`) |
| `thread_count` | `int \| None` | Number of threads used |
| `blas_backend` | `str \| None` | BLAS backend name (`None` for default BLAS-free wheels) |
| `cpu_features` | `list[str] \| None` | Detected CPU features (e.g., `["avx2", "fma"]`) |
| `params_json` | `Any \| None` | Algorithm-specific parameters as JSON |
| `pruning_stats` | `PruningStats \| None` | Pruning statistics (PELT/FPOP only) |
| `missing_policy_applied` | `str \| None` | Missing data policy used |
| `missing_fraction` | `float \| None` | Fraction of missing values in input |
| `effective_sample_count` | `int \| None` | Count of non-missing observations |

### Example

```python
result = cpd.Pelt(model="l2").fit(x).predict(pen="bic")
diag = result.diagnostics

print(f"Algorithm: {diag.algorithm}")
print(f"Cost model: {diag.cost_model}")
print(f"Runtime: {diag.runtime_ms}ms")
print(f"Repro mode: {diag.repro_mode}")

if diag.pruning_stats:
    print(f"Candidates pruned: {diag.pruning_stats.candidates_pruned}")

if diag.warnings:
    for w in diag.warnings:
        print(f"Warning: {w}")
```

---

## `PruningStats`

Pruning statistics for PELT and FPOP detectors.

### Properties

| Property | Type | Description |
|---|---|---|
| `candidates_considered` | `int` | Total number of candidate split points evaluated |
| `candidates_pruned` | `int` | Number of candidates removed by the pruning rule |

---

## `SegmentStats`

Per-segment statistics included in the result when available.

### Properties

| Property | Type | Description |
|---|---|---|
| `start` | `int` | Segment start index (inclusive) |
| `end` | `int` | Segment end index (exclusive) |
| `mean` | `list[float] \| None` | Per-dimension mean values |
| `variance` | `list[float] \| None` | Per-dimension variance values |
| `count` | `int` | Number of observations in the segment |
| `missing_count` | `int` | Number of missing values in the segment |

### Example

```python
if result.segments:
    for seg in result.segments:
        print(f"Segment [{seg.start}, {seg.end}): "
              f"mean={seg.mean}, count={seg.count}")
```
