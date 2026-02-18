# Crate Hierarchy

`changepoint-doctor` is organized as a Cargo workspace with 10 crates, each with a focused responsibility.

## Dependency graph

```{mermaid}
graph TD
    core[cpd-core]
    costs[cpd-costs]
    preprocess[cpd-preprocess]
    offline[cpd-offline]
    online[cpd-online]
    doctor[cpd-doctor]
    python[cpd-python]
    cli[cpd-cli]
    bench[cpd-bench]
    eval[cpd-eval]

    costs --> core
    preprocess --> core
    offline --> core
    offline --> costs
    online --> core
    online --> costs
    doctor --> core
    doctor --> costs
    doctor --> offline
    doctor --> online
    doctor --> preprocess
    python --> core
    python --> costs
    python --> offline
    python --> online
    python --> doctor
    python --> preprocess
    cli --> doctor
    bench --> core
    bench --> costs
    bench --> offline
    bench --> online
    eval --> core
    eval --> offline
    eval --> online
```

## Per-crate responsibilities

### `cpd-core`

Foundation types shared across the workspace:
- `TimeSeriesView<'a>`: zero-copy borrowed view over signal data with layout, missing policy, and time index
- `CpdError`: unified error type with variants for invalid input, resource limits, cancellation, and algorithm failures
- `ExecutionContext`: constraints, cancellation token, budget mode, repro mode, progress/telemetry sinks
- `Constraints`: min_segment_len, jump, max_change_points, budget controls
- `OfflineChangePointResult`, `OnlineStepResult`: result types
- `Diagnostics`, `PruningStats`: metadata
- `ReproMode`: strict/balanced/fast reproducibility contracts
- Numeric utilities: log_sum_exp, stable_mean, Kahan summation, prefix sums

### `cpd-costs`

All cost model implementations:
- `CostModel` trait and `CachedCost<C>` wrapper
- Implementations: `CostL2Mean`, `CostL1Median`, `CostNormalMeanVar`, `CostNormalFullCov`, `CostNIGMarginal`, `CostAR`, `CostBernoulli`, `CostPoissonRate`, `CostRank`, `CostCosine`

### `cpd-preprocess`

Signal preprocessing pipeline:
- Detrend (linear, polynomial)
- Deseasonalize (differencing, STL-like)
- Winsorize
- Robust scale (MAD-based)

### `cpd-offline`

Offline (batch) detector implementations:
- `Pelt`, `BinSeg`, `Fpop`, `Dynp` (SegNeigh), `Wbs`, `BottomUp`, `SlidingWindow`
- Each implements the `OfflineDetector` trait from cpd-core
- Experimental: `KernelCpd`, `GpCpd`, `ArgpCpd` (feature-gated)

### `cpd-online`

Online (streaming) detector implementations:
- `Bocpd` with conjugate observation models (`GaussianNigPrior`, `BernoulliBetaPrior`, `PoissonGammaPrior`)
- `Cusum`, `PageHinkley`
- Each implements the `OnlineDetector` trait from cpd-core
- Checkpoint/restore via `save_state()` / `load_state()`

### `cpd-doctor`

Recommendation engine:
- Signal diagnostics (distribution shape, autocorrelation, seasonality, missing patterns)
- Calibration families and per-family scoring
- Pipeline ranking with confidence intervals
- Preprocessing recommendations
- Pipeline execution helpers

### `cpd-python`

PyO3 extension module:
- Python class wrappers for all detectors
- NumPy array interop for `TimeSeriesView`
- GIL-release strategy for batched operations
- Result serialization/deserialization
- Module: `cpd._cpd_rs`

### `cpd-cli`

Command-line interface:
- `cpd doctor` for recommendation workflows
- CSV input, JSON output

### `cpd-bench`

Criterion benchmark harness:
- Offline detector benchmarks (various n, k, cost combinations)
- Cost model benchmarks (univariate + multivariate)
- SegNeigh scaling benchmarks

### `cpd-eval`

Evaluation framework:
- Parity testing against reference implementations (ruptures, bayesian_changepoint_detection)
- Corpus management and metric computation

## Build configuration

| Setting | Value |
|---|---|
| Rust edition | 2024 |
| MSRV | 1.93 |
| Resolver | 2 |
| License | MIT OR Apache-2.0 |
| PyO3 | 0.24.1 with abi3-py39 |

## Feature flags

| Feature | Crates affected | Description |
|---|---|---|
| `rayon` | cpd-offline | Parallel execution for supported algorithms |
| `serde` | cpd-core, cpd-costs, cpd-offline, cpd-online, cpd-doctor | JSON serialization |
| `tracing` | cpd-core, cpd-offline, cpd-online | Structured logging |
| `simd` | cpd-costs | SIMD-accelerated numeric kernels |
| `kernel` | cpd-offline, cpd-costs | Kernel-based detection (experimental) |
| `kernel-approx` | cpd-offline, cpd-costs | Approximate kernel methods |
| `blas` | cpd-costs | BLAS-accelerated linear algebra |
| `gp` | cpd-offline, cpd-costs | Gaussian process detection (experimental) |
| `preprocess` | cpd-preprocess, cpd-doctor, cpd-python | Signal preprocessing |
| `repro-strict` | cpd-costs | Deterministic numeric paths for strict mode |
