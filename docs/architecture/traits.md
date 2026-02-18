# Core Traits

The Rust core defines several traits that form the extension points of the system.

## `CostModel`

Defined in `cpd-costs`. The shared contract for segment cost computation.

```rust
pub trait CostModel {
    type Cache: Send + Sync;

    fn name(&self) -> &'static str;
    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError>;
    fn missing_support(&self) -> MissingSupport { MissingSupport::Reject }

    fn precompute(
        &self,
        x: &TimeSeriesView<'_>,
        policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError>;

    fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize;
    fn supports_approx_cache(&self) -> bool { false }

    fn penalty_params_per_segment(&self) -> usize { 2 }
    fn penalty_effective_params(&self, d: usize) -> Option<usize> {
        d.checked_mul(self.penalty_params_per_segment())
    }

    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64;

    fn segment_cost_batch(
        &self,
        cache: &Self::Cache,
        queries: &[(usize, usize)],
        out_costs: &mut [f64],
    ) { /* default: loops over segment_cost */ }
}
```

**Key design points:**
- **Two-phase pattern**: `precompute()` builds a cache once; `segment_cost()` queries it O(1) per segment
- **Half-open intervals**: segments are `[start, end)` by convention
- **Cache + Send + Sync**: enables safe sharing across threads when using Rayon
- **Penalty integration**: `penalty_params_per_segment()` drives BIC/AIC penalty computation; models with non-linear complexity (e.g., `CostNormalFullCov`) override `penalty_effective_params()`
- **Batch path**: `segment_cost_batch()` allows vectorized implementations; the default loops over `segment_cost()`

## `CachedCost<C>`

A convenience wrapper that bundles a `CostModel` with its precomputed cache:

```rust
pub struct CachedCost<C: CostModel> {
    model: C,
    cache: C::Cache,
}

impl<C: CostModel> CachedCost<C> {
    pub fn new(model: C, x: &TimeSeriesView<'_>, policy: &CachePolicy) -> Result<Self, CpdError>;
    pub fn from_parts(model: C, cache: C::Cache) -> Self;
    pub fn segment_cost(&self, start: usize, end: usize) -> f64;
    pub fn segment_cost_batch(&self, queries: &[(usize, usize)], out_costs: &mut [f64]);
    pub fn model(&self) -> &C;
    pub fn cache(&self) -> &C::Cache;
    pub fn into_parts(self) -> (C, C::Cache);
}
```

`CachedCost::new()` validates data compatibility (including missing-policy checks) and materializes the cache.

## `OfflineDetector`

Defined in `cpd-core`. Contract for batch detectors.

```rust
pub trait OfflineDetector {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError>;
}
```

All offline algorithms (PELT, BinSeg, FPOP, SegNeigh, WBS) implement this trait. The `ExecutionContext` carries constraints, cancellation, budget mode, and reproducibility settings.

## `OnlineDetector`

Defined in `cpd-core`. Contract for streaming detectors.

```rust
pub trait OnlineDetector {
    type State: Clone + std::fmt::Debug;

    fn reset(&mut self);
    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError>;
    fn save_state(&self) -> Self::State;
    fn load_state(&mut self, state: &Self::State);

    fn update_many(
        &mut self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Vec<OnlineStepResult>, CpdError> { /* default impl */ }
}
```

**Key design points:**
- **Stateful**: detectors maintain internal state (run-length distribution for BOCPD, cumulative sums for CUSUM)
- **Checkpoint/restore**: `save_state()` / `load_state()` enable fault tolerance
- **Default `update_many`**: iterates `update()` with cancellation checks; implementations can override for batched optimization
- **Multivariate via `x_t: &[f64]`**: single observation is a slice of dimension d

## `ExecutionContext`

Unified execution context threaded through all detector calls:

```rust
pub struct ExecutionContext<'a> {
    pub constraints: &'a Constraints,
    pub cancel: Option<&'a CancelToken>,
    pub budget_mode: BudgetMode,
    pub repro_mode: ReproMode,
    pub progress: Option<&'a dyn ProgressSink>,
    pub telemetry: Option<&'a dyn TelemetrySink>,
}
```

Builder pattern:

```rust
let ctx = ExecutionContext::new(&constraints)
    .with_cancel(&cancel)
    .with_budget_mode(BudgetMode::SoftDegrade)
    .with_repro_mode(ReproMode::Strict)
    .with_progress_sink(&progress)
    .with_telemetry_sink(&telemetry);
```

Methods for cooperative cancellation and budget enforcement:
- `check_cancelled()` / `check_cancelled_every(iter, every)`
- `check_cost_eval_budget(evals)`
- `check_time_budget(started_at)`
- `report_progress(fraction)` / `record_scalar(key, value)`

## `TimeSeriesView<'a>`

Zero-copy borrowed view over signal data:

```rust
pub struct TimeSeriesView<'a> {
    pub values: DTypeView<'a>,   // F32 or F64 slice
    pub n: usize,                // number of observations
    pub d: usize,                // number of dimensions
    pub layout: MemoryLayout,    // CContiguous, FContiguous, or Strided
    pub mask: Option<&'a [u8]>,  // optional NaN/missing mask
    pub time: TimeIndex<'a>,     // None, Uniform{t0,dt}, or Explicit
    pub missing: MissingPolicy,  // Error, Ignore, or Impute
}
```

This design avoids data copying when converting from NumPy arrays via PyO3.

## `CpdError`

Unified error type with categorized variants:

- `InvalidInput(String)` -- bad parameters, shape mismatches, unsupported configurations
- `ResourceLimit(String)` -- budget exceeded (cost evals, time, memory)
- `Cancelled` -- cooperative cancellation triggered
- `AlgorithmFailure(String)` -- numerical failures, convergence issues
