# cpd-rs: High-Performance Change Point Detection in Rust + Python

A production-grade change point detection (CPD) toolkit with a Rust core (speed, memory efficiency, safety) and a Python API (PyPI wheels, NumPy-native, ruptures-like ergonomics). The design is modular (cost models + search methods) and extensible (new algorithms plug in cleanly). It also ships a “Change Point Doctor” that recommends a pipeline based on time-series diagnostics, constraints, and user objectives.

## Positioning and Differentiators

### Target users and jobs-to-be-done

- Data/ML engineers who need fast offline segmentation for large time series (10^5 to 10^7 points) with reproducible outputs and predictable resource usage.
- Platform/infra engineers who need online detectors that can run in services (checkpointable state, bounded memory, stable alerts).
- Researchers who want a clean cost/algorithm abstraction and trustworthy reference implementations in a systems language.

### Why this toolkit (vs. existing libraries)

- Production-first: safety rails (constraints), cancellation, deterministic outputs, structured diagnostics, and operationally useful error messages.
- Fast-path focus: O(1) segment costs via caches + pruning + candidate thinning, with clearly documented tradeoffs.
- Two-tier API: ruptures-like high-level ergonomics plus a low-level “power user” API returning rich result objects.
- Doctor as a differentiator: explainable recommendations and optional quick validation to reduce parameter-tuning pain.

## Project Goals, Nongoals, Success Criteria, and Roadmap

### Goals

- Provide a production-grade CPD toolkit with a Rust core that is fast, memory-efficient, and safe.
- Provide a Python API that feels familiar to ruptures users (fit/predict style) while exposing richer outputs (scores/probabilities/diagnostics) when needed.
- Make correctness + numerical stability a first-class concern (especially for Bayesian / probabilistic methods).
- Make long-series scalability a first-class concern (10^6 points should be plausible for “fast path” algorithms like PELT on common costs, with appropriate constraints and candidate thinning).
- Make “operational reliability” a first-class concern: bounded resource usage, cancellation, stable diagnostics, and reproducibility.

### Nongoals (for v1)

- Full GPU acceleration and/or deep learning-based CPD (possible later, optional track).
- A default path for arbitrary Python callback cost functions (too slow, hard to stabilize). We can support this later as an explicitly “slow path” experimental feature.
- Implementing every CPD algorithm in the literature (focus on high-impact coverage + extensibility).

### Success criteria (measurable)

- Correctness: crosscheck segmentation outputs against reference implementations (ruptures + curated datasets) on a shared test corpus.
- Performance: publish benchmark results (runtime + peak memory) for representative sizes (1e4, 1e5, 1e6) across key algorithms (PELT, BinSeg/WBS, BOCPD).
- Packaging: ship manylinux/macOS/Windows wheels; pip install works without requiring a Rust toolchain.
- Operational: provide deterministic outputs by default and support checkpoint/restore for online detectors.
- Usability: ship end-to-end examples (offline + online) that cover realistic “messy data” cases (missingness, outliers, autocorrelation).
- Evaluation: publish offline + online accuracy metrics (F1, mean time to detect, false alarm rate) on curated datasets.
- Reliability: publish explicit runtime/memory/latency SLOs and enforce regression thresholds in CI/nightly.

### Phased roadmap

- MVP-A (v0.1, fast offline core):
  - TimeSeriesView + unified execution context + constraints + cost caching infrastructure
  - Costs: L2 mean shift + Normal mean+var
  - Algorithms: PELT + Binary Segmentation
  - Packaging: cross-platform wheels + typed Python API + baseline docs/examples
  - Exit gate: deterministic outputs in Balanced mode, budget enforcement tests, wheel smoke tests pass
- MVP-B (v0.2, robustness + breadth):
  - Add robust-ish O(1) option via conjugate marginal likelihood (Normal-Inverse-Gamma / Student-t predictive)
  - Add WBS and multivariate polish for common costs
  - Add differential/parity tests + benchmark SLO gates + schema fixtures
  - Exit gate: curated corpus parity threshold met + benchmark SLO pass
- MVP-C (v0.3, streaming + doctor beta):
  - BOCPD + one low-latency baseline streaming detector
  - Doctor beta with ranked recommendations, confidence, and abstain mode
  - Checkpoint/restore + event-time semantics for online detectors
  - Exit gate: online p99 latency SLO pass + checkpoint compatibility pass
- v1.0 (“production-ready” breadth):
  - Offline: Dynp, BottomUp, Sliding Window
  - Penalty helpers (BIC/AIC-like, user-specified, “known K” constraints)
  - Stable result objects (scores/diagnostics) + multivariate support for common costs
  - Doctor v1 (ranked recommendations + calibrated confidence + safe configs)
- v1.x+ (experimental / heavy, featureflagged):
  - Kernel CPD with scalable approximations (Nyström / random features) rather than O(n²) defaults
  - FPOP / functional pruning where assumptions apply
  - GP/ARGP Bayesian variants (experimental; high complexity)
  - Ensembles (optional) + automated validation utilities

## Architecture Overview

### Design principles

- Cost model + search method + constraint as the core decomposition (consistent with the ruptures conceptual framing and the broader CPD literature).
- “Fast path” focuses on:
  - O(1) segment costs via caches (prefix sums / sufficient statistics)
  - pruning / candidate thinning (jump / candidate_splits)
  - bounded memory in online Bayesian detection (truncation/pruning)
- Reproducibility modes:
  - Strict mode: deterministic reductions, fixed thread count, conservative numeric path
  - Balanced mode (default): deterministic algorithmic path with bounded floating-point drift
  - Fast mode: maximum throughput with potential non-deterministic reductions
  - always capture mode + thread/backend metadata in diagnostics
- Operational reliability:
  - cancellation + optional time/compute budgets
  - online detector checkpoint/restore to support long-running services
  - explicit degrade-vs-fail behavior under budget pressure via an ordered degradation plan
- Observability:
  - optional tracing/profiling hooks that don’t pollute the hot path when disabled
- Core crate does not depend on PyO3 (no FFI leakage into the compute core).
- Prefer “safe Rust” for algorithm code; keep `unsafe` rare, audited, and behind clearly documented invariants.
- Feature flags keep default builds lean; heavy algorithms are opt-in.

## Layered Crate Architecture + Feature Flags

Use a Cargo workspace with strict crate boundaries:

### Workspace layout (proposed)

```text
cpd/
  Cargo.toml              # workspace root
  crates/
    cpd-core/             # shared types, traits, constraints, results, numeric utils
    cpd-costs/            # built-in cost models + caches (keeps core lean)
    cpd-preprocess/       # optional: detrend/deseasonalize/robust scaling
    cpd-offline/
    cpd-online/
    cpd-doctor/           # diagnostics + recommendations + quick validation
    cpd-python/
    cpd-cli/              # optional
    cpd-bench/            # optional (criterion harness)
    cpd-eval/             # optional: dataset registry + metrics + baselines
  python/
    cpd/                  # optional pure-Python helpers, stubs, docs assets
  docs/
  tests/
```

### Crates

- cpd-core
- TimeSeriesView, constraints, penalty helpers, shared detector traits
- result types + diagnostics + error types (CpdError)
- shared numeric utilities (log-sum-exp, stable stats, stable variance)
- optional serde support for result serialization
- cpd-costs
- built-in cost models + caches (L2, Gaussian/NIG, Poisson/Bernoulli, etc.)
- cpd-preprocess
- optional preprocessing pipeline (detrend, deseasonalize, winsorize, robust scaling)
- cpd-offline
- offline search algorithms: PELT, BinSeg, WBS, Dynp, BottomUp, Window, KernelCPD
- cpd-online
- streaming detectors: BOCPD (+ optional classical streaming detectors if desired)
- cpd-doctor
- diagnostics + ranked pipeline recommendations + optional validation helpers
- cpd-python
- PyO3 wrappers, NumPy interop, Python-facing config/result objects
- cpd-cli (optional)
- batch segmentation CLI: JSON in/out, easy pipeline integration
- cpd-bench (optional)
- criterion benchmarks + profiling integration
- cpd-eval (optional)
- synthetic + real dataset registry, evaluation metrics (offline + online), baseline comparisons (for doctor + docs)

### Feature flags

- rayon — optional parallelism (be careful with Python + BLAS oversubscription)
- serde — JSON (results/configs) for debugging + CLI
- tracing — optional instrumentation (tracing spans/events) for profiling/ops
- simd — optional SIMD fast paths where it pays off (kept isolated + benchmark-driven)
- kernel — kernel CPD (default off)
- kernel-approx — Nyström / random features
- blas — heavy linear algebra dependencies (off by default)
- gp — GP/ARGP Bayesian models (experimental; off by default)
- preprocess — optional preprocessing helpers (off by default)
- repro-strict — strict deterministic reductions/config for reproducibility-sensitive runs

## Core API: Results, Errors, Diagnostics, Offline + Online Traits

Returning only Vec<usize> is too limiting. We want structured outputs for:
- confidence scores/probabilities
- segment stats
- debugging (pruning counts, candidate thinning, run-time notes)
- Doctor explanations and autotuning

### Types (sketch)

Offline and online detectors have different “natural” outputs. Avoid a single catch-all result type with lots of `Option<T>` fields; it complicates the API and can accidentally encourage huge allocations (e.g., per-timestep probabilities for offline runs).

```rust
// Convention: indices are 0-based.
// Offline result returns breakpoints as segment end indices,
// typically including n (ruptures-like); see "Result conventions".

#[derive(Clone, Debug)]
pub struct OfflineChangePointResult {
    pub breakpoints: Vec<usize>,           // sorted, ends of segments (often includes n)
    pub change_points: Vec<usize>,         // breakpoints excluding n (derived)
    pub scores: Option<Vec<f64>>,          // per change point score (offline)
    pub segments: Option<Vec<SegmentStats>>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub n: usize,
    pub d: usize,
    pub schema_version: u32,            // schema for results/configs (bump on change)
    pub engine_version: Option<String>, // crate version / git describe
    pub runtime_ms: Option<u64>,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub algorithm: &'static str,
    pub cost_model: &'static str,
    pub seed: Option<u64>,
    pub repro_mode: ReproMode,
    pub thread_count: Option<usize>,
    pub blas_backend: Option<String>,
    pub cpu_features: Option<Vec<String>>,
    pub params_json: Option<serde_json::Value>, // behind `serde` feature
    pub pruning_stats: Option<PruningStats>,
}

#[derive(Clone, Debug)]
pub struct PruningStats {
    pub candidates_considered: usize,
    pub candidates_pruned: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ReproMode {
    Strict,
    Balanced, // default
    Fast,
}

#[derive(Clone, Copy, Debug)]
pub enum BudgetMode {
    HardFail,
    SoftDegrade,
}

#[derive(thiserror::Error, Debug)]
pub enum CpdError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("numerical issue: {0}")]
    NumericalIssue(String),
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("cancelled")]
    Cancelled,
}

pub trait ProgressSink: Send + Sync {
    fn on_progress(&self, fraction: f32);
}

pub trait TelemetrySink: Send + Sync {
    // Implementations can use interior mutability if needed.
    fn record_scalar(&self, key: &'static str, value: f64);
}

pub struct ExecutionContext<'a> {
    pub constraints: &'a Constraints,
    pub cancel: Option<&'a CancelToken>,
    pub budget_mode: BudgetMode,
    pub repro_mode: ReproMode,
    pub progress: Option<&'a dyn ProgressSink>,
    pub telemetry: Option<&'a dyn TelemetrySink>,
}

// Offline algorithms: full series -> Result
pub trait OfflineDetector {
    fn detect(
        &self,
        x: &TimeSeriesView,
        ctx: &ExecutionContext,
    ) -> Result<OfflineChangePointResult, CpdError>;
}

// Online algorithms: stateful incremental update.
// Keep per-step outputs small; if callers want history, they can store it.
pub struct OnlineStepResult {
    pub t: usize,
    pub p_change: f64,
    pub alert: bool,              // derived from alert policy
    pub alert_reason: Option<String>,
    pub run_length_mode: usize,
    pub run_length_mean: f64,
    pub processing_latency_us: Option<u64>,
}

pub trait OnlineDetector {
    type State: Clone + std::fmt::Debug;
    fn reset(&mut self);
    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext,
    ) -> Result<OnlineStepResult, CpdError>;
    fn update_many(
        &mut self,
        x: &TimeSeriesView,
        ctx: &ExecutionContext,
    ) -> Result<Vec<OnlineStepResult>, CpdError>;
    fn save_state(&self) -> Self::State;
    fn load_state(&mut self, state: &Self::State);
}

```
## Formalize the Data Model: Univariate + Multivariate, Missing Values, ZeroCopy Python Interop

### TimeSeriesView contract

We want a predictable and safe boundary that supports both Rust users and NumPy users.
```rust
pub enum MissingPolicy {
    Error,          // default; safest
    ImputeZero,     // simple, explicit
    ImputeLast,     // streaming-friendly
    Ignore,         // allow missing-aware costs to skip NaNs where supported
    // later: linear, mean-per-dimension, drop (but beware index shifting)
}

pub enum TimeIndex<'a> {
    None,
    Uniform { t0_ns: i64, dt_ns: i64 },
    Explicit(&'a [i64]), // unix nanos, length = n
}

pub enum DTypeView<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

pub enum MemoryLayout {
    CContiguous,
    FContiguous,
    Strided { row_stride: isize, col_stride: isize },
}

pub struct TimeSeriesView<'a> {
    pub values: DTypeView<'a>,
    pub n: usize,
    pub d: usize,            // d=1 for univariate
    pub layout: MemoryLayout,
    pub missing_mask: Option<&'a [u8]>, // 1 = missing, len = n*d
    pub time: TimeIndex<'a>, // optional; used for reporting + online hazard when needed
    pub missing: MissingPolicy,
}

```
### Python interop rules

- Accept NumPy arrays (1D or 2D).
- Prefer zero-copy for contiguous arrays; otherwise copy with a diagnostic note.
- Validate:
  - dtype policy:
    - keep float32 zero-copy when detector/cost supports it
    - upcast to float64 only when required by model internals, and record `diagnostics.copy_reason`
  - shape ((n,) or (n, d))
  - layout/strides (C/F contiguous fast path; strided fallback may copy)
  - missing values policy (NaN default error unless configured otherwise)
  - optional time index:
    - accept `datetime64[ns]` or `int64` nanoseconds arrays (length = n) for reporting + online hazard functions
    - default to sample index (0..n-1) when not provided
  - Use the numpy crate (PyO3 ↔ NumPy bridge) which is built on PyO3 and ndarray.

### Missingness semantics (explicit, per-cost)

- Each cost model declares missing-data capability:
  - `Reject` (default)
  - `MaskAware`
  - `NaNIgnoreLossy`
- If requested `MissingPolicy` is unsupported by the selected cost model, fail validation instead of silently changing behavior.
- Diagnostics always report `missing_policy_applied`, `missing_fraction`, and `effective_sample_count`.

## Constraints and Safety Rails

Applies across algorithms (offline and online where relevant):
- min_segment_len: enforce everywhere to prevent degenerate segments / noise fitting.
- max_change_points / max_depth: bound recursion and runtime (especially BinSeg/WBS).
- candidate_splits / jump: restrict breakpoint candidates (huge speedup; aligns with ruptures’ jump for PELT). The ruptures docs explicitly support min_size and jump for PELT.
- time/compute budgets (optional but recommended in services):
  - time_budget_ms: stop early with a `ResourceLimit` error if exceeded
  - max_cost_evals: cap segment-cost evaluations to avoid pathological runtimes
- memory budgets (optional but recommended in services):
  - memory_budget_bytes: cap total memory; algorithm may return ResourceLimit or switch to compact caches
  - max_cache_bytes: cap cost-model cache size (avoid OOM on huge n)
- explicit cache policy:
  - full cache (max performance, max memory)
  - budgeted cache (bounded memory with eviction/partial materialization)
  - approximate cache (bounded memory with known error controls where supported)
- Cancellation + progress:
  - cancellation token (Arc<AtomicBool>) checked inside main loops
  - optional progress reporting (with careful Python callback handling to avoid reacquiring the GIL too often)

### Sketch

```rust
pub enum CachePolicy {
    Full,
    Budgeted { max_bytes: usize },
    Approximate { max_bytes: usize, error_tolerance: f64 },
}

pub struct Constraints {
    pub min_segment_len: usize,
    pub max_change_points: Option<usize>,
    pub max_depth: Option<usize>,
    pub candidate_splits: Option<Vec<usize>>, // sorted indices
    pub jump: usize,                          // convenience; candidate = multiples of jump
    pub time_budget_ms: Option<u64>,
    pub max_cost_evals: Option<usize>,
    pub memory_budget_bytes: Option<usize>,
    pub max_cache_bytes: Option<usize>,
    pub cache_policy: CachePolicy,
    pub degradation_plan: Vec<DegradationStep>,
    pub allow_algorithm_fallback: bool,
}

pub enum DegradationStep {
    IncreaseJump { factor: usize, max_jump: usize },
    DisableUncertaintyBands,
    SwitchCachePolicy(CachePolicy),
}

pub struct CancelToken(pub std::sync::Arc<std::sync::atomic::AtomicBool>);

```

### Constraint validation and canonicalization (required preflight)

- Validate once in a shared module before detector execution:
  - `jump >= 1`
  - `min_segment_len >= 1`
  - provided budgets are positive
  - `candidate_splits` are sorted, unique, and in-range
- Canonicalize effective candidates after applying `jump` + `min_segment_len`, so all detectors receive identical effective constraints.
- Emit structured validation errors with field names/values (improves Python and CLI UX).

## Cost Models: FirstClass, Cached, Extensible

Most CPD performance comes from fast segment cost computation. We make it explicit and reusable.

### CostModel trait + cache

- Expose a `CachedCost` wrapper so callers can precompute once and reuse across multiple runs/penalties.
- Surface cache size estimates so constraints and the Doctor can enforce memory budgets.

```rust
pub enum MissingSupport {
    Reject,
    MaskAware,
    NaNIgnoreLossy,
}

pub trait CostModel {
    type Cache: Send + Sync;

    fn name(&self) -> &'static str;

    fn validate(&self, x: &TimeSeriesView) -> Result<(), CpdError>;
    fn missing_support(&self) -> MissingSupport {
        MissingSupport::Reject
    }

    fn precompute(
        &self,
        x: &TimeSeriesView,
        policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError>;
    fn worst_case_cache_bytes(&self, x: &TimeSeriesView) -> usize;
    fn supports_approx_cache(&self) -> bool {
        false
    }

    /// Cost of segment [start, end) (end exclusive).
    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64;

    /// Optional fast-path for bulk queries (reduces call overhead + enables SIMD).
    fn segment_cost_batch(
        &self,
        cache: &Self::Cache,
        queries: &[(usize, usize)],
        out_costs: &mut [f64],
    ) {
        for (i, (start, end)) in queries.iter().enumerate() {
            out_costs[i] = self.segment_cost(cache, *start, *end);
        }
    }
}

```
### Built-in costs (roadmapped)

#### MVP

- CostL2Mean (piecewise constant mean, least squares)
- CostNormalMeanVar (Gaussian with MLE mean+variance; useful for volatility/regime shifts)
- CostNIGMarginal (conjugate Normal-Inverse-Gamma marginal likelihood; O(1) via sufficient stats; more robust than pure Gaussian MLE)

#### v1

- CostPoissonRate (count/rate changes; logs/traffic/telemetry)
- CostBernoulli (binary events; conversion/failure rates)
- CostLinear (piecewise linear trend; good for drifting sensors)
- CostAR (autoregressive residual cost; helps autocorrelated series; aligns with ruptures cost family)
- CostL1Median (robust but not O(1) without approximations; explicitly treated as “slow path”)
- CostRank / CostCosine (optional, depending on demand)

#### Experimental

- CostRbfKernel (distributional changes; default via approximations)
- “Student-t likelihood” variants (offline) if you want robust likelihood segmentation, but this can be expensive; consider as opt-in.

### Penalty & stopping helpers

- Penalty::BIC, Penalty::AIC, Penalty::Manual(f64)
- Stopping::KnownK(usize) vs Stopping::Penalized(Penalty)
- Stopping::PenaltyPath(Vec<Penalty>) (optional): compute a solution path for a penalty sweep in one run (PELT/FPOP)
- enforce constraints consistently (one shared enforcement module)

## Offline Algorithms: Roadmap / Maturity Levels

We implement a “small core” first, then expand.

### MVP-A/B (v0.1-v0.2, production-ready offline core)

- PELT (Pruned Exact Linear Time)
  - Penalized optimal partitioning with pruning; average linear time under conditions.
  - Supports `min_segment_len`, candidate thinning (`jump`/`candidate_splits`), and an optional `max_change_points`.
  - Includes a multi-resolution mode for huge n: run coarse (larger `jump`), then locally refine around candidate breakpoints.
  - Optional penalty-sweep mode to return multiple segmentations + costs without rerunning.
  - Primary engine for large offline series (finance/sensors/telemetry).
- Binary Segmentation (BinSeg)
  - Fast recursive splitting; good when changes are few and strong.
  - Supports `max_depth`, `max_change_points`, candidate splits, and deterministic tie-breaking.
- Wild Binary Segmentation (WBS)
  - Mitigates masking issues in vanilla BinSeg; handles multiple change points and short spacings.
  - Deterministic seeding and reproducibility; supports interval strategies (random, deterministic grid, stratified).
  - Planned in MVP-B after baseline PELT/BinSeg hardening.
- Shared post-processing utilities
  - Optional merge of near-duplicate breakpoints, enforce `min_segment_len`, and compute local uncertainty bands via re-scoring around each breakpoint.

### v1.0 additions (production-ready breadth)

- Dynp (Dynamic programming / optimal partitioning)
  - Exact but heavy; recommended for smaller n or validation.
- BottomUp segmentation
  - Merge-based; can be good for many small changes.
- Sliding Window
  - Local comparison; good for frequent short changes and streaming-adjacent workflows.
- FPOP (Functional Pruning Optimal Partitioning)
  - Strong penalized segmentation alternative for common L2 mean settings; high leverage when assumptions apply.
- Segment Neighborhood (SegNeigh) (optional)
  - Useful for known K, and as a refinement step around an initial segmentation.

### Experimental / heavy (featureflagged)

- Kernel CPD
- Make exact O(n²) mode opt-in.
- Default: scalable approximations (Nyström/random features) under kernel-approx.
- GP / ARGPCPstyle methods
- Extremely powerful but expensive and complex; implement only after core is stable.

## Online Algorithms: Production-Safe Streaming

### Bayesian Online Change Point Detection (BOCPD)

BOCPD maintains a runlength posterior and yields p(change at t) online.

### Implementation details for robustness

- Logspace probabilities + logsumexp everywhere (prevents underflow).
- Hazard function interface:
  - constant hazard
  - geometric hazard
  - parametrized hazard families
- Checkpoint/restore:
  - online detectors must expose a state object (behind `serde` feature for JSON/bincode)
  - makes long-running services restartable and debuggable
  - persist with a typed envelope: `detector_id`, `state_schema_version`, `engine_fingerprint`, `created_at_ns`, `payload_crc32`, `payload_codec`
  - require atomic persistence pattern (tmp + fsync + rename) for crash safety
- Event-time semantics:
  - `update(..., t_ns=...)` accepts optional event timestamps
  - configurable late-data policy:
    - Reject
    - BufferWithinWindow { max_delay_ns, max_buffer_items, on_overflow }
    - ReorderByTimestamp { max_delay_ns, max_buffer_items, on_overflow }
  - watermark tracked in state to guarantee deterministic replay behavior
  - overflow policy is explicit: DropOldest | DropNewest | Error
- Observation models (prioritized by usefulness + tractability):
  - Gaussian (unknown mean/variance, conjugate priors)
  - Poisson (rate changes)
  - Bernoulli (probability changes)
- Irregular sampling support:
  - hazard functions may optionally consume elapsed time (`TimeIndex`) rather than assuming uniform dt
- Truncation / pruning:
  - max_run_length
  - prune runlength states below a logprob threshold
  - Optional fixedlag smoothing for better retrospective estimates in streaming settings (bounded window).

### Outputs

- per update step: p_change, run-length mode/mean, alert flag, alert_reason, processing latency, and late/reorder counters.

### Alerting policy (ops-ready)

- AlertPolicy with threshold, hysteresis, cooldown, and min_run_length.
- Avoids flapping and makes alert semantics consistent across detectors.
- AlertPolicy is serialized with detector state so behavior is stable across restart.

### Baseline streaming detectors (optional but pragmatic)

These are often “good enough” for alerting and are cheaper/simpler than BOCPD:

- CUSUM (mean shift with drift/threshold parameters)
- Page-Hinkley (change detection for mean with robustness to gradual drift)
- EWMA/Shewhart-style detectors (fast and easy to reason about)

### Existing Rust crates (reference / possible reuse)

There is an existing Rust crate called changepoint that provides CPD tools including BOCPD and an autoregressive Gaussian CPD variant.
Recommendation: treat this as a reference implementation, not a hard dependency in the core path, unless you explicitly want to vendor/lock its behavior. For production libraries, keeping algorithm implementations inhouse usually makes correctness auditing, feature evolution, and numerical guarantees easier.

## Python Bindings: True NumPy Interop, GIL Release, Typed API, Streaming Interface

### Tooling

- PyO3 for Rust↔Python bindings.
- maturin for building and publishing wheels across platforms.
- numpy crate for safe NumPy CAPI bridging.

### Python API shape

Provide two layers:

#### High-level (ruptures-like)

- Pelt(model="l2").fit(x).predict(pen=...)
- Binseg(model="l2").fit(x).predict(n_bkps=...)
- Bocpd(model="gaussian_nig", hazard=...).update(x_t)  # streaming-friendly

#### Low-level (power users / Doctor integration)

- detect_offline(x, detector=..., cost=..., preprocess=..., constraints=..., stopping=..., return_diagnostics=True)
- returns a structured OfflineChangePointResult object
- online detectors expose checkpoint APIs:
  - state = detector.save_state()
  - detector.load_state(state)
  - detector.update(x_t, t_ns=...) for event-time aware streaming use-cases

### Result object in Python

- Return dataclass-like objects:
  - .breakpoints (list of ints, includes n by default)
  - .change_points (excludes n)
  - .scores, .diagnostics, .segments (offline)
  - .schema_version, .engine_version (for serialized compatibility)
  - OnlineStepResult: .t, .p_change, .alert, .run_length_mode, .run_length_mean (online)
  - Provide convenience: optional .plot() helper and .to_json()/.from_json() (when `serde` is enabled)

### Zerocopy + dtype handling

- Accept numpy.ndarray (float32/float64).
- Prefer zero-copy on contiguous arrays; copy otherwise (and record a diagnostic note).
- Preserve float32 when supported; avoid unconditional upcast to float64.
- Validate shape and enforce consistent memory layout expectations.

### Releasing the GIL

All heavy compute runs under py.allow_threads(|| ...), so Python remains responsive.

### Sketch

```rust
#[pymethods]
impl PyPelt {
    fn detect(&self, py: Python, x: &PyAny) -> PyResult<PyOfflineChangePointResult> {
        py.allow_threads(|| {
            let view = parse_numpy_to_view(x)?;  // zero-copy when possible
            let ctx = self.exec_context_for_python_call();
            let result = self.detector.detect(&view, &ctx)?;
            Ok(PyOfflineChangePointResult::from(result))
        })
    }
}

```
### Streaming-friendly Python interface

Online detectors are stateful Python objects:
- .update(x_t) where x_t can be scalar (univariate) or 1D array (multivariate at time t)
- .update(x_t, t_ns=...) to attach event time (optional)
- .update_many(x) for batch ingestion (reduces Python overhead)
- .reset()
- .save_state() / .load_state(state) for checkpoint/restore and debugging
- optional alert_policy=... or .set_alert_policy(...) to enable consistent alerting semantics

### Type hints + stability

- Ship py.typed and stub files or runtime type hints.
- Stable, documented config/result dataclasses.

## “Change Point Doctor”: Ranked Recommendations + Configs + Explanations + Optional Validation

### What it does

The Doctor returns a ranked set of recommended pipelines:
- detector (algorithm)
- cost model
- penalty/stopping rule
- preprocessing (optional)
- constraints (min_segment_len, candidates/jump)
- resource estimate + warnings (complexity, memory, stability)
- explanation (what diagnostics drove the recommendation)
This is explicitly aligned with how CPD methods are typically composed (cost + search + constraint).

### Doctor API (proposed)

- doctor.recommend(
    x,
    objective="balanced",
    online=False,
    constraints=...,
    min_confidence=0.6,
    allow_abstain=True,
  ) -> List[Recommendation]
- doctor.validate_top_k(x, k=3, downsample=..., seed=...) -> ValidationReport (optional)

### Recommendation object (sketch)

```rust
pub struct Recommendation {
    pub pipeline: PipelineConfig,         // typed config, easy to apply
    pub resource_estimate: ResourceEstimate,
    pub warnings: Vec<String>,
    pub explanation: Explanation,
    pub validation: Option<ValidationSummary>,
    pub confidence: f64,
    pub confidence_interval: (f64, f64),
    pub abstain_reason: Option<String>,
    pub objective_fit: Vec<(String, f64)>,
}

pub struct PipelineConfig {
    pub detector: DetectorConfig,
    pub cost: CostConfig,
    pub preprocess: Option<PreprocessPipeline>,
    pub constraints: Constraints,
    pub stopping: Stopping,
    pub seed: Option<u64>,
}

pub struct ResourceEstimate {
    pub time_class: String,   // e.g. "O(n)" / "O(n log n)" / "depends on candidates"
    pub memory_class: String, // e.g. "O(n)" / "O(W)" for windowed online
    pub notes: Vec<String>,
}

pub struct Explanation {
    pub signals: Vec<(String, f64)>,  // e.g. ("autocorr_lag1", 0.72)
    pub narrative: String,
}

pub struct ValidationSummary {
    pub stability_score: f64,
    pub agreement_score: f64,
    pub calibration_score: Option<f64>,
    pub notes: Vec<String>,
}

```
### Confidence semantics and calibration (required for Doctor v1)

- Define `confidence` as estimated probability that top-1 recommendation is within breakpoint tolerance for the stated objective.
- Calibrate confidence on held-out corpora and publish ECE/Brier metrics by data family.
- Add OOD gating: when diagnostics are outside calibration support, reduce confidence or abstain with an explicit reason.

### Diagnostics the Doctor computes

(Computed quickly; O(n) or O(n log n) max; supports sampling for n huge.)
- basic: n, d, sampling rate if provided
- missingness: NaN rate, longest NaN run
- noise/robustness: kurtosis proxy, outlier rate, MAD/STD ratio
- seasonality/proxies: dominant period hints, residual autocorr after detrending (cheap heuristics)
- heteroskedasticity: rolling variance drift, regime-change likelihood proxies
- autocorrelation: lag1/lagk autocorr, partial autocorr proxy
- stationarity proxies: rolling mean/var drift
- change “density” proxy: windowed divergence scores on a subsample
- user constraints: expected number of changes (if provided), latency requirements (online vs offline)

### Heuristic mapping (v1)

- Huge n (≥1e5) + offline: recommend PELT with cached L2/Normal costs, plus candidate thinning (jump) and sensible min_segment_len.
- Few strong changes: BinSeg or WBS (prefer WBS when masking risk detected).
- Many changes: PELT or BottomUp; Window for very local/frequent changes.
- Autocorrelated / trending sensors: prefer CostAR or CostLinear with PELT/BinSeg; treat GP methods as experimental.
- Heavy tails / outliers (finance): prefer robust-ish conjugate Bayesian costs (e.g., NIG marginal) or explicitly “slow path” robust costs; avoid fragile Gaussian-only assumptions.
- if diagnostics are low-signal/conflicting: return abstain recommendation with bounded "safe baseline" options.

### “Trust but verify” mode

validate_top_k runs a quick pass:
- potentially downsample or use candidate thinning
- run top 2–3 pipelines
- measure consensus/stability (e.g., overlap of breakpoints within tolerance, sensitivity to penalty)
- when available, use penalty-sweep paths to score stability across penalties
- returns a report that increases user trust and helps tune parameters
- includes confidence calibration summary (agreement vs held-out synthetic regimes) so ranking confidence is interpretable.

## Performance Plan: Benchmarking + Profiling + Budgets (No Speculative Claims)

Performance is treated as a deliverable.

### Benchmark suite

- sizes: 1e4, 1e5, 1e6 (and at least one “stress” size: 5e6 or 1e7 for fast-path PELT)
- cases: univariate + multivariate (d=8/16), piecewise constant/linear, heavy tails, autocorrelated, count/binary (Poisson/Bernoulli)
- algorithms: PELT, BinSeg/WBS, BOCPD (+ baseline streaming detectors)
- report:
  - runtime + peak RSS + allocations (where possible)
  - online throughput/latency (updates/sec, p99 update time) under bounded `max_run_length`
  - determinism checks by reproducibility mode (Strict/Balanced/Fast)

### Profiling

- identify hotspots:
  - segment cost computation
  - allocations (DP arrays, candidate lists)
  - kernel feature computation (if enabled)
- optimize based on evidence:
  - cache reuse
  - allocation reuse / bump allocators in tight loops (optional)
  - candidate thinning
  - selective parallelism (WBS interval scoring is a good target)
  - thread control: document how to bound threads (rayon pool sizing + Python integration)

### Performance budgets

Establish two benchmark gates:
- Absolute gate on a documented reference machine (product-level expectations).
- Relative gate in CI against rolling baseline on the same runner type.
- PELT + L2, n=1e6, d=1, jump=5, min_segment_len=20: <= 3.0s and <= 1.2 GB RSS
- BinSeg + L2, n=1e6, d=1: <= 1.5s and <= 700 MB RSS
- BOCPD (Gaussian/NIG), d=1, max_run_length=2000: p99 update <= 75 us and >= 150k updates/sec
- Balanced mode reproducibility: identical breakpoint sets for fixed seed/config/thread count on same target triple; score tolerance documented per cost model.
- Strict mode reproducibility: bitwise-identical outputs on same machine/toolchain/CPU features; cross-platform gate uses segmentation agreement thresholds.

### Regression prevention

- run microbenchmarks in CI and nightly to detect slowdowns.
- keep a low-noise benchmark mode (fixed seeds, pinned configs) so regressions are attributable.
- maintain “fast path” as a guarded invariant.
- fail CI if runtime regresses >10% or RSS regresses >15% versus rolling baseline for protected benchmark set.

## Quality, Testing, and Reliability Plan

### Testing strategy

- Unit tests per cost model:
  - validate O(1) cost vs naive recomputation on random segments
  - edge cases: constant series, small n, min_segment_len boundaries
- Unit tests per algorithm:
  - invariants and known small examples
- Propertybased tests (e.g., proptest):
  - breakpoints sorted, unique
  - all indices in bounds
  - respects min_segment_len
  - stable under constant series (no spurious cps given reasonable penalties)
- Metamorphic tests:
  - invariance under shift/scale transformations where theoretically expected
  - multivariate column permutation invariance (when model is symmetric)
- Numerical stress tests:
  - extreme magnitudes, denormals, near-zero variance, and NaN/Inf contamination paths
- Differential tests:
  - compare against ruptures / reference outputs on a curated corpus
  - include heavy tails, outliers, trends, missing values
- Fuzzing:
  - ensure no panics / UB from malformed inputs, especially across the FFI boundary
- Concurrency + soak:
  - stress tests with concurrent detector instances and cancellation/budget interrupts
  - long-run soak tests (>=24h) for online detectors with periodic checkpoint/restore
- Reproducibility:
  - deterministic behavior for randomized components (WBS interval sampling) with explicit seeds
- State + serialization:
  - roundtrip tests for (config, results, online detector state) when `serde` is enabled
  - Python↔Rust equivalence tests for JSON schemas (CLI/service compatibility)
- Fault-injection tests:
  - corrupted checkpoint payloads, truncated files, and partial writes fail safely with actionable errors
- Resource budget tests:
  - ensure `time_budget_ms` / `max_cost_evals` fail fast with a `ResourceLimit` error (not partial garbage)
- Release smoke tests:
  - wheel install/import tests on clean Linux/macOS/Windows runners
  - API parity tests across supported Python versions

## Security, Supply Chain, and Release Engineering

- Dependency governance:
  - `cargo audit` + `cargo deny` in CI
  - Python dependency scanning and pinned lower-bound tests
- Artifact provenance:
  - signed release artifacts
  - SBOM (CycloneDX/SPDX) for crates and wheels
- Vulnerability response:
  - triage SLA within 48h for high/critical issues
  - patch release process with backport guidance for supported versions
- Release checklist gates:
  - schema migration fixtures pass
  - benchmark SLO gates pass
  - wheel smoke tests pass
  - docs/examples CI pass

## Licensing, Attribution, and “Clean Room” Implementation Practices

### Upstream licenses (for reference behavior and attribution)

- hildensia/bayesian_changepoint_detection is marked as MIT licensed on GitHub.
- deepcharles/ruptures is under BSD 2Clause.

### Policy

- Prefer clean-room reimplementation from the underlying papers and public algorithm descriptions.
- Do not copy/paste code from upstream repos unless you fully comply with license terms and preserve notices.
- Include:
  - NOTICE / ATTRIBUTION file listing upstream inspirations
  - CITATION.cff with recommended citations (e.g., BOCPD, PELT, WBS, Truong et al survey)
- Pick a license for your project that is friendly to both Rust and Python ecosystems:
  - common Rust pattern: dual MIT/Apache-2.0
  - either is compatible with BSD2 and MIT upstream inspirations

## SemVer, Deprecation Policy, and Result Conventions

### Versioning & compatibility

- Follow SemVer:
  - v0.x can iterate faster but still avoid gratuitous breaks
  - v1.0: freeze public Python API (config/result objects) and Rust core traits
- Deprecation:
  - deprecate in one minor release, remove in the next major (Python users care a lot)

### Schema versioning

- Version config/result JSON with an explicit schema_version.
- Guarantee N-1 read compatibility in Rust and Python, and document migrations.
- Publish JSON Schema for config/result payloads and include a schema hash in release notes.
- Unknown-field policy:
  - N and N-1 readers should ignore unknown fields by default
  - preserve unknown fields where possible during roundtrip transforms

### Checkpoint compatibility policy

- Online checkpoints follow the same N-1 read-compatibility policy as configs/results.
- On mismatch or checksum failure, fail fast with actionable `CpdError` variants.
- Maintain checkpoint migration fixtures in CI.

### Result conventions (documented and enforced)

- 0based indices
- Segment boundaries use halfopen intervals: [start, end)
- Offline algorithms return:
  - breakpoints: sorted segment end indices including n by default (ruptures convention)
  - change_points: derived as breakpoints excluding n
  - Always specify whether 0 is included (it usually is not; segments start at 0 implicitly)
- Provide utilities:
  - segments_from_breakpoints(n, breakpoints) -> Vec<(start,end)>
  - validate_breakpoints(...) -> Result<(), CpdError>

### Cross-language serialization

- With serde feature, allow:
  - config serialization (for reproducible runs)
  - result JSON (for CLI/services/debug)
  - schema_version + engine_version fields, plus N-1 migration shims
- Maintain golden fixtures per schema version and enforce migration tests in CI.

### PipelineSpec (cross-language, first-class artifact)

- Introduce a typed `PipelineSpec` consumed by Rust API, Python API, CLI, and Doctor output.
- Doctor recommendations must be directly executable without manual field mapping.
- Add CLI execution path: `cpd run --pipeline spec.json --input data.npy` for reproducible runs across environments.

## Implementation Roadmap: Concrete Work Breakdown

### MVP-A (v0.1)

- Core abstractions (cpd-core)
  - TimeSeriesView parsing/validation with dtype + layout aware views
  - Constraints + CachePolicy + unified ExecutionContext
  - OfflineChangePointResult, OnlineStepResult, Diagnostics, CpdError
- Cost models (cpd-costs)
  - L2 mean (prefix sums/squares)
  - Normal mean+var (sufficient stats)
- Offline algorithms (cpd-offline)
  - PELT (generic over CostModel) + deterministic tie-breaking
  - BinSeg (generic over cost; supports constraints)
- Python bindings (cpd-python)
  - maturin project + wheels + py.typed
  - NumPy interop (zero-copy when contiguous) + GIL release
- Quality gates
  - unit + property tests + serialization fixtures
  - benchmark SLO baselines established and versioned

### MVP-B (v0.2)

- Robustness and breadth
  - NIG marginal likelihood cost (conjugate, O(1), robust-ish)
  - WBS (seeded; interval strategies; supports constraints)
  - multivariate polish across L2/Normal/NIG
- Evaluation + parity
  - differential tests against ruptures on curated corpus
  - docs with reproducibility modes and deterministic behavior contracts
- Release hardening
  - wheel smoke tests for Linux/macOS/Windows
  - schema migration test fixtures in CI

### MVP-C (v0.3)

- Online algorithms (cpd-online)
  - BOCPD (log-space, hazard interface, truncation/pruning)
  - event-time + late-data policy + checkpointable state
  - one baseline detector (CUSUM or Page-Hinkley) for cheap alerting
- Doctor beta (cpd-doctor)
  - ranked recommendation with confidence + abstain mode
  - validate_top_k with stability/agreement/calibration summary
- Evaluation utilities (cpd-eval)
  - offline + online metrics (F1, delay, false-alarm rate)
  - synthetic data generators + dataset registry (with license metadata)

### v1.0

- multivariate polish across costs/algorithms
- additional costs: Poisson, Bernoulli, CostLinear, CostAR, explicit slow-path robust costs
- Dynp / BottomUp / Window / FPOP / SegNeigh
- penalty helpers and consistent parameter naming with ruptures where possible
- Doctor v1 (typed pipelines, ranked recommendations, explanations, validation summaries)
- CLI (optional) and serde JSON outputs (results + configs + online state)

### v1.x+

- kernel approximations + opt-in exact kernel mode
- advanced pruning/path variants where assumptions apply
- GP/ARGP feature-flagged research module
- ensembles as optional doctor recommendation when uncertainty is high

## Notes on Naming and Packaging

- Choose a PyPI name that is short and unique (e.g., cpd-rs, rust-cpd, etc.).
- Provide cpd.doctor() and cpd.Pelt, cpd.Binseg, cpd.Bocpd for immediate familiarity.
- Mirror ruptures naming and conventions when sensible, but keep the richer result types as a differentiator.

## What You’ll Have at the End (Deliverable Summary)

By the end of MVP-C, you’ll have:
- A Rust core with stable APIs, cache policies, unified execution context, structured results, and robust errors
- Fast offline segmentation (PELT/BinSeg/WBS) with reproducibility modes and deterministic defaults
- Production-safe online detection (BOCPD + baseline detector) with event-time handling and checkpoint/restore
- A Python package on PyPI with NumPy-native interop, released GIL, and typed/stable results/configs
- A Doctor beta that ranks pipelines with explanations, confidence scoring, and abstain behavior
- Benchmarks + evaluation utilities + differential tests + release/security gates, with a clear path to expand without destabilizing the core
