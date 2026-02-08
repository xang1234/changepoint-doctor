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

### Phased roadmap

- MVP (v0.x):
  - TimeSeriesView + constraints + cost caching infrastructure
  - Costs: L2 mean shift, Normal mean+var, one robust-ish O(1) option via conjugate marginal likelihood (Normal-Inverse-Gamma / Student-t predictive)
  - Algorithms: PELT, Binary Segmentation, WBS (optional but high ROI), BOCPD
  - Python wheels + docs + tests + benchmark harness + metrics/evaluation utilities
- v1.0 (“production-ready” breadth):
  - Offline: Dynp, BottomUp, Sliding Window
  - Penalty helpers (BIC/AIC-like, user-specified, “known K” constraints)
  - Stable result objects (scores/diagnostics) + multivariate support for common costs
  - Doctor v1 (ranked recommendations + explanations + safe configs)
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
- Deterministic by default:
  - seeded randomness for any stochastic algorithm (WBS interval sampling, randomized features, etc.)
  - stable result conventions and explicit tie-breaking rules where relevant
- Operational reliability:
  - cancellation + optional time/compute budgets
  - online detector checkpoint/restore to support long-running services
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
    pub params_json: Option<serde_json::Value>, // behind `serde` feature
    pub pruning_stats: Option<PruningStats>,
}

#[derive(Clone, Debug)]
pub struct PruningStats {
    pub candidates_considered: usize,
    pub candidates_pruned: usize,
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

// Offline algorithms: full series -> Result
pub trait OfflineDetector {
    fn detect(&self, x: &TimeSeriesView) -> Result<OfflineChangePointResult, CpdError>;
}

// Online algorithms: stateful incremental update.
// Keep per-step outputs small; if callers want history, they can store it.
pub struct OnlineStepResult {
    pub t: usize,
    pub p_change: f64,
    pub alert: bool,              // derived from alert policy
    pub run_length_mode: usize,
    pub run_length_mean: f64,
}

pub trait OnlineDetector {
    type State: Clone + std::fmt::Debug;
    fn reset(&mut self);
    fn update(&mut self, x_t: &[f64]) -> Result<OnlineStepResult, CpdError>;
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

pub struct TimeSeriesView<'a> {
    pub values: &'a [f64],   // flattened row-major (C-order): [t0_d0, t0_d1, ...]
    pub n: usize,
    pub d: usize,            // d=1 for univariate
    pub stride: usize,       // default stride = d for row-major; supports future strided views if needed
    pub time: TimeIndex<'a>, // optional; used for reporting + online hazard when needed
    pub missing: MissingPolicy,
}

```
### Python interop rules

- Accept NumPy arrays (1D or 2D).
- Prefer zero-copy for contiguous arrays; otherwise copy with a diagnostic note.
- Validate:
  - dtype (float64 preferred; accept float32 but usually upcast to float64 for core algorithms)
  - shape ((n,) or (n, d))
  - missing values policy (NaN default error unless configured otherwise)
  - optional time index:
    - accept `datetime64[ns]` or `int64` nanoseconds arrays (length = n) for reporting + online hazard functions
    - default to sample index (0..n-1) when not provided
  - Use the numpy crate (PyO3 ↔ NumPy bridge) which is built on PyO3 and ndarray.

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
- Cancellation + progress:
  - cancellation token (Arc<AtomicBool>) checked inside main loops
  - optional progress reporting (with careful Python callback handling to avoid reacquiring the GIL too often)

### Sketch

```rust
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
}

pub struct CancelToken(pub std::sync::Arc<std::sync::atomic::AtomicBool>);

```
## Cost Models: FirstClass, Cached, Extensible

Most CPD performance comes from fast segment cost computation. We make it explicit and reusable.

### CostModel trait + cache

- Expose a `CachedCost` wrapper so callers can precompute once and reuse across multiple runs/penalties.
- Surface cache size estimates so constraints and the Doctor can enforce memory budgets.

```rust
pub trait CostModel {
    type Cache: Send + Sync;

    fn name(&self) -> &'static str;

    fn validate(&self, x: &TimeSeriesView) -> Result<(), CpdError>;

    fn precompute(&self, x: &TimeSeriesView) -> Result<Self::Cache, CpdError>;
    fn cache_memory_bytes(&self, x: &TimeSeriesView) -> usize;

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

### MVP (v0.x, production-ready)

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

- per update step: p_change, run-length mode/mean, alert flag (if policy configured).

### Alerting policy (ops-ready)

- AlertPolicy with threshold, hysteresis, cooldown, and min_run_length.
- Avoids flapping and makes alert semantics consistent across detectors.

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
            let result = self.detector.detect(&view)?;
            Ok(PyOfflineChangePointResult::from(result))
        })
    }
}

```
### Streaming-friendly Python interface

Online detectors are stateful Python objects:
- .update(x_t) where x_t can be scalar (univariate) or 1D array (multivariate at time t)
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

- doctor.recommend(x, objective="balanced", online=False, constraints=...) -> List[Recommendation]
- doctor.validate_top_k(x, k=3, downsample=..., seed=...) -> ValidationReport (optional)

### Recommendation object (sketch)

```rust
pub struct Recommendation {
    pub pipeline: PipelineConfig,         // typed config, easy to apply
    pub resource_estimate: ResourceEstimate,
    pub warnings: Vec<String>,
    pub explanation: Explanation,
    pub validation: Option<ValidationSummary>,
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
    pub notes: Vec<String>,
}

```
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

### “Trust but verify” mode

validate_top_k runs a quick pass:
- potentially downsample or use candidate thinning
- run top 2–3 pipelines
- measure consensus/stability (e.g., overlap of breakpoints within tolerance, sensitivity to penalty)
- when available, use penalty-sweep paths to score stability across penalties
- returns a report that increases user trust and helps tune parameters

## Performance Plan: Benchmarking + Profiling + Budgets (No Speculative Claims)

Performance is treated as a deliverable.

### Benchmark suite

- sizes: 1e4, 1e5, 1e6 (and at least one “stress” size: 5e6 or 1e7 for fast-path PELT)
- cases: univariate + multivariate (d=8/16), piecewise constant/linear, heavy tails, autocorrelated, count/binary (Poisson/Bernoulli)
- algorithms: PELT, BinSeg/WBS, BOCPD (+ baseline streaming detectors)
- report:
  - runtime + peak RSS + allocations (where possible)
  - online throughput/latency (updates/sec, p99 update time) under bounded `max_run_length`
  - determinism checks (same seed/config => identical outputs)

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

Establish target envelopes on a reference machine, e.g.:
- PELT + L2 on 1e6 points should run within X seconds and Y GB with jump and min_segment_len defaults.
(Exact numbers depend on your target hardware; the key is to define and publish them.)

### Regression prevention

- run microbenchmarks in CI (or nightly) to detect slowdowns.
- keep a low-noise benchmark mode (fixed seeds, pinned configs) so regressions are attributable.
- maintain “fast path” as a guarded invariant.

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
- Differential tests:
  - compare against ruptures / reference outputs on a curated corpus
  - include heavy tails, outliers, trends, missing values
- Fuzzing:
  - ensure no panics / UB from malformed inputs, especially across the FFI boundary
- Reproducibility:
  - deterministic behavior for randomized components (WBS interval sampling) with explicit seeds
- State + serialization:
  - roundtrip tests for (config, results, online detector state) when `serde` is enabled
  - Python↔Rust equivalence tests for JSON schemas (CLI/service compatibility)
- Resource budget tests:
  - ensure `time_budget_ms` / `max_cost_evals` fail fast with a `ResourceLimit` error (not partial garbage)

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

## Implementation Roadmap: Concrete Work Breakdown

### MVP (v0.x)

- Core abstractions (cpd-core)
  - TimeSeriesView + parsing/validation (including optional TimeIndex)
  - Constraints, Penalty, Stopping, cancellation/progress wiring, time/compute/memory budgets
  - CostModel trait + caches (+ optional batch cost API)
  - OfflineChangePointResult, OnlineStepResult, Diagnostics, CpdError
- Cost models (cpd-costs)
  - L2 mean (prefix sums/squares)
  - Normal mean+var (sufficient stats)
  - NIG marginal likelihood (conjugate, O(1) segment scores, more robust than pure MLE)
- Offline algorithms (cpd-offline)
  - PELT (generic over CostModel) + multi-resolution refinement for huge n
  - BinSeg (generic over cost; supports constraints)
  - WBS (seeded; interval sampling; supports cost)
  - shared post-processing utilities (merge + local uncertainty scoring)
- Online algorithms (cpd-online)
  - BOCPD (log-space math, hazard interface, truncation/pruning, checkpointable state)
  - one baseline detector (CUSUM or Page-Hinkley) to cover “cheap alerting” use-cases
- Python bindings (cpd-python)
  - maturin project + wheels
  - NumPy interop (zero-copy when contiguous) + GIL release
  - typed result/config objects + checkpoint APIs (`save_state`/`load_state`)
  - docs + minimal examples (offline + online)
- Evaluation utilities (cpd-eval)
  - synthetic data generators + dataset registry (with license metadata)
  - offline + online metrics (F1, delay, false-alarm rate) for docs + doctor validation
- Quality & benchmarks
  - unit + property tests + differential tests against ruptures where comparable
  - criterion benchmarks for fast path

### v1.0

- multivariate polish across costs/algorithms
- additional costs: Poisson, Bernoulli, CostLinear, CostAR, explicit slow-path robust costs
- Dynp / BottomUp / Window / FPOP / SegNeigh
- penalty helpers and consistent parameter naming with ruptures where possible
- Doctor v1 (typed pipelines, ranked recommendations, explanations, validation summaries)
- CLI (optional) and serde JSON outputs (results + configs + online state)

### v1.x+

- kernel approximations + opt-in exact kernel mode
- FPOP where applicable
- GP/ARGP feature-flagged research module
- ensembles as optional doctor recommendation when uncertainty is high

## Notes on Naming and Packaging

- Choose a PyPI name that is short and unique (e.g., cpd-rs, rust-cpd, etc.).
- Provide cpd.doctor() and cpd.Pelt, cpd.Binseg, cpd.Bocpd for immediate familiarity.
- Mirror ruptures naming and conventions when sensible, but keep the richer result types as a differentiator.

## What You’ll Have at the End (Deliverable Summary)

By the end of MVP, you’ll have:
- A Rust core with stable APIs, cached cost models, constraints, structured results, and robust errors
- Fast offline segmentation (PELT) and practical approximate methods (BinSeg/WBS)
- Production-safe online Bayesian detection (BOCPD in log-space with truncation) with checkpoint/restore
- A pragmatic baseline streaming detector for low-latency alerting
- A Python package on PyPI with NumPy-native interop, released GIL, and typed results/configs
- Benchmarks + evaluation utilities + differential tests, plus a clear path to expand without destabilizing the core
