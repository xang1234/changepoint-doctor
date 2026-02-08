# cpd-rs: HighPerformance Change Point Detection in Rust with Python Bindings

A productiongrade change point detection (CPD) toolkit with a Rust core (speed, memory efficiency, safety) and a Python API (PyPI wheels, NumPynative, ruptureslike ergonomics). The design is modular (cost models + search methods) and extensible (new algorithms plug in cleanly). It also ships a “Change Point Doctor” that recommends an algorithm pipeline based on timeseries diagnostics, constraints, and user objectives.

## Project Goals, Nongoals, Success Criteria, and Roadmap

### Goals

- Provide a productiongrade CPD toolkit with a Rust core that is fast, memoryefficient, and safe.
- Provide a Python API that feels familiar to ruptures users (fit/predict style) while exposing richer outputs (scores/probabilities/diagnostics) when needed.
- Make correctness + numerical stability a firstclass concern (especially for Bayesian methods).
- Make longseries scalability a firstclass concern (10⁶ points should be plausible for “fast path” algorithms like PELT on common costs, with appropriate constraints and candidate thinning).

### Nongoals (for v1)

- Full GPU acceleration and/or deep learningbased CPD (possible later, optional track).
- A default path for arbitrary Python callback cost functions (too slow, hard to stabilize). We can support this later as an explicitly “slow path” experimental feature.
- Implementing every CPD algorithm in the literature (focus on highimpact coverage + extensibility).

### Success criteria (measurable)

- Correctness: crosscheck segmentation outputs against reference implementations (ruptures + curated datasets) on a shared test corpus.
- Performance: publish benchmark results (runtime + peak memory) for representative sizes (1e4, 1e5, 1e6) across key algorithms (PELT, BinSeg/WBS, BOCPD).
- Packaging: ship manylinux/macOS/Windows wheels; pip install works without requiring a Rust toolchain.

### Phased roadmap

- MVP (v0.x):
  - TimeSeriesView + constraints + cost caching infrastructure
  - Costs: L2 mean shift, Normal mean+var, L1 robust (at least)
  - Algorithms: PELT, Binary Segmentation, WBS (optional but high ROI), BOCPD
  - Python wheels + docs + tests + benchmark harness
- v1.0 (“production-ready” breadth):
  - Offline: Dynp, BottomUp, Sliding Window
  - Penalty helpers (BIC/AIClike, userspecified, “known K” constraints)
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
  - Core crate does not depend on PyO3 (no FFI leakage into the compute core).
  - Feature flags to keep default builds lean; heavy algorithms are optin.

## Layered Crate Architecture + Feature Flags

Use a Cargo workspace with strict crate boundaries:

### Workspace layout (proposed)

```text
cpd/
  Cargo.toml              # workspace root
  crates/
    cpd-core/
    cpd-offline/
    cpd-online/
    cpd-python/
    cpd-cli/              # optional
    cpd-bench/            # optional (criterion harness)
  python/
    cpd/                  # optional pure-Python helpers, stubs, docs assets
  docs/
  tests/
```

### Crates

- cpd-core
- TimeSeriesView, constraints, penalty helpers
- CostModel trait + caches
- result types (ChangePointResult, Diagnostics)
- error types (CpdError)
- shared numeric utilities (log-sum-exp, stable stats)
- optional serde support for result serialization
- cpd-offline
- offline search algorithms: PELT, BinSeg, WBS, Dynp, BottomUp, Window, KernelCPD
- cpd-online
- streaming detectors: BOCPD (+ optional classical streaming detectors if desired)
- cpd-python
- PyO3 wrappers, NumPy interop, Python-facing config/result objects
- cpd-cli (optional)
- batch segmentation CLI: JSON in/out, easy pipeline integration
- cpd-bench (optional)
- criterion benchmarks + profiling integration

### Feature flags

- rayon — optional parallelism (be careful with Python + BLAS oversubscription)
- serde — JSON (results/configs) for debugging + CLI
- kernel — kernel CPD (default off)
- kernel-approx — Nyström / random features
- blas — heavy linear algebra dependencies (off by default)
- gp — GP/ARGP Bayesian models (experimental; off by default)

## Core API: Results, Errors, Diagnostics, Offline + Online Traits

Returning only Vec<usize> is too limiting. We want structured outputs for:
- confidence scores/probabilities
- segment stats
- debugging (pruning counts, candidate thinning, run-time notes)
- Doctor explanations and autotuning

### Types (sketch)

```rust
// Convention: indices are 0-based.
// Offline result returns breakpoints as segment end indices,
// typically including n (ruptures-like); see "Result conventions".

#[derive(Clone, Debug)]
pub struct ChangePointResult {
    pub breakpoints: Vec<usize>,           // sorted, ends of segments (often includes n)
    pub change_points: Vec<usize>,         // breakpoints excluding n (derived)
    pub scores: Option<Vec<f64>>,          // per change point score (offline)
    pub probabilities: Option<Vec<f64>>,   // per time-step or per cp (online/bayes)
    pub segments: Option<Vec<SegmentStats>>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub n: usize,
    pub d: usize,
    pub runtime_ms: Option<u64>,
    pub notes: Vec<String>,
    pub algorithm: &'static str,
    pub cost_model: &'static str,
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
    #[error("cancelled")]
    Cancelled,
}

// Offline algorithms: full series -> Result
pub trait OfflineDetector {
    fn detect(&self, x: &TimeSeriesView) -> Result<ChangePointResult, CpdError>;
}

// Online algorithms: stateful incremental update
pub struct OnlineStepResult {
    pub t: usize,
    pub p_change: f64,
    pub run_length_mode: usize,
    pub run_length_mean: f64,
}

pub trait OnlineDetector {
    fn reset(&mut self);
    fn update(&mut self, x_t: &[f64]) -> Result<OnlineStepResult, CpdError>;
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
    // later: linear, mean-per-dimension, drop (but beware index shifting)
}

pub struct TimeSeriesView<'a> {
    pub values: &'a [f64],   // flattened row-major (C-order): [t0_d0, t0_d1, ...]
    pub n: usize,
    pub d: usize,            // d=1 for univariate
    pub stride: usize,       // default stride = d for row-major; supports future strided views if needed
    pub missing: MissingPolicy,
}

```
### Python interop rules

- Accept NumPy arrays (1D or 2D).
- Prefer zerocopy for contiguous arrays; otherwise copy with a diagnostic note.
- Validate:
  - dtype (float64 preferred; accept float32 but usually upcast to float64 for core algorithms)
  - shape ((n,) or (n, d))
  - missing values policy (NaN default error unless configured otherwise)
  - Use the numpy crate (PyO3 ↔ NumPy bridge) which is built on PyO3 and ndarray.

## Constraints and Safety Rails

Applies across algorithms (offline and online where relevant):
- min_segment_len: enforce everywhere to prevent degenerate segments / noise fitting.
- max_change_points / max_depth: bound recursion and runtime (especially BinSeg/WBS).
- candidate_splits / jump: restrict breakpoint candidates (huge speedup; aligns with ruptures’ jump for PELT). The ruptures docs explicitly support min_size and jump for PELT.
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
}

pub struct CancelToken(pub std::sync::Arc<std::sync::atomic::AtomicBool>);

```
## Cost Models: FirstClass, Cached, Extensible

Most CPD performance comes from fast segment cost computation. We make it explicit and reusable.

### CostModel trait + cache

```rust
pub trait CostModel {
    type Cache: Send + Sync;

    fn name(&self) -> &'static str;

    fn validate(&self, x: &TimeSeriesView) -> Result<(), CpdError>;

    fn precompute(&self, x: &TimeSeriesView) -> Result<Self::Cache, CpdError>;

    /// Cost of segment [start, end) (end exclusive).
    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64;
}

```
### Builtin costs (roadmapped)

#### MVP

- CostL2Mean (piecewise constant mean, least squares)
- CostNormalMeanVar (Gaussian with MLE mean+variance; useful for volatility/regime shifts)
- CostL1Mean (robust to outliers; good for finance tails)

#### v1

- CostLinear (piecewise linear trend; good for drifting sensors)
- CostAR (autoregressive residual cost; helps autocorrelated series; aligns with ruptures cost family)
- CostRank / CostCosine (optional, depending on demand)

#### Experimental

- CostRbfKernel (distributional changes; default via approximations)
- “Studentt cost” (offline) if you want robust likelihood segmentation, but this can be expensive; consider as optin.

### Penalty & stopping helpers

- Penalty::BIC, Penalty::AIC, Penalty::Manual(f64)
- Stopping::KnownK(usize) vs Stopping::Penalized(Penalty)
- enforce constraints consistently (one shared enforcement module)

## Offline Algorithms: Roadmap / Maturity Levels

We implement a “small core” first, then expand.

### MVP (v0.x, productionready)

- PELT (Pruned Exact Linear Time)
- Penalized optimal partitioning with pruning; average linear time under conditions.
- Supports min_segment_len, candidate thinning (jump/candidate_splits), max cps.
- Primary engine for large offline series (finance/sensors).
- Binary Segmentation (BinSeg)
- Fast recursive splitting; good when changes are few and strong.
- Supports max_depth, max_change_points, and candidate splits.
- Wild Binary Segmentation (WBS)
- Addresses masking issues in vanilla BinSeg; designed for multiple change points and small spacings.
- Include deterministic seeding and reproducibility.

### v1.0 additions (productionready breadth)

- Dynp (Dynamic programming / optimal partitioning)
- Exact but heavy; recommended for smaller n or validation.
- BottomUp segmentation
- Mergebased; can be good for many small changes.
- Sliding Window
- Local comparison; good for frequent short changes and streamingadjacent workflows.
- FPOP / Functional pruning (where applicable)
- Strong penalized segmentation alternative; pruning techniques paper introduces FPOP and discusses scenarios where it shines.
- Implement for common L2 mean settings first (high leverage).

### Experimental / heavy (featureflagged)

- Kernel CPD
- Make exact O(n²) mode optin.
- Default: scalable approximations (Nyström/random features) under kernel-approx.
- GP / ARGPCPstyle methods
- Extremely powerful but expensive and complex; implement only after core is stable.

## Online Algorithms: BOCPD Done “ProductionSafe”

### Bayesian Online Change Point Detection (BOCPD)

BOCPD maintains a runlength posterior and yields p(change at t) online.

### Implementation details for robustness

- Logspace probabilities + logsumexp everywhere (prevents underflow).
- Hazard function interface:
  - constant hazard
  - geometric hazard
  - parametrized hazard families
- Truncation / pruning:
  - max_run_length
  - prune runlength states below a logprob threshold
  - Optional fixedlag smoothing for better retrospective estimates in streaming settings (bounded window).

### Outputs

- per update step: p_change, run-length mode/mean, optional topK run lengths for debugging.

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

#### High-level (ruptureslike)

- Pelt(model="l2").fit(x).predict(pen=...)
- Binseg(model="l2").fit(x).predict(n_bkps=...)
- Bocpd(model="student_t", hazard=...).update(x_t) streamingfriendly

#### Low-level (power users / Doctor integration)

- detect(x, detector=..., cost=..., constraints=..., return_diagnostics=True)
- returns a structured ChangePointResult object

### Result object in Python

- Return a dataclasslike object:
  - .breakpoints (list of ints, includes n by default)
  - .change_points (excludes n)
  - .scores, .probabilities, .diagnostics, .segments
  - Provide convenience: .plot() helper (optional pure Python layer)

### Zerocopy + dtype handling

- Accept numpy.ndarray (float32/float64).
- Prefer zerocopy on contiguous arrays; copy otherwise (and record a diagnostic note).
- Validate shape and enforce consistent memory layout expectations.

### Releasing the GIL

All heavy compute runs under py.allow_threads(|| ...), so Python remains responsive.

### Sketch

```rust
#[pymethods]
impl PyPelt {
    fn detect(&self, py: Python, x: &PyAny) -> PyResult<PyChangePointResult> {
        py.allow_threads(|| {
            let view = parse_numpy_to_view(x)?;  // zero-copy when possible
            let result = self.detector.detect(&view)?;
            Ok(PyChangePointResult::from(result))
        })
    }
}

```
### Streamingfriendly Python interface

Online detectors are stateful Python objects:
- .update(x_t) where x_t can be scalar (univariate) or 1D array (multivariate at time t)
- .reset()
- .state() debug export (optional; for monitoring services)

### Type hints + stability

- Ship py.typed and stub files or runtime type hints.
- Stable, documented config/result dataclasses.

## “Change Point Doctor”: Ranked Recommendations + Configs + Explanations + Optional Validation

### What it does

The Doctor returns a ranked set of recommended pipelines:
- detector (algorithm)
- cost model
- penalty/stopping rule
- constraints (min_segment_len, candidates/jump)
- warnings (complexity, memory)
- explanation (what diagnostics drove the recommendation)
This is explicitly aligned with how CPD methods are typically composed (cost + search + constraint).

### Doctor API (proposed)

- doctor.recommend(x, objective="balanced", online=False, constraints=...) -> List[Recommendation]
- doctor.validate_top_k(x, k=3, downsample=..., seed=...) -> ValidationReport (optional)

### Recommendation object (sketch)

```rust
pub struct Recommendation {
    pub detector: &'static str,      // "pelt", "wbs", "bocpd"
    pub cost: &'static str,          // "l2_mean", "normal_mean_var", ...
    pub params: serde_json::Value,   // or a typed enum/struct
    pub expected_complexity: String,
    pub warnings: Vec<String>,
    pub explanation: Explanation,
}

pub struct Explanation {
    pub signals: Vec<(String, f64)>,  // e.g. ("autocorr_lag1", 0.72)
    pub narrative: String,
}

```
### Diagnostics the Doctor computes

(Computed quickly; O(n) or O(n log n) max; supports sampling for n huge.)
- basic: n, d, sampling rate if provided
- missingness: NaN rate, longest NaN run
- noise/robustness: kurtosis proxy, outlier rate, MAD/STD ratio
- autocorrelation: lag1/lagk autocorr, partial autocorr proxy
- stationarity proxies: rolling mean/var drift
- change “density” proxy: windowed divergence scores on a subsample
- user constraints: expected number of changes (if provided), latency requirements (online vs offline)

### Heuristic mapping (v1)

- Huge n (≥1e5) + offline: recommend PELT with cached L2/Normal costs, plus candidate thinning (jump) and sensible min_segment_len.
- Few strong changes: BinSeg or WBS (prefer WBS when masking risk detected).
- Many changes: PELT or BottomUp; Window for very local/frequent changes.
- Autocorrelated / trending sensors: prefer CostAR or CostLinear with PELT/BinSeg; treat GP methods as experimental.
- Heavy tails / outliers (finance): robust costs (L1) or Bayesian models; avoid fragile Gaussian-only assumptions.

### “Trust but verify” mode

validate_top_k runs a quick pass:
- potentially downsample or use candidate thinning
- run top 2–3 pipelines
- measure consensus/stability (e.g., overlap of breakpoints within tolerance, sensitivity to penalty)
- returns a report that increases user trust and helps tune parameters

## Performance Plan: Benchmarking + Profiling + Budgets (No Speculative Claims)

Performance is treated as a deliverable.

### Benchmark suite

- sizes: 1e4, 1e5, 1e6
- cases: univariate + multivariate (d=8/16), piecewise constant/linear, heavy tails, autocorrelated
- algorithms: PELT, BinSeg/WBS, BOCPD
- report: runtime + peak RSS + allocations (where possible)

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

### Performance budgets

Establish target envelopes on a reference machine, e.g.:
- PELT + L2 on 1e6 points should run within X seconds and Y GB with jump and min_segment_len defaults.
(Exact numbers depend on your target hardware; the key is to define and publish them.)

### Regression prevention

- run microbenchmarks in CI (or nightly) to detect slowdowns.
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

## Implementation Roadmap: Concrete Work Breakdown

### MVP (v0.x)

- Core abstractions
- TimeSeriesView + parsing/validation
- Constraints, Penalty, Stopping
- CostModel trait + caches
- ChangePointResult, Diagnostics, CpdError
- cancellation/progress wiring
- Cost models
- L2 mean (prefix sums/squares)
- Normal mean+var (sufficient stats)
- L1 mean (prefix sums for median is not O(1); implement efficient approximations or restrict usage; alternatively keep L1 as v1 if you want strict O(1) focus)
- Offline algorithms
- PELT (generic over CostModel)
- BinSeg (generic over cost; supports constraints)
- WBS (seeded; interval sampling; supports cost)
- Online algorithms
- BOCPD:
  - log-space math
  - hazard interface
  - truncation/pruning
  - stable incremental likelihood updates for chosen observation models
  - Python bindings
  - maturin project + wheels
  - numpy array interop (zero-copy when contiguous)
  - GIL release
  - typed result objects + docs + minimal examples
  - Quality & benchmarks
  - unit + property tests
  - differential tests against ruptures for PELT/BinSeg where comparable
  - criterion benchmarks for fast path

### v1.0

- multivariate polish across costs/algorithms
- Dynp / BottomUp / Window
- penalty helpers and consistent parameter naming with ruptures where possible
- Doctor v1 (ranked pipelines + explanations + safe configs)
- CLI (optional) and serde JSON outputs

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
- Productionsafe online Bayesian detection (BOCPD in log-space with truncation)
- A Python package on PyPI with NumPy-native interop, released GIL, and typed results
- Benchmarks, differential tests, and a clear path to expand to heavier algorithms without destabilizing the core
