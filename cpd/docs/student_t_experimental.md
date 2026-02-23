# Student-t Cost (Experimental)

This note documents robustness validation, complexity tradeoffs, and benchmark guidance for `CostStudentT`.

## Robustness Coverage

Curated robustness fixture tests live in:

- `cpd/crates/cpd-costs/tests/student_t_robustness.rs`

Coverage includes:

- Outlier-contaminated fixture (`d in {1, 8}`) with Student-t vs Gaussian split-error comparison.
- Heavy-tail fixture (`d in {1, 8}`) with Student-t vs Gaussian split-error comparison.

The assertions are directional (`Student-t error <= Normal error`) to verify robustness claims without overfitting to exact split indices.

## Segment Query Complexity

`CostStudentT` supports two query paths:

- Exact cache (`CachePolicy::Full` or `CachePolicy::Budgeted`)
  - Query cost: `O(d * m)` for segment length `m` (per-value log-tail sum).
  - Cache bytes: `O(n * d)` prefix stats plus raw values.
- Approximate cache (`CachePolicy::Approximate`)
  - Query cost: `O(d * (m / B + 1))`, where `B` is a deterministic block size chosen from `error_tolerance` and `max_bytes`.
  - Method: second-order block-moment approximation of the log-tail sum, using cached per-block `(sum, sum_sq)`.
  - Cache bytes: `O(n * d)` prefix stats plus block summaries (no raw values).

Approximate mode is deterministic and reproducible for fixed input, policy, and repro mode, but is not numerically identical to exact mode.

## Experimental Caveats

- `CostStudentT` remains experimental; tune and validate on domain fixtures before production use.
- Approximate mode introduces controlled approximation error to reduce segment query time.
- Extremely small `error_tolerance` values are clamped to a minimum block size; they do not force exact evaluation.
- Detector compatibility remains unchanged: Student-t cost currently supports offline `pelt` and `binseg`.

## Reproducible Benchmarks (`n=1e5`, `d in {1, 8}`)

Benchmark harness:

- `cpd/crates/cpd-bench/benches/cost_student_t_robustness.rs`

Versioned benchmark manifest:

- `cpd/benchmarks/student_t_robustness_manifest.v1.json`

Run all Student-t robustness cases:

```bash
cargo bench -p cpd-bench --bench cost_student_t_robustness
```

Run a single case:

```bash
CPD_BENCH_EXACT=student_t_approx_heavy_tail_segment_cost_n1e5_d8_tol0p10 \
  cargo bench -p cpd-bench --bench cost_student_t_robustness
```

