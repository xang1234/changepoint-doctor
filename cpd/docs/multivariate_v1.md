# v1 Multivariate Semantics and Scaling

This note documents multivariate (`d > 1`) behavior for v1 costs, including diagonal vs full-covariance Normal options.

## Multivariate Semantics

Additive (independent-dimension) costs:

- `CostL2Mean`: sum of per-dimension SSE terms.
- `CostNormalMeanVar`: sum of per-dimension Gaussian negative log-likelihood terms (diagonal covariance).
- `CostNIGMarginal`: sum of per-dimension NIG log-marginal terms (diagonal covariance).
- `CostStudentT`: sum of per-dimension heavy-tail Student-t negative log-likelihood terms.
- `CostPoissonRate`: sum of per-dimension Poisson rate log-likelihood terms.
- `CostBernoulli`: sum of per-dimension Bernoulli log-likelihood terms.
- `CostLinear`: sum of per-dimension piecewise-linear residual terms.
- `CostAR`: sum of per-dimension AR residual likelihood terms.

Cross-dimension covariance-aware cost:

- `CostNormalFullCov`: multivariate Gaussian segment cost using a regularized full covariance estimate per segment.

## Choosing Diagonal vs Full Covariance

- Prefer `CostNormalMeanVar` (diagonal) when `d` is large, runtime/memory budget is tight, or cross-dimension covariance is weak/noisy.
- Prefer `CostNormalFullCov` when covariance structure between dimensions carries changepoint signal and `d` is moderate.
- `CostNormalFullCov` uses ridge regularization plus jitter escalation in Cholesky factorization to keep segment costs finite for near-singular segments (`m <= d`, collinearity).

## Cache Memory Scaling

Let:

- `n`: number of rows
- `d`: number of dimensions
- `P = (n + 1) * d`
- `T = d * (d + 1) / 2` (upper-triangle pair count)
- `F = sizeof(f64)`
- `U = sizeof(u64)`
- `Z = sizeof(usize)`

Worst-case cache bytes:

- `CostL2Mean`: `2 * P * F`
- `CostNormalMeanVar`: `2 * P * F`
- `CostNormalFullCov`: `(n + 1) * (d + T) * F`
- `CostNIGMarginal`: `P * U + 2 * P * F`
- `CostPoissonRate`: `P * F`
- `CostBernoulli`: `P * Z`
- `CostLinear`: `(n + 1) * 2 * F + P * 3 * F`
- `CostAR (order=1)`: `3 * P * F`
- `CostAR (order>1)`: `n * d * F`
- `CostStudentT (exact)`: `2 * P * F + (n * d * F)`
- `CostStudentT (approximate)`: `2 * P * F + 2 * ceil(n / B) * d * F` (`B`: approximate block length)

Asymptotically:

- Diagonal/additive costs are `O(n * d)` space.
- `CostNormalFullCov` is `O(n * d^2)` space.

## Time Scaling (Segment Query)

- `CostNormalMeanVar`: `O(d)` per segment query.
- `CostNormalFullCov`: `O(d^2)` covariance assembly + `O(d^3)` Cholesky log-det per segment query.
- `CostStudentT (exact)`: `O(d * m)` per segment query (`m = end - start`).
- `CostStudentT (approximate)`: `O(d * (m / B + 1))` per segment query.

## Penalty Scaling (BIC/AIC)

- `CostNormalMeanVar` uses linear model complexity (`effective_params = d * 3`).
- `CostNormalFullCov` uses model-aware complexity (`effective_params = 1 + d + d(d+1)/2`), so automatic penalties scale with full-covariance parameter growth.

## Verification Coverage

- Additive cross-model checks (`d=8`, `d=16`) live in `cpd/crates/cpd-costs/tests/multivariate_v1.rs`.
- `CostNormalFullCov` correctness/stability coverage lives in `cpd/crates/cpd-costs/src/normal.rs` tests.

## Performance Benchmarks (`d=8`, `d=16`)

Criterion harness:

- `cpd/crates/cpd-bench/benches/cost_multivariate.rs`

Run full matrix:

```bash
cargo bench -p cpd-bench --bench cost_multivariate
```

Run one benchmark id:

```bash
CPD_BENCH_EXACT=normal_full_cov_segment_cost_n5e4_d16 \
  cargo bench -p cpd-bench --bench cost_multivariate
```

Included families use `d in {1, 8, 16}` and include `normal_full_cov`.

Student-t robustness/performance harness (`n=1e5`, `d in {1, 8}`):

- `cpd/crates/cpd-bench/benches/cost_student_t_robustness.rs`
- `cpd/benchmarks/student_t_robustness_manifest.v1.json`
- `cpd/docs/student_t_experimental.md`

## Doctor Awareness

- Offline recommendations emit multivariate semantics guidance for selected cost models, including diagonal-vs-full-covariance tradeoffs for Normal costs.
- Online recommendations are rejected for multivariate inputs (`d>1`) with a clear guidance error.
