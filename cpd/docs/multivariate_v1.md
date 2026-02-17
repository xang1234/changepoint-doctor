# v1 Multivariate Semantics and Scaling

This note documents how v1 cost models behave for `d > 1`, how cache memory scales with dimension, and how to benchmark `d=8` and `d=16`.

## Multivariate Semantics

All current v1 costs use additive per-dimension scoring:

- `CostL2Mean`: sum of per-dimension SSE terms.
- `CostNormalMeanVar`: sum of per-dimension Gaussian negative log-likelihood terms.
- `CostNIGMarginal`: sum of per-dimension NIG log-marginal terms.
- `CostPoissonRate`: sum of per-dimension Poisson rate log-likelihood terms.
- `CostBernoulli`: sum of per-dimension Bernoulli log-likelihood terms.
- `CostLinear`: sum of per-dimension piecewise-linear residual terms.
- `CostAR`: sum of per-dimension AR residual likelihood terms.

Implication:

- Cross-dimension covariance is not modeled in v1 costs.
- For `CostNormalMeanVar` and `CostNIGMarginal`, this corresponds to a diagonal-covariance assumption.
- A full-covariance Normal cost remains future work (stretch goal from `CPD-xb5.14`).

## Cache Memory Scaling

Let:

- `n`: number of rows
- `d`: number of dimensions
- `P = (n + 1) * d`
- `F = sizeof(f64)`
- `U = sizeof(u64)`
- `Z = sizeof(usize)`

Worst-case cache bytes in current v1 implementations:

- `CostL2Mean`: `2 * P * F`
- `CostNormalMeanVar`: `2 * P * F`
- `CostNIGMarginal`: `P * U + 2 * P * F`
- `CostPoissonRate`: `P * F`
- `CostBernoulli`: `P * Z`
- `CostLinear`: `(n + 1) * 2 * F + P * 3 * F`
- `CostAR (order=1)`: `3 * P * F`
- `CostAR (order>1)`: `n * d * F`

All of the above are `O(n * d)` in space.

## Verification Coverage

`cpd-costs` now includes a cross-model integration test at:

- `cpd/crates/cpd-costs/tests/multivariate_v1.rs`

This verifies additive multivariate behavior at `d=8` and `d=16` for all v1 costs listed above.

## Performance Benchmarks (`d=8`, `d=16`)

Use the Criterion harness in:

- `cpd/crates/cpd-bench/benches/cost_multivariate.rs`

Run the full matrix:

```bash
cargo bench -p cpd-bench --bench cost_multivariate
```

Run a single exact benchmark id:

```bash
cargo bench -p cpd-bench --bench cost_multivariate -- --exact normal_segment_cost_n5e4_d16
```

The benchmark includes `d in {1, 8, 16}` for:

- `l2`, `normal`, `nig`, `poisson`, `bernoulli`, `linear`, `ar`

## Doctor Awareness

Doctor recommendations now emit explicit multivariate warnings:

- Offline recommendations explain additive/diagonal semantics for the selected cost.
- Online recommendations warn that current online detectors are univariate-only (`d=1`).
