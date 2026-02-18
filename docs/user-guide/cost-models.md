# Cost Models

Cost models define how segment homogeneity is measured. The choice of cost model determines what kind of distributional change the detector is sensitive to.

## Overview

| Cost model | API string | Description | Multivariate | Best for |
|---|---|---|---|---|
| L2 (mean) | `"l2"` | Sum of squared residuals from segment mean | Yes (additive) | Mean shifts in continuous data |
| L1 (median) | `"l1_median"` | Sum of absolute deviations from segment median | Yes (additive) | Robust mean estimation with outliers |
| Normal | `"normal"` | Gaussian negative log-likelihood (diagonal) | Yes (additive) | Mean + variance changes |
| Normal Full Cov | `"normal_full_cov"` | Multivariate Gaussian with full covariance | Yes (cross-dim) | Covariance structure changes |
| NIG | `"nig"` | Normal-Inverse-Gamma marginal likelihood | Yes (additive) | Bayesian mean/variance inference |
| AR | `"ar"` | Autoregressive residual likelihood | Yes (additive) | Changes in autocorrelated data |
| Bernoulli | `"bernoulli"` | Bernoulli log-likelihood | Yes (additive) | Binary event rate changes |
| Poisson | `"poisson"` | Poisson rate log-likelihood | Yes (additive) | Count data rate changes |
| Rank | `"rank"` | Rank-based non-parametric cost | Yes (additive) | Distribution-free change detection |
| Cosine | `"cosine"` | Cosine similarity-based cost | Yes (additive) | Directional/angular changes |

## Per-model details

### L2 (CostL2Mean)

The default and most widely used cost model. Measures the sum of squared deviations from the segment mean.

$$C(y_{a:b}) = \sum_{i=a}^{b-1} (y_i - \bar{y}_{a:b})^2$$

- **Multivariate:** Sum of per-dimension SSE (independent dimensions)
- **BIC/AIC params:** 2 per dimension (mean + residual variance proxy)
- **Cache scaling:** O(n * d) memory

### L1 Median (CostL1Median)

Uses median instead of mean, providing robustness to outliers.

- **Best for:** Data with occasional extreme values where L2 cost would be distorted
- **Multivariate:** Sum of per-dimension absolute deviations

### Normal (CostNormalMeanVar)

Gaussian negative log-likelihood modeling both mean and variance per segment. Detects both mean shifts and variance changes.

- **Multivariate:** Sum of per-dimension terms (diagonal covariance)
- **BIC/AIC params:** 3 per dimension (mean + variance + residual)
- **Cache scaling:** O(n * d) memory
- **Segment query:** O(d) per query

### Normal Full Covariance (CostNormalFullCov)

Multivariate Gaussian with full covariance estimation per segment. Detects changes in cross-dimensional correlations.

- **Multivariate:** Full covariance-aware (detects correlation structure changes)
- **BIC/AIC params:** Model-aware `1 + d + d(d+1)/2`
- **Cache scaling:** O(n * d^2) memory
- **Segment query:** O(d^2) covariance assembly + O(d^3) Cholesky
- **Regularization:** Uses ridge regularization + jitter escalation in Cholesky for near-singular segments

:::{tip}
Prefer `normal` (diagonal) when d is large, memory is constrained, or cross-dimension covariance is weak. Prefer `normal_full_cov` when covariance structure carries the change signal and d is moderate.
:::

### NIG (CostNIGMarginal)

Normal-Inverse-Gamma marginal likelihood. A Bayesian cost that integrates out the mean and variance parameters.

- **Multivariate:** Sum of per-dimension NIG marginal terms

### AR (CostAR)

Autoregressive residual likelihood for data with temporal correlation.

- **Best for:** Time series where autocorrelation is the dominant feature
- **Cache scaling:** O(n * d) for order 1; O(n * d) for higher orders

### Bernoulli (CostBernoulli)

Bernoulli log-likelihood for binary (0/1) event data.

- **Best for:** Binary event rate changes (error rates, click-through rates)

### Poisson (CostPoissonRate)

Poisson rate log-likelihood for count data.

- **Best for:** Count data rate changes (event counts per time period)

### Rank (CostRank)

Rank-based non-parametric cost function.

- **Best for:** Distribution-free change detection where parametric assumptions are unwanted

### Cosine (CostCosine)

Cosine similarity-based cost for directional data.

- **Best for:** Angular/directional changes in high-dimensional embeddings

## Decision tree: choosing a cost model

```
Is your data binary (0/1)?
  → Yes: Use "bernoulli"
Is your data count-valued (non-negative integers)?
  → Yes: Use "poisson"
Do you have outliers?
  → Yes: Use "l1_median" or "rank"
Do you want to detect variance changes (not just mean)?
  → Yes, univariate or independent dims: Use "normal"
  → Yes, cross-dimensional correlation: Use "normal_full_cov"
Is your data autocorrelated?
  → Yes: Use "ar"
Default:
  → Use "l2" (fastest, robust for mean shifts)
```

## Availability in Python

| Cost model | High-level classes | `detect_offline()` | Pipeline-only |
|---|---|---|---|
| `l2` | `Pelt`, `Binseg`, `Fpop` | Yes | -- |
| `l1_median` | -- | Yes | -- |
| `normal` | `Pelt`, `Binseg` | Yes | -- |
| `normal_full_cov` | `Pelt`, `Binseg` | Yes | -- |
| `nig` | -- | -- | Yes |
| `ar` | -- | -- | Yes |
| `rank` | -- | -- | Yes |
| `cosine` | -- | -- | Yes |
| `bernoulli` | -- | -- | Yes |
| `poisson` | -- | -- | Yes |
