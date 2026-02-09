# MVP-A Benchmark SLOs

This document defines the protected benchmark cases and CI gating policy for MVP-A offline core performance.

## CI Gate Policy

- Relative gate (PRs): compare current benchmark metrics against the latest rolling `main` baseline artifact.
  - Runtime regression budget: `+10%`
  - RSS regression budget: `+15%`
- Absolute gate (`main` pushes and nightly): enforce only explicitly defined product thresholds.
  - Cases without explicit thresholds are still measured and tracked in the relative gate.

## Benchmark Cases

The source of truth for cases and thresholds is:

- `cpd/benchmarks/mvp_a_benchmark_manifest.json`

Protected case IDs:

- `pelt_l2_n1e4_d1_jump1`
- `pelt_l2_n1e5_d1_jump1`
- `pelt_l2_n1e6_d1_jump5_minseg20`
- `binseg_l2_n1e4_d1_jump1`
- `binseg_l2_n1e5_d1_jump1`
- `binseg_l2_n1e6_d1_jump1`
- `cost_models/l2_precompute_n1e6_d1`
- `cost_models/normal_precompute_n1e6_d1`
- `cost_models/l2_segment_queries_n1e6_d1_1m`

## Absolute Thresholds

Thresholded cases:

- `pelt_l2_n1e4_d1_jump1`: runtime `<= 0.05s`
- `pelt_l2_n1e5_d1_jump1`: runtime `<= 0.5s`
- `pelt_l2_n1e6_d1_jump5_minseg20`: runtime `<= 3.0s`, RSS `<= 1171875 KiB`
- `binseg_l2_n1e6_d1_jump1`: runtime `<= 1.5s`, RSS `<= 683594 KiB`
- `cost_models/l2_precompute_n1e6_d1`: runtime `<= 0.02s`

## Memory Unit Convention

Product thresholds in MB/GB are interpreted using decimal SI units:

- `1 MB = 1,000,000 bytes`
- `1 GB = 1,000,000,000 bytes`

Converted manifest values:

- `1.2 GB = 1171875 KiB`
- `700 MB = 683594 KiB` (rounded to nearest whole KiB)

## Reference Baseline

Versioned reference baseline file:

- `cpd/benchmarks/mvp_a_reference_baseline.json`

This file is for historical/versioned benchmark context. The PR relative gate uses the rolling baseline artifact produced from successful `main` runs of `benchmark-gate.yml`.

## Refresh Procedure

Run from repository root:

```bash
python3 .github/scripts/benchmark_gate.py collect \
  --workspace cpd \
  --manifest cpd/benchmarks/mvp_a_benchmark_manifest.json \
  --out cpd/benchmarks/mvp_a_reference_baseline.json
```

Validate against thresholds:

```bash
python3 .github/scripts/benchmark_gate.py compare-absolute \
  --manifest cpd/benchmarks/mvp_a_benchmark_manifest.json \
  --current cpd/benchmarks/mvp_a_reference_baseline.json
```
