// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{
    CostAR, CostBernoulli, CostL2Mean, CostLinear, CostModel, CostNIGMarginal, CostNormalMeanVar,
    CostPoissonRate,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 50_000;
const GROUP_NAME: &str = "cost_models_multivariate_segment";
const BENCH_DIMS: [usize; 3] = [1, 8, 16];

fn generate_continuous_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let x = t as f64 + 1.0;
            let y = dim as f64 + 1.0;
            values.push((x * y) + (0.03 * x).sin() + (0.07 * y).cos());
        }
    }
    values
}

fn generate_count_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let raw = (t
                .wrapping_mul(dim.saturating_add(3))
                .wrapping_add(dim.saturating_mul(11)))
                % 37;
            values.push(raw as f64);
        }
    }
    values
}

fn generate_binary_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let bit = ((t + dim.saturating_mul(5)) % 2) as f64;
            values.push(bit);
        }
    }
    values
}

fn make_c_contiguous_view<'a>(values: &'a [f64], n: usize, d: usize) -> TimeSeriesView<'a> {
    TimeSeriesView::new(
        DTypeView::F64(values),
        n,
        d,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid")
}

fn exact_filter() -> Option<String> {
    std::env::var("CPD_BENCH_EXACT").ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn should_run_bench(bench_id: &str, exact: Option<&str>) -> bool {
    match exact {
        Some(filter) => {
            if bench_id == filter {
                return true;
            }
            let full_id = format!("{GROUP_NAME}/{bench_id}");
            full_id == filter
        }
        None => true,
    }
}

fn benchmark_multivariate_segment_scaling(c: &mut Criterion) {
    let l2_model = CostL2Mean::default();
    let normal_model = CostNormalMeanVar::default();
    let nig_model = CostNIGMarginal::default();
    let poisson_model = CostPoissonRate::default();
    let bernoulli_model = CostBernoulli::default();
    let linear_model = CostLinear::default();
    let ar_model = CostAR::default();

    let mut group = c.benchmark_group(GROUP_NAME);
    let start = N / 5;
    let end = (4 * N) / 5;
    let exact = exact_filter();

    for d in BENCH_DIMS {
        let l2_name = format!("l2_segment_cost_n5e4_d{d}");
        let normal_name = format!("normal_segment_cost_n5e4_d{d}");
        let nig_name = format!("nig_segment_cost_n5e4_d{d}");
        let poisson_name = format!("poisson_segment_cost_n5e4_d{d}");
        let bernoulli_name = format!("bernoulli_segment_cost_n5e4_d{d}");
        let linear_name = format!("linear_segment_cost_n5e4_d{d}");
        let ar_name = format!("ar_segment_cost_n5e4_d{d}");
        let run_l2 = should_run_bench(&l2_name, exact.as_deref());
        let run_normal = should_run_bench(&normal_name, exact.as_deref());
        let run_nig = should_run_bench(&nig_name, exact.as_deref());
        let run_poisson = should_run_bench(&poisson_name, exact.as_deref());
        let run_bernoulli = should_run_bench(&bernoulli_name, exact.as_deref());
        let run_linear = should_run_bench(&linear_name, exact.as_deref());
        let run_ar = should_run_bench(&ar_name, exact.as_deref());
        if !(run_l2
            || run_normal
            || run_nig
            || run_poisson
            || run_bernoulli
            || run_linear
            || run_ar)
        {
            continue;
        }

        let continuous_values = generate_continuous_values(N, d);
        let count_values = generate_count_values(N, d);
        let binary_values = generate_binary_values(N, d);
        let continuous_view = make_c_contiguous_view(continuous_values.as_slice(), N, d);
        let count_view = make_c_contiguous_view(count_values.as_slice(), N, d);
        let binary_view = make_c_contiguous_view(binary_values.as_slice(), N, d);

        if run_l2 {
            let l2_cache = l2_model
                .precompute(&continuous_view, &CachePolicy::Full)
                .expect("L2 precompute should succeed");
            group.bench_function(l2_name, |b| {
                b.iter(|| {
                    l2_model.segment_cost(black_box(&l2_cache), black_box(start), black_box(end))
                })
            });
        }

        if run_normal {
            let normal_cache = normal_model
                .precompute(&continuous_view, &CachePolicy::Full)
                .expect("Normal precompute should succeed");
            group.bench_function(normal_name, |b| {
                b.iter(|| {
                    normal_model.segment_cost(
                        black_box(&normal_cache),
                        black_box(start),
                        black_box(end),
                    )
                })
            });
        }

        if run_nig {
            let nig_cache = nig_model
                .precompute(&continuous_view, &CachePolicy::Full)
                .expect("NIG precompute should succeed");
            group.bench_function(nig_name, |b| {
                b.iter(|| {
                    nig_model.segment_cost(black_box(&nig_cache), black_box(start), black_box(end))
                })
            });
        }

        if run_poisson {
            let poisson_cache = poisson_model
                .precompute(&count_view, &CachePolicy::Full)
                .expect("Poisson precompute should succeed");
            group.bench_function(poisson_name, |b| {
                b.iter(|| {
                    poisson_model.segment_cost(
                        black_box(&poisson_cache),
                        black_box(start),
                        black_box(end),
                    )
                })
            });
        }

        if run_bernoulli {
            let bernoulli_cache = bernoulli_model
                .precompute(&binary_view, &CachePolicy::Full)
                .expect("Bernoulli precompute should succeed");
            group.bench_function(bernoulli_name, |b| {
                b.iter(|| {
                    bernoulli_model.segment_cost(
                        black_box(&bernoulli_cache),
                        black_box(start),
                        black_box(end),
                    )
                })
            });
        }

        if run_linear {
            let linear_cache = linear_model
                .precompute(&continuous_view, &CachePolicy::Full)
                .expect("Linear precompute should succeed");
            group.bench_function(linear_name, |b| {
                b.iter(|| {
                    linear_model.segment_cost(
                        black_box(&linear_cache),
                        black_box(start),
                        black_box(end),
                    )
                })
            });
        }

        if run_ar {
            let ar_cache = ar_model
                .precompute(&continuous_view, &CachePolicy::Full)
                .expect("AR precompute should succeed");
            group.bench_function(ar_name, |b| {
                b.iter(|| {
                    ar_model.segment_cost(black_box(&ar_cache), black_box(start), black_box(end))
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_multivariate_segment_scaling);
criterion_main!(benches);
