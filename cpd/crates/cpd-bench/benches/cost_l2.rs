// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostL2Mean, CostModel, CostNIGMarginal, CostNormalMeanVar};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 1_000_000;
const QUERY_COUNT: usize = 1_000_000;
const MULTI_N: usize = 200_000;
const MULTI_QUERY_COUNT: usize = 200_000;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn generate_queries(n: usize, count: usize) -> Vec<(usize, usize)> {
    let mut queries = Vec::with_capacity(count);
    let mut state = 0xfeed_f00d_dead_beef_u64;

    for _ in 0..count {
        let a = (lcg_next(&mut state) as usize) % n;
        let b = (lcg_next(&mut state) as usize) % n;
        let start = a.min(b);
        let mut end = a.max(b) + 1;
        if start == end {
            end = (start + 1).min(n);
        }
        queries.push((start, end));
    }

    queries
}

fn generate_multivariate_values(n: usize, d: usize) -> Vec<f64> {
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

fn benchmark_cost_models(c: &mut Criterion) {
    let values: Vec<f64> = (0..N)
        .map(|idx| {
            let x = idx as f64;
            x.sin() + x.cos() * 0.1
        })
        .collect();

    let view = TimeSeriesView::new(
        DTypeView::F64(&values),
        N,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid");

    let l2_model = CostL2Mean::default();
    let normal_model = CostNormalMeanVar::default();

    let mut group = c.benchmark_group("cost_models");

    group.bench_function("l2_precompute_n1e6_d1", |b| {
        b.iter(|| {
            let _cache = l2_model
                .precompute(black_box(&view), black_box(&CachePolicy::Full))
                .expect("precompute should succeed");
        })
    });

    group.bench_function("normal_precompute_n1e6_d1", |b| {
        b.iter(|| {
            let _cache = normal_model
                .precompute(black_box(&view), black_box(&CachePolicy::Full))
                .expect("normal precompute should succeed");
        })
    });

    let cache = l2_model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed");
    let queries = generate_queries(N, QUERY_COUNT);
    let mut out_costs = vec![0.0; queries.len()];

    group.bench_function("l2_segment_queries_n1e6_d1_1m", |b| {
        b.iter(|| {
            l2_model.segment_cost_batch(
                black_box(&cache),
                black_box(&queries),
                black_box(&mut out_costs),
            );
        })
    });

    group.finish();
}

fn benchmark_multivariate_segment_scaling(c: &mut Criterion) {
    let l2_model = CostL2Mean::default();
    let normal_model = CostNormalMeanVar::default();
    let nig_model = CostNIGMarginal::default();
    let queries = generate_queries(MULTI_N, MULTI_QUERY_COUNT);

    let mut group = c.benchmark_group("cost_models_multivariate");

    for d in [1_usize, 4, 16] {
        let values = generate_multivariate_values(MULTI_N, d);
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            MULTI_N,
            d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("benchmark view should be valid");

        let l2_cache = l2_model
            .precompute(&view, &CachePolicy::Full)
            .expect("L2 precompute should succeed");
        let normal_cache = normal_model
            .precompute(&view, &CachePolicy::Full)
            .expect("Normal precompute should succeed");
        let nig_cache = nig_model
            .precompute(&view, &CachePolicy::Full)
            .expect("NIG precompute should succeed");

        let mut out_l2 = vec![0.0; queries.len()];
        let mut out_normal = vec![0.0; queries.len()];
        let mut out_nig = vec![0.0; queries.len()];

        let l2_name = format!("l2_segment_queries_n2e5_d{d}_200k");
        group.bench_function(l2_name, |b| {
            b.iter(|| {
                l2_model.segment_cost_batch(
                    black_box(&l2_cache),
                    black_box(&queries),
                    black_box(&mut out_l2),
                );
            })
        });

        let normal_name = format!("normal_segment_queries_n2e5_d{d}_200k");
        group.bench_function(normal_name, |b| {
            b.iter(|| {
                normal_model.segment_cost_batch(
                    black_box(&normal_cache),
                    black_box(&queries),
                    black_box(&mut out_normal),
                );
            })
        });

        let nig_name = format!("nig_segment_queries_n2e5_d{d}_200k");
        group.bench_function(nig_name, |b| {
            b.iter(|| {
                nig_model.segment_cost_batch(
                    black_box(&nig_cache),
                    black_box(&queries),
                    black_box(&mut out_nig),
                );
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_cost_models,
    benchmark_multivariate_segment_scaling
);
criterion_main!(benches);
