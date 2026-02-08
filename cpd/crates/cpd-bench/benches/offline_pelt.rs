// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{Pelt, PeltConfig};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_view<'a>(values: &'a [f64], n: usize) -> TimeSeriesView<'a> {
    TimeSeriesView::new(
        DTypeView::F64(values),
        n,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid")
}

fn step_series(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n];
    for v in values.iter_mut().skip(n / 2) {
        *v = 5.0;
    }
    values
}

fn variance_shift_series(n: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n);
    let half = n / 2;
    for i in 0..half {
        values.push(if i % 2 == 0 { -1.0 } else { 1.0 });
    }
    for i in half..n {
        values.push(if i % 2 == 0 { -5.0 } else { 5.0 });
    }
    values
}

fn benchmark_pelt_l2_n1e5(c: &mut Criterion) {
    const N: usize = 100_000;
    let values = step_series(N);
    let view = make_view(&values, N);
    let constraints = Constraints {
        min_segment_len: 20,
        jump: 5,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = Pelt::new(
        CostL2Mean::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(10.0)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function("pelt_l2_n1e5", |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("PELT L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_pelt_l2_n1e6(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let values = step_series(N);
    let view = make_view(&values, N);
    let constraints = Constraints {
        min_segment_len: 50,
        jump: 10,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = Pelt::new(
        CostL2Mean::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(20.0)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function("pelt_l2_n1e6", |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("PELT L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_pelt_normal_n1e5(c: &mut Criterion) {
    const N: usize = 100_000;
    let values = variance_shift_series(N);
    let view = make_view(&values, N);
    let constraints = Constraints {
        min_segment_len: 20,
        jump: 5,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = Pelt::new(
        CostNormalMeanVar::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(10.0)),
            params_per_segment: 3,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function("pelt_normal_n1e5", |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("PELT Normal benchmark detect should succeed");
        })
    });
}

criterion_group!(
    benches,
    benchmark_pelt_l2_n1e5,
    benchmark_pelt_l2_n1e6,
    benchmark_pelt_normal_n1e5
);
criterion_main!(benches);
