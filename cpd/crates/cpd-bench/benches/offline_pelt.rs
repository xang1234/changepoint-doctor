// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
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

fn bench_pelt_l2(
    c: &mut Criterion,
    case_id: &str,
    n: usize,
    jump: usize,
    min_segment_len: usize,
    penalty: f64,
) {
    let values = step_series(n);
    let view = make_view(&values, n);
    let constraints = Constraints {
        min_segment_len,
        jump,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = Pelt::new(
        CostL2Mean::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(penalty)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function(case_id, |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("PELT L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_pelt_l2_n1e4_d1_jump1(c: &mut Criterion) {
    const N: usize = 10_000;
    bench_pelt_l2(c, "pelt_l2_n1e4_d1_jump1", N, 1, 2, 10.0);
}

fn benchmark_pelt_l2_n1e5_d1_jump1(c: &mut Criterion) {
    const N: usize = 100_000;
    bench_pelt_l2(c, "pelt_l2_n1e5_d1_jump1", N, 1, 2, 10.0);
}

fn benchmark_pelt_l2_n1e6_d1_jump5_minseg20(c: &mut Criterion) {
    const N: usize = 1_000_000;
    bench_pelt_l2(c, "pelt_l2_n1e6_d1_jump5_minseg20", N, 5, 20, 20.0);
}

criterion_group!(
    benches,
    benchmark_pelt_l2_n1e4_d1_jump1,
    benchmark_pelt_l2_n1e5_d1_jump1,
    benchmark_pelt_l2_n1e6_d1_jump5_minseg20
);
criterion_main!(benches);
