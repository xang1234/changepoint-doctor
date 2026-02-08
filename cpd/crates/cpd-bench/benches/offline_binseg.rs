// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{BinSeg, BinSegConfig};
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

fn benchmark_binseg_l2_n1e5(c: &mut Criterion) {
    const N: usize = 100_000;
    let values = step_series(N);
    let view = make_view(&values, N);
    let constraints = Constraints {
        min_segment_len: 20,
        jump: 5,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = BinSeg::new(
        CostL2Mean::default(),
        BinSegConfig {
            stopping: Stopping::Penalized(Penalty::Manual(10.0)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function("binseg_l2_n1e5", |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("BinSeg L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_binseg_l2_n1e6(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let values = step_series(N);
    let view = make_view(&values, N);
    let constraints = Constraints {
        min_segment_len: 50,
        jump: 10,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = BinSeg::new(
        CostL2Mean::default(),
        BinSegConfig {
            stopping: Stopping::Penalized(Penalty::Manual(20.0)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function("binseg_l2_n1e6", |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("BinSeg L2 benchmark detect should succeed");
        })
    });
}

criterion_group!(benches, benchmark_binseg_l2_n1e5, benchmark_binseg_l2_n1e6);
criterion_main!(benches);
