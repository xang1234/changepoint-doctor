// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{SegNeigh, SegNeighConfig};
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

fn staircase_series(n: usize, k: usize) -> Vec<f64> {
    let regimes = k + 1;
    let base_len = n / regimes;
    let mut values = Vec::with_capacity(n);

    for regime in 0..regimes {
        let level = match regime % 4 {
            0 => 0.0,
            1 => 8.0,
            2 => -4.0,
            _ => 12.0,
        };
        let len = if regime + 1 == regimes {
            n - values.len()
        } else {
            base_len
        };
        values.extend(std::iter::repeat_n(level, len));
    }

    values
}

fn bench_segneigh_l2_known_k(
    c: &mut Criterion,
    case_id: &str,
    n: usize,
    k: usize,
    jump: usize,
    min_segment_len: usize,
) {
    let values = staircase_series(n, k);
    let view = make_view(&values, n);
    let constraints = Constraints {
        min_segment_len,
        jump,
        max_change_points: Some(k),
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = SegNeigh::new(
        CostL2Mean::default(),
        SegNeighConfig {
            stopping: Stopping::KnownK(k),
            cancel_check_every: 1_000,
        },
    )
    .expect("SegNeigh config should be valid");

    let preflight = detector
        .detect(&view, &ctx)
        .expect("SegNeigh preflight detect should succeed");
    assert_eq!(
        preflight.change_points.len(),
        k,
        "benchmark fixture should produce exactly k change points"
    );

    c.bench_function(case_id, |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("SegNeigh L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_segneigh_l2_n2e3_k2_jump1_minseg4(c: &mut Criterion) {
    bench_segneigh_l2_known_k(c, "segneigh_l2_n2e3_k2_jump1_minseg4", 2_000, 2, 1, 4);
}

fn benchmark_segneigh_l2_n4e3_k6_jump2_minseg6(c: &mut Criterion) {
    bench_segneigh_l2_known_k(c, "segneigh_l2_n4e3_k6_jump2_minseg6", 4_000, 6, 2, 6);
}

fn benchmark_segneigh_l2_n8e3_k12_jump4_minseg8(c: &mut Criterion) {
    bench_segneigh_l2_known_k(c, "segneigh_l2_n8e3_k12_jump4_minseg8", 8_000, 12, 4, 8);
}

criterion_group!(
    benches,
    benchmark_segneigh_l2_n2e3_k2_jump1_minseg4,
    benchmark_segneigh_l2_n4e3_k6_jump2_minseg6,
    benchmark_segneigh_l2_n8e3_k12_jump4_minseg8
);
criterion_main!(benches);
