// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 1_000_000;

fn benchmark_prefix_sums(c: &mut Criterion) {
    let values: Vec<f64> = (0..N)
        .map(|idx| {
            let x = idx as f64;
            x.sin() + x.cos() * 0.1
        })
        .collect();

    let mut group = c.benchmark_group("numerics_prefix");

    group.bench_function("prefix_sums_n1e6", |b| {
        b.iter(|| cpd_core::prefix_sums(black_box(&values)))
    });

    group.bench_function("prefix_sums_kahan_n1e6", |b| {
        b.iter(|| cpd_core::prefix_sums_kahan(black_box(&values)))
    });

    group.finish();
}

criterion_group!(benches, benchmark_prefix_sums);
criterion_main!(benches);
