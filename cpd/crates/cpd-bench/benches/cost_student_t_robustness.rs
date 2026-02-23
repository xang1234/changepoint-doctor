// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostModel, CostNormalMeanVar, CostStudentT};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 100_000;
const BENCH_DIMS: [usize; 2] = [1, 8];
const GROUP_NAME: &str = "cost_student_t_robustness";
const APPROX_ERROR_TOLERANCE: f64 = 0.10;

#[derive(Clone, Copy)]
enum FixtureKind {
    Outlier,
    HeavyTail,
}

impl FixtureKind {
    fn id_suffix(self) -> &'static str {
        match self {
            Self::Outlier => "outlier",
            Self::HeavyTail => "heavy_tail",
        }
    }
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

fn cauchy_like_noise(t: usize, dim: usize) -> f64 {
    let mut state = (t as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add((dim as u64).wrapping_mul(1442695040888963407))
        .wrapping_add(1);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51afd7ed558ccd);
    let u = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
    let clamped = u.clamp(1e-6, 1.0 - 1e-6);
    let raw = (std::f64::consts::PI * (clamped - 0.5)).tan();
    raw.clamp(-8.0, 8.0)
}

fn generate_fixture_values(kind: FixtureKind, n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    let true_change = n / 2;
    for t in 0..n {
        for dim in 0..d {
            let value = match kind {
                FixtureKind::Outlier => {
                    let base = if t < true_change { 0.0 } else { 3.0 };
                    let trend = 0.02 * (dim as f64 + 1.0) * ((t as f64) * 0.003).sin();
                    let mut v = base + trend;
                    if (t + dim.saturating_mul(7)) % 97 == 11 {
                        let sign = if (t + dim) % 2 == 0 { 1.0 } else { -1.0 };
                        v += sign * 20.0;
                    }
                    v
                }
                FixtureKind::HeavyTail => {
                    let base = if t < true_change { -1.0 } else { 1.5 };
                    let seasonal = 0.05 * ((t as f64) * 0.009 + dim as f64).cos();
                    let heavy_tail = 0.35 * cauchy_like_noise(t, dim);
                    base + seasonal + heavy_tail
                }
            };
            values.push(value);
        }
    }
    values
}

fn benchmark_student_t_robustness(c: &mut Criterion) {
    let normal_model = CostNormalMeanVar::default();
    let student_t_model = CostStudentT::default();
    let mut group = c.benchmark_group(GROUP_NAME);
    let exact = exact_filter();
    let start = N / 5;
    let end = (4 * N) / 5;

    for kind in [FixtureKind::Outlier, FixtureKind::HeavyTail] {
        for d in BENCH_DIMS {
            let normal_name = format!("normal_{}_segment_cost_n1e5_d{d}", kind.id_suffix());
            let student_exact_name = format!(
                "student_t_exact_{}_segment_cost_n1e5_d{d}",
                kind.id_suffix()
            );
            let student_approx_name = format!(
                "student_t_approx_{}_segment_cost_n1e5_d{d}_tol0p10",
                kind.id_suffix()
            );

            let run_normal = should_run_bench(&normal_name, exact.as_deref());
            let run_student_exact = should_run_bench(&student_exact_name, exact.as_deref());
            let run_student_approx = should_run_bench(&student_approx_name, exact.as_deref());
            if !(run_normal || run_student_exact || run_student_approx) {
                continue;
            }

            let values = generate_fixture_values(kind, N, d);
            let view = make_c_contiguous_view(values.as_slice(), N, d);

            if run_normal {
                let cache = normal_model
                    .precompute(&view, &CachePolicy::Full)
                    .expect("normal precompute should succeed");
                group.bench_function(normal_name, |b| {
                    b.iter(|| {
                        normal_model.segment_cost(
                            black_box(&cache),
                            black_box(start),
                            black_box(end),
                        )
                    })
                });
            }

            if run_student_exact {
                let cache = student_t_model
                    .precompute(&view, &CachePolicy::Full)
                    .expect("Student-t exact precompute should succeed");
                group.bench_function(student_exact_name, |b| {
                    b.iter(|| {
                        student_t_model.segment_cost(
                            black_box(&cache),
                            black_box(start),
                            black_box(end),
                        )
                    })
                });
            }

            if run_student_approx {
                let cache = student_t_model
                    .precompute(
                        &view,
                        &CachePolicy::Approximate {
                            max_bytes: student_t_model.worst_case_cache_bytes(&view),
                            error_tolerance: APPROX_ERROR_TOLERANCE,
                        },
                    )
                    .expect("Student-t approximate precompute should succeed");
                group.bench_function(student_approx_name, |b| {
                    b.iter(|| {
                        student_t_model.segment_cost(
                            black_box(&cache),
                            black_box(start),
                            black_box(end),
                        )
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_student_t_robustness);
criterion_main!(benches);
