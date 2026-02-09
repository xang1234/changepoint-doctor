// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use cpd_core::{
    CachePolicy, Constraints, DTypeView, DegradationStep, ExecutionContext, MemoryLayout,
    MissingPolicy, OfflineDetector, TimeIndex, TimeSeriesView,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{BinSeg, BinSegConfig, Pelt, PeltConfig};
use libfuzzer_sys::fuzz_target;

fn build_cache_policy(seed: u8, n: usize, d: usize) -> CachePolicy {
    let required = n
        .saturating_add(1)
        .saturating_mul(d)
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_mul(2);

    match seed % 4 {
        0 => CachePolicy::Full,
        1 => CachePolicy::Budgeted {
            max_bytes: required.saturating_add(64),
        },
        2 => CachePolicy::Budgeted {
            max_bytes: usize::from(seed & 0x0F),
        },
        _ => CachePolicy::Approximate {
            max_bytes: required.saturating_add(64),
            error_tolerance: if seed & 1 == 0 { 0.05 } else { 0.0 },
        },
    }
}

fn build_degradation(seed: u8) -> Vec<DegradationStep> {
    match seed % 4 {
        0 => vec![],
        1 => vec![DegradationStep::IncreaseJump {
            factor: usize::from(seed & 0x07),
            max_jump: usize::from((seed >> 1) & 0x0F),
        }],
        2 => vec![DegradationStep::DisableUncertaintyBands],
        _ => vec![DegradationStep::SwitchCachePolicy(CachePolicy::Budgeted {
            max_bytes: usize::from(seed & 0x03),
        })],
    }
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    let n_seed = cursor.next_u8();
    let d_seed = cursor.next_u8();
    let min_segment_seed = cursor.next_u8();
    let jump_seed = cursor.next_u8();
    let options_seed = cursor.next_u8();
    let cache_seed = cursor.next_u8();
    let stopping_seed = cursor.next_u8();
    let beta_seed = cursor.next_u8();
    let path_seed = cursor.next_u8();
    let params_seed = cursor.next_u8();
    let cancel_seed = cursor.next_u8();
    let time_seed = cursor.next_u8();

    let payload_len = common::bounded(cursor.next_u8(), 1, 192).saturating_mul(8);
    let candidate_len = common::bounded(cursor.next_u8(), 0, 64);
    let explicit_count = common::bounded(cursor.next_u8(), 0, 64);

    let payload = cursor.take_padded(payload_len);
    let candidate_bytes = cursor.take_padded(candidate_len);

    let mut explicit_time = Vec::with_capacity(explicit_count);
    for _ in 0..explicit_count {
        explicit_time.push(cursor.next_i64());
    }

    let n = common::bounded(n_seed, 1, 128);
    let d = common::bounded(d_seed, 1, 3);
    let expected_len = n.saturating_mul(d).min(common::MAX_VALUE_LEN);

    let mut values = common::decode_f64_chunks(&payload, expected_len);
    common::ensure_len(&mut values, expected_len);
    values.truncate(expected_len);

    let explicit_len = if time_seed & 1 == 0 {
        n
    } else {
        n.saturating_add(1)
    }
    .min(common::MAX_VALUE_LEN);
    common::ensure_len_i64(&mut explicit_time, explicit_len);
    explicit_time.truncate(explicit_len);

    let time = match time_seed % 3 {
        0 => TimeIndex::None,
        1 => TimeIndex::Uniform {
            t0_ns: 0,
            dt_ns: i64::from((time_seed % 7) as i8) - 3,
        },
        _ => TimeIndex::Explicit(explicit_time.as_slice()),
    };

    let view = TimeSeriesView::new(
        DTypeView::F64(values.as_slice()),
        n,
        d,
        MemoryLayout::CContiguous,
        None,
        time,
        MissingPolicy::Error,
    );
    let Ok(view) = view else {
        return;
    };

    let mut candidate_splits = candidate_bytes
        .iter()
        .take(24)
        .map(|v| usize::from(*v) % n.saturating_add(1))
        .collect::<Vec<_>>();
    if options_seed & 1 == 0 {
        candidate_splits.sort_unstable();
        candidate_splits.dedup();
    }

    let constraints = Constraints {
        min_segment_len: usize::from(min_segment_seed % 8),
        max_change_points: if options_seed & 2 == 0 {
            None
        } else {
            Some(usize::from(options_seed % 10))
        },
        max_depth: if options_seed & 4 == 0 {
            None
        } else {
            Some(usize::from(options_seed % 6))
        },
        candidate_splits: if options_seed & 8 == 0 {
            None
        } else {
            Some(candidate_splits)
        },
        jump: usize::from(jump_seed % 8),
        time_budget_ms: if options_seed & 16 == 0 {
            None
        } else {
            Some(u64::from(options_seed % 4))
        },
        max_cost_evals: if options_seed & 32 == 0 {
            None
        } else {
            Some(usize::from(options_seed % 6))
        },
        memory_budget_bytes: if options_seed & 64 == 0 {
            None
        } else {
            Some(usize::from(options_seed % 64))
        },
        max_cache_bytes: if options_seed & 128 == 0 {
            None
        } else {
            Some(usize::from(options_seed % 64))
        },
        cache_policy: build_cache_policy(cache_seed, n, d),
        degradation_plan: build_degradation(cache_seed),
        allow_algorithm_fallback: options_seed & 1 != 0,
    };

    let ctx = ExecutionContext::new(&constraints);
    let stopping = common::choose_stopping(stopping_seed, beta_seed, path_seed);

    let pelt_config = PeltConfig {
        stopping: stopping.clone(),
        params_per_segment: usize::from(params_seed % 5),
        cancel_check_every: usize::from(cancel_seed),
    };

    let binseg_config = BinSegConfig {
        stopping,
        params_per_segment: usize::from((params_seed >> 1) % 5),
        cancel_check_every: usize::from(cancel_seed.wrapping_add(1)),
    };

    if options_seed & 1 == 0 {
        if let Ok(detector) = Pelt::new(CostL2Mean::default(), pelt_config.clone()) {
            let _ = detector.detect(&view, &ctx);
        }
        if let Ok(detector) = BinSeg::new(CostL2Mean::default(), binseg_config) {
            let _ = detector.detect(&view, &ctx);
        }
    } else {
        if let Ok(detector) = Pelt::new(CostNormalMeanVar::default(), pelt_config) {
            let _ = detector.detect(&view, &ctx);
        }
        if let Ok(detector) = BinSeg::new(CostNormalMeanVar::default(), binseg_config) {
            let _ = detector.detect(&view, &ctx);
        }
    }
});
