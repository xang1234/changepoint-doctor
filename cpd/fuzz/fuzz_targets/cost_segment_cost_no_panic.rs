// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, TimeIndex, TimeSeriesView};
use cpd_costs::{CostL2Mean, CostModel, CostNormalMeanVar};
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
        2 => CachePolicy::Budgeted { max_bytes: 1 },
        _ => CachePolicy::Approximate {
            max_bytes: required.saturating_add(64),
            error_tolerance: if seed & 1 == 0 { 0.05 } else { 0.0 },
        },
    }
}

fn build_queries(query_bytes: &[u8], n: usize) -> Vec<(usize, usize)> {
    if n < 2 {
        return vec![];
    }

    let mut queries = Vec::with_capacity(16);
    for chunk in query_bytes.chunks(2).take(16) {
        let start = usize::from(chunk[0]) % (n - 1);
        let width_seed = chunk.get(1).copied().unwrap_or(0);
        let span = 1 + (usize::from(width_seed) % (n - start));
        let end = start + span;
        queries.push((start, end));
    }

    if queries.is_empty() {
        queries.push((0, 1));
    }

    queries
}

fn exercise_model<C: CostModel>(
    model: C,
    view: &TimeSeriesView<'_>,
    policy: &CachePolicy,
    query_bytes: &[u8],
) {
    if model.validate(view).is_err() {
        return;
    }

    let Ok(cache) = model.precompute(view, policy) else {
        return;
    };

    let queries = build_queries(query_bytes, view.n);
    if queries.is_empty() {
        return;
    }

    for (start, end) in queries.iter().copied() {
        let _ = model.segment_cost(&cache, start, end);
    }

    let mut out = vec![0.0; queries.len()];
    model.segment_cost_batch(&cache, &queries, &mut out);
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    let n_seed = cursor.next_u8();
    let d_seed = cursor.next_u8();
    let dtype_seed = cursor.next_u8();
    let layout_seed = cursor.next_u8();
    let policy_seed = cursor.next_u8();
    let mask_seed = cursor.next_u8();

    let payload_len = common::bounded(cursor.next_u8(), 1, 192).saturating_mul(8);
    let query_len = common::bounded(cursor.next_u8(), 0, 64);
    let mask_len = common::bounded(cursor.next_u8(), 0, 160);

    let payload = cursor.take_padded(payload_len);
    let query_bytes = cursor.take_padded(query_len);
    let mask_bytes = cursor.take_padded(mask_len);

    let n = common::bounded(n_seed, 1, 128);
    let d = common::bounded(d_seed, 1, 4);
    let expected_len = n.saturating_mul(d).min(common::MAX_VALUE_LEN);

    let mut values_f64 = common::decode_f64_chunks(&payload, expected_len);
    common::ensure_len(&mut values_f64, expected_len);
    values_f64.truncate(expected_len);
    let values_f32 = common::decode_f32_from_f64(&values_f64);

    let layout = match layout_seed % 4 {
        0 => MemoryLayout::CContiguous,
        1 => MemoryLayout::FContiguous,
        2 => MemoryLayout::Strided {
            row_stride: isize::try_from(d).unwrap_or(1),
            col_stride: 1,
        },
        _ => MemoryLayout::Strided {
            row_stride: 0,
            col_stride: 1,
        },
    };

    let missing_mask = if mask_seed % 3 == 0 {
        None
    } else {
        let target_mask_len = if mask_seed & 1 == 0 {
            expected_len
        } else {
            expected_len.saturating_add(1)
        }
        .min(common::MAX_VALUE_LEN);

        let mut mask = mask_bytes;
        mask.resize(target_mask_len, 0);
        if mask_seed & 1 == 0 {
            for value in &mut mask {
                *value &= 1;
            }
        }
        Some(mask)
    };

    let missing = common::choose_missing_policy(mask_seed);
    let mask_ref = missing_mask.as_deref();

    let view = if dtype_seed & 1 == 0 {
        TimeSeriesView::new(
            DTypeView::F64(values_f64.as_slice()),
            n,
            d,
            layout,
            mask_ref,
            TimeIndex::None,
            missing,
        )
    } else {
        TimeSeriesView::new(
            DTypeView::F32(values_f32.as_slice()),
            n,
            d,
            layout,
            mask_ref,
            TimeIndex::None,
            missing,
        )
    };

    let Ok(view) = view else {
        return;
    };

    let policy = build_cache_policy(policy_seed, n, d);
    exercise_model(CostL2Mean::default(), &view, &policy, &query_bytes);
    exercise_model(CostNormalMeanVar::default(), &view, &policy, &query_bytes);
});
