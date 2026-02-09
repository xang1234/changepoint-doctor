// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use cpd_core::{DTypeView, MemoryLayout, TimeIndex, TimeSeriesView};
use libfuzzer_sys::fuzz_target;

fn as_stride(seed: u8) -> isize {
    isize::from((i16::from(seed % 9)) - 4)
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    let n_seed = cursor.next_u8();
    let d_seed = cursor.next_u8();
    let dtype_seed = cursor.next_u8();
    let value_shape_seed = cursor.next_u8();
    let layout_seed = cursor.next_u8();
    let missing_seed = cursor.next_u8();
    let mask_mode = cursor.next_u8();
    let time_seed = cursor.next_u8();
    let dt_seed = cursor.next_i16();
    let inject_inf = cursor.next_u8() & 1 == 0;

    let payload_len = common::bounded(cursor.next_u8(), 0, 160).saturating_mul(8);
    let mask_len = common::bounded(cursor.next_u8(), 0, 128);
    let explicit_count = common::bounded(cursor.next_u8(), 0, 64);

    let payload = cursor.take_padded(payload_len);
    let mask_bytes = cursor.take_padded(mask_len);
    let mut explicit_time = Vec::with_capacity(explicit_count);
    for _ in 0..explicit_count {
        explicit_time.push(cursor.next_i64());
    }

    let n = usize::from(n_seed % 33);
    let d = usize::from(d_seed % 9);
    let expected_len = n.saturating_mul(d).min(common::MAX_VALUE_LEN);

    let mut values_f64 = common::decode_f64_chunks(&payload, common::MAX_VALUE_LEN);
    let requested_value_len = match value_shape_seed % 5 {
        0 => expected_len,
        1 => expected_len.saturating_sub(1),
        2 => expected_len.saturating_add(1),
        3 => usize::from(value_shape_seed),
        _ => values_f64.len(),
    }
    .min(common::MAX_VALUE_LEN);
    common::ensure_len(&mut values_f64, requested_value_len);
    values_f64.truncate(requested_value_len);

    if inject_inf && !values_f64.is_empty() {
        values_f64[0] = if layout_seed & 1 == 0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }

    let values_f32 = common::decode_f32_from_f64(&values_f64);

    let mask_storage = if mask_mode % 4 == 0 {
        None
    } else {
        let target_mask_len = match mask_mode % 4 {
            1 => expected_len,
            2 => expected_len.saturating_add(1),
            _ => usize::from(n_seed),
        }
        .min(common::MAX_VALUE_LEN);

        let mut mask = mask_bytes;
        mask.resize(target_mask_len, 0);
        if mask_mode & 1 == 0 {
            for value in &mut mask {
                *value &= 1;
            }
        }
        Some(mask)
    };

    let explicit_len = match time_seed % 3 {
        0 => n,
        1 => n.saturating_add(1),
        _ => usize::from(d_seed),
    }
    .min(common::MAX_VALUE_LEN);
    common::ensure_len_i64(&mut explicit_time, explicit_len);
    explicit_time.truncate(explicit_len);

    let time_index = match time_seed % 4 {
        0 => TimeIndex::None,
        1 => TimeIndex::Uniform {
            t0_ns: i64::from(dt_seed),
            dt_ns: i64::from(dt_seed),
        },
        2 => TimeIndex::Explicit(explicit_time.as_slice()),
        _ => TimeIndex::Uniform { t0_ns: 0, dt_ns: 0 },
    };

    let layout = match layout_seed % 4 {
        0 => MemoryLayout::CContiguous,
        1 => MemoryLayout::FContiguous,
        2 => MemoryLayout::Strided {
            row_stride: as_stride(layout_seed),
            col_stride: as_stride(time_seed),
        },
        _ => MemoryLayout::Strided {
            row_stride: 0,
            col_stride: 1,
        },
    };

    let missing = common::choose_missing_policy(missing_seed);
    let mask_ref = mask_storage.as_deref();

    if dtype_seed & 1 == 0 {
        let _ = TimeSeriesView::new(
            DTypeView::F64(values_f64.as_slice()),
            n,
            d,
            layout,
            mask_ref,
            time_index,
            missing,
        );
    } else {
        let _ = TimeSeriesView::new(
            DTypeView::F32(values_f32.as_slice()),
            n,
            d,
            layout,
            mask_ref,
            time_index,
            missing,
        );
    }
});
