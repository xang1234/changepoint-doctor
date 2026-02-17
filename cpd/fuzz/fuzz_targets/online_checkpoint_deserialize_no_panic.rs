// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
use cpd_online::{
    decode_checkpoint_envelope, encode_checkpoint_envelope, load_bocpd_checkpoint,
    load_cusum_checkpoint, load_page_hinkley_checkpoint, save_bocpd_checkpoint,
    save_cusum_checkpoint, save_page_hinkley_checkpoint, BocpdConfig, BocpdDetector,
    CheckpointEnvelope, CusumConfig, CusumDetector, PageHinkleyConfig, PageHinkleyDetector,
    PayloadCodec,
};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const CHECKPOINT_V0_FIXTURE: &str =
    include_str!("../../tests/fixtures/migrations/checkpoint/online_detector_checkpoint.v0.json");
const CHECKPOINT_V1_FIXTURE: &str =
    include_str!("../../tests/fixtures/migrations/checkpoint/online_detector_checkpoint.v1.json");

fn ctx() -> ExecutionContext<'static> {
    static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
    let constraints = CONSTRAINTS.get_or_init(Constraints::default);
    ExecutionContext::new(constraints)
}

fn payload_codec_from_seed(seed: u8) -> PayloadCodec {
    if seed & 1 == 0 {
        PayloadCodec::Json
    } else {
        PayloadCodec::Bincode
    }
}

fn should_replay_fixture(data: &[u8], byte_index: usize) -> bool {
    data.get(byte_index).copied().unwrap_or(u8::MAX) % 4 == 0
}

fn next_observation(cursor: &mut common::ByteCursor<'_>) -> f64 {
    (f64::from(cursor.next_i16()) / 16.0).clamp(-1_000.0, 1_000.0)
}

fn next_timestamp(cursor: &mut common::ByteCursor<'_>) -> Option<i64> {
    match cursor.next_u8() % 5 {
        0 => None,
        1 => Some(i64::from(cursor.next_i16())),
        2 => Some(i64::from(cursor.next_u8())),
        3 => Some(cursor.next_i64()),
        _ => Some(0),
    }
}

fn advance_detector<D: OnlineDetector>(detector: &mut D, cursor: &mut common::ByteCursor<'_>) {
    let steps = common::bounded(cursor.next_u8(), 1, 24);
    for _ in 0..steps {
        let x = next_observation(cursor);
        let t_ns = next_timestamp(cursor);
        let _ = detector.update(&[x], t_ns, &ctx());

        if cursor.next_u8() % 11 == 0 {
            let saved = detector.save_state();
            detector.load_state(&saved);
        }
    }
}

fn mutate_input(base: &[u8], cursor: &mut common::ByteCursor<'_>) -> Vec<u8> {
    let mut out = base.to_vec();
    let flips = common::bounded(cursor.next_u8(), 0, 24);
    for _ in 0..flips {
        if out.is_empty() {
            break;
        }
        let idx = usize::from(cursor.next_u8()) % out.len();
        out[idx] ^= cursor.next_u8();
    }

    match cursor.next_u8() % 5 {
        0 => {}
        1 => {
            let new_len = common::bounded(cursor.next_u8(), 0, out.len());
            out.truncate(new_len);
        }
        2 => {
            if !out.is_empty() {
                let drop_len = common::bounded(cursor.next_u8(), 0, out.len() - 1);
                out.drain(..drop_len);
            }
        }
        3 => {
            let append_len = common::bounded(cursor.next_u8(), 0, 128);
            let append = cursor.take_padded(append_len);
            out.extend_from_slice(&append);
        }
        _ => {
            if out.len() >= 2 {
                let i = usize::from(cursor.next_u8()) % out.len();
                let j = usize::from(cursor.next_u8()) % out.len();
                out.swap(i, j);
            }
        }
    }

    out
}

fn build_encoded_checkpoint(cursor: &mut common::ByteCursor<'_>) -> Option<Vec<u8>> {
    let codec = payload_codec_from_seed(cursor.next_u8());
    let envelope = match cursor.next_u8() % 3 {
        0 => {
            let mut detector = BocpdDetector::new(BocpdConfig::default()).ok()?;
            advance_detector(&mut detector, cursor);
            save_bocpd_checkpoint(&detector, codec).ok()?
        }
        1 => {
            let mut detector = CusumDetector::new(CusumConfig::default()).ok()?;
            advance_detector(&mut detector, cursor);
            save_cusum_checkpoint(&detector, codec).ok()?
        }
        _ => {
            let mut detector = PageHinkleyDetector::new(PageHinkleyConfig::default()).ok()?;
            advance_detector(&mut detector, cursor);
            save_page_hinkley_checkpoint(&detector, codec).ok()?
        }
    };

    encode_checkpoint_envelope(&envelope).ok()
}

fn exercise_restore_paths(envelope: &CheckpointEnvelope, cursor: &mut common::ByteCursor<'_>) {
    if let Ok(mut detector) = BocpdDetector::new(BocpdConfig::default()) {
        if load_bocpd_checkpoint(&mut detector, envelope).is_ok() {
            advance_detector(&mut detector, cursor);
            let _ = save_bocpd_checkpoint(&detector, payload_codec_from_seed(cursor.next_u8()));
        }
    }

    if let Ok(mut detector) = CusumDetector::new(CusumConfig::default()) {
        if load_cusum_checkpoint(&mut detector, envelope).is_ok() {
            advance_detector(&mut detector, cursor);
            let _ = save_cusum_checkpoint(&detector, payload_codec_from_seed(cursor.next_u8()));
        }
    }

    if let Ok(mut detector) = PageHinkleyDetector::new(PageHinkleyConfig::default()) {
        if load_page_hinkley_checkpoint(&mut detector, envelope).is_ok() {
            advance_detector(&mut detector, cursor);
            let _ =
                save_page_hinkley_checkpoint(&detector, payload_codec_from_seed(cursor.next_u8()));
        }
    }
}

fn try_decode_and_restore(candidate: &[u8], cursor: &mut common::ByteCursor<'_>) {
    if let Ok(envelope) = decode_checkpoint_envelope(candidate) {
        let _ = encode_checkpoint_envelope(&envelope);
        exercise_restore_paths(&envelope, cursor);
    }
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    try_decode_and_restore(data, &mut cursor);
    if should_replay_fixture(data, 0) {
        try_decode_and_restore(CHECKPOINT_V0_FIXTURE.as_bytes(), &mut cursor);
    }
    if should_replay_fixture(data, 1) {
        try_decode_and_restore(CHECKPOINT_V1_FIXTURE.as_bytes(), &mut cursor);
    }

    let mutated_v0 = mutate_input(CHECKPOINT_V0_FIXTURE.as_bytes(), &mut cursor);
    try_decode_and_restore(&mutated_v0, &mut cursor);

    let mutated_v1 = mutate_input(CHECKPOINT_V1_FIXTURE.as_bytes(), &mut cursor);
    try_decode_and_restore(&mutated_v1, &mut cursor);

    if let Some(encoded) = build_encoded_checkpoint(&mut cursor) {
        try_decode_and_restore(&encoded, &mut cursor);

        let variants = common::bounded(cursor.next_u8(), 1, 8);
        for _ in 0..variants {
            let mutated = mutate_input(&encoded, &mut cursor);
            try_decode_and_restore(&mutated, &mut cursor);
        }
    }

    let tail_len = common::bounded(cursor.next_u8(), 0, 256);
    let tail = cursor.take_padded(tail_len);
    try_decode_and_restore(&tail, &mut cursor);
});
