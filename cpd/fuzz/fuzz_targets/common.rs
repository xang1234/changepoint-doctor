// SPDX-License-Identifier: MIT OR Apache-2.0

use cpd_core::{MissingPolicy, Penalty, Stopping};

pub const MAX_VALUE_LEN: usize = 2048;

pub struct ByteCursor<'a> {
    data: &'a [u8],
    idx: usize,
}

impl<'a> ByteCursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, idx: 0 }
    }

    pub fn next_u8(&mut self) -> u8 {
        let value = self.data.get(self.idx).copied().unwrap_or(0);
        self.idx = self.idx.saturating_add(1);
        value
    }

    pub fn next_i16(&mut self) -> i16 {
        let bytes = [self.next_u8(), self.next_u8()];
        i16::from_le_bytes(bytes)
    }

    pub fn next_i64(&mut self) -> i64 {
        let bytes = [
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
            self.next_u8(),
        ];
        i64::from_le_bytes(bytes)
    }

    pub fn take_padded(&mut self, len: usize) -> Vec<u8> {
        let mut out = vec![0_u8; len];
        let available = self.data.len().saturating_sub(self.idx);
        let copy_len = available.min(len);
        if copy_len > 0 {
            let start = self.idx;
            let end = start + copy_len;
            out[..copy_len].copy_from_slice(&self.data[start..end]);
            self.idx = end;
        }
        out
    }

    pub fn remaining(&self) -> &[u8] {
        if self.idx >= self.data.len() {
            &[]
        } else {
            &self.data[self.idx..]
        }
    }
}

pub fn decode_f64_chunks(bytes: &[u8], max_values: usize) -> Vec<f64> {
    let capped = max_values.min(MAX_VALUE_LEN);
    let mut out = Vec::with_capacity(capped);

    for chunk in bytes.chunks_exact(8).take(capped) {
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(chunk);
        let value = f64::from_le_bytes(raw);
        if value.is_finite() {
            out.push(value.clamp(-1.0e9, 1.0e9));
        } else {
            out.push(0.0);
        }
    }

    out
}

pub fn decode_f32_from_f64(values: &[f64]) -> Vec<f32> {
    values.iter().copied().map(|v| v as f32).collect()
}

pub fn ensure_len(values: &mut Vec<f64>, len: usize) {
    let target = len.min(MAX_VALUE_LEN);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    while values.len() < target {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let scaled = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
        values.push((scaled * 2_000.0) - 1_000.0);
    }
}

pub fn ensure_len_i64(values: &mut Vec<i64>, len: usize) {
    let target = len.min(MAX_VALUE_LEN);
    let mut cursor = values.last().copied().unwrap_or(0);
    while values.len() < target {
        cursor = cursor.saturating_add(1_000_000);
        values.push(cursor);
    }
}

pub fn choose_missing_policy(seed: u8) -> MissingPolicy {
    match seed % 4 {
        0 => MissingPolicy::Error,
        1 => MissingPolicy::ImputeZero,
        2 => MissingPolicy::ImputeLast,
        _ => MissingPolicy::Ignore,
    }
}

pub fn choose_stopping(kind: u8, beta_seed: u8, path_seed: u8) -> Stopping {
    match kind % 5 {
        0 => Stopping::KnownK(usize::from(path_seed % 8)),
        1 => Stopping::Penalized(Penalty::BIC),
        2 => Stopping::Penalized(Penalty::AIC),
        3 => {
            let beta = (f64::from(beta_seed) / 12.0) - 2.0;
            Stopping::Penalized(Penalty::Manual(beta))
        }
        _ => {
            if path_seed % 3 == 0 {
                Stopping::PenaltyPath(vec![])
            } else {
                let beta = (f64::from(beta_seed) / 12.0) - 2.0;
                Stopping::PenaltyPath(vec![Penalty::BIC, Penalty::Manual(beta)])
            }
        }
    }
}

pub fn bounded(seed: u8, min: usize, max_inclusive: usize) -> usize {
    if max_inclusive <= min {
        min
    } else {
        min + (usize::from(seed) % (max_inclusive - min + 1))
    }
}
