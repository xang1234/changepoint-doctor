// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Optional callback for reporting algorithm progress in `[0.0, 1.0]`.
pub trait ProgressSink: Send + Sync {
    fn on_progress(&self, fraction: f32);
}

/// Optional sink for low-overhead scalar telemetry.
pub trait TelemetrySink: Send + Sync {
    fn record_scalar(&self, key: &'static str, value: f64);
}

/// No-op progress sink for call sites that want an explicit sink object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoopProgressSink;

impl ProgressSink for NoopProgressSink {
    fn on_progress(&self, _fraction: f32) {}
}

/// No-op telemetry sink for call sites that want an explicit sink object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoopTelemetrySink;

impl TelemetrySink for NoopTelemetrySink {
    fn record_scalar(&self, _key: &'static str, _value: f64) {}
}

#[cfg(test)]
mod tests {
    use super::{NoopProgressSink, NoopTelemetrySink, ProgressSink, TelemetrySink};

    #[test]
    fn noop_sinks_accept_calls_without_panicking() {
        let progress = NoopProgressSink;
        let telemetry = NoopTelemetrySink;

        progress.on_progress(0.0);
        progress.on_progress(0.5);
        progress.on_progress(1.0);

        telemetry.record_scalar("runtime_ms", 12.5);
        telemetry.record_scalar("cost_evals", 42.0);
    }
}
