# Rust Usage Examples

Using `changepoint-doctor` crates directly from Rust.

## Running PELT from Rust

```rust
use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout,
    MissingPolicy, ReproMode, TimeIndex, TimeSeriesView,
};
use cpd_costs::{CostL2Mean, CostModel, CachedCost};
use cpd_core::CachePolicy;
use cpd_offline::{Pelt, PeltConfig};
use cpd_core::{OfflineDetector, Stopping, Penalty};

fn main() -> Result<(), cpd_core::CpdError> {
    // Build a test signal: 3 segments
    let mut signal = vec![0.0f64; 50];
    signal.extend(vec![5.0; 50]);
    signal.extend(vec![-2.0; 50]);

    // Create a zero-copy view
    let view = TimeSeriesView::new(
        DTypeView::F64(&signal),
        signal.len(),
        1,  // univariate
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )?;

    // Configure constraints and execution context
    let constraints = Constraints {
        min_segment_len: 2,
        jump: 1,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints)
        .with_repro_mode(ReproMode::Balanced);

    // Build the detector with BIC penalty
    let config = PeltConfig {
        stopping: Stopping::Penalized(Penalty::Bic),
        ..PeltConfig::default()
    };
    let detector = Pelt::<CostL2Mean>::new(config, CostL2Mean);

    // Run detection
    let result = detector.detect(&view, &ctx)?;
    println!("Breakpoints: {:?}", result.breakpoints);
    println!("Algorithm: {}", result.diagnostics.algorithm);

    Ok(())
}
```

## Running BOCPD from Rust

```rust
use cpd_core::{Constraints, ExecutionContext, ReproMode};
use cpd_online::{Bocpd, BocpdConfig, GaussianNigPrior, ObservationModel};
use cpd_core::OnlineDetector;

fn main() -> Result<(), cpd_core::CpdError> {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints)
        .with_repro_mode(ReproMode::Balanced);

    let config = BocpdConfig {
        observation_model: ObservationModel::GaussianNig(GaussianNigPrior::default()),
        hazard: 1.0 / 200.0,
        max_run_length: 512,
        ..BocpdConfig::default()
    };

    let mut bocpd = Bocpd::new(config);

    // Process observations
    let observations = [0.1, 0.2, -0.1, 0.3, 5.2, 5.1, 4.9, 5.3];
    for (t, &x) in observations.iter().enumerate() {
        let step = bocpd.update(&[x], None, &ctx)?;
        if step.alert {
            println!("Alert at t={}: p_change={:.3}", t, step.p_change);
        }
    }

    // Checkpoint
    let state = bocpd.save_state();
    println!("State saved: {:?}", state);

    Ok(())
}
```

## Custom `CostModel` implementation

```rust
use cpd_core::{CachePolicy, CpdError, MissingSupport, TimeSeriesView};
use cpd_costs::CostModel;

/// Example: constant-cost model (useful as a template).
struct CostConstant {
    value: f64,
}

impl CostModel for CostConstant {
    type Cache = ();  // no precomputation needed

    fn name(&self) -> &'static str { "constant" }

    fn validate(&self, _x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        Ok(())
    }

    fn precompute(
        &self,
        _x: &TimeSeriesView<'_>,
        _policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError> {
        Ok(())
    }

    fn worst_case_cache_bytes(&self, _x: &TimeSeriesView<'_>) -> usize {
        0
    }

    fn segment_cost(&self, _cache: &Self::Cache, _start: usize, _end: usize) -> f64 {
        self.value
    }
}
```

The real implementation would typically:
1. Build prefix sums in `precompute()` for O(1) segment queries
2. Override `penalty_params_per_segment()` to match the model's parameter count
3. Override `missing_support()` if the model handles NaN values

## `ExecutionContext` configuration

```rust
use cpd_core::{
    CancelToken, Constraints, ExecutionContext, ReproMode,
    control::BudgetMode,
};
use std::sync::Arc;

// Basic context
let constraints = Constraints::default();
let ctx = ExecutionContext::new(&constraints);

// Full configuration
let cancel = CancelToken::new();
let ctx = ExecutionContext::new(&constraints)
    .with_cancel(&cancel)
    .with_budget_mode(BudgetMode::SoftDegrade)
    .with_repro_mode(ReproMode::Strict);

// Cancel from another thread
let cancel_clone = cancel.clone();
std::thread::spawn(move || {
    std::thread::sleep(std::time::Duration::from_secs(5));
    cancel_clone.cancel();
});
```
