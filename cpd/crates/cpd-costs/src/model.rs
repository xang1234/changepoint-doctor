// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    CachePolicy, CpdError, MissingSupport, TimeSeriesView, check_missing_compatibility,
};

/// Shared contract for cost models used by offline and online change-point algorithms.
///
/// Segment conventions use half-open intervals: `[start, end)`.
pub trait CostModel {
    type Cache: Send + Sync;

    fn name(&self) -> &'static str;

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError>;

    fn missing_support(&self) -> MissingSupport {
        MissingSupport::Reject
    }

    fn precompute(
        &self,
        x: &TimeSeriesView<'_>,
        policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError>;

    fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize;

    fn supports_approx_cache(&self) -> bool {
        false
    }

    /// Effective per-dimension parameter count used by BIC/AIC penalties.
    ///
    /// Detectors multiply this by series dimensionality (`d`) to get the
    /// effective model complexity term.
    fn penalty_params_per_segment(&self) -> usize {
        2
    }

    /// Returns the cost for segment `[start, end)`.
    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64;

    /// Optional bulk fast-path. Default behavior loops over `segment_cost`.
    fn segment_cost_batch(
        &self,
        cache: &Self::Cache,
        queries: &[(usize, usize)],
        out_costs: &mut [f64],
    ) {
        assert_eq!(
            queries.len(),
            out_costs.len(),
            "segment_cost_batch length mismatch: queries={}, out_costs={}",
            queries.len(),
            out_costs.len()
        );

        for (idx, (start, end)) in queries.iter().copied().enumerate() {
            out_costs[idx] = self.segment_cost(cache, start, end);
        }
    }
}

/// A model plus precomputed cache for repeated segment-cost queries.
#[derive(Debug)]
pub struct CachedCost<C: CostModel> {
    model: C,
    cache: C::Cache,
}

impl<C: CostModel> CachedCost<C> {
    /// Validates data compatibility and materializes the model cache.
    pub fn new(model: C, x: &TimeSeriesView<'_>, policy: &CachePolicy) -> Result<Self, CpdError> {
        model.validate(x)?;
        check_missing_compatibility(x.missing, model.missing_support())?;
        let cache = model.precompute(x, policy)?;
        Ok(Self { model, cache })
    }

    /// Creates a cached wrapper from already-constructed parts.
    pub fn from_parts(model: C, cache: C::Cache) -> Self {
        Self { model, cache }
    }

    /// Returns a shared reference to the underlying model.
    pub fn model(&self) -> &C {
        &self.model
    }

    /// Returns a shared reference to the precomputed cache.
    pub fn cache(&self) -> &C::Cache {
        &self.cache
    }

    /// Decomposes into `(model, cache)`.
    pub fn into_parts(self) -> (C, C::Cache) {
        (self.model, self.cache)
    }

    /// Returns the cost for segment `[start, end)`.
    pub fn segment_cost(&self, start: usize, end: usize) -> f64 {
        self.model.segment_cost(&self.cache, start, end)
    }

    /// Computes costs for many `[start, end)` queries.
    pub fn segment_cost_batch(&self, queries: &[(usize, usize)], out_costs: &mut [f64]) {
        self.model
            .segment_cost_batch(&self.cache, queries, out_costs);
    }
}

#[cfg(test)]
mod tests {
    use super::{CachedCost, CostModel};
    use cpd_core::{
        CachePolicy, CpdError, DTypeView, MemoryLayout, MissingPolicy, MissingSupport, TimeIndex,
        TimeSeriesView,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct CallCounters {
        validate_calls: AtomicUsize,
        precompute_calls: AtomicUsize,
        segment_cost_calls: AtomicUsize,
    }

    #[derive(Clone, Debug)]
    struct MockCostModel {
        name: &'static str,
        counters: Arc<CallCounters>,
        missing_support: Option<MissingSupport>,
    }

    impl MockCostModel {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                counters: Arc::new(CallCounters::default()),
                missing_support: None,
            }
        }

        fn with_missing_support(mut self, missing_support: MissingSupport) -> Self {
            self.missing_support = Some(missing_support);
            self
        }
    }

    impl CostModel for MockCostModel {
        type Cache = Vec<f64>;

        fn name(&self) -> &'static str {
            self.name
        }

        fn validate(&self, _x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
            self.counters.validate_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn missing_support(&self) -> MissingSupport {
            self.missing_support.unwrap_or(MissingSupport::Reject)
        }

        fn precompute(
            &self,
            x: &TimeSeriesView<'_>,
            policy: &CachePolicy,
        ) -> Result<Self::Cache, CpdError> {
            self.counters
                .precompute_calls
                .fetch_add(1, Ordering::SeqCst);
            let policy_flag = match policy {
                CachePolicy::Full => 0.0,
                CachePolicy::Budgeted { .. } => 1.0,
                CachePolicy::Approximate { .. } => 2.0,
            };
            Ok(vec![x.n as f64 + policy_flag])
        }

        fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize {
            (x.n + 1) * std::mem::size_of::<f64>()
        }

        fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64 {
            self.counters
                .segment_cost_calls
                .fetch_add(1, Ordering::SeqCst);
            cache[0] + start as f64 + end as f64
        }
    }

    fn make_view<'a>(values: &'a [f64], missing: MissingPolicy) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F64(values),
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    #[test]
    fn trait_defaults_match_contract() {
        let model = MockCostModel::new("mock-defaults");
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
        assert_eq!(model.penalty_params_per_segment(), 2);
    }

    #[test]
    fn default_batch_path_delegates_to_segment_cost() {
        let model = MockCostModel::new("mock-batch");
        let cache = vec![10.0];
        let queries = [(0, 1), (2, 4), (3, 9)];
        let mut out = vec![0.0; queries.len()];

        model.segment_cost_batch(&cache, &queries, &mut out);

        assert_eq!(out, vec![11.0, 16.0, 22.0]);
        assert_eq!(
            model.counters.segment_cost_calls.load(Ordering::SeqCst),
            queries.len()
        );
    }

    #[test]
    #[should_panic(expected = "segment_cost_batch length mismatch")]
    fn default_batch_path_panics_on_length_mismatch() {
        let model = MockCostModel::new("mock-batch-mismatch");
        let cache = vec![1.0];
        let queries = [(0, 1), (1, 2)];
        let mut out = vec![0.0; 1];
        model.segment_cost_batch(&cache, &queries, &mut out);
    }

    #[test]
    fn cached_cost_new_validates_compatibility_and_precomputes() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let view = make_view(&values, MissingPolicy::Error);
        let model = MockCostModel::new("mock-new");
        let counters = Arc::clone(&model.counters);

        let cached = CachedCost::new(model, &view, &CachePolicy::Budgeted { max_bytes: 128 })
            .expect("cached construction should succeed");

        assert_eq!(counters.validate_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counters.precompute_calls.load(Ordering::SeqCst), 1);
        assert_eq!(cached.cache(), &vec![5.0]);
    }

    #[test]
    fn cached_cost_new_rejects_missing_policy_mismatch() {
        let values = [1.0, 2.0, 3.0];
        let view = make_view(&values, MissingPolicy::Ignore);
        let model = MockCostModel::new("mock-reject-missing");

        let err = CachedCost::new(model, &view, &CachePolicy::Full)
            .expect_err("Ignore + Reject support should fail");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("policy=Ignore"));
    }

    #[test]
    fn cached_cost_delegates_segment_cost_and_batch() {
        let model =
            MockCostModel::new("mock-delegate").with_missing_support(MissingSupport::MaskAware);
        let cached = CachedCost::from_parts(model.clone(), vec![4.0]);

        let single = cached.segment_cost(2, 5);
        assert_eq!(single, 11.0);

        let queries = [(0, 1), (3, 4)];
        let mut out = vec![0.0; queries.len()];
        cached.segment_cost_batch(&queries, &mut out);

        assert_eq!(out, vec![5.0, 11.0]);
        assert_eq!(model.counters.segment_cost_calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn cached_cost_from_parts_accessors_and_into_parts_roundtrip() {
        let model = MockCostModel::new("mock-roundtrip");
        let cached = CachedCost::from_parts(model, vec![42.0]);

        assert_eq!(cached.model().name(), "mock-roundtrip");
        assert_eq!(cached.cache().as_slice(), &[42.0]);

        let (model, cache) = cached.into_parts();
        assert_eq!(model.name(), "mock-roundtrip");
        assert_eq!(cache, vec![42.0]);
    }
}
