// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::synthetic::{
    Ar1Config, BinaryConfig, CountConfig, HeavyTailConfig, MultivariateConfig,
    PiecewiseLinearConfig, PiecewiseMeanConfig, PiecewiseVarianceConfig, ar1_with_changes,
    binary_data, count_data, evenly_spaced_breakpoints, heavy_tailed, multivariate,
    piecewise_constant_mean, piecewise_constant_variance, piecewise_linear,
};
use cpd_core::CpdError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MANIFEST_JSON: &str = include_str!("../datasets/registry_manifest.v1.json");
const MANIFEST_VERSION: u32 = 1;

/// Dataset source category.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    /// Deterministic synthetic dataset generated from a fixed seed.
    Synthetic,
    /// Real-world public benchmark slice with license attribution metadata.
    RealWorld,
}

/// Public dataset metadata exposed by the registry API.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetMetadata {
    pub name: String,
    pub display_name: String,
    pub kind: DatasetKind,
    pub description: String,
    pub length: usize,
    pub dimensions: usize,
    pub true_change_points: Option<Vec<usize>>,
    pub license: Option<String>,
    pub source_url: Option<String>,
    pub citation: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct Manifest {
    manifest_version: u32,
    datasets: Vec<ManifestEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestEntry {
    name: String,
    display_name: String,
    kind: DatasetKind,
    description: String,
    length: usize,
    dimensions: usize,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    source_url: Option<String>,
    #[serde(default)]
    citation: Option<String>,
    #[serde(default)]
    synthetic: Option<SyntheticSpec>,
    #[serde(default)]
    real_world: Option<RealWorldSpec>,
}

#[derive(Clone, Debug, Deserialize)]
struct RealWorldSpec {
    fixture_path: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SyntheticSpec {
    PiecewiseConstantMean {
        n: usize,
        d: usize,
        n_changes: usize,
        snr: f64,
        noise_std: f64,
        seed: u64,
    },
    PiecewiseConstantVariance {
        n: usize,
        d: usize,
        n_changes: usize,
        mean: f64,
        base_std: f64,
        std_step: f64,
        seed: u64,
    },
    PiecewiseLinear {
        n: usize,
        d: usize,
        n_changes: usize,
        slope_step: f64,
        noise_std: f64,
        seed: u64,
    },
    Ar1WithChanges {
        n: usize,
        d: usize,
        n_changes: usize,
        phi: f64,
        mean_step: f64,
        base_std: f64,
        std_step: f64,
        seed: u64,
    },
    HeavyTailed {
        n: usize,
        d: usize,
        n_changes: usize,
        degrees_of_freedom: usize,
        mean_step: f64,
        scale: f64,
        seed: u64,
    },
    CountData {
        n: usize,
        d: usize,
        n_changes: usize,
        base_rate: f64,
        rate_step: f64,
        seed: u64,
    },
    BinaryData {
        n: usize,
        d: usize,
        n_changes: usize,
        base_prob: f64,
        prob_step: f64,
        seed: u64,
    },
    Multivariate {
        n: usize,
        d: usize,
        n_changes: usize,
        snr: f64,
        noise_std: f64,
        correlation: Option<f64>,
        seed: u64,
    },
}

#[derive(Clone, Debug, Deserialize)]
struct RealWorldFixture {
    values: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CachedDataset {
    manifest_version: u32,
    name: String,
    kind: DatasetKind,
    length: usize,
    dimensions: usize,
    values: Vec<f64>,
    #[serde(default)]
    true_change_points: Vec<usize>,
}

static REGISTRY_MANIFEST: OnceLock<Manifest> = OnceLock::new();

/// Returns all datasets declared in the registry manifest.
pub fn list_datasets() -> Result<Vec<DatasetMetadata>, CpdError> {
    manifest()?
        .datasets
        .iter()
        .map(ManifestEntry::list_metadata)
        .collect()
}

/// Loads a dataset by name and returns `(flattened_values, metadata)`.
///
/// Loading is lazy: the first load generates/reads and caches a local JSON copy,
/// and subsequent loads read from cache.
pub fn load_dataset(name: &str) -> Result<(Vec<f64>, DatasetMetadata), CpdError> {
    let cache_root = default_cache_root()?;
    load_dataset_with_cache_root(name, cache_root.as_path())
}

fn load_dataset_with_cache_root(
    name: &str,
    cache_root: &Path,
) -> Result<(Vec<f64>, DatasetMetadata), CpdError> {
    let manifest = manifest()?;
    let entry = manifest
        .datasets
        .iter()
        .find(|dataset| dataset.name == name)
        .ok_or_else(|| CpdError::invalid_input(format!("unknown dataset '{name}'")))?;

    fs::create_dir_all(cache_root)
        .map_err(|err| io_error("failed to create cache directory", cache_root, err))?;

    let cache_path = cache_path(cache_root, name);
    if let Some(cached) = try_read_cache(cache_path.as_path(), entry)? {
        let metadata = entry.loaded_metadata(cached.true_change_points.clone());
        return Ok((cached.values, metadata));
    }

    let (values, true_change_points) = entry.load_uncached()?;

    let cached = CachedDataset {
        manifest_version: MANIFEST_VERSION,
        name: entry.name.clone(),
        kind: entry.kind.clone(),
        length: entry.length,
        dimensions: entry.dimensions,
        values: values.clone(),
        true_change_points: true_change_points.clone(),
    };
    write_cache(cache_path.as_path(), &cached)?;

    let metadata = entry.loaded_metadata(true_change_points);
    Ok((values, metadata))
}

fn cache_path(cache_root: &Path, name: &str) -> PathBuf {
    cache_root.join(format!("{name}.json"))
}

fn try_read_cache(
    cache_path: &Path,
    entry: &ManifestEntry,
) -> Result<Option<CachedDataset>, CpdError> {
    if !cache_path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(cache_path)
        .map_err(|err| io_error("failed to read cache file", cache_path, err))?;
    let cached: CachedDataset = serde_json::from_str(raw.as_str())
        .map_err(|_| CpdError::invalid_input("cache file is not valid JSON".to_string()))?;
    if validate_cached_dataset(entry, &cached).is_err() {
        return Ok(None);
    }

    Ok(Some(cached))
}

fn validate_cached_dataset(entry: &ManifestEntry, cached: &CachedDataset) -> Result<(), CpdError> {
    if cached.manifest_version != MANIFEST_VERSION {
        return Err(CpdError::invalid_input(format!(
            "cache manifest version mismatch: expected {MANIFEST_VERSION}, got {}",
            cached.manifest_version
        )));
    }
    if cached.name != entry.name {
        return Err(CpdError::invalid_input(format!(
            "cache dataset name mismatch for '{}': got '{}'",
            entry.name, cached.name
        )));
    }
    if cached.kind != entry.kind {
        return Err(CpdError::invalid_input(format!(
            "cache dataset kind mismatch for '{}': got {:?}",
            entry.name, cached.kind
        )));
    }
    if cached.length != entry.length || cached.dimensions != entry.dimensions {
        return Err(CpdError::invalid_input(format!(
            "cache shape mismatch for '{}': expected {}x{}, got {}x{}",
            entry.name, entry.length, entry.dimensions, cached.length, cached.dimensions
        )));
    }

    let expected_len = entry.length.checked_mul(entry.dimensions).ok_or_else(|| {
        CpdError::invalid_input(format!("shape overflow for '{}': n*d", entry.name))
    })?;
    if cached.values.len() != expected_len {
        return Err(CpdError::invalid_input(format!(
            "cache data length mismatch for '{}': expected {}, got {}",
            entry.name,
            expected_len,
            cached.values.len()
        )));
    }

    Ok(())
}

fn write_cache(cache_path: &Path, cached: &CachedDataset) -> Result<(), CpdError> {
    let encoded = serde_json::to_vec_pretty(cached)
        .map_err(|err| CpdError::invalid_input(format!("failed to encode cache JSON: {err}")))?;
    fs::write(cache_path, encoded)
        .map_err(|err| io_error("failed to write cache file", cache_path, err))
}

fn default_cache_root() -> Result<PathBuf, CpdError> {
    if let Ok(path) = env::var("CPD_EVAL_DATASET_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(path) = env::var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path).join("cpd-eval").join("datasets"));
    }
    if let Ok(path) = env::var("HOME") {
        return Ok(PathBuf::from(path)
            .join(".cache")
            .join("cpd-eval")
            .join("datasets"));
    }

    Err(CpdError::resource_limit(
        "unable to resolve cache directory; set CPD_EVAL_DATASET_CACHE_DIR".to_string(),
    ))
}

fn manifest() -> Result<&'static Manifest, CpdError> {
    if let Some(existing) = REGISTRY_MANIFEST.get() {
        return Ok(existing);
    }

    let parsed = parse_manifest()?;
    let _ = REGISTRY_MANIFEST.set(parsed);
    REGISTRY_MANIFEST.get().ok_or_else(|| {
        CpdError::resource_limit("failed to initialize dataset manifest".to_string())
    })
}

fn parse_manifest() -> Result<Manifest, CpdError> {
    let manifest: Manifest = serde_json::from_str(MANIFEST_JSON).map_err(|err| {
        CpdError::invalid_input(format!("dataset manifest is not valid JSON: {err}"))
    })?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &Manifest) -> Result<(), CpdError> {
    if manifest.manifest_version != MANIFEST_VERSION {
        return Err(CpdError::invalid_input(format!(
            "dataset manifest version mismatch: expected {MANIFEST_VERSION}, got {}",
            manifest.manifest_version
        )));
    }

    let mut seen_names = HashSet::new();
    let mut synthetic_count = 0usize;
    let mut real_world_count = 0usize;

    for entry in &manifest.datasets {
        if !seen_names.insert(entry.name.as_str()) {
            return Err(CpdError::invalid_input(format!(
                "dataset '{}' appears multiple times in the manifest",
                entry.name
            )));
        }
        if entry.length == 0 || entry.dimensions == 0 {
            return Err(CpdError::invalid_input(format!(
                "dataset '{}' must have positive length and dimensions",
                entry.name
            )));
        }

        match entry.kind {
            DatasetKind::Synthetic => {
                synthetic_count += 1;
                let synthetic = entry.synthetic_spec()?;
                let (n, d) = synthetic.shape();
                if n != entry.length || d != entry.dimensions {
                    return Err(CpdError::invalid_input(format!(
                        "dataset '{}' shape mismatch between manifest and synthetic config",
                        entry.name
                    )));
                }
                let _ = synthetic.expected_change_points()?;
                if entry.real_world.is_some() {
                    return Err(CpdError::invalid_input(format!(
                        "dataset '{}' cannot define real_world config for synthetic kind",
                        entry.name
                    )));
                }
            }
            DatasetKind::RealWorld => {
                real_world_count += 1;
                let real_world = entry.real_world_spec()?;
                if real_world.fixture_path.trim().is_empty() {
                    return Err(CpdError::invalid_input(format!(
                        "dataset '{}' has empty real_world.fixture_path",
                        entry.name
                    )));
                }
                require_non_empty(entry.license.as_deref(), "license", entry.name.as_str())?;
                require_non_empty(
                    entry.source_url.as_deref(),
                    "source_url",
                    entry.name.as_str(),
                )?;
                require_non_empty(entry.citation.as_deref(), "citation", entry.name.as_str())?;
                if entry.synthetic.is_some() {
                    return Err(CpdError::invalid_input(format!(
                        "dataset '{}' cannot define synthetic config for real_world kind",
                        entry.name
                    )));
                }
            }
        }
    }

    if synthetic_count < 10 {
        return Err(CpdError::invalid_input(format!(
            "manifest must declare at least 10 synthetic datasets; found {synthetic_count}"
        )));
    }
    if real_world_count < 5 {
        return Err(CpdError::invalid_input(format!(
            "manifest must declare at least 5 real-world datasets; found {real_world_count}"
        )));
    }

    Ok(())
}

fn require_non_empty(value: Option<&str>, label: &str, name: &str) -> Result<(), CpdError> {
    if value.is_some_and(|item| !item.trim().is_empty()) {
        return Ok(());
    }

    Err(CpdError::invalid_input(format!(
        "dataset '{name}' missing non-empty {label}"
    )))
}

fn workspace_root() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root
}

fn io_error(context: &str, path: &Path, err: std::io::Error) -> CpdError {
    CpdError::resource_limit(format!("{context} at '{}': {err}", path.display()))
}

impl ManifestEntry {
    fn list_metadata(&self) -> Result<DatasetMetadata, CpdError> {
        let true_change_points = match self.kind {
            DatasetKind::Synthetic => Some(self.synthetic_spec()?.expected_change_points()?),
            DatasetKind::RealWorld => None,
        };

        Ok(DatasetMetadata {
            name: self.name.clone(),
            display_name: self.display_name.clone(),
            kind: self.kind.clone(),
            description: self.description.clone(),
            length: self.length,
            dimensions: self.dimensions,
            true_change_points,
            license: self.license.clone(),
            source_url: self.source_url.clone(),
            citation: self.citation.clone(),
        })
    }

    fn loaded_metadata(&self, true_change_points: Vec<usize>) -> DatasetMetadata {
        DatasetMetadata {
            name: self.name.clone(),
            display_name: self.display_name.clone(),
            kind: self.kind.clone(),
            description: self.description.clone(),
            length: self.length,
            dimensions: self.dimensions,
            true_change_points: match self.kind {
                DatasetKind::Synthetic => Some(true_change_points),
                DatasetKind::RealWorld => None,
            },
            license: self.license.clone(),
            source_url: self.source_url.clone(),
            citation: self.citation.clone(),
        }
    }

    fn synthetic_spec(&self) -> Result<&SyntheticSpec, CpdError> {
        self.synthetic.as_ref().ok_or_else(|| {
            CpdError::invalid_input(format!(
                "dataset '{}' is missing synthetic config",
                self.name
            ))
        })
    }

    fn real_world_spec(&self) -> Result<&RealWorldSpec, CpdError> {
        self.real_world.as_ref().ok_or_else(|| {
            CpdError::invalid_input(format!(
                "dataset '{}' is missing real_world config",
                self.name
            ))
        })
    }

    fn load_uncached(&self) -> Result<(Vec<f64>, Vec<usize>), CpdError> {
        match self.kind {
            DatasetKind::Synthetic => self.synthetic_spec()?.generate(),
            DatasetKind::RealWorld => {
                let spec = self.real_world_spec()?;
                let path = workspace_root().join(spec.fixture_path.as_str());
                let raw = fs::read_to_string(path.as_path()).map_err(|err| {
                    io_error("failed to read real-world fixture", path.as_path(), err)
                })?;
                let fixture: RealWorldFixture =
                    serde_json::from_str(raw.as_str()).map_err(|err| {
                        CpdError::invalid_input(format!(
                            "real-world fixture '{}' is not valid JSON: {err}",
                            spec.fixture_path
                        ))
                    })?;

                let expected_len = self.length.checked_mul(self.dimensions).ok_or_else(|| {
                    CpdError::invalid_input(format!("shape overflow for '{}': n*d", self.name))
                })?;
                if fixture.values.len() != expected_len {
                    return Err(CpdError::invalid_input(format!(
                        "real-world fixture '{}' length mismatch for '{}': expected {}, got {}",
                        spec.fixture_path,
                        self.name,
                        expected_len,
                        fixture.values.len()
                    )));
                }

                Ok((fixture.values, Vec::new()))
            }
        }
    }
}

impl SyntheticSpec {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::PiecewiseConstantMean { n, d, .. }
            | Self::PiecewiseConstantVariance { n, d, .. }
            | Self::PiecewiseLinear { n, d, .. }
            | Self::Ar1WithChanges { n, d, .. }
            | Self::HeavyTailed { n, d, .. }
            | Self::CountData { n, d, .. }
            | Self::BinaryData { n, d, .. }
            | Self::Multivariate { n, d, .. } => (*n, *d),
        }
    }

    fn n_changes(&self) -> usize {
        match self {
            Self::PiecewiseConstantMean { n_changes, .. }
            | Self::PiecewiseConstantVariance { n_changes, .. }
            | Self::PiecewiseLinear { n_changes, .. }
            | Self::Ar1WithChanges { n_changes, .. }
            | Self::HeavyTailed { n_changes, .. }
            | Self::CountData { n_changes, .. }
            | Self::BinaryData { n_changes, .. }
            | Self::Multivariate { n_changes, .. } => *n_changes,
        }
    }

    fn expected_change_points(&self) -> Result<Vec<usize>, CpdError> {
        let (n, _) = self.shape();
        evenly_spaced_breakpoints(n, self.n_changes())
    }

    fn generate(&self) -> Result<(Vec<f64>, Vec<usize>), CpdError> {
        match self {
            Self::PiecewiseConstantMean {
                n,
                d,
                n_changes,
                snr,
                noise_std,
                seed,
            } => piecewise_constant_mean(&PiecewiseMeanConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                snr: *snr,
                noise_std: *noise_std,
                seed: *seed,
            }),
            Self::PiecewiseConstantVariance {
                n,
                d,
                n_changes,
                mean,
                base_std,
                std_step,
                seed,
            } => piecewise_constant_variance(&PiecewiseVarianceConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                mean: *mean,
                base_std: *base_std,
                std_step: *std_step,
                seed: *seed,
            }),
            Self::PiecewiseLinear {
                n,
                d,
                n_changes,
                slope_step,
                noise_std,
                seed,
            } => piecewise_linear(&PiecewiseLinearConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                slope_step: *slope_step,
                noise_std: *noise_std,
                seed: *seed,
            }),
            Self::Ar1WithChanges {
                n,
                d,
                n_changes,
                phi,
                mean_step,
                base_std,
                std_step,
                seed,
            } => ar1_with_changes(&Ar1Config {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                phi: *phi,
                mean_step: *mean_step,
                base_std: *base_std,
                std_step: *std_step,
                seed: *seed,
            }),
            Self::HeavyTailed {
                n,
                d,
                n_changes,
                degrees_of_freedom,
                mean_step,
                scale,
                seed,
            } => heavy_tailed(&HeavyTailConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                degrees_of_freedom: *degrees_of_freedom,
                mean_step: *mean_step,
                scale: *scale,
                seed: *seed,
            }),
            Self::CountData {
                n,
                d,
                n_changes,
                base_rate,
                rate_step,
                seed,
            } => count_data(&CountConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                base_rate: *base_rate,
                rate_step: *rate_step,
                seed: *seed,
            }),
            Self::BinaryData {
                n,
                d,
                n_changes,
                base_prob,
                prob_step,
                seed,
            } => binary_data(&BinaryConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                base_prob: *base_prob,
                prob_step: *prob_step,
                seed: *seed,
            }),
            Self::Multivariate {
                n,
                d,
                n_changes,
                snr,
                noise_std,
                correlation,
                seed,
            } => multivariate(&MultivariateConfig {
                n: *n,
                d: *d,
                n_changes: *n_changes,
                snr: *snr,
                noise_std: *noise_std,
                correlation: *correlation,
                seed: *seed,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CachedDataset, DatasetKind, cache_path, list_datasets, load_dataset_with_cache_root,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_cache_root(name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cpd-eval-registry-{name}-{}-{ts}",
            std::process::id()
        ));
        fs::create_dir_all(path.as_path()).expect("temp cache root should be creatable");
        path
    }

    #[test]
    fn manifest_meets_minimum_dataset_counts() {
        let datasets = list_datasets().expect("dataset list should load");
        let synthetic = datasets
            .iter()
            .filter(|dataset| dataset.kind == DatasetKind::Synthetic)
            .count();
        let real_world = datasets
            .iter()
            .filter(|dataset| dataset.kind == DatasetKind::RealWorld)
            .count();

        assert!(synthetic >= 10, "expected >= 10 synthetic datasets");
        assert!(real_world >= 5, "expected >= 5 real-world datasets");
    }

    #[test]
    fn real_world_entries_include_license_metadata() {
        let datasets = list_datasets().expect("dataset list should load");
        for dataset in datasets
            .iter()
            .filter(|dataset| dataset.kind == DatasetKind::RealWorld)
        {
            assert!(
                dataset
                    .license
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "{} missing license",
                dataset.name
            );
            assert!(
                dataset
                    .source_url
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "{} missing source_url",
                dataset.name
            );
            assert!(
                dataset
                    .citation
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "{} missing citation",
                dataset.name
            );
        }
    }

    #[test]
    fn synthetic_dataset_reads_from_cache_after_first_load() {
        let cache_root = unique_cache_root("synthetic");
        let dataset_name = list_datasets()
            .expect("dataset list should load")
            .into_iter()
            .find(|dataset| dataset.kind == DatasetKind::Synthetic)
            .expect("manifest should include synthetic datasets")
            .name;

        let (first_values, _) =
            load_dataset_with_cache_root(dataset_name.as_str(), cache_root.as_path())
                .expect("first synthetic load should succeed");
        let path = cache_path(cache_root.as_path(), dataset_name.as_str());
        assert!(path.exists(), "cache file should be created on first load");

        let mut cached: CachedDataset = serde_json::from_str(
            fs::read_to_string(path.as_path())
                .expect("cache should be readable")
                .as_str(),
        )
        .expect("cache JSON should parse");
        let sentinel = first_values[0] + 1234.5;
        cached.values[0] = sentinel;
        fs::write(
            path.as_path(),
            serde_json::to_vec_pretty(&cached).expect("cache should serialize"),
        )
        .expect("cache should be writable");

        let (second_values, _) =
            load_dataset_with_cache_root(dataset_name.as_str(), cache_root.as_path())
                .expect("second synthetic load should succeed");
        assert_eq!(second_values[0], sentinel);

        fs::remove_dir_all(cache_root).expect("temp cache root should be removable");
    }

    #[test]
    fn real_world_dataset_reads_from_cache_after_first_load() {
        let cache_root = unique_cache_root("real-world");
        let dataset_name = list_datasets()
            .expect("dataset list should load")
            .into_iter()
            .find(|dataset| dataset.kind == DatasetKind::RealWorld)
            .expect("manifest should include real-world datasets")
            .name;

        let (_first_values, _) =
            load_dataset_with_cache_root(dataset_name.as_str(), cache_root.as_path())
                .expect("first real-world load should succeed");
        let path = cache_path(cache_root.as_path(), dataset_name.as_str());
        assert!(path.exists(), "cache file should be created on first load");

        let mut cached: CachedDataset = serde_json::from_str(
            fs::read_to_string(path.as_path())
                .expect("cache should be readable")
                .as_str(),
        )
        .expect("cache JSON should parse");
        let sentinel = -987.25;
        cached.values[0] = sentinel;
        fs::write(
            path.as_path(),
            serde_json::to_vec_pretty(&cached).expect("cache should serialize"),
        )
        .expect("cache should be writable");

        let (second_values, _) =
            load_dataset_with_cache_root(dataset_name.as_str(), cache_root.as_path())
                .expect("second real-world load should succeed");
        assert_eq!(second_values[0], sentinel);

        fs::remove_dir_all(cache_root).expect("temp cache root should be removable");
    }

    #[test]
    fn load_dataset_rejects_unknown_dataset_name() {
        let cache_root = unique_cache_root("unknown");
        let err = load_dataset_with_cache_root("does_not_exist", cache_root.as_path())
            .expect_err("unknown dataset should return an error");

        assert!(err.to_string().contains("unknown dataset 'does_not_exist'"));

        fs::remove_dir_all(cache_root).expect("temp cache root should be removable");
    }
}
