# Offline Result JSON Contract

This document defines the stable JSON contract for
`OfflineChangePointResult` payloads emitted by Python and Rust adapters.

## Canonical Schema + Version Marker

- Canonical JSON schema:
  `cpd/schemas/result/offline_change_point_result.v1.schema.json`
- Canonical schema marker in `0.x` payloads: `diagnostics.schema_version`
- Current writer version: `1`
- Reader compatibility window: `1..=2` (current + additive forward-compatible
  fixture window in `0.x`), enforced by `cpd-core` schema migration helpers.

## Top-Level Field Contract

Required fields:

- `breakpoints: int[]` (strictly increasing, final element equals `diagnostics.n`)
- `change_points: int[]` (must equal `breakpoints` excluding terminal `n`)
- `diagnostics: object`

Optional fields:

- `scores: float[] | null` (if present, length must equal `change_points` length)
- `segments: SegmentStats[] | null` (if present, length must equal
  `breakpoints` length and segment boundaries/counts must be valid)

## Diagnostics Contract

Required diagnostics fields:

- `n: int`
- `d: int`
- `schema_version: int >= 1`
- `algorithm: string`
- `cost_model: string`
- `repro_mode: string`

Optional diagnostics fields:

- `engine_version`, `runtime_ms`, `notes`, `warnings`, `seed`, `thread_count`
- `blas_backend`, `cpu_features`, `params_json`, `pruning_stats`
- `missing_policy_applied`, `missing_fraction`, `effective_sample_count`

## Backward/Forward Compatibility Expectations

- Writers emit the current canonical version (`schema_version=1`) until a
  migration explicitly bumps the contract.
- Readers accept current + additive forward-compatible fixtures within the
  supported window (`1..=2`).
- Unknown fields are preserved in wire structs and round-tripped where
  round-trip APIs are available.
- Removing or renaming required fields requires a schema migration and fixture
  updates.

When bumping schema versions:

1. Add migration notes using
   `cpd/docs/templates/schema_migration.md`.
2. Add/refresh fixtures in
   `cpd/tests/fixtures/migrations/result/`.
3. Keep compatibility-window tests green in `cpd-core` and Python contract
   tests.

## Validation Failure + Error Messaging Contract

Implementations must fail fast with actionable, field-specific messages.

Required failure classes:

- Unsupported schema version:
  - Must include artifact name, offending `schema_version`, supported range, and
    migration guidance path.
- Structural validation failures:
  - Missing required fields.
  - Wrong JSON types.
- Semantic validation failures:
  - Invalid breakpoints (`n` terminal requirement, ordering, bounds).
  - `change_points` mismatch with derived values.
  - `scores` or `segments` length/shape mismatches.
  - Segment boundary/count invariants.

Python APIs must surface these as `ValueError` with messages that keep the key
artifact/field identifiers intact for user debugging.

## Compatibility Fixtures

Canonical fixtures for migration and compatibility checks:

- Current: `cpd/tests/fixtures/migrations/result/offline_result.v1.json`
- Additive compatibility:
  `cpd/tests/fixtures/migrations/result/offline_result.v2.additive.json`
- Diagnostics-only fixtures:
  - `cpd/tests/fixtures/migrations/result/diagnostics.v1.json`
  - `cpd/tests/fixtures/migrations/result/diagnostics.v2.additive.json`
