# crates.io Publication Workflow

This project publishes core Rust crates through the release workflow in `.github/workflows/release.yml`.

## Publish Scope and Order

Only these crates are published to crates.io, in strict dependency order:

1. `cpd-core`
2. `cpd-costs`
3. `cpd-preprocess`
4. `cpd-offline`
5. `cpd-online`
6. `cpd-doctor`

All other workspace crates are internal-only (`publish = false`).

## Release Gates

Publication is gated behind release bundle verification (`verify-release-bundle`) and split into two stages:

1. `verify-crates-publish-dry-run`
2. `publish-crates-io`

`verify-crates-publish-dry-run` runs `cargo publish --dry-run --no-verify` to validate package assembly and manifest wiring without requiring already-published internal crate versions.

Behavior by trigger:

- Tag push (`v*`): crates dry-run and crates publish run automatically.
- `workflow_dispatch`: crates jobs run only when `publish_to_crates_io=true`.

## Authentication Policy

`publish-crates-io` uses OIDC-first authentication with token fallback:

1. Attempt trusted publishing (OIDC) without `CARGO_REGISTRY_TOKEN`.
2. If auth fails and `CARGO_REGISTRY_TOKEN` is set in the `crates-io` environment, retry with token fallback.
3. If both methods fail, the workflow blocks.

Fallback secret contract:

- Environment: `crates-io`
- Secret: `CARGO_REGISTRY_TOKEN`
- Scope: publish permission for CPD crates

## Artifact Alignment Guard

Before publishing, `verify-alignment` re-packages publishable crates and compares SHA256 hashes against `release-artifacts/crates/*.crate` from the signed release bundle.

If any hash differs, publication is blocked.

## Reruns and Partial Publishes

Publish runs are idempotent:

- If a crate version is already published, the job logs a skip and continues.
- Retry/backoff is applied for retryable index/propagation failures.
- If a run partially succeeds, rerun the same workflow for the same tag/version; already-published crates are skipped.
