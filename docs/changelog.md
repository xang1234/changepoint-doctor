# Changelog

## Unreleased

- Added explicit Python optional extras (`plot`, `notebooks`, `parity`, `dev`) and migrated onboarding docs to extras-first install commands.
- Standardized install/import naming guidance: install `changepoint-doctor`, import `cpd`, with optional `changepoint_doctor` compatibility alias.
- Added notebook/docs path guardrails and normalized notebook onboarding to clone-safe, repo-relative instructions.
- Published product-grade CLI documentation (`cpd/docs/cli.md` and docs-site CLI getting-started page), including JSON output mode controls.
- Aligned Cargo workspace metadata and `CITATION.cff` to canonical repository identity with explicit provenance notes.
- Enabled preprocess support in default Python wheel builds, converted integration coverage to the positive preprocess path, and extended wheel-smoke checks to fail if preprocess support regresses.

See the [GitHub Releases](https://github.com/xang1234/changepoint-doctor/releases) page for the full changelog.
