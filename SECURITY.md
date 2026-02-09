# Security Policy

## Dependency Vulnerability Triage SLA

High and critical dependency vulnerabilities (RustSec advisories) are triaged within 48 hours of detection.

## Reporting

Open a bead issue and tag it with `security` and `ci` when a dependency security gate fails.

## Triage Workflow

1. Confirm the failing advisory IDs from the `dependency-security / cargo-audit` job output.
2. Open or update a bead issue with:
   - advisory ID(s)
   - affected crate(s)
   - current dependency chain
   - mitigation options
3. Assign an owner immediately.
4. Choose a mitigation:
   - upgrade to a fixed dependency version
   - replace the affected dependency
   - add a temporary ignore with a clear expiry date and justification
5. Link the mitigation PR to the bead issue and include validation evidence from CI.
6. Close the issue once the advisory no longer appears in CI.

## Temporary Ignores

Temporary ignores are allowed only with:

- a documented reason
- an expiry date
- an owning bead issue

Expired ignores must be removed or renewed with a fresh justification.
