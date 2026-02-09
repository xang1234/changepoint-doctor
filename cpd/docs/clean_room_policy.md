# Clean-Room Reimplementation Policy

This project follows a clean-room approach for algorithm implementations.
Contributors should derive implementations from peer-reviewed papers, public
algorithm descriptions, and first-principles derivations.

## Policy Rules

1. Do not copy/paste source code from third-party repositories into this
   codebase unless the license obligations are fully satisfied and attribution
   is recorded in `NOTICE` and the relevant file history.
2. Prefer independent implementations based on primary papers.
3. Preserve provenance notes in PR descriptions for new detector and cost-model
   implementations.
4. Keep SPDX license headers on Rust source files.
5. When adapting ideas from external code, rewrite from scratch and cite the
   source paper and any upstream project that informed design decisions.

## Prohibited Practices

- Unattributed copying of code, tests, comments, or docs from external repos.
- Porting implementation details line-by-line from upstream code.
- Introducing external snippets with incompatible or unknown licenses.

## Required Attribution Behavior

- Update `NOTICE` when a new upstream project materially informs API shape,
  testing strategy, or behavior.
- Update `CITATION.cff` and this document when new algorithm families are
  implemented.
- Include references in PR notes for algorithmic additions.

## Algorithm-to-Paper Mapping

| Algorithm / Component | Status | Primary reference(s) |
| --- | --- | --- |
| PELT (`cpd-offline`) | Implemented | Killick, Fearnhead, Eckley (2012), JASA, DOI: 10.1080/01621459.2012.737745 |
| Binary Segmentation (`cpd-offline`) | Implemented | General binary segmentation family; survey context in Truong, Oudre, Vayatis (2020), Signal Processing, DOI: 10.1016/j.sigpro.2019.107299 |
| Wild Binary Segmentation (WBS) | Planned | Fryzlewicz (2014), Annals of Statistics, DOI: 10.1214/14-AOS1245 |
| Bayesian Online Changepoint Detection (BOCPD) | Planned | Adams, MacKay (2007), arXiv:0710.3742 |
| Offline CPD taxonomy and benchmarking context | Cross-cutting guidance | Truong, Oudre, Vayatis (2020), Signal Processing, DOI: 10.1016/j.sigpro.2019.107299 |

## Contributor Compliance Checklist

Before opening or merging a PR that adds or changes algorithmic behavior:

1. Confirm implementation was written clean-room from papers/specs.
2. Confirm no unattributed external code was copied.
3. Add/update citations in `CITATION.cff` if algorithm scope changed.
4. Add/update `NOTICE` for any new upstream project inspirations.
5. Keep Rust SPDX headers intact for touched files.
