#!/usr/bin/env python3
"""Enforce per-file line-coverage thresholds from LCOV reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class CoverageMetric:
    covered: int
    total: int

    @property
    def pct(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * (self.covered / self.total)


@dataclass(frozen=True)
class CoverageTarget:
    file_suffix: str
    min_pct: float


def parse_lcov(path: Path) -> dict[str, CoverageMetric]:
    records: dict[str, CoverageMetric] = {}

    current_file: str | None = None
    covered = 0
    total = 0

    def flush() -> None:
        nonlocal current_file, covered, total
        if current_file is None:
            return

        existing = records.get(current_file)
        if existing is None:
            records[current_file] = CoverageMetric(covered=covered, total=total)
        else:
            records[current_file] = CoverageMetric(
                covered=existing.covered + covered,
                total=existing.total + total,
            )

        current_file = None
        covered = 0
        total = 0

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("SF:"):
            flush()
            current_file = line[3:]
            covered = 0
            total = 0
            continue
        if line.startswith("DA:"):
            _, payload = line.split(":", 1)
            parts = payload.split(",")
            if len(parts) < 2:
                raise ValueError(f"invalid DA record: {line!r}")
            count = int(parts[1])
            total += 1
            if count > 0:
                covered += 1
            continue
        if line == "end_of_record":
            flush()
            continue

    flush()
    return records


def _parse_target(raw: str) -> CoverageTarget:
    if ":" not in raw:
        raise ValueError(
            f"invalid --target value {raw!r}; expected format '<file-suffix>:<min-pct>'"
        )
    suffix, pct_raw = raw.rsplit(":", 1)
    suffix = suffix.strip()
    if not suffix:
        raise ValueError("target file suffix must be non-empty")
    min_pct = float(pct_raw)
    if min_pct < 0.0 or min_pct > 100.0:
        raise ValueError(f"target threshold must be within [0, 100]; got {min_pct}")
    return CoverageTarget(file_suffix=suffix, min_pct=min_pct)


def _find_metric(records: dict[str, CoverageMetric], suffix: str) -> CoverageMetric | None:
    normalized = suffix.replace("\\", "/")
    for path, metric in records.items():
        candidate = path.replace("\\", "/")
        if candidate.endswith(normalized):
            return metric
        if candidate.endswith("/" + normalized):
            return metric
    return None


def evaluate_targets(
    records: dict[str, CoverageMetric],
    targets: list[CoverageTarget],
) -> list[str]:
    failures: list[str] = []

    for target in targets:
        metric = _find_metric(records, target.file_suffix)
        if metric is None:
            failures.append(f"missing coverage data for {target.file_suffix}")
            continue

        pct = metric.pct
        if pct + 1e-9 < target.min_pct:
            failures.append(
                f"{target.file_suffix}: {pct:.2f}% < required {target.min_pct:.2f}%"
            )

    return failures


def run_check_lcov(lcov_path: Path, targets: list[CoverageTarget]) -> int:
    try:
        records = parse_lcov(lcov_path)
    except Exception as exc:  # pragma: no cover - defensive CI guard
        print(f"BLOCK: failed to parse LCOV report: {exc}", file=sys.stderr)
        return 1

    failures = evaluate_targets(records=records, targets=targets)
    if failures:
        print("BLOCK: coverage thresholds not met")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS: coverage thresholds met")
    for target in targets:
        metric = _find_metric(records, target.file_suffix)
        assert metric is not None
        print(
            f"  - {target.file_suffix}: {metric.pct:.2f}% "
            f"(required {target.min_pct:.2f}%)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCOV per-file coverage gate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-lcov", help="Evaluate LCOV coverage thresholds")
    check.add_argument("--lcov", required=True, help="Path to LCOV report")
    check.add_argument(
        "--target",
        action="append",
        required=True,
        help="Coverage requirement in '<file-suffix>:<min-pct>' format",
    )

    args = parser.parse_args(argv)

    if args.command == "check-lcov":
        targets = [_parse_target(raw) for raw in args.target]
        return run_check_lcov(lcov_path=Path(args.lcov), targets=targets)

    raise ValueError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
