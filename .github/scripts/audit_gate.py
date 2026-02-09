#!/usr/bin/env python3
"""Gate cargo-audit JSON output on high/critical vulnerabilities."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BLOCKING_LEVELS = {"high", "critical", "unknown"}


def _normalize_severity(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    alias = {
        "moderate": "medium",
        "informational": "low",
        "none": "low",
    }
    value = alias.get(value, value)
    if value in {"low", "medium", "high", "critical", "unknown"}:
        return value
    return None


def _extract_float(raw: Any) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        match = re.search(r"\d+(\.\d+)?", raw)
        if match:
            return float(match.group(0))
    if isinstance(raw, dict):
        for key in ("score", "baseScore", "cvss", "value"):
            value = _extract_float(raw.get(key))
            if value is not None:
                return value
    if isinstance(raw, list):
        for item in raw:
            value = _extract_float(item)
            if value is not None:
                return value
    return None


def _severity_from_cvss(raw: Any) -> str | None:
    score = _extract_float(raw)
    if score is None:
        return None
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


def _vulnerability_list(report: dict[str, Any]) -> list[dict[str, Any]]:
    data = report.get("vulnerabilities")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("list", "items", "vulnerabilities"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _advisory_id(vuln: dict[str, Any], index: int) -> str:
    advisory = vuln.get("advisory")
    if isinstance(advisory, dict):
        for key in ("id", "advisory_id"):
            value = advisory.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("advisory_id", "id"):
        value = vuln.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"unknown-advisory-{index}"


def _severity(vuln: dict[str, Any]) -> str:
    advisory = vuln.get("advisory")
    if not isinstance(advisory, dict):
        advisory = {}

    direct = _normalize_severity(advisory.get("severity"))
    if direct:
        return direct
    direct = _normalize_severity(vuln.get("severity"))
    if direct:
        return direct

    candidates = [
        advisory.get("cvss"),
        advisory.get("cvss_score"),
        vuln.get("cvss"),
        vuln.get("cvss_score"),
    ]
    metadata = advisory.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("cvss"))
        candidates.append(metadata.get("cvss_score"))

    for candidate in candidates:
        parsed = _severity_from_cvss(candidate)
        if parsed:
            return parsed

    return "unknown"


def evaluate_report(report: dict[str, Any]) -> tuple[list[str], int]:
    blocking: list[str] = []
    non_blocking = 0

    for index, vuln in enumerate(_vulnerability_list(report), start=1):
        advisory_id = _advisory_id(vuln, index)
        severity = _severity(vuln)
        if severity in BLOCKING_LEVELS:
            blocking.append(advisory_id)
        else:
            non_blocking += 1

    return blocking, non_blocking


def _load_report(path: str) -> dict[str, Any]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("cargo-audit report must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail if cargo-audit report includes high/critical vulnerabilities."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="-",
        help="Path to cargo-audit JSON report (default: stdin).",
    )
    args = parser.parse_args(argv)

    try:
        report = _load_report(args.report)
    except Exception as exc:  # pragma: no cover - defensive CI guard
        print(f"BLOCK: failed to parse cargo-audit report: {exc}", file=sys.stderr)
        return 1

    blocking, non_blocking = evaluate_report(report)
    if blocking:
        print(
            "BLOCK: "
            f"{len(blocking)} blocking advisories (high/critical/unknown): "
            f"{', '.join(blocking)}"
        )
        return 1

    print(f"PASS: no blocking advisories. Non-blocking advisories: {non_blocking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
