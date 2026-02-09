#!/usr/bin/env python3
"""Collect and gate benchmark regressions for CI."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    bench: str
    criterion_id: str
    max_runtime_seconds: float | None
    max_rss_kib: int | None


@dataclass(frozen=True)
class BenchmarkMetric:
    id: str
    runtime_seconds: float
    max_rss_kib: int


_GNU_MAX_RSS_RE = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$")
_BSD_MAX_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$")
_DEFAULT_MANIFEST_VERSION = 1


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object in {path}")
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_thresholds(raw: Any, case_id: str) -> tuple[float | None, int | None]:
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise ValueError(f"manifest case {case_id!r} has non-object absolute_thresholds")

    runtime = raw.get("max_runtime_seconds")
    rss = raw.get("max_rss_kib")

    if runtime is not None:
        if not isinstance(runtime, (int, float)) or not float(runtime) > 0.0:
            raise ValueError(
                f"manifest case {case_id!r} has invalid max_runtime_seconds={runtime!r}"
            )
        runtime = float(runtime)

    if rss is not None:
        if not isinstance(rss, int) or rss <= 0:
            raise ValueError(f"manifest case {case_id!r} has invalid max_rss_kib={rss!r}")

    return runtime, rss


def load_manifest(path: Path) -> tuple[int, list[BenchmarkCase]]:
    payload = _load_json(path)

    manifest_version = payload.get("manifest_version", _DEFAULT_MANIFEST_VERSION)
    if not isinstance(manifest_version, int) or manifest_version <= 0:
        raise ValueError(f"manifest_version must be a positive integer; got {manifest_version!r}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest is missing non-empty 'cases' list")

    out: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"manifest case index {index} is not an object")

        case_id = raw_case.get("id")
        bench = raw_case.get("bench")
        criterion_id = raw_case.get("criterion_id", case_id)

        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"manifest case index {index} has invalid id={case_id!r}")
        if not isinstance(bench, str) or not bench:
            raise ValueError(f"manifest case {case_id!r} has invalid bench={bench!r}")
        if not isinstance(criterion_id, str) or not criterion_id:
            raise ValueError(
                f"manifest case {case_id!r} has invalid criterion_id={criterion_id!r}"
            )
        if case_id in seen_ids:
            raise ValueError(f"manifest contains duplicate case id {case_id!r}")

        max_runtime_seconds, max_rss_kib = _validate_thresholds(
            raw_case.get("absolute_thresholds"), case_id
        )

        out.append(
            BenchmarkCase(
                id=case_id,
                bench=bench,
                criterion_id=criterion_id,
                max_runtime_seconds=max_runtime_seconds,
                max_rss_kib=max_rss_kib,
            )
        )
        seen_ids.add(case_id)

    return manifest_version, out


def _is_gnu_time(binary: str) -> bool:
    try:
        process = subprocess.run(
            [binary, "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if process.returncode != 0:
        return False
    haystack = f"{process.stdout}\n{process.stderr}"
    return "GNU time" in haystack


def _select_time_command() -> tuple[list[str], str]:
    if _is_gnu_time("/usr/bin/time"):
        return ["/usr/bin/time", "-v"], "gnu"

    gtime = shutil.which("gtime")
    if gtime is not None and _is_gnu_time(gtime):
        return [gtime, "-v"], "gnu"

    if Path("/usr/bin/time").exists():
        return ["/usr/bin/time", "-l"], "bsd"

    raise ValueError("unable to find supported time command (/usr/bin/time or gtime)")


def _parse_max_rss_kib(time_output: str, mode: str) -> int:
    max_rss_raw: int | None = None
    for line in time_output.splitlines():
        if mode == "gnu":
            match = _GNU_MAX_RSS_RE.match(line)
            if match is not None:
                max_rss_raw = int(match.group(1))
                break
        elif mode == "bsd":
            match = _BSD_MAX_RSS_RE.match(line)
            if match is not None:
                max_rss_raw = int(match.group(1))
                break
        else:
            raise ValueError(f"unsupported time mode {mode!r}")

    if max_rss_raw is None:
        raise ValueError(f"missing max RSS in time output (mode={mode})")

    if mode == "gnu":
        return max_rss_raw

    # On macOS /usr/bin/time -l reports bytes; convert to KiB.
    # If the value is already KiB on another BSD variant, this heuristic keeps it unchanged.
    if max_rss_raw > 16_000_000:
        return max(1, (max_rss_raw + 1023) // 1024)
    return max_rss_raw


def _criterion_estimates_path(workspace: Path, criterion_id: str) -> Path:
    out = workspace / "target" / "criterion"
    for part in criterion_id.split("/"):
        if not part:
            raise ValueError(f"invalid empty criterion path segment in {criterion_id!r}")
        out = out / part
    return out / "new" / "estimates.json"


def _parse_runtime_ns(estimates_path: Path) -> float:
    payload = _load_json(estimates_path)
    median = payload.get("median")
    if not isinstance(median, dict):
        raise ValueError(f"estimates missing median object: {estimates_path}")
    point_estimate = median.get("point_estimate")
    if not isinstance(point_estimate, (int, float)) or not float(point_estimate) > 0.0:
        raise ValueError(
            f"estimates missing positive median.point_estimate in {estimates_path}"
        )
    return float(point_estimate)


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_checked(command: list[str], cwd: Path, label: str) -> None:
    process = _run_command(command, cwd=cwd)
    if process.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {process.returncode}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )


def _prepare_bench_binaries(workspace: Path) -> None:
    _run_checked(
        ["cargo", "bench", "-p", "cpd-bench", "--no-run"],
        cwd=workspace,
        label="bench prebuild",
    )


def _run_case(
    workspace: Path,
    case: BenchmarkCase,
    time_command: list[str],
    time_mode: str,
) -> dict[str, Any]:
    command = [
        *time_command,
        "cargo",
        "bench",
        "-p",
        "cpd-bench",
        "--bench",
        case.bench,
        "--",
        "--exact",
        case.criterion_id,
        "--noplot",
    ]
    process = _run_command(command, cwd=workspace)
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark command failed for {case.id!r} with exit code {process.returncode}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )

    combined_output = f"{process.stdout}\n{process.stderr}"
    max_rss_kib = _parse_max_rss_kib(combined_output, mode=time_mode)
    estimates_path = _criterion_estimates_path(workspace=workspace, criterion_id=case.criterion_id)
    if not estimates_path.exists():
        raise ValueError(
            f"criterion estimates not found for case {case.id!r}: {estimates_path}"
        )
    runtime_ns = _parse_runtime_ns(estimates_path)
    runtime_seconds = runtime_ns / 1_000_000_000.0

    return {
        "id": case.id,
        "bench": case.bench,
        "criterion_id": case.criterion_id,
        "runtime_ns": runtime_ns,
        "runtime_seconds": runtime_seconds,
        "max_rss_kib": max_rss_kib,
    }


def _rustc_version(workspace: Path) -> str:
    process = _run_command(["rustc", "--version"], cwd=workspace)
    if process.returncode != 0:
        return f"unavailable(exit={process.returncode})"
    return process.stdout.strip() or "unknown"


def collect_metrics(
    workspace: Path,
    manifest_path: Path,
    manifest_version: int,
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    _prepare_bench_binaries(workspace)
    time_command, time_mode = _select_time_command()

    metrics: list[dict[str, Any]] = []
    for case in cases:
        metrics.append(_run_case(workspace=workspace, case=case, time_command=time_command, time_mode=time_mode))

    return {
        "metadata": {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustc": _rustc_version(workspace),
            "manifest_path": str(manifest_path),
            "manifest_version": manifest_version,
            "time_mode": time_mode,
        },
        "benchmarks": metrics,
    }


def _extract_benchmark_map(payload: dict[str, Any], label: str) -> dict[str, BenchmarkMetric]:
    data = payload.get("benchmarks")
    if not isinstance(data, list):
        raise ValueError(f"{label} is missing 'benchmarks' list")

    out: dict[str, BenchmarkMetric] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"{label} contains non-object benchmark entry")

        case_id = item.get("id")
        runtime = item.get("runtime_seconds")
        max_rss = item.get("max_rss_kib")

        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} benchmark entry missing non-empty 'id'")
        if not isinstance(runtime, (int, float)):
            raise ValueError(f"{label} benchmark {case_id!r} missing numeric runtime_seconds")
        if not isinstance(max_rss, int):
            raise ValueError(f"{label} benchmark {case_id!r} missing integer max_rss_kib")
        if runtime <= 0:
            raise ValueError(f"{label} benchmark {case_id!r} has non-positive runtime_seconds")
        if max_rss <= 0:
            raise ValueError(f"{label} benchmark {case_id!r} has non-positive max_rss_kib")
        if case_id in out:
            raise ValueError(f"{label} has duplicate benchmark id {case_id!r}")

        out[case_id] = BenchmarkMetric(
            id=case_id,
            runtime_seconds=float(runtime),
            max_rss_kib=max_rss,
        )

    return out


def compare_relative_metrics(
    baseline_payload: dict[str, Any],
    current_payload: dict[str, Any],
    max_runtime_regression_pct: float,
    max_rss_regression_pct: float,
) -> list[str]:
    baseline = _extract_benchmark_map(baseline_payload, "baseline")
    current = _extract_benchmark_map(current_payload, "current")

    failures: list[str] = []

    baseline_ids = set(baseline)
    current_ids = set(current)
    missing_in_baseline = sorted(current_ids - baseline_ids)
    missing_in_current = sorted(baseline_ids - current_ids)

    for case_id in missing_in_baseline:
        failures.append(f"missing baseline metric for benchmark {case_id!r}")
    for case_id in missing_in_current:
        failures.append(f"missing current metric for benchmark {case_id!r}")

    for case_id in sorted(baseline_ids & current_ids):
        baseline_metric = baseline[case_id]
        current_metric = current[case_id]

        runtime_regression = (
            (current_metric.runtime_seconds - baseline_metric.runtime_seconds)
            / baseline_metric.runtime_seconds
            * 100.0
        )
        rss_regression = (
            (current_metric.max_rss_kib - baseline_metric.max_rss_kib)
            / baseline_metric.max_rss_kib
            * 100.0
        )

        if runtime_regression > max_runtime_regression_pct:
            failures.append(
                f"{case_id}: runtime regression {runtime_regression:.2f}% exceeds "
                f"threshold {max_runtime_regression_pct:.2f}% "
                f"(baseline={baseline_metric.runtime_seconds:.6f}s, "
                f"current={current_metric.runtime_seconds:.6f}s)"
            )

        if rss_regression > max_rss_regression_pct:
            failures.append(
                f"{case_id}: RSS regression {rss_regression:.2f}% exceeds "
                f"threshold {max_rss_regression_pct:.2f}% "
                f"(baseline={baseline_metric.max_rss_kib} KiB, "
                f"current={current_metric.max_rss_kib} KiB)"
            )

    return failures


def compare_absolute_metrics(manifest_cases: list[BenchmarkCase], current_payload: dict[str, Any]) -> list[str]:
    current = _extract_benchmark_map(current_payload, "current")
    failures: list[str] = []

    for case in manifest_cases:
        if case.max_runtime_seconds is None and case.max_rss_kib is None:
            continue

        metric = current.get(case.id)
        if metric is None:
            failures.append(f"missing current metric for thresholded benchmark {case.id!r}")
            continue

        if case.max_runtime_seconds is not None and metric.runtime_seconds > case.max_runtime_seconds:
            failures.append(
                f"{case.id}: runtime {metric.runtime_seconds:.6f}s exceeds absolute threshold "
                f"{case.max_runtime_seconds:.6f}s"
            )

        if case.max_rss_kib is not None and metric.max_rss_kib > case.max_rss_kib:
            failures.append(
                f"{case.id}: RSS {metric.max_rss_kib} KiB exceeds absolute threshold "
                f"{case.max_rss_kib} KiB"
            )

    return failures


def _select_cases(cases: list[BenchmarkCase], requested_ids: list[str] | None) -> list[BenchmarkCase]:
    if not requested_ids:
        return list(cases)

    by_id = {case.id: case for case in cases}
    selected: list[BenchmarkCase] = []
    seen: set[str] = set()

    for case_id in requested_ids:
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"requested case {case_id!r} is not present in manifest")
        if case_id in seen:
            continue
        selected.append(case)
        seen.add(case_id)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark collection and regression gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run benchmark cases and collect metrics.")
    collect_parser.add_argument("--workspace", required=True, help="Workspace directory.")
    collect_parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON path.")
    collect_parser.add_argument("--out", required=True, help="Output JSON path.")
    collect_parser.add_argument(
        "--case",
        action="append",
        help="Benchmark case id to collect. Repeat for multiple cases; defaults to all.",
    )

    relative_parser = subparsers.add_parser(
        "compare-relative",
        help="Compare current metrics to baseline for relative regressions.",
    )
    relative_parser.add_argument("--baseline", required=True, help="Baseline JSON path.")
    relative_parser.add_argument("--current", required=True, help="Current JSON path.")
    relative_parser.add_argument(
        "--max-runtime-regression-pct",
        type=float,
        required=True,
        help="Maximum allowed runtime regression percentage.",
    )
    relative_parser.add_argument(
        "--max-rss-regression-pct",
        type=float,
        required=True,
        help="Maximum allowed RSS regression percentage.",
    )

    absolute_parser = subparsers.add_parser(
        "compare-absolute",
        help="Compare current metrics against manifest absolute thresholds.",
    )
    absolute_parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON path.")
    absolute_parser.add_argument("--current", required=True, help="Current JSON path.")

    args = parser.parse_args(argv)

    try:
        if args.command == "collect":
            workspace = Path(args.workspace).resolve()
            if not workspace.exists() or not workspace.is_dir():
                raise ValueError(f"workspace does not exist or is not a directory: {workspace}")

            manifest_path = Path(args.manifest).resolve()
            out_path = Path(args.out)

            manifest_version, all_cases = load_manifest(manifest_path)
            selected_cases = _select_cases(all_cases, args.case)
            if not selected_cases:
                raise ValueError("no benchmark cases selected")

            payload = collect_metrics(
                workspace=workspace,
                manifest_path=manifest_path,
                manifest_version=manifest_version,
                cases=selected_cases,
            )
            _write_json(out_path, payload)

            print(f"Wrote benchmark metrics to {out_path}")
            for row in payload["benchmarks"]:
                print(
                    f"- {row['id']}: runtime={row['runtime_seconds']:.6f}s "
                    f"({row['runtime_ns']:.0f}ns), max_rss={row['max_rss_kib']} KiB"
                )
            return 0

        if args.command == "compare-relative":
            baseline_payload = _load_json(Path(args.baseline))
            current_payload = _load_json(Path(args.current))
            failures = compare_relative_metrics(
                baseline_payload=baseline_payload,
                current_payload=current_payload,
                max_runtime_regression_pct=args.max_runtime_regression_pct,
                max_rss_regression_pct=args.max_rss_regression_pct,
            )
            if failures:
                print("BLOCK: benchmark relative regressions detected:", file=sys.stderr)
                for failure in failures:
                    print(f"  - {failure}", file=sys.stderr)
                return 1
            print("PASS: relative benchmark regressions are within configured thresholds")
            return 0

        if args.command == "compare-absolute":
            _, cases = load_manifest(Path(args.manifest))
            current_payload = _load_json(Path(args.current))
            failures = compare_absolute_metrics(cases, current_payload)
            if failures:
                print("BLOCK: benchmark absolute thresholds violated:", file=sys.stderr)
                for failure in failures:
                    print(f"  - {failure}", file=sys.stderr)
                return 1
            print("PASS: benchmark absolute thresholds satisfied")
            return 0

        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
