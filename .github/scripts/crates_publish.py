#!/usr/bin/env python3
"""Crates.io dry-run, alignment, and publish helpers for release workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from typing import Callable

PUBLISHABLE_CRATES: tuple[str, ...] = (
    "cpd-core",
    "cpd-costs",
    "cpd-preprocess",
    "cpd-offline",
    "cpd-online",
    "cpd-doctor",
)

ALREADY_PUBLISHED_MARKERS: tuple[str, ...] = (
    "already uploaded",
    "already exists",
    "already published",
    "previously uploaded",
)

AUTH_FAILURE_MARKERS: tuple[str, ...] = (
    "please run cargo login",
    "please run `cargo login`",
    "failed to authenticate",
    "authentication required",
    "no token found",
    "token is invalid",
    "unauthorized",
    "forbidden",
)

RETRYABLE_MARKERS: tuple[str, ...] = (
    "no matching package named",
    "failed to select a version for the requirement",
    "doesn't match any versions",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "status 429",
    "http code 429",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CPD_ROOT = REPO_ROOT / "cpd"
TARGET_PACKAGE_DIR = CPD_ROOT / "target" / "package"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


Runner = Callable[[list[str], Path | None, dict[str, str] | None], CommandResult]
SleepFn = Callable[[float], None]


@functools.lru_cache(maxsize=None)
def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected TOML table at {path}")
    return data


@functools.lru_cache(maxsize=1)
def workspace_version() -> str:
    workspace_manifest = _load_toml(CPD_ROOT / "Cargo.toml")
    workspace = workspace_manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("cpd/Cargo.toml missing [workspace] table")
    package = workspace.get("package")
    if not isinstance(package, dict):
        raise ValueError("cpd/Cargo.toml missing [workspace.package] table")
    version = package.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("cpd/Cargo.toml [workspace.package].version must be a string")
    return version


def crate_manifest(crate: str) -> Path:
    manifest = CPD_ROOT / "crates" / crate / "Cargo.toml"
    if not manifest.is_file():
        raise ValueError(f"missing crate manifest: {manifest}")
    return manifest


def crate_version(crate: str) -> str:
    manifest = _load_toml(crate_manifest(crate))
    package = manifest.get("package")
    if not isinstance(package, dict):
        raise ValueError(f"{crate}: missing [package] table")

    version = package.get("version")
    if isinstance(version, str) and version.strip():
        return version

    if isinstance(version, dict) and version.get("workspace") is True:
        return workspace_version()

    raise ValueError(f"{crate}: unable to resolve package version")


def crate_archive_name(crate: str) -> str:
    return f"{crate}-{crate_version(crate)}.crate"


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    print(f"+ {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    return CommandResult(returncode=completed.returncode, output=output)


def _require_success(result: CommandResult, context: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{context} failed with exit code {result.returncode}")


def _contains_any_marker(output: str, markers: tuple[str, ...]) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in markers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dry_run_publishable_crates(runner: Runner = run_command) -> None:
    for crate in PUBLISHABLE_CRATES:
        manifest = crate_manifest(crate)
        result = runner(
            [
                "cargo",
                "publish",
                "--dry-run",
                "--locked",
                "--no-verify",
                "--manifest-path",
                str(manifest),
            ],
            REPO_ROOT,
            None,
        )
        _require_success(result, f"dry-run publish for {crate}")


def _package_crate(crate: str, runner: Runner) -> Path:
    archive_path = TARGET_PACKAGE_DIR / crate_archive_name(crate)
    if archive_path.exists():
        archive_path.unlink()

    manifest = crate_manifest(crate)
    result = runner(
        [
            "cargo",
            "package",
            "--locked",
            "--allow-dirty",
            "--no-verify",
            "--manifest-path",
            str(manifest),
        ],
        REPO_ROOT,
        None,
    )
    _require_success(result, f"cargo package for {crate}")

    if not archive_path.is_file():
        raise RuntimeError(f"expected package artifact missing: {archive_path}")
    return archive_path


def verify_alignment(release_artifacts_dir: Path, runner: Runner = run_command) -> None:
    crates_dir = release_artifacts_dir / "crates"
    if not crates_dir.is_dir():
        raise ValueError(f"missing crates directory: {crates_dir}")

    for crate in PUBLISHABLE_CRATES:
        built = _package_crate(crate, runner)
        signed = crates_dir / built.name
        if not signed.is_file():
            raise RuntimeError(f"missing signed crate artifact for {crate}: {signed}")

        built_sha = _sha256(built)
        signed_sha = _sha256(signed)
        if built_sha != signed_sha:
            raise RuntimeError(
                f"artifact alignment failed for {crate}: "
                f"built={built_sha} signed={signed_sha}"
            )
        print(f"PASS: alignment verified for {crate} ({built.name})")


def _publish_env(use_token: bool, fallback_token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CARGO_REGISTRY_TOKEN", None)
    if use_token and fallback_token:
        env["CARGO_REGISTRY_TOKEN"] = fallback_token
    return env


def _publish_one_crate(
    crate: str,
    *,
    fallback_token: str | None,
    max_attempts: int,
    retry_delay_seconds: float,
    runner: Runner,
    sleep_fn: SleepFn,
) -> str:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if retry_delay_seconds <= 0:
        raise ValueError("retry_delay_seconds must be > 0")

    manifest = crate_manifest(crate)
    cmd = ["cargo", "publish", "--locked", "--manifest-path", str(manifest)]

    attempt = 1
    use_token = False

    while attempt <= max_attempts:
        result = runner(cmd, REPO_ROOT, _publish_env(use_token=use_token, fallback_token=fallback_token))

        if result.returncode == 0:
            print(f"PASS: published {crate}")
            return "published"

        if _contains_any_marker(result.output, ALREADY_PUBLISHED_MARKERS):
            print(f"PASS: {crate} already published; continuing")
            return "already-published"

        if not use_token and fallback_token and _contains_any_marker(result.output, AUTH_FAILURE_MARKERS):
            print(
                f"INFO: trusted publishing failed for {crate}; retrying with CARGO_REGISTRY_TOKEN fallback"
            )
            use_token = True
            continue

        if not use_token and not fallback_token and _contains_any_marker(result.output, AUTH_FAILURE_MARKERS):
            raise RuntimeError(
                f"trusted publishing failed for {crate} and CARGO_REGISTRY_TOKEN is not configured"
            )

        if use_token and _contains_any_marker(result.output, AUTH_FAILURE_MARKERS):
            raise RuntimeError(f"token fallback authentication failed for {crate}")

        if attempt < max_attempts and _contains_any_marker(result.output, RETRYABLE_MARKERS):
            delay = retry_delay_seconds * (2 ** (attempt - 1))
            print(
                f"INFO: retryable publish failure for {crate} "
                f"(attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
            )
            sleep_fn(delay)
            attempt += 1
            continue

        raise RuntimeError(f"publish failed for {crate} on attempt {attempt}")

    raise RuntimeError(f"publish failed for {crate}; exceeded {max_attempts} attempts")


def publish_crates(
    *,
    max_attempts: int,
    retry_delay_seconds: float,
    runner: Runner = run_command,
    sleep_fn: SleepFn = time.sleep,
) -> list[tuple[str, str]]:
    fallback_token = os.environ.get("CARGO_REGISTRY_TOKEN", "").strip() or None

    statuses: list[tuple[str, str]] = []
    for crate in PUBLISHABLE_CRATES:
        status = _publish_one_crate(
            crate,
            fallback_token=fallback_token,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            runner=runner,
            sleep_fn=sleep_fn,
        )
        statuses.append((crate, status))

    print("Publish summary:")
    for crate, status in statuses:
        print(f"  - {crate}: {status}")
    return statuses


def _resolve_release_artifacts_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manage crates.io release workflow checks and publishing for core CPD crates."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "dry-run",
        help="Run cargo publish --dry-run for publishable crates in dependency order.",
    )

    verify_parser = subparsers.add_parser(
        "verify-alignment",
        help="Re-package publishable crates and compare SHA256 against signed release artifacts.",
    )
    verify_parser.add_argument(
        "--release-artifacts-dir",
        default="release-artifacts",
        help="Release artifact root directory that contains crates/*.crate (default: release-artifacts).",
    )

    publish_parser = subparsers.add_parser(
        "publish",
        help="Publish crates in dependency order with OIDC-first auth and token fallback.",
    )
    publish_parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum publish attempts per crate for retryable failures.",
    )
    publish_parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=15.0,
        help="Base retry delay in seconds (exponential backoff).",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "dry-run":
            dry_run_publishable_crates()
        elif args.command == "verify-alignment":
            verify_alignment(_resolve_release_artifacts_dir(args.release_artifacts_dir))
        elif args.command == "publish":
            publish_crates(
                max_attempts=args.max_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
            )
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"unsupported command: {args.command}")
    except (RuntimeError, ValueError) as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
