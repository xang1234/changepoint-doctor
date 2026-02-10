import os
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crates_publish  # noqa: E402


def _result(returncode: int = 0, output: str = "") -> crates_publish.CommandResult:
    return crates_publish.CommandResult(returncode=returncode, output=output)


class FakeRunner:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    def __call__(
        self,
        cmd: list[str],
        cwd: Path | None,
        env: dict[str, str] | None,
    ) -> crates_publish.CommandResult:
        self.calls.append((cmd, cwd, env))
        if not self._responses:
            return _result()
        return self._responses.pop(0)


class CratesPublishTests(unittest.TestCase):
    def test_dry_run_executes_in_publish_order(self):
        runner = FakeRunner([_result() for _ in crates_publish.PUBLISHABLE_CRATES])

        crates_publish.dry_run_publishable_crates(runner=runner)

        called_crates = [Path(cmd[-1]).parent.name for cmd, _, _ in runner.calls]
        self.assertEqual(called_crates, list(crates_publish.PUBLISHABLE_CRATES))
        for cmd, _, _ in runner.calls:
            self.assertIn("--dry-run", cmd)
            self.assertIn("--locked", cmd)
            self.assertIn("--no-verify", cmd)

    def test_publish_falls_back_to_token_after_auth_failure(self):
        responses = [
            _result(returncode=101, output="error: no token found"),
            _result(returncode=0, output="published cpd-core"),
        ] + [_result(returncode=0, output="ok") for _ in crates_publish.PUBLISHABLE_CRATES[1:]]
        runner = FakeRunner(responses)

        with mock.patch.dict(os.environ, {"CARGO_REGISTRY_TOKEN": "token-value"}, clear=False):
            statuses = crates_publish.publish_crates(
                max_attempts=3,
                retry_delay_seconds=0.1,
                runner=runner,
                sleep_fn=lambda _seconds: None,
            )

        self.assertEqual(statuses[0], ("cpd-core", "published"))
        first_env = runner.calls[0][2]
        second_env = runner.calls[1][2]
        self.assertIsNotNone(first_env)
        self.assertIsNotNone(second_env)
        self.assertNotIn("CARGO_REGISTRY_TOKEN", first_env)
        self.assertEqual(second_env.get("CARGO_REGISTRY_TOKEN"), "token-value")

    def test_publish_skips_already_uploaded_versions(self):
        responses = [
            _result(returncode=101, output="error: crate version already uploaded"),
        ] + [_result(returncode=0, output="ok") for _ in crates_publish.PUBLISHABLE_CRATES[1:]]
        runner = FakeRunner(responses)

        statuses = crates_publish.publish_crates(
            max_attempts=3,
            retry_delay_seconds=0.1,
            runner=runner,
            sleep_fn=lambda _seconds: None,
        )

        self.assertEqual(statuses[0], ("cpd-core", "already-published"))
        self.assertEqual(len(statuses), len(crates_publish.PUBLISHABLE_CRATES))

    def test_publish_retries_dependency_propagation_errors(self):
        responses = [
            _result(returncode=101, output="error: no matching package named `cpd-core` found"),
            _result(returncode=0, output="published"),
        ] + [_result(returncode=0, output="ok") for _ in crates_publish.PUBLISHABLE_CRATES[1:]]
        runner = FakeRunner(responses)
        sleeps: list[float] = []

        statuses = crates_publish.publish_crates(
            max_attempts=3,
            retry_delay_seconds=2.0,
            runner=runner,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(statuses[0], ("cpd-core", "published"))
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(len(runner.calls), len(crates_publish.PUBLISHABLE_CRATES) + 1)

    def test_publish_errors_when_auth_fails_without_token_fallback(self):
        runner = FakeRunner([_result(returncode=101, output="error: please run cargo login")])

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CARGO_REGISTRY_TOKEN"):
                crates_publish.publish_crates(
                    max_attempts=2,
                    retry_delay_seconds=1.0,
                    runner=runner,
                    sleep_fn=lambda _seconds: None,
                )


if __name__ == "__main__":
    unittest.main()
