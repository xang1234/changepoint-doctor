import json
import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_gate  # noqa: E402


def _payload(entries):
    return {"metadata": {"source": "unit-test"}, "benchmarks": entries}


def _entry(case_id, runtime_seconds, max_rss_kib):
    return {
        "id": case_id,
        "runtime_seconds": runtime_seconds,
        "max_rss_kib": max_rss_kib,
    }


def _manifest(cases):
    return {"manifest_version": 1, "cases": cases}


def _case(case_id, bench, criterion_id=None, absolute_thresholds=None):
    payload = {"id": case_id, "bench": bench}
    if criterion_id is not None:
        payload["criterion_id"] = criterion_id
    if absolute_thresholds is not None:
        payload["absolute_thresholds"] = absolute_thresholds
    return payload


class BenchmarkGateTests(unittest.TestCase):
    def test_load_manifest_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    _manifest(
                        [
                            _case("dup", "offline_pelt"),
                            _case("dup", "offline_binseg"),
                        ]
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                benchmark_gate.load_manifest(path)

    def test_parse_runtime_ns_reads_criterion_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimates.json"
            path.write_text(
                json.dumps({"median": {"point_estimate": 1234567.0}}),
                encoding="utf-8",
            )
            runtime_ns = benchmark_gate._parse_runtime_ns(path)
            self.assertEqual(runtime_ns, 1234567.0)

    def test_parse_max_rss_kib_gnu(self):
        out = "Maximum resident set size (kbytes): 4242"
        self.assertEqual(benchmark_gate._parse_max_rss_kib(out, "gnu"), 4242)

    def test_compare_relative_pass_within_thresholds(self):
        baseline = _payload([_entry("offline_pelt", 10.0, 1000)])
        current = _payload([_entry("offline_pelt", 10.9, 1140)])

        failures = benchmark_gate.compare_relative_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )
        self.assertEqual(failures, [])

    def test_compare_relative_fails_missing_in_current(self):
        baseline = _payload(
            [
                _entry("offline_pelt", 10.0, 1000),
                _entry("offline_binseg", 1.0, 200),
            ]
        )
        current = _payload([_entry("offline_pelt", 10.0, 1000)])

        failures = benchmark_gate.compare_relative_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("missing current metric", failures[0])

    def test_compare_absolute_enforces_thresholded_cases_only(self):
        cases = [
            benchmark_gate.BenchmarkCase(
                id="thresholded",
                bench="offline_pelt",
                criterion_id="thresholded",
                max_runtime_seconds=0.5,
                max_rss_kib=2000,
            ),
            benchmark_gate.BenchmarkCase(
                id="baseline_only",
                bench="offline_binseg",
                criterion_id="baseline_only",
                max_runtime_seconds=None,
                max_rss_kib=None,
            ),
        ]

        current = _payload(
            [
                _entry("thresholded", 0.7, 1500),
                _entry("baseline_only", 99.0, 999999),
            ]
        )

        failures = benchmark_gate.compare_absolute_metrics(cases, current)
        self.assertEqual(len(failures), 1)
        self.assertIn("runtime", failures[0])
        self.assertIn("thresholded", failures[0])

    def test_main_compare_absolute_fails_on_threshold_breach(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            _case(
                                "case_a",
                                "offline_pelt",
                                absolute_thresholds={"max_runtime_seconds": 1.0},
                            )
                        ]
                    )
                ),
                encoding="utf-8",
            )
            current_path = tmp_path / "current.json"
            current_path.write_text(
                json.dumps(_payload([_entry("case_a", 2.0, 1234)])),
                encoding="utf-8",
            )

            exit_code = benchmark_gate.main(
                [
                    "compare-absolute",
                    "--manifest",
                    str(manifest_path),
                    "--current",
                    str(current_path),
                ]
            )
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
