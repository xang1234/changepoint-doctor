import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import coverage_gate  # noqa: E402


class CoverageGateTests(unittest.TestCase):
    def _write_lcov(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp.write(text)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_parse_lcov_collects_line_counts(self):
        path = self._write_lcov(
            """TN:
SF:/repo/cpd/crates/cpd-costs/src/l2.rs
DA:1,1
DA:2,0
end_of_record
SF:/repo/cpd/crates/cpd-costs/src/normal.rs
DA:1,1
DA:2,1
end_of_record
"""
        )

        records = coverage_gate.parse_lcov(path)
        l2 = records["/repo/cpd/crates/cpd-costs/src/l2.rs"]
        normal = records["/repo/cpd/crates/cpd-costs/src/normal.rs"]

        self.assertEqual((l2.covered, l2.total), (1, 2))
        self.assertEqual((normal.covered, normal.total), (2, 2))

    def test_evaluate_targets_passes_when_thresholds_met(self):
        records = {
            "/repo/cpd/crates/cpd-costs/src/l2.rs": coverage_gate.CoverageMetric(
                covered=97, total=100
            ),
            "/repo/cpd/crates/cpd-offline/src/pelt.rs": coverage_gate.CoverageMetric(
                covered=91, total=100
            ),
        }
        targets = [
            coverage_gate.CoverageTarget("crates/cpd-costs/src/l2.rs", 95.0),
            coverage_gate.CoverageTarget("crates/cpd-offline/src/pelt.rs", 90.0),
        ]

        failures = coverage_gate.evaluate_targets(records, targets)
        self.assertEqual(failures, [])

    def test_evaluate_targets_reports_missing_and_under_threshold(self):
        records = {
            "/repo/cpd/crates/cpd-costs/src/l2.rs": coverage_gate.CoverageMetric(
                covered=90, total=100
            )
        }
        targets = [
            coverage_gate.CoverageTarget("crates/cpd-costs/src/l2.rs", 95.0),
            coverage_gate.CoverageTarget("crates/cpd-offline/src/binseg.rs", 90.0),
        ]

        failures = coverage_gate.evaluate_targets(records, targets)
        self.assertEqual(len(failures), 2)
        self.assertIn("l2.rs", failures[0])
        self.assertIn("missing coverage data", failures[1])


if __name__ == "__main__":
    unittest.main()
