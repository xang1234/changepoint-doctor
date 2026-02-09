import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_gate  # noqa: E402


def _report(vulns):
    return {"vulnerabilities": {"list": vulns}}


def _vuln(advisory_id, severity=None, cvss=None):
    advisory = {"id": advisory_id}
    if severity is not None:
        advisory["severity"] = severity
    if cvss is not None:
        advisory["cvss"] = cvss
    return {"advisory": advisory}


class AuditGateTests(unittest.TestCase):
    def test_no_vulnerabilities(self):
        blocking, non_blocking = audit_gate.evaluate_report(_report([]))
        self.assertEqual(blocking, [])
        self.assertEqual(non_blocking, 0)

    def test_medium_vulnerability_is_non_blocking(self):
        report = _report([_vuln("RUSTSEC-0001-0001", severity="medium")])
        blocking, non_blocking = audit_gate.evaluate_report(report)
        self.assertEqual(blocking, [])
        self.assertEqual(non_blocking, 1)

    def test_high_vulnerability_blocks(self):
        report = _report([_vuln("RUSTSEC-0002-0002", severity="high")])
        blocking, non_blocking = audit_gate.evaluate_report(report)
        self.assertEqual(blocking, ["RUSTSEC-0002-0002"])
        self.assertEqual(non_blocking, 0)

    def test_critical_vulnerability_blocks(self):
        report = _report([_vuln("RUSTSEC-0003-0003", severity="critical")])
        blocking, non_blocking = audit_gate.evaluate_report(report)
        self.assertEqual(blocking, ["RUSTSEC-0003-0003"])
        self.assertEqual(non_blocking, 0)

    def test_missing_severity_and_cvss_is_blocking(self):
        report = _report([_vuln("RUSTSEC-0004-0004")])
        blocking, non_blocking = audit_gate.evaluate_report(report)
        self.assertEqual(blocking, ["RUSTSEC-0004-0004"])
        self.assertEqual(non_blocking, 0)


if __name__ == "__main__":
    unittest.main()
