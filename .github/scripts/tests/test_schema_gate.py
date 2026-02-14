import copy
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema_gate  # noqa: E402


class SchemaGateTests(unittest.TestCase):
    @staticmethod
    def _valid_config_fixture() -> dict:
        return {
            "schema_version": 0,
            "kind": "pipeline_spec",
            "payload": {
                "preprocess": {
                    "detrend": {"method": "polynomial", "degree": 2},
                    "deseasonalize": {"method": "stl_like", "period": 4},
                    "winsorize": {"lower_quantile": 0.05, "upper_quantile": 0.95},
                    "robust_scale": {"mad_epsilon": 1.0e-9, "normal_consistency": 1.4826},
                }
            },
        }

    @staticmethod
    def _load_config_schema() -> dict:
        schema_path = (
            schema_gate.CPD_ROOT / "schemas" / "config" / "pipeline_spec.v0.schema.json"
        )
        return schema_gate._as_dict(schema_gate._read_json(schema_path), str(schema_path))

    @staticmethod
    def _valid_constraints_migration_fixture() -> dict:
        return {
            "schema_version": 1,
            "min_segment_len": 2,
            "max_change_points": None,
            "max_depth": None,
            "candidate_splits": None,
            "jump": 1,
            "time_budget_ms": None,
            "max_cost_evals": None,
            "memory_budget_bytes": None,
            "max_cache_bytes": None,
            "cache_policy": "Full",
            "degradation_plan": [],
            "allow_algorithm_fallback": False,
        }

    @staticmethod
    def _valid_offline_config_migration_fixture() -> dict:
        return {
            "schema_version": 1,
            "stopping": {"Penalized": "BIC"},
            "params_per_segment": 2,
            "cancel_check_every": 1000,
        }

    @staticmethod
    def _valid_result_fixture_with_version(schema_version: int) -> dict:
        return {
            "breakpoints": [10, 20],
            "change_points": [10],
            "scores": [0.5],
            "segments": [
                {
                    "start": 0,
                    "end": 10,
                    "mean": [1.0],
                    "variance": [0.1],
                    "count": 10,
                    "missing_count": 0,
                },
                {
                    "start": 10,
                    "end": 20,
                    "mean": [2.0],
                    "variance": [0.2],
                    "count": 10,
                    "missing_count": 0,
                },
            ],
            "diagnostics": {
                "n": 20,
                "d": 1,
                "schema_version": schema_version,
                "algorithm": "pelt",
                "cost_model": "l2_mean",
                "repro_mode": "Balanced",
                "notes": [],
                "warnings": [],
            },
        }

    @staticmethod
    def _valid_diagnostics_migration_fixture(schema_version: int) -> dict:
        return {
            "n": 20,
            "d": 1,
            "schema_version": schema_version,
            "algorithm": "pelt",
            "cost_model": "l2_mean",
            "repro_mode": "Balanced",
            "notes": [],
            "warnings": [],
        }

    def test_validate_repo_passes_for_workspace(self):
        errors = schema_gate.validate_repo(schema_gate.REPO_ROOT)
        self.assertEqual(errors, [])

    def test_config_fixture_requires_schema_version_marker(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            schema_gate.validate_config_fixture(
                {
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_config_fixture_rejects_wrong_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema_version must be 0"):
            schema_gate.validate_config_fixture(
                {
                    "schema_version": 1,
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_config_schema_rejects_missing_preprocess_contract(self):
        schema = self._load_config_schema()
        broken = copy.deepcopy(schema)
        del broken["properties"]["payload"]["properties"]["preprocess"]
        with self.assertRaisesRegex(ValueError, "preprocess"):
            schema_gate.validate_config_schema(broken)

    def test_config_fixture_accepts_valid_preprocess_contract(self):
        schema_gate.validate_config_fixture(self._valid_config_fixture())

    def test_config_fixture_rejects_unknown_preprocess_stage(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["unknown_stage"] = {}
        with self.assertRaisesRegex(ValueError, "unsupported preprocess keys"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_invalid_detrend_method(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["detrend"] = {"method": "cubic"}
        with self.assertRaisesRegex(ValueError, "detrend\\.method"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_polynomial_without_degree(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["detrend"] = {"method": "polynomial"}
        with self.assertRaisesRegex(ValueError, "degree >= 1"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_stl_like_period_below_two(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["deseasonalize"] = {
            "method": "stl_like",
            "period": 1,
        }
        with self.assertRaisesRegex(ValueError, ">= 2"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_invalid_winsorize_quantiles(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["winsorize"] = {
            "lower_quantile": 0.8,
            "upper_quantile": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "lower_quantile < upper_quantile"):
            schema_gate.validate_config_fixture(fixture)

    def test_config_fixture_rejects_nonpositive_robust_scale(self):
        fixture = self._valid_config_fixture()
        fixture["payload"]["preprocess"]["robust_scale"] = {
            "mad_epsilon": 0.0,
            "normal_consistency": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            schema_gate.validate_config_fixture(fixture)

    def test_checkpoint_fixture_rejects_bad_crc(self):
        with self.assertRaisesRegex(ValueError, "payload_crc32"):
            schema_gate.validate_checkpoint_fixture(
                {
                    "schema_version": 0,
                    "detector_id": "bocpd",
                    "engine_version": "0.1.0",
                    "created_at_ns": 1,
                    "payload_codec": "json",
                    "payload_crc32": "DEADBEEF",
                    "payload": {},
                }
            )

    def test_result_fixture_rejects_v2_by_default(self):
        fixture = self._valid_result_fixture_with_version(2)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            schema_gate.validate_result_fixture(fixture)

    def test_result_fixture_accepts_v2_when_requested(self):
        fixture = self._valid_result_fixture_with_version(2)
        schema_gate.validate_result_fixture(
            fixture, schema_gate.MIGRATION_SUPPORTED_SCHEMA_VERSIONS
        )

    def test_constraints_migration_fixture_rejects_unsupported_version(self):
        fixture = self._valid_constraints_migration_fixture()
        fixture["schema_version"] = 7
        with self.assertRaisesRegex(ValueError, "schema_version"):
            schema_gate.validate_constraints_migration_fixture(fixture)

    def test_offline_config_migration_fixture_requires_stopping_object(self):
        fixture = self._valid_offline_config_migration_fixture()
        fixture["stopping"] = "Penalized"
        with self.assertRaisesRegex(ValueError, "must be an object"):
            schema_gate.validate_offline_config_migration_fixture(
                fixture, "pelt migration fixture"
            )

    def test_diagnostics_migration_fixture_accepts_v2(self):
        fixture = self._valid_diagnostics_migration_fixture(2)
        schema_gate.validate_diagnostics_migration_fixture(fixture)

    def test_constraints_migration_fixture_rejects_invalid_cache_policy_shape(self):
        fixture = self._valid_constraints_migration_fixture()
        fixture["cache_policy"] = "Budgeted"
        with self.assertRaisesRegex(ValueError, "string variant must be 'Full'"):
            schema_gate.validate_constraints_migration_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
