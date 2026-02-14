#!/usr/bin/env python3
"""Gate schema and fixture compatibility contracts for CPD-kvd.7."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
CRC32_RE = re.compile(r"^[0-9a-f]{8}$")

RESULT_REQUIRED_ROOT = {"breakpoints", "change_points", "diagnostics"}
MIGRATION_SUPPORTED_SCHEMA_VERSIONS = {1, 2}
RESULT_DIAGNOSTICS_REQUIRED = {
    "n",
    "d",
    "schema_version",
    "algorithm",
    "cost_model",
    "repro_mode",
}
SEGMENT_REQUIRED = {"start", "end", "count", "missing_count"}
CONFIG_REQUIRED = {"schema_version", "kind", "payload"}
CONSTRAINTS_MIGRATION_REQUIRED = {
    "schema_version",
    "min_segment_len",
    "max_change_points",
    "max_depth",
    "candidate_splits",
    "jump",
    "time_budget_ms",
    "max_cost_evals",
    "memory_budget_bytes",
    "max_cache_bytes",
    "cache_policy",
    "degradation_plan",
    "allow_algorithm_fallback",
}
OFFLINE_CONFIG_MIGRATION_REQUIRED = {
    "schema_version",
    "stopping",
    "params_per_segment",
    "cancel_check_every",
}
CHECKPOINT_REQUIRED = {
    "schema_version",
    "detector_id",
    "engine_version",
    "created_at_ns",
    "payload_codec",
    "payload_crc32",
    "payload",
}
PREPROCESS_STAGE_KEYS = {"detrend", "deseasonalize", "winsorize", "robust_scale"}
PREPROCESS_STAGE_REFS = {
    "detrend": "#/$defs/detrendConfig",
    "deseasonalize": "#/$defs/deseasonalizeConfig",
    "winsorize": "#/$defs/winsorizeConfig",
    "robust_scale": "#/$defs/robustScaleConfig",
}
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99
DEFAULT_MAD_EPSILON = 1.0e-9
DEFAULT_NORMAL_CONSISTENCY = 1.4826

REPO_ROOT = Path(__file__).resolve().parents[2]
CPD_ROOT = REPO_ROOT / "cpd"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _as_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}")
    except OSError as exc:  # pragma: no cover - defensive CI guard
        raise ValueError(f"failed to read {path}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc


def _require_keys(payload: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _format_versions(versions: set[int]) -> str:
    return ", ".join(str(version) for version in sorted(versions))


def _validate_schema_version(
    schema_version: Any, allowed_versions: set[int], context: str
) -> None:
    _require(
        isinstance(schema_version, int) and schema_version in allowed_versions,
        f"{context} must be one of {{{_format_versions(allowed_versions)}}}",
    )


def _validate_penalty_payload(payload: Any, context: str) -> None:
    if isinstance(payload, str):
        _require(
            payload in {"BIC", "AIC"},
            f"{context} must be 'BIC', 'AIC', or {{'Manual': <float>}}",
        )
        return

    penalty = _as_dict(payload, context)
    _require(
        set(penalty) == {"Manual"},
        f"{context} object variant must contain only 'Manual'",
    )
    manual = penalty.get("Manual")
    _require(
        _is_finite_number(manual) and float(manual) > 0.0,
        f"{context}.Manual must be finite and > 0",
    )


def _validate_stopping_payload(payload: Any, context: str) -> None:
    stopping = _as_dict(payload, context)
    _require(len(stopping) == 1, f"{context} must contain exactly one variant")
    variant, value = next(iter(stopping.items()))

    if variant == "KnownK":
        _require(
            isinstance(value, int) and value >= 1,
            f"{context}.KnownK must be int >= 1",
        )
        return

    if variant == "Penalized":
        _validate_penalty_payload(value, f"{context}.Penalized")
        return

    if variant == "PenaltyPath":
        penalties = _as_list(value, f"{context}.PenaltyPath")
        _require(
            len(penalties) > 0,
            f"{context}.PenaltyPath must be non-empty",
        )
        for idx, penalty in enumerate(penalties):
            _validate_penalty_payload(penalty, f"{context}.PenaltyPath[{idx}]")
        return

    raise ValueError(f"{context} has unsupported variant '{variant}'")


def _validate_cache_policy_payload(payload: Any, context: str) -> None:
    if isinstance(payload, str):
        _require(payload == "Full", f"{context} string variant must be 'Full'")
        return

    cache_policy = _as_dict(payload, context)
    _require(len(cache_policy) == 1, f"{context} must contain exactly one variant")
    variant, value = next(iter(cache_policy.items()))
    variant_payload = _as_dict(value, f"{context}.{variant}")

    if variant == "Budgeted":
        _require(
            set(variant_payload) == {"max_bytes"},
            f"{context}.Budgeted must contain only max_bytes",
        )
        _require(
            isinstance(variant_payload.get("max_bytes"), int)
            and variant_payload.get("max_bytes") >= 1,
            f"{context}.Budgeted.max_bytes must be int >= 1",
        )
        return

    if variant == "Approximate":
        _require(
            set(variant_payload) == {"max_bytes", "error_tolerance"},
            f"{context}.Approximate must contain max_bytes and error_tolerance",
        )
        _require(
            isinstance(variant_payload.get("max_bytes"), int)
            and variant_payload.get("max_bytes") >= 1,
            f"{context}.Approximate.max_bytes must be int >= 1",
        )
        tolerance = variant_payload.get("error_tolerance")
        _require(
            _is_finite_number(tolerance) and float(tolerance) > 0.0,
            f"{context}.Approximate.error_tolerance must be finite and > 0",
        )
        return

    raise ValueError(f"{context} has unsupported variant '{variant}'")


def _validate_preprocess_fixture(preprocess: dict[str, Any], context: str) -> None:
    unknown_stage_keys = sorted(set(preprocess) - PREPROCESS_STAGE_KEYS)
    _require(
        not unknown_stage_keys,
        f"{context} has unsupported preprocess keys: {', '.join(unknown_stage_keys)}",
    )
    missing_stage_keys = sorted(PREPROCESS_STAGE_KEYS - set(preprocess))
    _require(
        not missing_stage_keys,
        f"{context} missing preprocess stage coverage: {', '.join(missing_stage_keys)}",
    )

    detrend = _as_dict(preprocess.get("detrend"), f"{context}.detrend")
    detrend_keys = set(detrend)
    _require(
        detrend_keys.issubset({"method", "degree"}),
        f"{context}.detrend contains unsupported keys",
    )
    detrend_method = detrend.get("method")
    _require(
        isinstance(detrend_method, str),
        f"{context}.detrend.method must be a string",
    )
    if detrend_method == "linear":
        _require(
            "degree" not in detrend,
            f"{context}.detrend.method=linear must not include degree",
        )
    elif detrend_method == "polynomial":
        degree = detrend.get("degree")
        _require(
            isinstance(degree, int) and degree >= 1,
            f"{context}.detrend.method=polynomial requires degree >= 1",
        )
    else:
        raise ValueError(
            f"{context}.detrend.method must be one of linear|polynomial, got {detrend_method!r}"
        )

    deseasonalize = _as_dict(preprocess.get("deseasonalize"), f"{context}.deseasonalize")
    deseasonalize_keys = set(deseasonalize)
    _require(
        deseasonalize_keys.issubset({"method", "period"}),
        f"{context}.deseasonalize contains unsupported keys",
    )
    deseasonalize_method = deseasonalize.get("method")
    _require(
        isinstance(deseasonalize_method, str),
        f"{context}.deseasonalize.method must be a string",
    )
    period = deseasonalize.get("period")
    _require(
        isinstance(period, int),
        f"{context}.deseasonalize.period must be an integer",
    )
    if deseasonalize_method == "differencing":
        _require(period >= 1, f"{context}.deseasonalize.period must be >= 1 for differencing")
    elif deseasonalize_method == "stl_like":
        _require(period >= 2, f"{context}.deseasonalize.period must be >= 2 for stl_like")
    else:
        raise ValueError(
            f"{context}.deseasonalize.method must be one of differencing|stl_like, got {deseasonalize_method!r}"
        )

    winsorize = _as_dict(preprocess.get("winsorize"), f"{context}.winsorize")
    winsorize_keys = set(winsorize)
    _require(
        winsorize_keys.issubset({"lower_quantile", "upper_quantile"}),
        f"{context}.winsorize contains unsupported keys",
    )
    lower_quantile_raw = winsorize.get("lower_quantile", DEFAULT_WINSOR_LOWER)
    upper_quantile_raw = winsorize.get("upper_quantile", DEFAULT_WINSOR_UPPER)
    _require(
        _is_finite_number(lower_quantile_raw) and _is_finite_number(upper_quantile_raw),
        f"{context}.winsorize quantiles must be finite numbers",
    )
    lower_quantile = float(lower_quantile_raw)
    upper_quantile = float(upper_quantile_raw)
    _require(
        0.0 <= lower_quantile <= 1.0 and 0.0 <= upper_quantile <= 1.0,
        f"{context}.winsorize quantiles must satisfy 0.0 <= q <= 1.0",
    )
    _require(
        lower_quantile < upper_quantile,
        f"{context}.winsorize requires lower_quantile < upper_quantile",
    )

    robust_scale = _as_dict(preprocess.get("robust_scale"), f"{context}.robust_scale")
    robust_scale_keys = set(robust_scale)
    _require(
        robust_scale_keys.issubset({"mad_epsilon", "normal_consistency"}),
        f"{context}.robust_scale contains unsupported keys",
    )
    mad_epsilon_raw = robust_scale.get("mad_epsilon", DEFAULT_MAD_EPSILON)
    normal_consistency_raw = robust_scale.get(
        "normal_consistency", DEFAULT_NORMAL_CONSISTENCY
    )
    _require(
        _is_finite_number(mad_epsilon_raw) and _is_finite_number(normal_consistency_raw),
        f"{context}.robust_scale parameters must be finite numbers",
    )
    _require(
        float(mad_epsilon_raw) > 0.0 and float(normal_consistency_raw) > 0.0,
        f"{context}.robust_scale parameters must be > 0",
    )


def validate_result_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "result schema must declare JSON Schema Draft 2020-12",
    )
    _require(schema.get("type") == "object", "result schema root type must be object")
    _require(
        schema.get("additionalProperties") is True,
        "result schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "result schema.required"))
    missing_required = RESULT_REQUIRED_ROOT - required
    _require(
        not missing_required,
        f"result schema.required missing: {', '.join(sorted(missing_required))}",
    )

    properties = _as_dict(schema.get("properties"), "result schema.properties")
    for field in RESULT_REQUIRED_ROOT.union({"scores", "segments"}):
        _require(field in properties, f"result schema.properties missing '{field}'")

    defs = _as_dict(schema.get("$defs"), "result schema.$defs")
    diagnostics = _as_dict(defs.get("diagnostics"), "result schema.$defs.diagnostics")
    diag_required = set(_as_list(diagnostics.get("required"), "result diagnostics.required"))
    missing_diag_required = RESULT_DIAGNOSTICS_REQUIRED - diag_required
    _require(
        not missing_diag_required,
        "result diagnostics.required missing: "
        + ", ".join(sorted(missing_diag_required)),
    )

    diag_properties = _as_dict(
        diagnostics.get("properties"), "result diagnostics.properties"
    )
    schema_version = _as_dict(
        diag_properties.get("schema_version"), "result diagnostics.schema_version"
    )
    _require(
        schema_version.get("type") == "integer",
        "result diagnostics.schema_version type must be integer",
    )


def validate_config_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "config schema must declare JSON Schema Draft 2020-12",
    )
    _require(schema.get("type") == "object", "config schema root type must be object")
    _require(
        schema.get("additionalProperties") is True,
        "config schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "config schema.required"))
    _require_keys({key: True for key in required}, CONFIG_REQUIRED, "config schema.required")

    properties = _as_dict(schema.get("properties"), "config schema.properties")
    schema_version = _as_dict(properties.get("schema_version"), "config schema.schema_version")
    kind = _as_dict(properties.get("kind"), "config schema.kind")
    _require(
        schema_version.get("const") == 0,
        "config schema.schema_version const must be 0",
    )
    _require(
        kind.get("const") == "pipeline_spec",
        "config schema.kind const must be 'pipeline_spec'",
    )

    payload = _as_dict(properties.get("payload"), "config schema.payload")
    _require(payload.get("type") == "object", "config schema.payload type must be object")
    payload_properties = _as_dict(
        payload.get("properties"), "config schema.payload.properties"
    )
    preprocess_ref = _as_dict(
        payload_properties.get("preprocess"), "config schema.payload.preprocess"
    )
    _require(
        preprocess_ref.get("$ref") == "#/$defs/preprocessConfig",
        "config schema.payload.preprocess must reference #/$defs/preprocessConfig",
    )

    defs = _as_dict(schema.get("$defs"), "config schema.$defs")
    preprocess_config = _as_dict(defs.get("preprocessConfig"), "config schema.$defs.preprocessConfig")
    _require(
        preprocess_config.get("type") == "object",
        "config schema.$defs.preprocessConfig type must be object",
    )
    _require(
        preprocess_config.get("additionalProperties") is False,
        "config schema.$defs.preprocessConfig must set additionalProperties=false",
    )
    preprocess_properties = _as_dict(
        preprocess_config.get("properties"), "config schema.$defs.preprocessConfig.properties"
    )
    _require(
        set(preprocess_properties) == PREPROCESS_STAGE_KEYS,
        "config schema.$defs.preprocessConfig.properties must include only detrend, deseasonalize, winsorize, robust_scale",
    )

    for stage in sorted(PREPROCESS_STAGE_KEYS):
        stage_schema = _as_dict(
            preprocess_properties.get(stage),
            f"config schema.$defs.preprocessConfig.properties.{stage}",
        )
        any_of = _as_list(stage_schema.get("anyOf"), f"config schema preprocess stage {stage}.anyOf")
        _require(len(any_of) == 2, f"config schema preprocess stage {stage} must have exactly two anyOf entries")
        stage_ref = PREPROCESS_STAGE_REFS[stage]
        has_ref = any(
            isinstance(entry, dict) and entry.get("$ref") == stage_ref for entry in any_of
        )
        has_null = any(
            isinstance(entry, dict) and entry.get("type") == "null" for entry in any_of
        )
        _require(
            has_ref and has_null,
            f"config schema preprocess stage {stage} must allow null or {stage_ref}",
        )

    detrend_schema = _as_dict(defs.get("detrendConfig"), "config schema.$defs.detrendConfig")
    _require(
        detrend_schema.get("type") == "object",
        "config schema.$defs.detrendConfig type must be object",
    )
    detrend_variants = _as_list(detrend_schema.get("oneOf"), "config schema.$defs.detrendConfig.oneOf")
    _require(
        len(detrend_variants) == 2,
        "config schema.$defs.detrendConfig.oneOf must contain exactly 2 variants",
    )
    detrend_by_method: dict[str, dict[str, Any]] = {}
    for idx, variant_value in enumerate(detrend_variants):
        variant = _as_dict(
            variant_value, f"config schema.$defs.detrendConfig.oneOf[{idx}]"
        )
        _require(
            variant.get("additionalProperties") is False,
            f"config schema.$defs.detrendConfig.oneOf[{idx}] must set additionalProperties=false",
        )
        variant_properties = _as_dict(
            variant.get("properties"),
            f"config schema.$defs.detrendConfig.oneOf[{idx}].properties",
        )
        method = _as_dict(
            variant_properties.get("method"),
            f"config schema.$defs.detrendConfig.oneOf[{idx}].properties.method",
        ).get("const")
        _require(
            isinstance(method, str),
            f"config schema.$defs.detrendConfig.oneOf[{idx}] method const must be a string",
        )
        detrend_by_method[method] = variant
    _require(
        set(detrend_by_method) == {"linear", "polynomial"},
        "config schema.$defs.detrendConfig must define linear and polynomial variants",
    )
    polynomial_props = _as_dict(
        detrend_by_method["polynomial"].get("properties"),
        "config schema.$defs.detrendConfig polynomial properties",
    )
    degree_schema = _as_dict(
        polynomial_props.get("degree"),
        "config schema.$defs.detrendConfig polynomial degree",
    )
    _require(
        degree_schema.get("type") == "integer" and degree_schema.get("minimum") == 1,
        "config schema detrend polynomial degree must be integer with minimum=1",
    )

    deseasonalize_schema = _as_dict(
        defs.get("deseasonalizeConfig"), "config schema.$defs.deseasonalizeConfig"
    )
    _require(
        deseasonalize_schema.get("type") == "object",
        "config schema.$defs.deseasonalizeConfig type must be object",
    )
    deseasonalize_variants = _as_list(
        deseasonalize_schema.get("oneOf"), "config schema.$defs.deseasonalizeConfig.oneOf"
    )
    _require(
        len(deseasonalize_variants) == 2,
        "config schema.$defs.deseasonalizeConfig.oneOf must contain exactly 2 variants",
    )
    deseasonalize_by_method: dict[str, dict[str, Any]] = {}
    for idx, variant_value in enumerate(deseasonalize_variants):
        variant = _as_dict(
            variant_value, f"config schema.$defs.deseasonalizeConfig.oneOf[{idx}]"
        )
        _require(
            variant.get("additionalProperties") is False,
            f"config schema.$defs.deseasonalizeConfig.oneOf[{idx}] must set additionalProperties=false",
        )
        variant_properties = _as_dict(
            variant.get("properties"),
            f"config schema.$defs.deseasonalizeConfig.oneOf[{idx}].properties",
        )
        method = _as_dict(
            variant_properties.get("method"),
            f"config schema.$defs.deseasonalizeConfig.oneOf[{idx}].properties.method",
        ).get("const")
        _require(
            isinstance(method, str),
            f"config schema.$defs.deseasonalizeConfig.oneOf[{idx}] method const must be a string",
        )
        deseasonalize_by_method[method] = variant_properties
    _require(
        set(deseasonalize_by_method) == {"differencing", "stl_like"},
        "config schema.$defs.deseasonalizeConfig must define differencing and stl_like variants",
    )
    diff_period = _as_dict(
        deseasonalize_by_method["differencing"].get("period"),
        "config schema.$defs.deseasonalizeConfig differencing period",
    )
    stl_period = _as_dict(
        deseasonalize_by_method["stl_like"].get("period"),
        "config schema.$defs.deseasonalizeConfig stl_like period",
    )
    _require(
        diff_period.get("type") == "integer" and diff_period.get("minimum") == 1,
        "config schema differencing period must be integer with minimum=1",
    )
    _require(
        stl_period.get("type") == "integer" and stl_period.get("minimum") == 2,
        "config schema stl_like period must be integer with minimum=2",
    )

    winsorize_schema = _as_dict(
        defs.get("winsorizeConfig"), "config schema.$defs.winsorizeConfig"
    )
    _require(
        winsorize_schema.get("type") == "object"
        and winsorize_schema.get("additionalProperties") is False,
        "config schema.$defs.winsorizeConfig must be closed object",
    )
    winsorize_properties = _as_dict(
        winsorize_schema.get("properties"), "config schema.$defs.winsorizeConfig.properties"
    )
    for key in ("lower_quantile", "upper_quantile"):
        quantile = _as_dict(
            winsorize_properties.get(key),
            f"config schema.$defs.winsorizeConfig.{key}",
        )
        _require(
            quantile.get("type") == "number"
            and quantile.get("minimum") == 0.0
            and quantile.get("maximum") == 1.0,
            f"config schema.$defs.winsorizeConfig.{key} must be number within [0, 1]",
        )

    robust_scale_schema = _as_dict(
        defs.get("robustScaleConfig"), "config schema.$defs.robustScaleConfig"
    )
    _require(
        robust_scale_schema.get("type") == "object"
        and robust_scale_schema.get("additionalProperties") is False,
        "config schema.$defs.robustScaleConfig must be closed object",
    )
    robust_scale_properties = _as_dict(
        robust_scale_schema.get("properties"),
        "config schema.$defs.robustScaleConfig.properties",
    )
    for key in ("mad_epsilon", "normal_consistency"):
        value_schema = _as_dict(
            robust_scale_properties.get(key),
            f"config schema.$defs.robustScaleConfig.{key}",
        )
        _require(
            value_schema.get("type") == "number"
            and value_schema.get("exclusiveMinimum") == 0.0,
            f"config schema.$defs.robustScaleConfig.{key} must be number with exclusiveMinimum=0",
        )


def validate_checkpoint_schema(schema: dict[str, Any]) -> None:
    _require(
        schema.get("$schema") == SCHEMA_DRAFT_2020_12,
        "checkpoint schema must declare JSON Schema Draft 2020-12",
    )
    _require(
        schema.get("type") == "object", "checkpoint schema root type must be object"
    )
    _require(
        schema.get("additionalProperties") is True,
        "checkpoint schema must allow additionalProperties=true",
    )

    required = set(_as_list(schema.get("required"), "checkpoint schema.required"))
    missing_required = CHECKPOINT_REQUIRED - required
    _require(
        not missing_required,
        "checkpoint schema.required missing: " + ", ".join(sorted(missing_required)),
    )

    properties = _as_dict(schema.get("properties"), "checkpoint schema.properties")
    schema_version = _as_dict(
        properties.get("schema_version"), "checkpoint schema.schema_version"
    )
    payload_crc32 = _as_dict(
        properties.get("payload_crc32"), "checkpoint schema.payload_crc32"
    )
    _require(
        schema_version.get("const") == 0,
        "checkpoint schema.schema_version const must be 0",
    )
    _require(
        payload_crc32.get("pattern") == CRC32_RE.pattern,
        "checkpoint schema.payload_crc32 pattern must be ^[0-9a-f]{8}$",
    )


def _validate_diagnostics_object(
    diagnostics: dict[str, Any], context: str, allowed_schema_versions: set[int]
) -> tuple[int, int]:
    _require_keys(diagnostics, RESULT_DIAGNOSTICS_REQUIRED, context)

    n = diagnostics.get("n")
    d = diagnostics.get("d")
    schema_version = diagnostics.get("schema_version")
    _require(isinstance(n, int) and n >= 0, f"{context}.n must be int >= 0")
    _require(isinstance(d, int) and d >= 0, f"{context}.d must be int >= 0")
    _validate_schema_version(
        schema_version, allowed_schema_versions, f"{context}.schema_version"
    )

    _require(
        isinstance(diagnostics.get("algorithm"), str)
        and bool(diagnostics.get("algorithm")),
        f"{context}.algorithm must be a non-empty string",
    )
    _require(
        isinstance(diagnostics.get("cost_model"), str)
        and bool(diagnostics.get("cost_model")),
        f"{context}.cost_model must be a non-empty string",
    )
    _require(
        isinstance(diagnostics.get("repro_mode"), str)
        and bool(diagnostics.get("repro_mode")),
        f"{context}.repro_mode must be a non-empty string",
    )
    return n, d


def validate_result_fixture(
    payload: dict[str, Any], allowed_schema_versions: set[int] | None = None
) -> None:
    _require_keys(payload, RESULT_REQUIRED_ROOT, "result fixture")

    breakpoints = _as_list(payload.get("breakpoints"), "result fixture.breakpoints")
    change_points = _as_list(payload.get("change_points"), "result fixture.change_points")
    diagnostics = _as_dict(payload.get("diagnostics"), "result fixture.diagnostics")
    versions = allowed_schema_versions if allowed_schema_versions is not None else {1}
    n, d = _validate_diagnostics_object(
        diagnostics, "result fixture.diagnostics", versions
    )

    _require(
        all(isinstance(bp, int) and bp >= 0 for bp in breakpoints),
        "result fixture.breakpoints must contain int >= 0",
    )
    _require(
        all(
            breakpoints[idx] > breakpoints[idx - 1]
            for idx in range(1, len(breakpoints))
        ),
        "result fixture.breakpoints must be strictly increasing",
    )
    if n == 0:
        _require(not breakpoints, "result fixture.breakpoints must be empty when n=0")
    else:
        _require(breakpoints, "result fixture.breakpoints must be non-empty when n>0")
        _require(
            breakpoints[-1] == n,
            "result fixture.breakpoints must include n as final element",
        )
        _require(
            all(bp > 0 for bp in breakpoints),
            "result fixture.breakpoints must be > 0 when n>0",
        )

    _require(
        all(isinstance(cp, int) and cp >= 0 for cp in change_points),
        "result fixture.change_points must contain int >= 0",
    )
    expected_change_points = [bp for bp in breakpoints if bp < n]
    _require(
        change_points == expected_change_points,
        "result fixture.change_points must equal breakpoints excluding n",
    )

    scores = payload.get("scores")
    if scores is not None:
        scores_array = _as_list(scores, "result fixture.scores")
        _require(
            all(_is_number(score) for score in scores_array),
            "result fixture.scores must contain numbers",
        )
        _require(
            len(scores_array) == len(change_points),
            "result fixture.scores length must equal change_points length",
        )

    segments = payload.get("segments")
    if segments is None:
        return

    segments_array = _as_list(segments, "result fixture.segments")
    _require(
        len(segments_array) == len(breakpoints),
        "result fixture.segments length must equal breakpoints length",
    )

    expected_start = 0
    for idx, segment_value in enumerate(segments_array):
        segment = _as_dict(segment_value, f"result fixture.segments[{idx}]")
        _require_keys(segment, SEGMENT_REQUIRED, f"result fixture.segments[{idx}]")
        start = segment.get("start")
        end = segment.get("end")
        count = segment.get("count")
        missing_count = segment.get("missing_count")

        _require(
            isinstance(start, int) and isinstance(end, int) and start >= 0 and end >= start,
            f"result fixture.segments[{idx}] start/end must satisfy 0 <= start <= end",
        )
        _require(
            isinstance(count, int) and count == (end - start),
            f"result fixture.segments[{idx}].count must equal end-start",
        )
        _require(
            isinstance(missing_count, int) and 0 <= missing_count <= count,
            f"result fixture.segments[{idx}].missing_count must be within [0, count]",
        )

        expected_end = breakpoints[idx]
        _require(
            start == expected_start and end == expected_end,
            f"result fixture.segments[{idx}] boundaries must match breakpoints",
        )
        expected_start = expected_end

        for field in ("mean", "variance"):
            value = segment.get(field)
            if value is None:
                continue
            series = _as_list(value, f"result fixture.segments[{idx}].{field}")
            _require(
                all(_is_number(item) for item in series),
                f"result fixture.segments[{idx}].{field} must contain numbers",
            )
            if d > 0:
                _require(
                    len(series) == d,
                    f"result fixture.segments[{idx}].{field} length must equal diagnostics.d",
                )


def validate_constraints_migration_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, CONSTRAINTS_MIGRATION_REQUIRED, "constraints migration fixture")
    _validate_schema_version(
        payload.get("schema_version"),
        MIGRATION_SUPPORTED_SCHEMA_VERSIONS,
        "constraints migration fixture.schema_version",
    )

    min_segment_len = payload.get("min_segment_len")
    jump = payload.get("jump")
    _require(
        isinstance(min_segment_len, int) and min_segment_len >= 1,
        "constraints migration fixture.min_segment_len must be int >= 1",
    )
    _require(
        isinstance(jump, int) and jump >= 1,
        "constraints migration fixture.jump must be int >= 1",
    )

    for field in (
        "max_change_points",
        "max_depth",
        "time_budget_ms",
        "max_cost_evals",
        "memory_budget_bytes",
        "max_cache_bytes",
    ):
        value = payload.get(field)
        _require(
            value is None or (isinstance(value, int) and value >= 0),
            f"constraints migration fixture.{field} must be null or int >= 0",
        )

    candidate_splits = payload.get("candidate_splits")
    if candidate_splits is not None:
        splits = _as_list(candidate_splits, "constraints migration fixture.candidate_splits")
        _require(
            all(isinstance(split, int) and split > 0 for split in splits),
            "constraints migration fixture.candidate_splits must contain int > 0",
        )
        _require(
            all(splits[idx] > splits[idx - 1] for idx in range(1, len(splits))),
            "constraints migration fixture.candidate_splits must be strictly increasing and unique",
        )

    cache_policy = payload.get("cache_policy")
    _validate_cache_policy_payload(
        cache_policy, "constraints migration fixture.cache_policy"
    )

    degradation_plan = payload.get("degradation_plan")
    degradation_steps = _as_list(
        degradation_plan, "constraints migration fixture.degradation_plan"
    )
    for idx, step in enumerate(degradation_steps):
        context = f"constraints migration fixture.degradation_plan[{idx}]"
        if isinstance(step, str):
            _require(
                step == "DisableUncertaintyBands",
                f"{context} string variant must be 'DisableUncertaintyBands'",
            )
            continue

        step_obj = _as_dict(step, context)
        _require(len(step_obj) == 1, f"{context} must contain exactly one variant")
        variant, variant_payload = next(iter(step_obj.items()))
        if variant == "IncreaseJump":
            payload_obj = _as_dict(variant_payload, f"{context}.IncreaseJump")
            _require(
                set(payload_obj) == {"factor", "max_jump"},
                f"{context}.IncreaseJump must contain factor and max_jump",
            )
            _require(
                isinstance(payload_obj.get("factor"), int) and payload_obj.get("factor") >= 1,
                f"{context}.IncreaseJump.factor must be int >= 1",
            )
            _require(
                isinstance(payload_obj.get("max_jump"), int)
                and payload_obj.get("max_jump") >= 1,
                f"{context}.IncreaseJump.max_jump must be int >= 1",
            )
            continue

        if variant == "SwitchCachePolicy":
            _validate_cache_policy_payload(
                variant_payload, f"{context}.SwitchCachePolicy"
            )
            continue

        raise ValueError(f"{context} has unsupported variant '{variant}'")

    _require(
        isinstance(payload.get("allow_algorithm_fallback"), bool),
        "constraints migration fixture.allow_algorithm_fallback must be bool",
    )


def validate_offline_config_migration_fixture(
    payload: dict[str, Any], context: str
) -> None:
    _require_keys(payload, OFFLINE_CONFIG_MIGRATION_REQUIRED, context)
    _validate_schema_version(
        payload.get("schema_version"),
        MIGRATION_SUPPORTED_SCHEMA_VERSIONS,
        f"{context}.schema_version",
    )
    _validate_stopping_payload(payload.get("stopping"), f"{context}.stopping")
    _require(
        isinstance(payload.get("params_per_segment"), int)
        and payload.get("params_per_segment") >= 1,
        f"{context}.params_per_segment must be int >= 1",
    )
    _require(
        isinstance(payload.get("cancel_check_every"), int)
        and payload.get("cancel_check_every") >= 1,
        f"{context}.cancel_check_every must be int >= 1",
    )


def validate_diagnostics_migration_fixture(payload: dict[str, Any]) -> None:
    _validate_diagnostics_object(
        payload,
        "diagnostics migration fixture",
        MIGRATION_SUPPORTED_SCHEMA_VERSIONS,
    )


def validate_config_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, CONFIG_REQUIRED, "config fixture")
    _require(
        payload.get("schema_version") == 0, "config fixture schema_version must be 0"
    )
    _require(
        payload.get("kind") == "pipeline_spec",
        "config fixture kind must be 'pipeline_spec'",
    )
    fixture_payload = _as_dict(payload.get("payload"), "config fixture.payload")
    preprocess = _as_dict(fixture_payload.get("preprocess"), "config fixture.payload.preprocess")
    _validate_preprocess_fixture(preprocess, "config fixture.payload.preprocess")


def validate_checkpoint_fixture(payload: dict[str, Any]) -> None:
    _require_keys(payload, CHECKPOINT_REQUIRED, "checkpoint fixture")
    _require(
        payload.get("schema_version") == 0,
        "checkpoint fixture schema_version must be 0",
    )
    _require(
        isinstance(payload.get("detector_id"), str) and bool(payload.get("detector_id")),
        "checkpoint fixture detector_id must be a non-empty string",
    )
    _require(
        isinstance(payload.get("engine_version"), str)
        and bool(payload.get("engine_version")),
        "checkpoint fixture engine_version must be a non-empty string",
    )
    created_at_ns = payload.get("created_at_ns")
    _require(
        isinstance(created_at_ns, int) and created_at_ns >= 0,
        "checkpoint fixture created_at_ns must be int >= 0",
    )
    _require(
        isinstance(payload.get("payload_codec"), str) and bool(payload.get("payload_codec")),
        "checkpoint fixture payload_codec must be a non-empty string",
    )
    payload_crc32 = payload.get("payload_crc32")
    _require(
        isinstance(payload_crc32, str) and bool(CRC32_RE.fullmatch(payload_crc32)),
        "checkpoint fixture payload_crc32 must match ^[0-9a-f]{8}$",
    )
    _as_dict(payload.get("payload"), "checkpoint fixture.payload")


def _validate_required_coverage(
    fixture: dict[str, Any], schema: dict[str, Any], context: str
) -> None:
    required = _as_list(schema.get("required"), f"{context} schema.required")
    for key in required:
        _require(key in fixture, f"{context} fixture missing required schema field '{key}'")


def validate_repo(repo_root: Path) -> list[str]:
    cpd_root = repo_root / "cpd"
    result_schema_path = (
        cpd_root / "schemas" / "result" / "offline_change_point_result.v1.schema.json"
    )
    config_schema_path = cpd_root / "schemas" / "config" / "pipeline_spec.v0.schema.json"
    checkpoint_schema_path = (
        cpd_root
        / "schemas"
        / "checkpoint"
        / "online_detector_checkpoint.v0.schema.json"
    )
    result_fixture_path = (
        cpd_root / "crates" / "cpd-python" / "tests" / "fixtures" / "offline_result_v1.json"
    )
    config_fixture_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "schemas"
        / "config"
        / "pipeline_spec.v0.stub.json"
    )
    checkpoint_fixture_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "schemas"
        / "checkpoint"
        / "online_detector_checkpoint.v0.stub.json"
    )
    migration_constraints_v1_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "config" / "constraints.v1.json"
    )
    migration_constraints_v2_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "migrations"
        / "config"
        / "constraints.v2.additive.json"
    )
    migration_pelt_v1_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "config" / "pelt.v1.json"
    )
    migration_pelt_v2_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "config" / "pelt.v2.additive.json"
    )
    migration_binseg_v1_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "config" / "binseg.v1.json"
    )
    migration_binseg_v2_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "migrations"
        / "config"
        / "binseg.v2.additive.json"
    )
    migration_result_v1_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "result" / "offline_result.v1.json"
    )
    migration_result_v2_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "migrations"
        / "result"
        / "offline_result.v2.additive.json"
    )
    migration_diagnostics_v1_path = (
        cpd_root / "tests" / "fixtures" / "migrations" / "result" / "diagnostics.v1.json"
    )
    migration_diagnostics_v2_path = (
        cpd_root
        / "tests"
        / "fixtures"
        / "migrations"
        / "result"
        / "diagnostics.v2.additive.json"
    )

    errors: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    json_objects = {
        "result_schema": result_schema_path,
        "config_schema": config_schema_path,
        "checkpoint_schema": checkpoint_schema_path,
        "result_fixture": result_fixture_path,
        "config_fixture": config_fixture_path,
        "checkpoint_fixture": checkpoint_fixture_path,
        "migration_constraints_v1": migration_constraints_v1_path,
        "migration_constraints_v2": migration_constraints_v2_path,
        "migration_pelt_v1": migration_pelt_v1_path,
        "migration_pelt_v2": migration_pelt_v2_path,
        "migration_binseg_v1": migration_binseg_v1_path,
        "migration_binseg_v2": migration_binseg_v2_path,
        "migration_result_v1": migration_result_v1_path,
        "migration_result_v2": migration_result_v2_path,
        "migration_diagnostics_v1": migration_diagnostics_v1_path,
        "migration_diagnostics_v2": migration_diagnostics_v2_path,
    }

    for label, path in json_objects.items():
        try:
            loaded[label] = _as_dict(_read_json(path), str(path))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        return errors

    try:
        validate_result_schema(loaded["result_schema"])
    except ValueError as exc:
        errors.append(f"{result_schema_path}: {exc}")

    try:
        validate_config_schema(loaded["config_schema"])
    except ValueError as exc:
        errors.append(f"{config_schema_path}: {exc}")

    try:
        validate_checkpoint_schema(loaded["checkpoint_schema"])
    except ValueError as exc:
        errors.append(f"{checkpoint_schema_path}: {exc}")

    try:
        validate_result_fixture(loaded["result_fixture"])
    except ValueError as exc:
        errors.append(f"{result_fixture_path}: {exc}")

    try:
        validate_config_fixture(loaded["config_fixture"])
    except ValueError as exc:
        errors.append(f"{config_fixture_path}: {exc}")

    try:
        validate_checkpoint_fixture(loaded["checkpoint_fixture"])
    except ValueError as exc:
        errors.append(f"{checkpoint_fixture_path}: {exc}")

    try:
        validate_constraints_migration_fixture(loaded["migration_constraints_v1"])
    except ValueError as exc:
        errors.append(f"{migration_constraints_v1_path}: {exc}")

    try:
        validate_constraints_migration_fixture(loaded["migration_constraints_v2"])
    except ValueError as exc:
        errors.append(f"{migration_constraints_v2_path}: {exc}")

    try:
        validate_offline_config_migration_fixture(
            loaded["migration_pelt_v1"], "pelt migration fixture"
        )
    except ValueError as exc:
        errors.append(f"{migration_pelt_v1_path}: {exc}")

    try:
        validate_offline_config_migration_fixture(
            loaded["migration_pelt_v2"], "pelt migration fixture"
        )
    except ValueError as exc:
        errors.append(f"{migration_pelt_v2_path}: {exc}")

    try:
        validate_offline_config_migration_fixture(
            loaded["migration_binseg_v1"], "binseg migration fixture"
        )
    except ValueError as exc:
        errors.append(f"{migration_binseg_v1_path}: {exc}")

    try:
        validate_offline_config_migration_fixture(
            loaded["migration_binseg_v2"], "binseg migration fixture"
        )
    except ValueError as exc:
        errors.append(f"{migration_binseg_v2_path}: {exc}")

    try:
        validate_result_fixture(
            loaded["migration_result_v1"], MIGRATION_SUPPORTED_SCHEMA_VERSIONS
        )
    except ValueError as exc:
        errors.append(f"{migration_result_v1_path}: {exc}")

    try:
        validate_result_fixture(
            loaded["migration_result_v2"], MIGRATION_SUPPORTED_SCHEMA_VERSIONS
        )
    except ValueError as exc:
        errors.append(f"{migration_result_v2_path}: {exc}")

    try:
        validate_diagnostics_migration_fixture(loaded["migration_diagnostics_v1"])
    except ValueError as exc:
        errors.append(f"{migration_diagnostics_v1_path}: {exc}")

    try:
        validate_diagnostics_migration_fixture(loaded["migration_diagnostics_v2"])
    except ValueError as exc:
        errors.append(f"{migration_diagnostics_v2_path}: {exc}")

    if loaded["migration_constraints_v1"].get("schema_version") != 1:
        errors.append(
            f"{migration_constraints_v1_path}: schema_version must be exactly 1 for v1 fixture"
        )
    if loaded["migration_constraints_v2"].get("schema_version") != 2:
        errors.append(
            f"{migration_constraints_v2_path}: schema_version must be exactly 2 for v2 fixture"
        )
    if loaded["migration_pelt_v1"].get("schema_version") != 1:
        errors.append(
            f"{migration_pelt_v1_path}: schema_version must be exactly 1 for v1 fixture"
        )
    if loaded["migration_pelt_v2"].get("schema_version") != 2:
        errors.append(
            f"{migration_pelt_v2_path}: schema_version must be exactly 2 for v2 fixture"
        )
    if loaded["migration_binseg_v1"].get("schema_version") != 1:
        errors.append(
            f"{migration_binseg_v1_path}: schema_version must be exactly 1 for v1 fixture"
        )
    if loaded["migration_binseg_v2"].get("schema_version") != 2:
        errors.append(
            f"{migration_binseg_v2_path}: schema_version must be exactly 2 for v2 fixture"
        )

    migration_result_v1_diag = loaded["migration_result_v1"].get("diagnostics")
    migration_result_v2_diag = loaded["migration_result_v2"].get("diagnostics")
    if not isinstance(migration_result_v1_diag, dict):
        errors.append(f"{migration_result_v1_path}: diagnostics must be an object")
    elif migration_result_v1_diag.get("schema_version") != 1:
        errors.append(
            f"{migration_result_v1_path}: diagnostics.schema_version must be exactly 1 for v1 fixture"
        )
    if not isinstance(migration_result_v2_diag, dict):
        errors.append(f"{migration_result_v2_path}: diagnostics must be an object")
    elif migration_result_v2_diag.get("schema_version") != 2:
        errors.append(
            f"{migration_result_v2_path}: diagnostics.schema_version must be exactly 2 for v2 fixture"
        )
    if loaded["migration_diagnostics_v1"].get("schema_version") != 1:
        errors.append(
            f"{migration_diagnostics_v1_path}: schema_version must be exactly 1 for v1 fixture"
        )
    if loaded["migration_diagnostics_v2"].get("schema_version") != 2:
        errors.append(
            f"{migration_diagnostics_v2_path}: schema_version must be exactly 2 for v2 fixture"
        )

    if errors:
        return errors

    try:
        _validate_required_coverage(
            loaded["result_fixture"], loaded["result_schema"], "result"
        )
        result_defs = _as_dict(loaded["result_schema"].get("$defs"), "result schema.$defs")
        result_diag_schema = _as_dict(
            result_defs.get("diagnostics"), "result schema.$defs.diagnostics"
        )
        result_diag = _as_dict(
            loaded["result_fixture"].get("diagnostics"), "result fixture.diagnostics"
        )
        _validate_required_coverage(result_diag, result_diag_schema, "result diagnostics")
        _validate_required_coverage(
            loaded["config_fixture"], loaded["config_schema"], "config"
        )
        _validate_required_coverage(
            loaded["checkpoint_fixture"], loaded["checkpoint_schema"], "checkpoint"
        )
        _validate_required_coverage(
            loaded["migration_result_v1"], loaded["result_schema"], "migration result v1"
        )
        _validate_required_coverage(
            loaded["migration_result_v2"], loaded["result_schema"], "migration result v2"
        )
        if loaded["result_fixture"] != loaded["migration_result_v1"]:
            raise ValueError(
                "cpd-python offline_result_v1 fixture must match tests/fixtures/migrations/result/offline_result.v1.json"
            )
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate schema + fixture compatibility contracts."
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root containing cpd/ (default: auto-detected).",
    )
    args = parser.parse_args(argv)

    errors = validate_repo(Path(args.repo_root))
    if errors:
        print("BLOCK: schema/fixture contract checks failed")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("PASS: schema/fixture contracts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
