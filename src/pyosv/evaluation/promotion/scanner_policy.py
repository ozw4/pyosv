"""Scanner-thinning policy contracts for synthetic-quality promotion.

The policy identifiers in this module are derived from the saved synthetic-quality
configuration.  They are comparison metadata only; they are deliberately not
synthetic-quality variants or fields in the synthetic-quality report schema.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCANNER_THINNING_POLICY_PROFILE = "scanner-thinning-policy-v1"
REFERENCE_SCANNER_POLICY_ID = "quality_scanner_reference_v1"
NORMAL_SCANNER_POLICY_ID = "quality_scanner_thin_normal_v1"
ALLOWED_CONFIG_DIFFERENCE_PATHS = ("config.scanner.scanner_thin_mode",)

_MISSING_VALUE = "<missing>"

_COMMON_POLICY_CONDITIONS: tuple[tuple[str, Any], ...] = (
    ("config.case_set", "extended"),
    ("config.input_mode", "both"),
    ("config.workflow_mode", "quality"),
    ("config.shape", [49, 49, 49]),
    ("config.scanner.backend", "quality"),
    ("config.scanner.phi_min", 0.0),
    ("config.scanner.phi_max", 180.0),
    ("config.scanner.theta_min", 45.0),
    ("config.scanner.theta_max", 90.0),
    ("config.scanner.sigma1", 2.0),
    ("config.scanner.sigma2", 2.0),
    ("config.scanner.refinement_factor", 2),
    ("config.scanner.remove_edge_effects", True),
    ("config.scanner.input.background", 1.0),
    ("config.scanner.input.fault_contrast", 0.85),
    ("config.scanner.input.noise_sigma", 0.0),
    ("config.scanner.input.seed", 20260706),
    ("config.scanner.input.clip_min", 0.0),
    ("config.scanner.input.clip_max", 1.0),
)


def effective_remove_edge_effects(
    scanner_thin_mode: str, requested_remove_edge_effects: bool
) -> bool | None:
    """Return the effective edge-cleanup setting for a scanner thinning mode.

    Edge cleanup is implemented only by reference thinning.  ``None`` records
    that the requested flag is not applicable to normal or unthinned output.
    """

    if not isinstance(requested_remove_edge_effects, bool):
        raise ValueError("requested_remove_edge_effects must be a bool")
    if scanner_thin_mode == "reference":
        return requested_remove_edge_effects
    return None


def identify_scanner_policy(config: Mapping[str, Any]) -> str | None:
    """Derive a formal scanner policy ID from a saved report config.

    A policy is identified only when every common 49^3 quality-run condition and
    the requested edge-cleanup flag match exactly.  In particular, JSON booleans
    are not accepted in place of numbers and integers are not accepted in place
    of the expected floating-point values (or vice versa).
    """

    if not isinstance(config, Mapping):
        return None
    if _condition_mismatches(config, _COMMON_POLICY_CONDITIONS):
        return None
    scanner_mode = _value_at_path(config, "config.scanner.scanner_thin_mode")
    if _strict_equal(scanner_mode, "reference"):
        return REFERENCE_SCANNER_POLICY_ID
    if _strict_equal(scanner_mode, "normal"):
        return NORMAL_SCANNER_POLICY_ID
    return None


def load_metrics_report(metrics_path: str | Path, *, context: str = "metrics") -> dict[str, Any]:
    """Load and structurally validate a synthetic-quality ``metrics.json``.

    Policy value mismatches are intentionally not rejected here: they are
    preserved as contract failures.  Malformed evidence, an unsupported report
    format, and non-finite config values are input errors.
    """

    path = Path(metrics_path)
    try:
        payload_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(f"{context} metrics JSON does not exist: {path}") from error
    except OSError as error:
        raise ValueError(f"could not read {context} metrics JSON {path}: {error}") from error
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid {context} metrics JSON {path}: "
            f"line {error.lineno} column {error.colno}: {error.msg}"
        ) from error
    return validate_metrics_report(payload, context=f"{context} metrics JSON {path}")


def validate_metrics_report(report: object, *, context: str = "metrics report") -> dict[str, Any]:
    """Validate in-memory metrics evidence and return it as a plain mapping."""

    if not isinstance(report, Mapping):
        raise ValueError(f"{context} root must be a mapping")
    format_version = report.get("format_version", _MISSING_VALUE)
    if not _strict_equal(format_version, 1):
        raise ValueError(f"{context} format_version must be integer 1")
    config = report.get("config", _MISSING_VALUE)
    if not isinstance(config, Mapping):
        raise ValueError(f"{context} config must be a mapping")
    scanner = config.get("scanner", _MISSING_VALUE)
    if not isinstance(scanner, Mapping):
        raise ValueError(f"{context} config.scanner must be a mapping")
    _reject_non_finite(config, path="config", context=context)
    return dict(report)


def recursive_config_differences(
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    *,
    allowed_paths: Sequence[str] = ALLOWED_CONFIG_DIFFERENCE_PATHS,
) -> list[dict[str, Any]]:
    """Return deterministic, strict, recursively discovered config differences."""

    if not isinstance(baseline_config, Mapping):
        raise ValueError("baseline config must be a mapping")
    if not isinstance(candidate_config, Mapping):
        raise ValueError("candidate config must be a mapping")
    _reject_non_finite(baseline_config, path="config", context="baseline config")
    _reject_non_finite(candidate_config, path="config", context="candidate config")
    differences: list[dict[str, Any]] = []
    _collect_differences(
        baseline_config,
        candidate_config,
        path="config",
        allowed_paths=frozenset(allowed_paths),
        differences=differences,
    )
    return differences


def build_scanner_policy_contract(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_variant: str,
    candidate_variant: str,
) -> dict[str, Any]:
    """Build the scanner-thinning policy contract for a report comparison."""

    baseline_report = validate_metrics_report(baseline_metrics, context="baseline metrics report")
    candidate_report = validate_metrics_report(
        candidate_metrics, context="candidate metrics report"
    )
    baseline_config = baseline_report["config"]
    candidate_config = candidate_report["config"]

    differences = recursive_config_differences(baseline_config, candidate_config)
    allowed_differences = [item for item in differences if item["allowed"]]
    disallowed_differences = [item for item in differences if not item["allowed"]]
    baseline_conditions = (
        *_COMMON_POLICY_CONDITIONS,
        ("config.scanner.scanner_thin_mode", "reference"),
    )
    candidate_conditions = (
        *_COMMON_POLICY_CONDITIONS,
        ("config.scanner.scanner_thin_mode", "normal"),
    )
    baseline_mismatches = _condition_mismatches(baseline_config, baseline_conditions)
    candidate_mismatches = _condition_mismatches(candidate_config, candidate_conditions)

    reasons: list[str] = []
    if baseline_variant != candidate_variant:
        reasons.append(
            "selected variants differ: "
            f"baseline={baseline_variant!r}, candidate={candidate_variant!r}"
        )
    if baseline_variant != "current_default":
        reasons.append(
            f"baseline selected variant must be 'current_default': got {baseline_variant!r}"
        )
    if candidate_variant != "current_default":
        reasons.append(
            f"candidate selected variant must be 'current_default': got {candidate_variant!r}"
        )
    reasons.extend(_mismatch_reason("baseline", item) for item in baseline_mismatches)
    reasons.extend(_mismatch_reason("candidate", item) for item in candidate_mismatches)
    reasons.extend(
        f"disallowed config difference: {difference['path']}"
        for difference in disallowed_differences
    )

    return {
        "name": SCANNER_THINNING_POLICY_PROFILE,
        "passed": not reasons,
        "baseline": _policy_summary(baseline_config),
        "candidate": _policy_summary(candidate_config),
        "allowed_config_difference_paths": list(ALLOWED_CONFIG_DIFFERENCE_PATHS),
        "observed_config_differences": differences,
        "allowed_config_differences": allowed_differences,
        "disallowed_config_differences": disallowed_differences,
        "reasons": reasons,
    }


def _policy_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    scanner_mode = _value_at_path(config, "config.scanner.scanner_thin_mode")
    requested = _value_at_path(config, "config.scanner.remove_edge_effects")
    effective = None
    if isinstance(scanner_mode, str) and isinstance(requested, bool):
        effective = effective_remove_edge_effects(scanner_mode, requested)
    return {
        "policy_id": identify_scanner_policy(config),
        "scanner_thin_mode": None if scanner_mode == _MISSING_VALUE else scanner_mode,
        "requested_remove_edge_effects": None if requested == _MISSING_VALUE else requested,
        "effective_remove_edge_effects": effective,
    }


def _condition_mismatches(
    config: Mapping[str, Any], conditions: Sequence[tuple[str, Any]]
) -> list[dict[str, Any]]:
    mismatches = []
    for path, expected in conditions:
        actual = _value_at_path(config, path)
        if not _strict_equal(actual, expected):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    return mismatches


def _mismatch_reason(role: str, mismatch: Mapping[str, Any]) -> str:
    actual = mismatch["actual"]
    actual_text = _MISSING_VALUE if actual == _MISSING_VALUE else repr(actual)
    return (
        f"{role} policy condition mismatch: {mismatch['path']} "
        f"expected {mismatch['expected']!r}, got {actual_text}"
    )


def _value_at_path(config: Mapping[str, Any], path: str) -> Any:
    parts = path.split(".")
    if parts and parts[0] == "config":
        parts = parts[1:]
    value: Any = config
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING_VALUE
        value = value[part]
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_strict_equal(left[key], right[key]) for key in left)
    if _is_sequence(left) or _is_sequence(right):
        if not _is_sequence(left) or not _is_sequence(right) or len(left) != len(right):
            return False
        return all(_strict_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _reject_non_finite(value: Any, *, path: str, context: str) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            _reject_non_finite(value[key], path=f"{path}.{key}", context=context)
        return
    if _is_sequence(value):
        for index, item in enumerate(value):
            _reject_non_finite(item, path=f"{path}[{index}]", context=context)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains non-finite config value at {path}")


def _collect_differences(
    baseline: Any,
    candidate: Any,
    *,
    path: str,
    allowed_paths: frozenset[str],
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        for key in sorted(set(baseline) | set(candidate)):
            child_path = f"{path}.{key}"
            if key not in baseline:
                differences.append(
                    _difference(
                        child_path,
                        baseline=_MISSING_VALUE,
                        candidate=candidate[key],
                        kind="missing_baseline_key",
                        allowed_paths=allowed_paths,
                    )
                )
            elif key not in candidate:
                differences.append(
                    _difference(
                        child_path,
                        baseline=baseline[key],
                        candidate=_MISSING_VALUE,
                        kind="missing_candidate_key",
                        allowed_paths=allowed_paths,
                    )
                )
            else:
                _collect_differences(
                    baseline[key],
                    candidate[key],
                    path=child_path,
                    allowed_paths=allowed_paths,
                    differences=differences,
                )
        return
    if _is_sequence(baseline) and _is_sequence(candidate):
        common_length = min(len(baseline), len(candidate))
        for index in range(common_length):
            _collect_differences(
                baseline[index],
                candidate[index],
                path=f"{path}[{index}]",
                allowed_paths=allowed_paths,
                differences=differences,
            )
        for index in range(common_length, len(baseline)):
            differences.append(
                _difference(
                    f"{path}[{index}]",
                    baseline=baseline[index],
                    candidate=_MISSING_VALUE,
                    kind="missing_candidate_item",
                    allowed_paths=allowed_paths,
                )
            )
        for index in range(common_length, len(candidate)):
            differences.append(
                _difference(
                    f"{path}[{index}]",
                    baseline=_MISSING_VALUE,
                    candidate=candidate[index],
                    kind="missing_baseline_item",
                    allowed_paths=allowed_paths,
                )
            )
        return
    if not _strict_equal(baseline, candidate):
        differences.append(
            _difference(
                path,
                baseline=baseline,
                candidate=candidate,
                kind="value_mismatch",
                allowed_paths=allowed_paths,
            )
        )


def _difference(
    path: str,
    *,
    baseline: Any,
    candidate: Any,
    kind: str,
    allowed_paths: frozenset[str],
) -> dict[str, Any]:
    result = {
        "path": path,
        "baseline": baseline,
        "candidate": candidate,
        "allowed": path in allowed_paths,
    }
    if kind != "value_mismatch":
        result["kind"] = kind
    return result
