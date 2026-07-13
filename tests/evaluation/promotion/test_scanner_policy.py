from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pyosv.evaluation.promotion.scanner_policy import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    NORMAL_SCANNER_POLICY_ID,
    QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    REFERENCE_SCANNER_POLICY_ID,
    REFERENCE_LIKE_NORMAL_SCANNER_POLICY_ID,
    REFERENCE_LIKE_REFERENCE_SCANNER_POLICY_ID,
    SCANNER_POLICY_PROFILES,
    SCANNER_THINNING_POLICY_PROFILE,
    build_scanner_policy_contract,
    effective_remove_edge_effects,
    identify_scanner_policy,
    load_metrics_report,
    recursive_config_differences,
)


def _config(mode: str = "reference") -> dict[str, object]:
    return {
        "case_set": "extended",
        "input_mode": "both",
        "workflow_mode": "quality",
        "shape": [49, 49, 49],
        "variants": ["current_default"],
        "variant_preset": "default",
        "voting": {"voter_thin_mode": "reference"},
        "truth_metrics": {"buffer_radius": 2.0},
        "skinning": {"method": "quality", "enabled": True},
        "scanner_backend_matrix": False,
        "scanner_downstream_diagnostics": True,
        "scanner": {
            "backend": "quality",
            "phi_min": 0.0,
            "phi_max": 180.0,
            "theta_min": 45.0,
            "theta_max": 90.0,
            "sigma1": 2.0,
            "sigma2": 2.0,
            "refinement_factor": 2,
            "scanner_thin_mode": mode,
            "remove_edge_effects": True,
            "input": {
                "background": 1.0,
                "fault_contrast": 0.85,
                "noise_sigma": 0.0,
                "seed": 20260706,
                "clip_min": 0.0,
                "clip_max": 1.0,
            },
        },
    }


def _metrics(mode: str = "reference") -> dict[str, object]:
    return {"format_version": 1, "config": _config(mode), "cases": []}


def _reference_like_config(mode: str = "reference") -> dict[str, object]:
    config = _config(mode)
    scanner = config["scanner"]
    assert isinstance(scanner, dict)
    scanner["backend"] = "reference-like"
    return config


def _reference_like_metrics(mode: str = "reference") -> dict[str, object]:
    return {"format_version": 1, "config": _reference_like_config(mode), "cases": []}


def _set_path(config: dict[str, object], path: str, value: object) -> None:
    target: dict[str, object] = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[assignment]
    target[parts[-1]] = value


def test_identifies_exact_reference_and_normal_policies() -> None:
    assert identify_scanner_policy(_config("reference")) == REFERENCE_SCANNER_POLICY_ID
    assert identify_scanner_policy(_config("normal")) == NORMAL_SCANNER_POLICY_ID


def test_identifies_reference_like_policies_for_quality_workflow_profile() -> None:
    assert SCANNER_POLICY_PROFILES == (
        SCANNER_THINNING_POLICY_PROFILE,
        QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    )
    assert (
        identify_scanner_policy(
            _reference_like_config("reference"),
            comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
        )
        == REFERENCE_LIKE_REFERENCE_SCANNER_POLICY_ID
    )
    assert (
        identify_scanner_policy(
            _reference_like_config("normal"),
            comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
        )
        == REFERENCE_LIKE_NORMAL_SCANNER_POLICY_ID
    )


def test_policy_identification_default_keeps_existing_quality_backend_contract() -> None:
    assert identify_scanner_policy(_reference_like_config()) is None
    assert (
        identify_scanner_policy(
            _config(),
            comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
        )
        is None
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("case_set", "geometry"),
        ("input_mode", "scanner"),
        ("workflow_mode", "reference"),
        ("shape", [21, 21, 21]),
        ("variants", ["current_default", "voter_thin_normal"]),
        ("scanner.backend", "quality"),
        ("scanner.remove_edge_effects", False),
        ("scanner.remove_edge_effects", 1),
    ],
)
def test_reference_like_policy_identification_rejects_contract_values(
    path: str, value: object
) -> None:
    config = _reference_like_config()
    _set_path(config, path, value)
    assert (
        identify_scanner_policy(
            config,
            comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
        )
        is None
    )


def test_policy_identification_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown scanner policy comparison profile"):
        identify_scanner_policy(_config(), comparison_profile="unknown")


@pytest.mark.parametrize(
    ("mode", "requested", "expected"),
    [
        ("reference", True, True),
        ("reference", False, False),
        ("normal", True, None),
        ("normal", False, None),
        ("none", True, None),
        ("none", False, None),
    ],
)
def test_effective_remove_edge_effects_semantics(
    mode: str, requested: bool, expected: bool | None
) -> None:
    assert effective_remove_edge_effects(mode, requested) is expected


def test_effective_remove_edge_effects_rejects_non_boolean_request() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        effective_remove_edge_effects("reference", 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("scanner.backend", "fast"),
        ("scanner.refinement_factor", 3),
        ("scanner.sigma1", 2.5),
        ("scanner.sigma2", 2.5),
        ("scanner.phi_min", 1.0),
        ("scanner.theta_max", 89.0),
        ("scanner.input.background", 0.9),
        ("scanner.input.fault_contrast", 0.8),
        ("scanner.input.noise_sigma", 0.1),
        ("scanner.input.seed", 1),
        ("scanner.input.clip_min", 0.1),
        ("scanner.input.clip_max", 0.9),
        ("scanner.remove_edge_effects", False),
    ],
)
def test_policy_identification_rejects_non_policy_values(path: str, value: object) -> None:
    config = _config()
    _set_path(config, path, value)
    assert identify_scanner_policy(config) is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("scanner.remove_edge_effects", 1),
        ("scanner.refinement_factor", True),
        ("scanner.phi_min", False),
        ("scanner.input.seed", True),
        ("scanner.sigma1", 2),
    ],
)
def test_policy_identification_does_not_conflate_json_scalar_types(
    path: str, value: object
) -> None:
    config = _config()
    _set_path(config, path, value)
    assert identify_scanner_policy(config) is None


def test_recursive_diff_records_only_scanner_mode_as_allowed() -> None:
    differences = recursive_config_differences(_config("reference"), _config("normal"))
    assert ALLOWED_CONFIG_DIFFERENCE_PATHS == ("config.scanner.scanner_thin_mode",)
    assert differences == [
        {
            "path": "config.scanner.scanner_thin_mode",
            "baseline": "reference",
            "candidate": "normal",
            "allowed": True,
        }
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("scanner.sigma1", 3.0),
        ("scanner.input.noise_sigma", 0.1),
        ("voting.voter_thin_mode", "normal"),
        ("skinning.method", "reference"),
        ("scanner.remove_edge_effects", False),
    ],
)
def test_recursive_diff_records_disallowed_config_changes(path: str, value: object) -> None:
    baseline = _config()
    candidate = copy.deepcopy(baseline)
    _set_path(candidate, path, value)
    differences = recursive_config_differences(baseline, candidate)
    assert [item["path"] for item in differences] == [f"config.{path}"]
    assert differences[0]["allowed"] is False


def test_recursive_diff_distinguishes_missing_keys_items_and_values() -> None:
    baseline = {"nested": {"baseline_only": 1}, "items": [1, 2]}
    candidate = {"nested": {"candidate_only": 2}, "items": [1, 3, 4]}
    assert recursive_config_differences(baseline, candidate) == [
        {
            "path": "config.items[1]",
            "baseline": 2,
            "candidate": 3,
            "allowed": False,
        },
        {
            "path": "config.items[2]",
            "baseline": "<missing>",
            "candidate": 4,
            "allowed": False,
            "kind": "missing_baseline_item",
        },
        {
            "path": "config.nested.baseline_only",
            "baseline": 1,
            "candidate": "<missing>",
            "allowed": False,
            "kind": "missing_candidate_key",
        },
        {
            "path": "config.nested.candidate_only",
            "baseline": "<missing>",
            "candidate": 2,
            "allowed": False,
            "kind": "missing_baseline_key",
        },
    ]


def test_recursive_diff_order_does_not_depend_on_mapping_insertion_order() -> None:
    baseline = {"z": 1, "a": {"z": 1, "a": 1}}
    candidate_forward = {"a": {"a": 2, "z": 2}, "z": 2}
    candidate_reverse = {"z": 2, "a": {"z": 2, "a": 2}}
    forward = recursive_config_differences(baseline, candidate_forward)
    reverse = recursive_config_differences(baseline, candidate_reverse)
    assert forward == reverse
    assert [item["path"] for item in forward] == ["config.a.a", "config.a.z", "config.z"]


def test_recursive_diff_treats_int_and_float_as_different() -> None:
    assert recursive_config_differences({"value": 1}, {"value": 1.0}) == [
        {"path": "config.value", "baseline": 1, "candidate": 1.0, "allowed": False}
    ]


def test_valid_scanner_policy_contract() -> None:
    contract = build_scanner_policy_contract(
        _metrics("reference"),
        _metrics("normal"),
        "current_default",
        "current_default",
    )
    assert contract["name"] == SCANNER_THINNING_POLICY_PROFILE
    assert contract["passed"] is True
    assert contract["baseline"] == {
        "policy_id": REFERENCE_SCANNER_POLICY_ID,
        "scanner_thin_mode": "reference",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": True,
    }
    assert contract["candidate"] == {
        "policy_id": NORMAL_SCANNER_POLICY_ID,
        "scanner_thin_mode": "normal",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": None,
    }
    assert contract["allowed_config_differences"] == contract["observed_config_differences"]
    assert contract["disallowed_config_differences"] == []
    assert contract["reasons"] == []


def test_valid_reference_like_scanner_policy_contract() -> None:
    contract = build_scanner_policy_contract(
        _reference_like_metrics("reference"),
        _reference_like_metrics("normal"),
        "current_default",
        "current_default",
        comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    )
    assert contract["name"] == QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE
    assert contract["passed"] is True
    assert contract["baseline"] == {
        "policy_id": REFERENCE_LIKE_REFERENCE_SCANNER_POLICY_ID,
        "scanner_thin_mode": "reference",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": True,
    }
    assert contract["candidate"] == {
        "policy_id": REFERENCE_LIKE_NORMAL_SCANNER_POLICY_ID,
        "scanner_thin_mode": "normal",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": None,
    }
    assert contract["observed_config_differences"] == [
        {
            "path": "config.scanner.scanner_thin_mode",
            "baseline": "reference",
            "candidate": "normal",
            "allowed": True,
        }
    ]
    assert contract["reasons"] == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("scanner.phi_min", 5.0),
        ("scanner.sigma1", 3.0),
        ("scanner.refinement_factor", 3),
        ("scanner.input.seed", 1),
        ("voting.voter_thin_mode", "normal"),
        ("skinning.method", "reference"),
        ("truth_metrics.buffer_radius", 3.0),
        ("scanner_downstream_diagnostics", False),
    ],
)
def test_reference_like_contract_rejects_every_other_config_difference(
    path: str, value: object
) -> None:
    baseline = _reference_like_metrics("reference")
    candidate = _reference_like_metrics("normal")
    _set_path(candidate["config"], path, value)  # type: ignore[arg-type]
    contract = build_scanner_policy_contract(
        baseline,
        candidate,
        "current_default",
        "current_default",
        comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    )
    assert contract["passed"] is False
    assert [difference["path"] for difference in contract["disallowed_config_differences"]] == [
        f"config.{path}"
    ]
    assert any(f"config.{path}" in reason for reason in contract["reasons"])


def test_reference_like_contract_requires_exact_current_default_variant_list() -> None:
    candidate = _reference_like_metrics("normal")
    candidate_config = candidate["config"]
    assert isinstance(candidate_config, dict)
    candidate_config["variants"] = ["current_default", "voter_thin_normal"]
    contract = build_scanner_policy_contract(
        _reference_like_metrics("reference"),
        candidate,
        "current_default",
        "current_default",
        comparison_profile=QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    )
    assert contract["passed"] is False
    assert any("config.variants" in reason for reason in contract["reasons"])


@pytest.mark.parametrize(
    (
        "baseline_mode",
        "candidate_mode",
        "baseline_variant",
        "candidate_variant",
        "path",
        "value",
        "reason",
    ),
    [
        (
            "reference",
            "reference",
            "current_default",
            "current_default",
            None,
            None,
            "candidate policy condition mismatch",
        ),
        (
            "normal",
            "normal",
            "current_default",
            "current_default",
            None,
            None,
            "baseline policy condition mismatch",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "scanner.backend",
            "fast",
            "config.scanner.backend",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "scanner.refinement_factor",
            3,
            "config.scanner.refinement_factor",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "scanner.input.seed",
            9,
            "config.scanner.input.seed",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "voting.voter_thin_mode",
            "normal",
            "config.voting.voter_thin_mode",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "skinning.method",
            "other",
            "config.skinning.method",
        ),
        (
            "reference",
            "normal",
            "current_default",
            "current_default",
            "scanner.remove_edge_effects",
            False,
            "config.scanner.remove_edge_effects",
        ),
        ("reference", "normal", "current_default", "other", None, None, "selected variants differ"),
        ("reference", "normal", "other", "other", None, None, "baseline selected variant"),
    ],
)
def test_contract_failures_have_deterministic_reasons(
    baseline_mode: str,
    candidate_mode: str,
    baseline_variant: str,
    candidate_variant: str,
    path: str | None,
    value: object,
    reason: str,
) -> None:
    baseline = _metrics(baseline_mode)
    candidate = _metrics(candidate_mode)
    if path is not None:
        _set_path(candidate["config"], path, value)  # type: ignore[arg-type]
    contract = build_scanner_policy_contract(
        baseline, candidate, baseline_variant, candidate_variant
    )
    assert contract["passed"] is False
    assert any(reason in item for item in contract["reasons"])


def test_load_metrics_report_reads_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metrics()), encoding="utf-8")
    assert load_metrics_report(path, context="baseline") == _metrics()


def test_load_metrics_report_rejects_missing_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match=r"baseline metrics JSON does not exist: .*missing.json"):
        load_metrics_report(path, context="baseline")


def test_load_metrics_report_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid candidate metrics JSON .*line 1 column"):
        load_metrics_report(path, context="candidate")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a mapping"),
        ({"config": {"scanner": {}}}, "format_version must be integer 1"),
        ({"format_version": True, "config": {"scanner": {}}}, "format_version must be integer 1"),
        ({"format_version": 2, "config": {"scanner": {}}}, "format_version must be integer 1"),
        ({"format_version": 1}, "config must be a mapping"),
        ({"format_version": 1, "config": []}, "config must be a mapping"),
        ({"format_version": 1, "config": {}}, "config.scanner must be a mapping"),
        ({"format_version": 1, "config": {"scanner": []}}, "config.scanner must be a mapping"),
        (
            {"format_version": 1, "config": {"scanner": {"sigma1": float("nan")}}},
            "contains non-finite config value at config.scanner.sigma1",
        ),
        (
            {"format_version": 1, "config": {"scanner": {"sigma1": float("inf")}}},
            "contains non-finite config value at config.scanner.sigma1",
        ),
    ],
)
def test_load_metrics_report_rejects_malformed_evidence(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_metrics_report(path)
