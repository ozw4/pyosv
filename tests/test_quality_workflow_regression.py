from __future__ import annotations

import math
from typing import Any

import pytest

from pyosv.cli import synthetic_quality


SHAPE = (21, 21, 21)
CASE_IDS = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)
FVT_BUFFERED_F1_R2 = (
    "quality",
    "fvt_top_truth_count",
    "buffered_overlap_radius2",
    "buffered_f1",
)
FVT_POSITIVE_BUFFERED_F1_R2 = (
    "quality",
    "fvt_positive_top_truth_count",
    "buffered_overlap_radius2",
    "buffered_f1",
)
FVT_POSITIVE_CANDIDATE_COUNT = (
    "quality",
    "fvt_positive_top_truth_count",
    "surface_distance",
    "candidate_count",
)
FVT_POSITIVE_DISTANCE_P95 = (
    "quality",
    "fvt_positive_top_truth_count",
    "surface_distance",
    "candidate_to_truth_p95",
)
FVT_POSITIVE_EDGE_FALSE_POSITIVE_FRACTION = (
    "quality",
    "edge_false_positive",
    "fvt_positive_top_truth_count",
    "edge_false_positive_fraction_of_candidates",
)
SKIN_BUFFERED_F1_R2 = (
    "quality",
    "skin",
    "buffered_overlap_radius2",
    "buffered_f1",
)
SCANNER_FT_BUFFERED_F1_R2 = (
    "scanner_quality",
    "ft_top_truth_count",
    "buffered_overlap_radius2",
    "buffered_f1",
)
SCANNER_FVT_TO_FT_DISTANCE_P95 = (
    "scanner_downstream",
    "fvt_candidate_to_scanner_ft_distance_p95",
)
FVT_EDGE_FALSE_POSITIVE_FRACTION = (
    "quality",
    "edge_false_positive",
    "fvt_top_truth_count",
    "edge_false_positive_fraction_of_candidates",
)


def _load_report_module() -> object:
    return synthetic_quality


@pytest.fixture(scope="module")
def workflow_reports() -> dict[str, dict[str, Any]]:
    module = _load_report_module()
    return {
        mode: module.build_report(
            case_set="extended",
            shape=SHAPE,
            variants=("current_default",),
            workflow_mode=mode,
        )
        for mode in ("reference", "quality")
    }


@pytest.fixture(scope="module")
def quality_skinner_v2_report() -> dict[str, Any]:
    module = _load_report_module()
    return module.build_report(
        case_set="extended",
        shape=SHAPE,
        variants=("current_default", "quality_skinner_v2"),
        workflow_mode="quality",
    )


@pytest.fixture(scope="module")
def scanner_quality_report() -> dict[str, Any]:
    module = _load_report_module()
    return module.build_report(
        case_set="extended",
        shape=SHAPE,
        variants=("current_default",),
        workflow_mode="quality",
        input_mode="scanner",
        scanner_config=module.SyntheticScannerConfig(
            backend="quality",
            refinement_factor=2,
        ),
        include_scanner_downstream_diagnostics=True,
    )


def _cases_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = {case["case_id"]: case for case in report["cases"]}
    assert tuple(cases) == CASE_IDS
    return cases


def _nested(report: dict[str, Any], *path: str) -> Any:
    value: Any = report
    for key in path:
        value = value[key]
    return value


def _float_metric(report: dict[str, Any], *path: str) -> float:
    value = float(_nested(report, *path))
    assert math.isfinite(value)
    return value


def _metric_path(*path: str) -> str:
    return ".".join(path)


def _assert_quality_at_least_reference_delta(
    reference_cases: dict[str, dict[str, Any]],
    quality_cases: dict[str, dict[str, Any]],
    case_id: str,
    path: tuple[str, ...],
    min_delta: float,
) -> None:
    reference = _float_metric(reference_cases[case_id], *path)
    quality = _float_metric(quality_cases[case_id], *path)
    threshold = reference + min_delta
    assert quality >= threshold, (
        f"{case_id} {_metric_path(*path)}: quality={quality:.6g}, "
        f"reference={reference:.6g}, expected quality >= {threshold:.6g}"
    )


def _assert_quality_at_most_reference_delta(
    reference_cases: dict[str, dict[str, Any]],
    quality_cases: dict[str, dict[str, Any]],
    case_id: str,
    path: tuple[str, ...],
    max_delta: float,
) -> None:
    reference = _float_metric(reference_cases[case_id], *path)
    quality = _float_metric(quality_cases[case_id], *path)
    threshold = reference + max_delta
    assert quality <= threshold, (
        f"{case_id} {_metric_path(*path)}: quality={quality:.6g}, "
        f"reference={reference:.6g}, expected quality <= {threshold:.6g}"
    )


def _assert_finite_metric_tree(value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(float(value))
        return
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_metric_tree(child)
        return
    if value is None:
        return
    raise AssertionError(f"unexpected metric value type: {type(value)!r}")


def test_quality_workflow_effective_settings_are_recorded(
    workflow_reports: dict[str, dict[str, Any]],
) -> None:
    reference = workflow_reports["reference"]["config"]
    assert reference["workflow_mode"] == "reference"
    assert reference["voting"]["voter_thin_mode"] == "reference"
    assert reference["voting"]["surface_support_min_fraction"] == 0.0
    assert reference["voting"]["surface_support_exponent"] == 0.0
    assert reference["skinning"]["method"] == "reference"
    assert reference["skinning"]["growth_source"] == "thinned"
    assert reference["skinning"]["adaptive_min_likelihood"] is False
    assert reference["skinning"]["seed_min_ep"] == 0.8
    assert reference["skinning"]["accepted_occupancy_radius"] is None
    assert reference["skinning"]["effective_accepted_occupancy_radius"] == 5

    quality = workflow_reports["quality"]["config"]
    assert quality["workflow_mode"] == "quality"
    assert quality["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert quality["voting"]["surface_support_min_fraction"] == 0.0
    assert quality["voting"]["surface_support_exponent"] == 0.0
    assert quality["skinning"]["method"] == "quality"
    assert quality["skinning"]["growth_source"] == "pre_thin"
    assert quality["skinning"]["min_likelihood"] is None
    assert quality["skinning"]["adaptive_min_likelihood"] is True
    assert quality["skinning"]["seed_min_ep"] == 0.5
    assert quality["skinning"]["accepted_occupancy_radius"] == 1
    assert quality["skinning"]["effective_accepted_occupancy_radius"] == 1
    assert quality["skinning"]["boundary_skinner_fallback"] is True


def test_quality_workflow_key_metrics_are_finite(
    workflow_reports: dict[str, dict[str, Any]],
) -> None:
    for report in workflow_reports.values():
        assert report["config"]["shape"] == list(SHAPE)
        assert report["config"]["variants"] == ["current_default"]
        for case in _cases_by_id(report).values():
            assert case["variants"]["current_default"]["quality"] == case["quality"]
            skinning_report = case["variants"]["current_default"]["skinning"]
            assert skinning_report["enabled"] is True
            assert "diagnostics" in skinning_report

            quality = case["quality"]
            for field in (
                "fv_top_truth_count",
                "fvt_top_truth_count",
                "fv_positive_top_truth_count",
                "fvt_positive_top_truth_count",
            ):
                _assert_finite_metric_tree(quality[field]["buffered_overlap_radius2"])
                _assert_finite_metric_tree(quality[field]["surface_distance"])
                _assert_finite_metric_tree(quality[field]["orientation_error"])

            edge_false_positive = quality["edge_false_positive"]
            _assert_finite_metric_tree(edge_false_positive["fv_top_truth_count"])
            _assert_finite_metric_tree(edge_false_positive["fvt_top_truth_count"])
            _assert_finite_metric_tree(edge_false_positive["fv_positive_top_truth_count"])
            _assert_finite_metric_tree(edge_false_positive["fvt_positive_top_truth_count"])

            skin = quality["skin"]
            assert skin is not None
            diagnostics = skinning_report["diagnostics"]
            skin_count = skin["topology"]["skin_count"]
            if diagnostics["fallback_used"]:
                assert diagnostics["fallback_skin_count"] == skin_count
            else:
                assert diagnostics["accepted_skin_count"] == skin_count
            for field in (
                "topology",
                "buffered_overlap_radius2",
                "surface_distance",
                "orientation_error",
            ):
                assert field in skin
                _assert_finite_metric_tree(skin[field])


def test_scanner_quality_extended_report_builds_successfully(
    scanner_quality_report: dict[str, Any],
) -> None:
    config = scanner_quality_report["config"]
    assert config["case_set"] == "extended"
    assert config["shape"] == list(SHAPE)
    assert config["workflow_mode"] == "quality"
    assert config["variants"] == ["current_default"]
    assert config["input_mode"] == "scanner"
    assert config["scanner"]["backend"] == "quality"
    assert config["scanner"]["refinement_factor"] == 2
    assert config["scanner_downstream_diagnostics"] is True
    assert tuple(case["case_id"] for case in scanner_quality_report["cases"]) == CASE_IDS


def test_scanner_quality_extended_guardrail_metrics_are_finite(
    scanner_quality_report: dict[str, Any],
) -> None:
    for case in _cases_by_id(scanner_quality_report).values():
        variant = case["variants"]["current_default"]
        assert case["quality"] == variant["quality"]
        assert "scanner_quality" in variant
        assert "scanner_downstream" in variant
        _assert_finite_metric_tree(variant["scanner_quality"])
        for field in (
            "scanner_ft_positive_candidate_count",
            "scanner_fet_positive_candidate_count",
            "fv_positive_candidate_count",
            "fvt_positive_candidate_count",
            "fvt_candidate_to_scanner_ft_distance_p95",
            "fvt_positive_edge_candidate_fraction",
            "fvt_positive_edge_false_positive_fraction",
        ):
            assert math.isfinite(float(variant["scanner_downstream"][field]))
        assert _float_metric(case, *FVT_POSITIVE_CANDIDATE_COUNT) >= 0.0
        assert math.isfinite(_float_metric(case, *FVT_POSITIVE_BUFFERED_F1_R2))

        skinning_report = variant["skinning"]
        assert skinning_report["enabled"] is True
        skin = variant["quality"]["skin"]
        assert skin is not None
        for field in (
            "topology",
            "buffered_overlap_radius2",
            "surface_distance",
            "orientation_error",
        ):
            _assert_finite_metric_tree(skin[field])


def test_scanner_quality_boundary_guardrails(
    scanner_quality_report: dict[str, Any],
) -> None:
    boundary = _cases_by_id(scanner_quality_report)["boundary_plane"]
    variant = boundary["variants"]["current_default"]
    diagnostics = variant["skinning"]["diagnostics"]

    assert _float_metric(boundary, *SCANNER_FT_BUFFERED_F1_R2) >= 0.50
    assert _float_metric(boundary, *FVT_POSITIVE_CANDIDATE_COUNT) > 0.0
    assert math.isfinite(_float_metric(boundary, *SCANNER_FVT_TO_FT_DISTANCE_P95))
    assert variant["scanner"]["config"]["backend"] == "quality"
    assert "fvt_candidate_to_scanner_ft_distance_p95" in variant["scanner_downstream"]
    assert "skin_fvt_to_scanner_target_distance_p95" in diagnostics
    assert math.isfinite(float(diagnostics["skin_fvt_to_scanner_target_distance_p95"]))


def test_scanner_quality_non_boundary_skins_do_not_collapse(
    scanner_quality_report: dict[str, Any],
) -> None:
    cases = _cases_by_id(scanner_quality_report)
    for case_id, case in cases.items():
        if case_id == "boundary_plane":
            continue
        skin = case["variants"]["current_default"]["quality"]["skin"]
        assert skin is not None
        assert _float_metric(case, "quality", "skin", "topology", "skin_count") > 0.0
        assert _float_metric(case, *SKIN_BUFFERED_F1_R2) >= 0.40


def test_quality_skinner_v2_extended_skin_metrics_are_finite(
    quality_skinner_v2_report: dict[str, Any],
) -> None:
    assert quality_skinner_v2_report["config"]["shape"] == list(SHAPE)
    assert quality_skinner_v2_report["config"]["variants"] == [
        "current_default",
        "quality_skinner_v2",
    ]
    for case in _cases_by_id(quality_skinner_v2_report).values():
        assert (
            case["variants"]["current_default"]["config"]["skinning"]
            == case["variants"]["quality_skinner_v2"]["config"]["skinning"]
        )
        variant = case["variants"]["quality_skinner_v2"]
        skinning = variant["config"]["skinning"]
        assert skinning["method"] == "quality"
        assert skinning["growth_source"] == "pre_thin"
        assert skinning["effective_accepted_occupancy_radius"] == 1
        skin = variant["quality"]["skin"]
        assert skin is not None
        for field in (
            "topology",
            "buffered_overlap_radius2",
            "surface_distance",
            "orientation_error",
        ):
            _assert_finite_metric_tree(skin[field])


def test_quality_skinner_v2_skin_f1_guardrail(
    quality_skinner_v2_report: dict[str, Any],
) -> None:
    cases = _cases_by_id(quality_skinner_v2_report)
    comparisons: dict[str, tuple[float, float]] = {}
    for case_id in ("curved_surface", "crossing_planes"):
        current_default = _float_metric(
            cases[case_id],
            "variants",
            "current_default",
            *SKIN_BUFFERED_F1_R2,
        )
        quality_skinner_v2 = _float_metric(
            cases[case_id],
            "variants",
            "quality_skinner_v2",
            *SKIN_BUFFERED_F1_R2,
        )
        comparisons[case_id] = (current_default, quality_skinner_v2)
    assert any(
        quality_skinner_v2 >= current_default
        for current_default, quality_skinner_v2 in comparisons.values()
    ), "quality_skinner_v2 skin buffered F1 was below current_default for " + ", ".join(
        f"{case_id}: {quality_skinner_v2:.6g} < {current_default:.6g}"
        for case_id, (current_default, quality_skinner_v2) in comparisons.items()
    )


def test_quality_boundary_skinner_fallback_recovers_boundary_skin() -> None:
    module = _load_report_module()
    report = module.build_report(
        case_set="extended",
        shape=(33, 33, 33),
        variants=("current_default", "quality_boundary_skinner_fallback"),
        workflow_mode="quality",
    )
    cases = _cases_by_id(report)

    assert report["config"]["variants"] == [
        "current_default",
        "quality_boundary_skinner_fallback",
    ]
    boundary = cases["boundary_plane"]["variants"]["quality_boundary_skinner_fallback"]
    diagnostics = boundary["skinning"]["diagnostics"]
    skin = boundary["quality"]["skin"]

    assert _float_metric(boundary, *FVT_POSITIVE_CANDIDATE_COUNT) > 0.0
    assert diagnostics["fallback_enabled"] is True
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "empty_primary_skin_with_positive_fvt"
    assert diagnostics["fallback_method"] == "connected_component_on_fvt"
    assert diagnostics["fallback_input"] == "fvt"
    assert diagnostics["accepted_skin_count"] == 0
    assert diagnostics["skin_primary_count"] == 0
    assert diagnostics["skin_primary_cell_count"] == 0
    assert diagnostics["skin_primary_degraded_candidate"] is True
    assert diagnostics["skin_primary_degraded_reasons"] == [
        "empty_primary_skin",
        "low_fvt_positive_coverage",
    ]
    assert diagnostics["fallback_skin_count"] == skin["topology"]["skin_count"]
    assert diagnostics["fallback_cell_count"] == skin["topology"]["cell_count"]
    assert skin["topology"]["skin_count"] > 0
    assert skin["buffered_overlap_radius2"]["buffered_f1"] > 0.5

    current_default = cases["boundary_plane"]["variants"]["current_default"]
    current_diagnostics = current_default["skinning"]["diagnostics"]
    current_skin = current_default["quality"]["skin"]
    assert current_default["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert current_diagnostics["fallback_enabled"] is True
    assert current_diagnostics["fallback_used"] is True
    assert current_diagnostics["fallback_skin_count"] == current_skin["topology"]["skin_count"]
    assert current_skin["topology"]["skin_count"] > 0
    assert current_skin["buffered_overlap_radius2"]["buffered_f1"] >= 0.5


def test_quality_boundary_skinner_fallback_does_not_run_when_primary_succeeds() -> None:
    module = _load_report_module()
    report = module.build_report(
        case_set="minimal",
        shape=SHAPE,
        variants=("quality_boundary_skinner_fallback",),
        workflow_mode="quality",
    )
    case = report["cases"][0]
    variant = case["variants"]["quality_boundary_skinner_fallback"]
    diagnostics = variant["skinning"]["diagnostics"]
    skin = variant["quality"]["skin"]

    assert skin["topology"]["skin_count"] > 0
    assert diagnostics["accepted_skin_count"] == skin["topology"]["skin_count"]
    assert diagnostics["skin_primary_count"] == skin["topology"]["skin_count"]
    assert diagnostics["skin_primary_cell_count"] == skin["topology"]["cell_count"]
    assert diagnostics["fallback_enabled"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_nonempty"
    assert diagnostics["fallback_skin_count"] == 0
    assert diagnostics["fallback_cell_count"] == 0


def test_quality_workflow_case_specific_guardrails(
    workflow_reports: dict[str, dict[str, Any]],
) -> None:
    reference_cases = _cases_by_id(workflow_reports["reference"])
    quality_cases = _cases_by_id(workflow_reports["quality"])

    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "single_vertical_plane",
        FVT_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "single_vertical_plane",
        FVT_POSITIVE_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "parallel_planes",
        FVT_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "single_dipping_plane",
        SKIN_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "curved_surface",
        SKIN_BUFFERED_F1_R2,
        0.10,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "crossing_planes",
        SKIN_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "weak_noisy_plane",
        SKIN_BUFFERED_F1_R2,
        -0.08,
    )
    _assert_quality_at_most_reference_delta(
        reference_cases,
        quality_cases,
        "boundary_plane",
        FVT_EDGE_FALSE_POSITIVE_FRACTION,
        0.05,
    )
    boundary = quality_cases["boundary_plane"]
    assert _float_metric(boundary, *FVT_POSITIVE_CANDIDATE_COUNT) > 0.0
    assert _float_metric(boundary, *FVT_POSITIVE_BUFFERED_F1_R2) >= 0.98
    assert _float_metric(boundary, *FVT_POSITIVE_DISTANCE_P95) <= 1.0
    assert _float_metric(boundary, *FVT_POSITIVE_EDGE_FALSE_POSITIVE_FRACTION) <= 1.0e-12
    diagnostics = boundary["variants"]["current_default"]["skinning"]["diagnostics"]
    skin_count = _float_metric(boundary, "quality", "skin", "topology", "skin_count")
    assert skin_count > 0
    assert _float_metric(boundary, *SKIN_BUFFERED_F1_R2) >= 0.5
    assert (
        _float_metric(boundary, "quality", "skin", "surface_distance", "candidate_to_truth_p95")
        <= 3.0
    )
    assert diagnostics["fallback_enabled"] is True
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "empty_primary_skin_with_positive_fvt"
    assert diagnostics["fallback_skin_count"] == skin_count
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "boundary_plane",
        SKIN_BUFFERED_F1_R2,
        -0.02,
    )
