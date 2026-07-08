from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "report_3d_synthetic_quality.py"
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
SKIN_BUFFERED_F1_R2 = (
    "quality",
    "skin",
    "buffered_overlap_radius2",
    "buffered_f1",
)
FVT_EDGE_FALSE_POSITIVE_FRACTION = (
    "quality",
    "edge_false_positive",
    "fvt_top_truth_count",
    "edge_false_positive_fraction_of_candidates",
)


def _load_report_module() -> object:
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert reference["skinning"]["adaptive_min_likelihood"] is False
    assert reference["skinning"]["seed_min_ep"] == 0.8

    quality = workflow_reports["quality"]["config"]
    assert quality["workflow_mode"] == "quality"
    assert quality["voting"]["voter_thin_mode"] == "hybrid"
    assert quality["voting"]["surface_support_min_fraction"] == 0.0
    assert quality["voting"]["surface_support_exponent"] == 0.0
    assert quality["skinning"]["method"] == "quality"
    assert quality["skinning"]["min_likelihood"] is None
    assert quality["skinning"]["adaptive_min_likelihood"] is True
    assert quality["skinning"]["seed_min_ep"] == 0.5


def test_quality_workflow_key_metrics_are_finite(
    workflow_reports: dict[str, dict[str, Any]],
) -> None:
    for report in workflow_reports.values():
        assert report["config"]["shape"] == list(SHAPE)
        assert report["config"]["variants"] == ["current_default"]
        for case in _cases_by_id(report).values():
            assert case["variants"]["current_default"]["quality"] == case["quality"]
            assert case["variants"]["current_default"]["skinning"] == {"enabled": True}

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
            for field in (
                "topology",
                "buffered_overlap_radius2",
                "surface_distance",
                "orientation_error",
            ):
                assert field in skin
                _assert_finite_metric_tree(skin[field])


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
        "parallel_planes",
        FVT_BUFFERED_F1_R2,
        -0.02,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "curved_surface",
        FVT_BUFFERED_F1_R2,
        0.15,
    )
    _assert_quality_at_least_reference_delta(
        reference_cases,
        quality_cases,
        "crossing_planes",
        SKIN_BUFFERED_F1_R2,
        -0.10,
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
