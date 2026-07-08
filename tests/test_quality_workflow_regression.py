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
    assert quality["voting"]["surface_support_min_fraction"] == 0.5
    assert quality["voting"]["surface_support_exponent"] == 1.0
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
            for field in ("fv_top_truth_count", "fvt_top_truth_count"):
                _assert_finite_metric_tree(quality[field]["buffered_overlap_radius2"])
                _assert_finite_metric_tree(quality[field]["surface_distance"])
                _assert_finite_metric_tree(quality[field]["orientation_error"])

            edge_false_positive = quality["edge_false_positive"]
            _assert_finite_metric_tree(edge_false_positive["fv_top_truth_count"])
            _assert_finite_metric_tree(edge_false_positive["fvt_top_truth_count"])

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


def test_quality_workflow_broad_guardrails(
    workflow_reports: dict[str, dict[str, Any]],
) -> None:
    reference_cases = _cases_by_id(workflow_reports["reference"])
    quality_cases = _cases_by_id(workflow_reports["quality"])

    curved_reference_f1 = _float_metric(
        reference_cases["curved_surface"],
        "quality",
        "fvt_top_truth_count",
        "buffered_overlap_radius2",
        "buffered_f1",
    )
    curved_quality_f1 = _float_metric(
        quality_cases["curved_surface"],
        "quality",
        "fvt_top_truth_count",
        "buffered_overlap_radius2",
        "buffered_f1",
    )
    assert curved_quality_f1 >= curved_reference_f1 - 0.05

    vertical_reference_f1 = _float_metric(
        reference_cases["single_vertical_plane"],
        "quality",
        "fvt_top_truth_count",
        "buffered_overlap_radius2",
        "buffered_f1",
    )
    vertical_quality_f1 = _float_metric(
        quality_cases["single_vertical_plane"],
        "quality",
        "fvt_top_truth_count",
        "buffered_overlap_radius2",
        "buffered_f1",
    )
    assert vertical_quality_f1 >= vertical_reference_f1 - 0.08

    boundary_reference_edge_fraction = _float_metric(
        reference_cases["boundary_plane"],
        "quality",
        "edge_false_positive",
        "fvt_top_truth_count",
        "edge_false_positive_fraction_of_candidates",
    )
    boundary_quality_edge_fraction = _float_metric(
        quality_cases["boundary_plane"],
        "quality",
        "edge_false_positive",
        "fvt_top_truth_count",
        "edge_false_positive_fraction_of_candidates",
    )
    assert boundary_quality_edge_fraction <= boundary_reference_edge_fraction + 0.05
