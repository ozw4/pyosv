from __future__ import annotations

import pytest

from pyosv.evaluation.synthetic_quality.application import build_report
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
)


def test_extended_report_preserves_legacy_weak_noisy_case_and_default_metrics() -> None:
    report = build_report(
        case_set="extended",
        shape=(5, 5, 5),
        variants=("current_default",),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )

    assert report["format_version"] == 1
    assert tuple(case["case_id"] for case in report["cases"]) == (
        "single_vertical_plane",
        "single_dipping_plane",
        "curved_surface",
        "parallel_planes",
        "crossing_planes",
        "boundary_plane",
        "weak_noisy_plane",
    )
    weak_noisy = report["cases"][-1]
    overlap = weak_noisy["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"]
    assert overlap["candidate_count"] == 33
    assert overlap["truth_count"] == 61
    assert overlap["intersection_count"] == 32
    assert overlap["f1"] == pytest.approx(0.6808510638297871)


def test_scanner_backend_matrix_input_mode_preserves_legacy_report_structure() -> None:
    report = build_report(
        case_set="minimal",
        shape=(3, 3, 3),
        variants=("current_default",),
        input_mode="scanner",
        scanner_backend_matrix=True,
        scanner_config=SyntheticScannerConfig(
            backend="fast",
            phi_min=0.0,
            phi_max=0.0,
            theta_min=90.0,
            theta_max=90.0,
            sigma1=1.0,
            sigma2=1.0,
            scanner_thin_mode="none",
        ),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )

    assert report["format_version"] == 1
    assert report["config"]["input_mode"] == "scanner"
    case = report["cases"][0]
    matrix = case["scanner_backend_matrix"]
    assert tuple(matrix["backends"]) == ("reference-like", "quality", "fast")
    assert case["scanner"] == matrix["backends"]["fast"]["scanner"]
    assert case["scanner_quality"] == matrix["backends"]["fast"]["scanner_quality"]
