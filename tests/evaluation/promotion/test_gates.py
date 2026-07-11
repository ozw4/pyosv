from __future__ import annotations

from pathlib import Path

from pyosv.evaluation.promotion.comparison import compare_rows
from pyosv.evaluation.promotion.gates import build_promotion_report, evaluate_gate
from pyosv.evaluation.promotion.rows import SummaryRow


FIXTURE = Path("tests/fixtures/synthetic_quality_refactor/known_49_quality_summary.csv")


def test_none_gate_exact_json_contract() -> None:
    assert evaluate_gate("none", {}, {}) == {
        "name": "none",
        "passed": True,
        "boundary_plane": None,
        "non_boundary_regressions": [],
        "oracle_regressions": [],
        "false_fallback_replacements": [],
        "topology_regressions": [],
        "reasons": [],
    }


def test_known_49_boundary_aware_voter_fails_with_stable_reasons() -> None:
    report = build_promotion_report(
        FIXTURE, FIXTURE, candidate_variants=("boundary_aware_voter_v1",)
    )
    gate = report["promotion_gate"]["candidates"]["boundary_aware_voter_v1"]

    assert gate["passed"] is False
    assert gate["coverage"]["passed"] is True
    assert gate["reasons"] == [
        "skin_buffered_f1_r2 below 0.90",
        "skin_count above 3",
        "skin_cell_to_fvt_positive_candidate_ratio outside [0.75, 1.25]",
        "fvt_positive_buffered_f1_r2 below 0.90 after FVT change",
        "fvt_positive_distance_p95 above 2.0 after FVT change",
    ]


def test_required_coverage_and_strict_missing_rows_are_reported() -> None:
    report = build_promotion_report(
        FIXTURE,
        FIXTURE,
        candidate_variants=("missing",),
        strict_missing_rows=True,
    )
    candidate_gate = report["promotion_gate"]["candidates"]["missing"]

    assert candidate_gate["coverage"]["passed"] is False
    assert candidate_gate["coverage"]["checks"][0]["missing_case_ids"] == ["boundary_plane"]
    assert report["promotion_gate"]["missing_rows"]["missing"]["missing_candidate_rows"]
    assert "missing: missing matched rows" in report["promotion_gate"]["reasons"]


def _row(**values: object) -> SummaryRow:
    return SummaryRow.from_mapping(
        {
            "case_id": "boundary_plane",
            "pipeline": "scanner",
            "input_mode": "synthetic",
            "workflow_mode": "quality",
            "scanner_backend": "quality",
            "scanner_refinement_factor": 2,
            "shape_n3": 49,
            "shape_n2": 49,
            "shape_n1": 49,
            "skin_buffered_f1_r2": 0.95,
            "skin_count": 2,
            "skin_cell_count": 100,
            "fvt_positive_candidate_count": 100,
            "fvt_positive_buffered_f1_r2": 0.95,
            "fvt_positive_distance_p95": 1,
            "skin_distance_p95": 1,
            "skin_fallback_replaced_primary": False,
            "skin_over_merge_count": 0,
            "skin_over_split_count": 0,
            **values,
        }
    )


def _comparison(case_id: str, pipeline: str = "scanner", **candidate_values: object):
    baseline = _row(case_id=case_id, pipeline=pipeline)
    candidate = _row(case_id=case_id, pipeline=pipeline, **candidate_values)
    return candidate, compare_rows(baseline.key, baseline, candidate, "base", "candidate")


def test_material_topology_and_false_fallback_regressions() -> None:
    boundary, boundary_comparison = _comparison(
        "boundary_plane",
        skin_buffered_f1_r2=0.95,
        skin_count=2,
        skin_cell_count=100,
        fvt_positive_candidate_count=100,
    )
    material, material_comparison = _comparison(
        "single_vertical_plane",
        skin_buffered_f1_r2=0.92,
        skin_fallback_replaced_primary=True,
    )
    topology, topology_comparison = _comparison("parallel_planes", skin_over_merge_count=1)
    rows = {row.key: row for row in (boundary, material, topology)}
    comparisons = {
        row.key: comparison
        for row, comparison in (
            (boundary, boundary_comparison),
            (material, material_comparison),
            (topology, topology_comparison),
        )
    }

    gate = evaluate_gate("scanner-boundary", rows, comparisons)

    assert gate["non_boundary_regressions"][0]["metric"] == "skin_buffered_f1_r2"
    assert gate["false_fallback_replacements"][0]["key"]["case_id"] == "single_vertical_plane"
    assert gate["topology_regressions"][0]["metric"] == "skin_over_merge_count"
