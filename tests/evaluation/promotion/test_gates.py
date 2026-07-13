from __future__ import annotations

import csv
from pathlib import Path

from pyosv.evaluation.promotion.comparison import compare_rows
from pyosv.evaluation.promotion.gates import (
    add_required_coverage,
    build_promotion_report,
    evaluate_gate,
)
from pyosv.evaluation.promotion.rows import SummaryRow
from pyosv.evaluation.promotion.specifications import EXTENDED_CASES
from pyosv.evaluation.reporting import SUMMARY_CSV_V1_FIELDS


FIXTURE = Path("tests/fixtures/synthetic_quality_refactor/known_49_quality_summary.csv")


def test_known_49_fixture_is_a_full_report_artifact() -> None:
    with FIXTURE.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == SUMMARY_CSV_V1_FIELDS
    assert len(rows) == 28
    assert {row["variant"] for row in rows} == {
        "current_default",
        "boundary_aware_voter_v1",
    }
    assert {row["shape_n3"] for row in rows} == {"49"}
    for row in rows:
        for field in ("skin_count", "skin_cell_count", "fvt_positive_candidate_count"):
            if row[field]:
                int(row[field])


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


def _comparison(
    case_id: str,
    pipeline: str = "scanner",
    *,
    scanner_backend: str = "quality",
    **candidate_values: object,
):
    baseline = _row(
        case_id=case_id,
        pipeline=pipeline,
        scanner_backend=scanner_backend,
    )
    candidate = _row(
        case_id=case_id,
        pipeline=pipeline,
        scanner_backend=scanner_backend,
        **candidate_values,
    )
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


def _gate_inputs(
    comparisons: list[tuple[SummaryRow, dict[str, object]]],
) -> tuple[dict[tuple[str, ...], SummaryRow], dict[tuple[str, ...], dict[str, object]]]:
    return (
        {row.key: row for row, _ in comparisons},
        {row.key: comparison for row, comparison in comparisons},
    )


def test_reference_like_gate_selects_only_reference_like_backend() -> None:
    quality_boundary = _comparison("boundary_plane")
    rows, comparisons = _gate_inputs([quality_boundary])

    gate = evaluate_gate("scanner-boundary-reference-like", rows, comparisons)

    assert gate["passed"] is False
    assert gate["boundary_plane"] is None
    assert gate["reasons"] == [
        "missing boundary_plane scanner reference-like refinement-2 49^3 candidate row"
    ]


def test_reference_like_gate_always_checks_boundary_fvt_thresholds() -> None:
    reference_like_baseline = _row(
        scanner_backend="reference-like",
        fvt_positive_buffered_f1_r2=0.80,
        fvt_positive_distance_p95=3.0,
    )
    reference_like_candidate = _row(
        scanner_backend="reference-like",
        fvt_positive_buffered_f1_r2=0.80,
        fvt_positive_distance_p95=3.0,
    )
    reference_like_boundary = (
        reference_like_candidate,
        compare_rows(
            reference_like_candidate.key,
            reference_like_baseline,
            reference_like_candidate,
            "base",
            "candidate",
        ),
    )
    rows, comparisons = _gate_inputs([reference_like_boundary])

    gate = evaluate_gate("scanner-boundary-reference-like", rows, comparisons)

    assert gate["boundary_plane"]["fvt_changed"] is False
    assert gate["reasons"] == [
        "fvt_positive_buffered_f1_r2 below 0.90",
        "fvt_positive_distance_p95 above 2.0",
    ]

    quality_baseline = _row(
        fvt_positive_buffered_f1_r2=0.80,
        fvt_positive_distance_p95=3.0,
    )
    quality_candidate = _row(
        fvt_positive_buffered_f1_r2=0.80,
        fvt_positive_distance_p95=3.0,
    )
    quality_boundary = (
        quality_candidate,
        compare_rows(
            quality_candidate.key,
            quality_baseline,
            quality_candidate,
            "base",
            "candidate",
        ),
    )
    rows, comparisons = _gate_inputs([quality_boundary])
    legacy_gate = evaluate_gate("scanner-boundary", rows, comparisons)
    assert legacy_gate["reasons"] == []


def test_reference_like_gate_requires_exact_oracle_numeric_values() -> None:
    reference_like_boundary = _comparison("boundary_plane", scanner_backend="reference-like")
    reference_like_oracle = _comparison(
        "single_vertical_plane",
        pipeline="oracle",
        scanner_backend="",
        skin_buffered_f1_r2=0.950000001,
    )
    rows, comparisons = _gate_inputs([reference_like_boundary, reference_like_oracle])

    gate = evaluate_gate("scanner-boundary-reference-like", rows, comparisons)

    assert gate["passed"] is False
    assert len(gate["oracle_regressions"]) == 1
    oracle_regression = gate["oracle_regressions"][0]
    assert {key: value for key, value in oracle_regression.items() if key != "delta"} == {
        "key": reference_like_oracle[1]["key"],
        "metric": "skin_buffered_f1_r2",
        "baseline": 0.95,
        "candidate": 0.950000001,
        "threshold": 0,
    }
    assert oracle_regression["delta"] > 0
    assert gate["reasons"] == ["oracle regression: skin_buffered_f1_r2"]

    quality_boundary = _comparison("boundary_plane")
    quality_oracle = _comparison(
        "single_vertical_plane",
        pipeline="oracle",
        scanner_backend="",
        skin_buffered_f1_r2=0.950000001,
    )
    rows, comparisons = _gate_inputs([quality_boundary, quality_oracle])
    assert evaluate_gate("scanner-boundary", rows, comparisons)["oracle_regressions"] == []


def test_reference_like_gate_does_not_excuse_improved_false_fallback() -> None:
    reference_like_boundary = _comparison("boundary_plane", scanner_backend="reference-like")
    reference_like_stable = _comparison(
        "single_vertical_plane",
        scanner_backend="reference-like",
        skin_buffered_f1_r2=0.98,
        skin_fallback_replaced_primary=True,
    )
    rows, comparisons = _gate_inputs([reference_like_boundary, reference_like_stable])

    gate = evaluate_gate("scanner-boundary-reference-like", rows, comparisons)

    assert [item["key"]["case_id"] for item in gate["false_fallback_replacements"]] == [
        "single_vertical_plane"
    ]

    quality_boundary = _comparison("boundary_plane")
    quality_stable = _comparison(
        "single_vertical_plane",
        skin_buffered_f1_r2=0.98,
        skin_fallback_replaced_primary=True,
    )
    rows, comparisons = _gate_inputs([quality_boundary, quality_stable])
    assert evaluate_gate("scanner-boundary", rows, comparisons)["false_fallback_replacements"] == []


def test_reference_like_gate_requires_all_fourteen_rows() -> None:
    scanner_comparisons = [
        _comparison(case_id, scanner_backend="reference-like") for case_id in EXTENDED_CASES
    ]
    oracle_comparisons = [
        _comparison(case_id, pipeline="oracle", scanner_backend="") for case_id in EXTENDED_CASES
    ]
    rows, comparisons = _gate_inputs([*scanner_comparisons, *oracle_comparisons])
    gate = evaluate_gate("scanner-boundary-reference-like", rows, comparisons)
    report = add_required_coverage(
        {
            "config": {"promotion_gate": "scanner-boundary-reference-like"},
            "comparisons": list(comparisons.values()),
            "promotion_gate": gate,
        }
    )

    assert len(report["comparisons"]) == 14
    assert report["promotion_gate"]["coverage"]["passed"] is True
    assert all(
        check["scanner_backend"] in {None, "reference-like"}
        for check in report["promotion_gate"]["coverage"]["checks"]
    )

    incomplete_report = {
        **report,
        "promotion_gate": {key: value for key, value in gate.items() if key != "coverage"},
        "comparisons": report["comparisons"][:-1],
    }
    incomplete_report = add_required_coverage(incomplete_report)
    assert incomplete_report["promotion_gate"]["coverage"]["passed"] is False
    assert incomplete_report["promotion_gate"]["coverage"]["checks"][2]["missing_case_ids"] == [
        "weak_noisy_plane"
    ]
