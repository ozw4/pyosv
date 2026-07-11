from __future__ import annotations

import csv
from pathlib import Path

from pyosv.evaluation.promotion.comparison import compare_reports, compare_rows
from pyosv.evaluation.promotion.rows import SummaryRow


def _row(**values: object) -> SummaryRow:
    return SummaryRow.from_mapping(
        {
            "case_id": "boundary_plane",
            "pipeline": "scanner",
            "workflow_mode": "quality",
            "scanner_backend": "quality",
            "scanner_refinement_factor": 2,
            "shape_n3": 49,
            "shape_n2": 49,
            "shape_n1": 49,
            **values,
        }
    )


def test_numeric_delta_ratio_and_missing_metric_contract() -> None:
    baseline = _row(skin_buffered_f1_r2=0.45, skin_cell_count=60, fvt_positive_candidate_count=100)
    candidate = _row(
        skin_buffered_f1_r2=0.91, skin_cell_count=100, fvt_positive_candidate_count=100
    )

    comparison = compare_rows(baseline.key, baseline, candidate, "base", "candidate")

    assert comparison["metrics"]["skin_buffered_f1_r2"] == {
        "baseline": 0.45,
        "candidate": 0.91,
        "delta": 0.46,
    }
    assert comparison["derived"]["skin_cell_to_fvt_positive_candidate_ratio"] == {
        "baseline": 0.6,
        "candidate": 1.0,
        "delta": 0.4,
    }
    assert comparison["metrics"]["scanner_ft_distance_p95"] == {
        "baseline": None,
        "candidate": None,
        "delta": None,
    }


def test_compare_reports_exact_json_contract_and_strict_missing_rows(tmp_path: Path) -> None:
    fields = (
        "case_id",
        "pipeline",
        "variant",
        "input_mode",
        "workflow_mode",
        "scanner_backend",
        "scanner_refinement_factor",
        "shape_n3",
        "shape_n2",
        "shape_n1",
    )
    summary = tmp_path / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "boundary_plane",
                "pipeline": "scanner",
                "variant": "base",
                "input_mode": "",
                "workflow_mode": "quality",
                "scanner_backend": "quality",
                "scanner_refinement_factor": 2,
                "shape_n3": 49,
                "shape_n2": 49,
                "shape_n1": 49,
            }
        )

    report = compare_reports(summary, summary, "base", "missing", strict_missing_rows=True)

    assert report == {
        "format_version": 1,
        "config": {
            "baseline_summary": str(summary),
            "candidate_summary": str(summary),
            "baseline_variant": "base",
            "candidate_variant": "missing",
            "promotion_gate": "none",
            "strict_missing_rows": True,
        },
        "row_count": 0,
        "missing_baseline_rows": [],
        "missing_candidate_rows": [
            {
                "case_id": "boundary_plane",
                "pipeline": "scanner",
                "input_mode": "",
                "workflow_mode": "quality",
                "scanner_backend": "quality",
                "scanner_refinement_factor": "2",
                "shape_n3": "49",
                "shape_n2": "49",
                "shape_n1": "49",
            }
        ],
        "comparisons": [],
        "promotion_gate": {
            "name": "none",
            "passed": True,
            "boundary_plane": None,
            "non_boundary_regressions": [],
            "oracle_regressions": [],
            "false_fallback_replacements": [],
            "topology_regressions": [],
            "reasons": [],
        },
    }
