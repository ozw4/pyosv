from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
EXTENDED_CASE_IDS = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)


BASE_FIELDS = (
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
    "skin_buffered_f1_r2",
    "skin_count",
    "skin_cell_count",
    "fvt_positive_candidate_count",
    "fvt_positive_buffered_f1_r2",
    "fvt_positive_distance_p95",
    "skin_distance_p95",
    "skin_fallback_replaced_primary",
    "skin_over_merge_count",
    "skin_over_split_count",
)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "case_id": "boundary_plane",
        "pipeline": "scanner",
        "variant": "current_default",
        "input_mode": "synthetic",
        "workflow_mode": "quality",
        "scanner_backend": "quality",
        "scanner_refinement_factor": 2,
        "shape_n3": 49,
        "shape_n2": 49,
        "shape_n1": 49,
        "skin_buffered_f1_r2": 0.8,
        "skin_count": 4,
        "skin_cell_count": 50,
        "fvt_positive_candidate_count": 100,
        "fvt_positive_buffered_f1_r2": 0.95,
        "fvt_positive_distance_p95": 1.0,
        "skin_distance_p95": 2.0,
        "skin_fallback_replaced_primary": False,
        "skin_over_merge_count": 0,
        "skin_over_split_count": 0,
    }
    row.update(overrides)
    return row


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = BASE_FIELDS
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _promotion_gate_rows(candidate_variant: str) -> list[dict[str, Any]]:
    rows = []
    for pipeline in ("scanner", "oracle"):
        for case_id in EXTENDED_CASE_IDS:
            baseline = _row(case_id=case_id, pipeline=pipeline)
            candidate = _row(case_id=case_id, pipeline=pipeline, variant=candidate_variant)
            if pipeline == "oracle":
                baseline["scanner_backend"] = ""
                candidate["scanner_backend"] = ""
                baseline["scanner_refinement_factor"] = ""
                candidate["scanner_refinement_factor"] = ""
            if pipeline == "scanner" and case_id == "boundary_plane":
                candidate.update(
                    skin_buffered_f1_r2=0.91,
                    skin_count=2,
                    skin_cell_count=100,
                )
            rows.extend((baseline, candidate))
    return rows


def _run_compare(
    tmp_path: Path,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *args: str,
    fields: tuple[str, ...] = BASE_FIELDS,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "quality_delta.json"
    _write_csv(baseline, baseline_rows, fields=fields)
    _write_csv(candidate, candidate_rows, fields=fields)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(baseline),
            str(candidate),
            "--output-json",
            str(output),
            *args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    return result, data, output


def test_compare_quality_reports_writes_json(tmp_path: Path) -> None:
    result, data, output = _run_compare(tmp_path, [_row()], [_row(skin_buffered_f1_r2=0.9)])

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert data["format_version"] == 1
    assert data["row_count"] == 1
    assert data["comparisons"][0]["metrics"]["skin_buffered_f1_r2"]["candidate"] == 0.9


def test_variant_is_excluded_from_match_key(tmp_path: Path) -> None:
    result, data, _ = _run_compare(
        tmp_path,
        [_row(variant="current_default")],
        [_row(variant="boundary_edge_thin_v1", skin_buffered_f1_r2=0.91)],
        "--candidate-variant",
        "boundary_edge_thin_v1",
    )

    assert result.returncode == 0, result.stderr
    comparison = data["comparisons"][0]
    assert data["row_count"] == 1
    assert comparison["baseline_variant"] == "current_default"
    assert comparison["candidate_variant"] == "boundary_edge_thin_v1"
    assert "variant" not in comparison["key"]


def test_numeric_delta_and_skin_to_fvt_ratio_are_computed(tmp_path: Path) -> None:
    result, data, _ = _run_compare(
        tmp_path,
        [_row(skin_buffered_f1_r2=0.45, skin_cell_count=60, fvt_positive_candidate_count=100)],
        [_row(skin_buffered_f1_r2=0.91, skin_cell_count=100, fvt_positive_candidate_count=100)],
    )

    assert result.returncode == 0, result.stderr
    comparison = data["comparisons"][0]
    assert comparison["metrics"]["skin_buffered_f1_r2"]["delta"] == 0.46
    ratio = comparison["derived"]["skin_cell_to_fvt_positive_candidate_ratio"]
    assert ratio == {"baseline": 0.6, "candidate": 1.0, "delta": 0.4}


def test_scanner_boundary_gate_fails_when_candidate_misses_conditions(tmp_path: Path) -> None:
    result, data, _ = _run_compare(
        tmp_path,
        [_row(skin_buffered_f1_r2=0.91, skin_count=2, skin_cell_count=100)],
        [_row(skin_buffered_f1_r2=0.80, skin_count=4, skin_cell_count=50)],
        "--promotion-gate",
        "scanner-boundary",
    )

    assert result.returncode == 0, result.stderr
    gate = data["promotion_gate"]
    assert gate["passed"] is False
    assert gate["boundary_plane"]["passed"] is False
    assert "skin_buffered_f1_r2 below 0.90" in gate["reasons"]


def test_scanner_boundary_gate_passes_for_candidate_that_meets_conditions(
    tmp_path: Path,
) -> None:
    passing_row = _row(skin_buffered_f1_r2=0.91, skin_count=2, skin_cell_count=100)
    result, data, _ = _run_compare(
        tmp_path,
        [passing_row],
        [passing_row],
        "--promotion-gate",
        "scanner-boundary",
    )

    assert result.returncode == 0, result.stderr
    gate = data["promotion_gate"]
    assert gate["passed"] is True
    assert gate["boundary_plane"]["fvt_changed"] is False
    assert gate["reasons"] == []


def test_missing_metric_columns_are_null_instead_of_errors(tmp_path: Path) -> None:
    fields = (
        "case_id",
        "pipeline",
        "variant",
        "input_mode",
        "workflow_mode",
        "shape_n3",
        "shape_n2",
        "shape_n1",
    )
    result, data, _ = _run_compare(
        tmp_path,
        [_row(scanner_backend="")],
        [_row(scanner_backend="")],
        fields=fields,
    )

    assert result.returncode == 0, result.stderr
    metrics = data["comparisons"][0]["metrics"]
    assert metrics["skin_buffered_f1_r2"] == {
        "baseline": None,
        "candidate": None,
        "delta": None,
    }
    assert metrics["scanner_ft_distance_p95"] == {
        "baseline": None,
        "candidate": None,
        "delta": None,
    }


def test_fail_on_gate_failure_returns_exit_code_2(tmp_path: Path) -> None:
    result, data, _ = _run_compare(
        tmp_path,
        [_row()],
        [_row(skin_buffered_f1_r2=0.5)],
        "--promotion-gate",
        "scanner-boundary",
        "--fail-on-gate-failure",
    )

    assert result.returncode == 2
    assert data["promotion_gate"]["passed"] is False


def test_scanner_boundary_gate_flags_false_non_boundary_fallback_replacement(
    tmp_path: Path,
) -> None:
    baseline_rows = [
        _row(),
        _row(case_id="single_vertical_plane", skin_fallback_replaced_primary=False),
    ]
    candidate_rows = [
        _row(skin_buffered_f1_r2=0.91, skin_count=2, skin_cell_count=100),
        _row(case_id="single_vertical_plane", skin_fallback_replaced_primary=True),
    ]
    result, data, _ = _run_compare(
        tmp_path,
        baseline_rows,
        candidate_rows,
        "--promotion-gate",
        "scanner-boundary",
    )

    assert result.returncode == 0, result.stderr
    gate = data["promotion_gate"]
    assert gate["passed"] is False
    assert gate["false_fallback_replacements"][0]["key"]["case_id"] == "single_vertical_plane"
    assert "stable non-boundary false fallback replacement" in gate["reasons"][0]


def test_scanner_boundary_gate_flags_parallel_crossing_topology_regression(
    tmp_path: Path,
) -> None:
    baseline_rows = [_row(), _row(case_id="parallel_planes", skin_over_merge_count=0)]
    candidate_rows = [
        _row(skin_buffered_f1_r2=0.91, skin_count=2, skin_cell_count=100),
        _row(case_id="parallel_planes", skin_over_merge_count=1),
    ]
    result, data, _ = _run_compare(
        tmp_path,
        baseline_rows,
        candidate_rows,
        "--promotion-gate",
        "scanner-boundary",
    )

    assert result.returncode == 0, result.stderr
    gate = data["promotion_gate"]
    assert gate["passed"] is False
    assert gate["topology_regressions"][0]["metric"] == "skin_over_merge_count"


def test_check_synthetic_quality_promotion_gate_help() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--candidate-variant" in result.stdout


def test_check_synthetic_quality_promotion_gate_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    output_json = tmp_path / "promotion_gate.json"
    output_markdown = tmp_path / "promotion_gate.md"
    _write_csv(
        summary,
        _promotion_gate_rows("quality_boundary_skinner_fallback_v5"),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--baseline-summary",
            str(summary),
            "--candidate-summary",
            str(summary),
            "--candidate-variant",
            "quality_boundary_skinner_fallback_v5",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["promotion_gate"]["passed"] is True
    assert data["promotion_gate"]["promotable_candidates"] == [
        "quality_boundary_skinner_fallback_v5"
    ]
    candidate_gate = data["promotion_gate"]["candidates"]["quality_boundary_skinner_fallback_v5"]
    assert candidate_gate["coverage"]["passed"] is True
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "quality_boundary_skinner_fallback_v5" in markdown
    assert "scanner-boundary" in markdown
    assert "oracle_49" in markdown


def test_check_synthetic_quality_promotion_gate_requires_full_gate_coverage(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    output_json = tmp_path / "promotion_gate.json"
    _write_csv(
        summary,
        [
            _row(variant="current_default"),
            _row(
                variant="quality_boundary_skinner_fallback_v5",
                skin_buffered_f1_r2=0.91,
                skin_count=2,
                skin_cell_count=100,
            ),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--baseline-summary",
            str(summary),
            "--candidate-summary",
            str(summary),
            "--candidate-variant",
            "quality_boundary_skinner_fallback_v5",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))
    gate = data["promotion_gate"]
    candidate_gate = gate["candidates"]["quality_boundary_skinner_fallback_v5"]
    assert gate["passed"] is False
    assert gate["promotable_candidates"] == []
    assert candidate_gate["coverage"]["passed"] is False
    assert any("missing required gate coverage: oracle_49" in reason for reason in gate["reasons"])


def test_check_synthetic_quality_promotion_gate_requires_quality_backend_and_refinement(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    output_json = tmp_path / "promotion_gate.json"
    rows = _promotion_gate_rows("quality_boundary_skinner_fallback_v5")
    for row in rows:
        if row["pipeline"] == "scanner":
            row["scanner_backend"] = "fast"
            row["scanner_refinement_factor"] = 1
    _write_csv(summary, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--baseline-summary",
            str(summary),
            "--candidate-summary",
            str(summary),
            "--candidate-variant",
            "quality_boundary_skinner_fallback_v5",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))
    gate = data["promotion_gate"]
    candidate_gate = gate["candidates"]["quality_boundary_skinner_fallback_v5"]
    scanner_checks = [
        check for check in candidate_gate["coverage"]["checks"] if check["pipeline"] == "scanner"
    ]
    assert gate["passed"] is False
    assert scanner_checks
    assert all(check["passed"] is False for check in scanner_checks)
    assert all(check["scanner_backend"] == "quality" for check in scanner_checks)
    assert all(check["scanner_refinement_factor"] == "2" for check in scanner_checks)
