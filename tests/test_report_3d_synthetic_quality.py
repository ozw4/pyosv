from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "report_3d_synthetic_quality.py"
GEOMETRY_CASE_IDS = ("single_vertical_plane", "single_dipping_plane", "curved_surface")
EXTENDED_CASE_IDS = (
    *GEOMETRY_CASE_IDS,
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)
DIAGNOSTIC_VARIANTS = (
    "current_default",
    "no_surface_orientation_smoothing",
    "final_norm_smoothing_1",
    "voter_thin_normal",
)
EXPECTED_VOLUME_FILES = (
    "truth_fault_mask.dat",
    "truth_distance.dat",
    "truth_strike.dat",
    "truth_dip.dat",
    "ft_oracle.dat",
    "pt_oracle.dat",
    "tt_oracle.dat",
    "fv_py.dat",
    "vp_py.dat",
    "vt_py.dat",
    "fvt_py.dat",
    "skin_mask_py.dat",
)
EXPECTED_SCANNER_VOLUME_FILES = (
    "scanner_input.dat",
    "ft_scan.dat",
    "pt_scan.dat",
    "tt_scan.dat",
    "ft_used.dat",
    "pt_used.dat",
    "tt_used.dat",
)
EXPECTED_I3_FIGURES = (
    "ft_oracle_i3_center.png",
    "fv_py_i3_center.png",
    "fvt_py_i3_center.png",
    "skin_mask_py_i3_center.png",
    "truth_vs_fvt_overlay_i3_center.png",
    "truth_vs_skin_overlay_i3_center.png",
)
EXPECTED_SCANNER_I3_FIGURES = (
    "scanner_input_i3_center.png",
    "ft_scan_i3_center.png",
    "ft_used_i3_center.png",
    "truth_vs_ft_scan_overlay_i3_center.png",
    "truth_vs_ft_used_overlay_i3_center.png",
    "truth_vs_fvt_overlay_i3_center.png",
)
EXPECTED_SKIN_SUMMARY_FIELDS = (
    "skin_enabled",
    "skin_count",
    "skin_cell_count",
    "skin_unique_cell_count",
    "skin_duplicate_cell_count",
    "skin_largest_size",
    "skin_largest_fraction",
    "skin_small_count",
    "skin_small_cell_fraction",
    "skin_buffered_f1_r2",
    "skin_buffered_precision_r2",
    "skin_buffered_recall_r2",
    "skin_distance_candidate_to_truth_p95",
    "skin_distance_truth_to_candidate_p95",
    "skin_distance_hausdorff_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "skin_buffered_f1_delta_vs_baseline",
    "skin_distance_p95_delta_vs_baseline",
    "skin_strike_median_error_delta_vs_baseline",
    "skin_dip_median_error_delta_vs_baseline",
    "skin_count_delta_vs_baseline",
)
SKIN_NUMERIC_SUMMARY_FIELDS = (
    "skin_count",
    "skin_cell_count",
    "skin_unique_cell_count",
    "skin_duplicate_cell_count",
    "skin_largest_size",
    "skin_largest_fraction",
    "skin_small_count",
    "skin_small_cell_fraction",
    "skin_buffered_f1_r2",
    "skin_buffered_precision_r2",
    "skin_buffered_recall_r2",
    "skin_distance_candidate_to_truth_p95",
    "skin_distance_truth_to_candidate_p95",
    "skin_distance_hausdorff_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
)
SKIN_EMPTY_WHEN_DISABLED_FIELDS = (
    "skin_buffered_f1_r2",
    "skin_buffered_precision_r2",
    "skin_buffered_recall_r2",
    "skin_distance_p95",
    "skin_distance_candidate_to_truth_p95",
    "skin_distance_truth_to_candidate_p95",
    "skin_distance_hausdorff_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "skin_buffered_f1_delta_vs_baseline",
    "skin_distance_p95_delta_vs_baseline",
    "skin_strike_median_error_delta_vs_baseline",
    "skin_dip_median_error_delta_vs_baseline",
    "skin_count_delta_vs_baseline",
)
EXPECTED_SCANNER_SUMMARY_FIELDS = (
    "input_mode",
    "scanner_backend",
    "scanner_thin_mode",
    "scanner_ft_buffered_f1_r2",
    "scanner_ft_distance_p95",
    "scanner_strike_median_error",
    "scanner_dip_median_error",
    "scanner_input_contrast",
)
EXPECTED_THINNING_DIAGNOSTIC_SUMMARY_FIELDS = (
    "thinning_diag_reference_fvt_buffered_f1_r2",
    "thinning_diag_normal_fvt_buffered_f1_r2",
    "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2",
    "thinning_diag_reference_fvt_distance_p95",
    "thinning_diag_normal_fvt_distance_p95",
    "thinning_diag_normal_minus_reference_fvt_distance_p95",
    "thinning_diag_reference_count",
    "thinning_diag_normal_count",
    "thinning_diag_intersection_count",
    "thinning_diag_reference_only_count",
    "thinning_diag_normal_only_count",
    "thinning_diag_jaccard",
)
EXPECTED_THINNING_DIAGNOSTIC_VOLUME_FILES = (
    "fvt_reference.dat",
    "fvt_normal.dat",
    "keep_reference.dat",
    "keep_normal.dat",
    "keep_both.dat",
    "keep_reference_only.dat",
    "keep_normal_only.dat",
)
EXPECTED_THINNING_DIAGNOSTIC_I3_FIGURES = (
    "truth_vs_fvt_reference_overlay_i3_center.png",
    "truth_vs_fvt_normal_overlay_i3_center.png",
    "truth_vs_keep_reference_only_overlay_i3_center.png",
    "truth_vs_keep_normal_only_overlay_i3_center.png",
    "fvt_reference_vs_normal_i3_center.png",
)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, str(SCRIPT.relative_to(REPO_ROOT)), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_enabled_skin_summary_row(row: dict[str, str]) -> None:
    for field in EXPECTED_SKIN_SUMMARY_FIELDS:
        assert field in row
    assert row["skin_enabled"] == "True"
    for field in SKIN_NUMERIC_SUMMARY_FIELDS:
        assert math.isfinite(float(row[field]))


def _assert_scanner_quality_contract(scanner_quality: dict[str, object]) -> None:
    ft_quality = scanner_quality["ft_top_truth_count"]
    assert "buffered_overlap_radius2" in ft_quality
    assert "surface_distance" in ft_quality
    assert ft_quality["buffered_overlap_radius2"]["candidate_count"] > 0
    assert ft_quality["surface_distance"]["candidate_count"] > 0

    orientation_error = scanner_quality["orientation_error"]
    for name in ("raw_scan_top_truth_count", "used_attributes_top_truth_count"):
        assert orientation_error[name]["count"] > 0
        assert math.isfinite(float(orientation_error[name]["strike_median"]))
        assert math.isfinite(float(orientation_error[name]["dip_median"]))

    input_association = scanner_quality["input_association"]
    assert math.isfinite(float(input_association["truth_surface_mean"]))
    assert math.isfinite(float(input_association["far_from_truth_mean"]))
    assert math.isfinite(float(input_association["contrast"]))


def _assert_top_truth_quality_has_orientation(quality: dict[str, object]) -> None:
    assert "buffered_f1" in quality["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in quality["surface_distance"]
    orientation = quality["orientation_error"]
    assert orientation["count"] > 0
    assert math.isfinite(float(orientation["strike_median"]))
    assert math.isfinite(float(orientation["dip_median"]))


def test_report_3d_synthetic_quality_help_exits_successfully() -> None:
    result = _run_script("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--case-set" in result.stdout
    assert "geometry" in result.stdout
    assert "extended" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--shape" in result.stdout
    assert "--variants" in result.stdout
    assert "--input-mode" in result.stdout
    assert "--scanner-backend" in result.stdout
    assert "--scanner-thin-mode" in result.stdout
    assert "--save-volumes" in result.stdout
    assert "--save-figures" in result.stdout
    assert "--write-markdown-index" in result.stdout
    assert "--voter-thin-mode" in result.stdout
    assert "--thinning-diagnostics" in result.stdout
    assert "--include-thinning-diagnostic" in result.stdout
    assert "--thinning-diagnostic-cases" in result.stdout
    assert "--truth-surface-half-width" in result.stdout
    assert "--buffer-radius" in result.stdout
    assert "--skip-skinning" in result.stdout
    assert "--skinner-min-likelihood" in result.stdout
    assert "--skinner-ru" in result.stdout
    assert "--no-skinner-reskin" in result.stdout
    assert "--small-skin-size" in result.stdout


def test_report_3d_synthetic_quality_minimal_case_writes_contract_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--pretty",
    )

    assert result.returncode == 0, result.stderr
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.csv"
    assert metrics_path.is_file()
    assert summary_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["format_version"] == 1
    assert metrics["config"]["case_set"] == "minimal"
    assert metrics["config"]["shape"] == [17, 17, 17]
    case = metrics["cases"][0]
    assert case["case_id"] == "single_vertical_plane"
    assert case["shape"] == [17, 17, 17]
    assert set(case["variants"]) == {"current_default"}
    assert case["truth"]["fault_voxel_count"] > case["truth"]["surface_voxel_count"] > 0
    quality = case["quality"]
    fv_quality = quality["fv_top_truth_count"]
    _assert_top_truth_quality_has_orientation(fv_quality)
    fvt_quality = quality["fvt_top_truth_count"]
    _assert_top_truth_quality_has_orientation(fvt_quality)
    skin_quality = quality["skin"]
    assert skin_quality is not None
    assert "buffered_f1" in skin_quality["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in skin_quality["surface_distance"]
    assert "strike_median" in skin_quality["orientation_error"]
    skins = case["pyosv"]["skins"]
    assert isinstance(skins["skin_count"], int)
    assert isinstance(skins["cell_count"], int)
    assert skins["skin_count"] >= 0
    assert skins["cell_count"] >= 0
    assert skins == skin_quality["topology"]
    for name in ("fv", "fvt"):
        summary = case["pyosv"][name]
        assert summary["shape"] == [17, 17, 17]
        assert summary["finite_count"] == 17 * 17 * 17
        assert summary["finite_fraction"] == 1.0
        assert summary["max"] > 0.0
        assert 0.0 < summary["nonzero_fraction"] <= 1.0

    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["case_id"] == "single_vertical_plane"
    assert rows[0]["variant"] == "current_default"
    assert rows[0]["shape_n3"] == "17"
    assert rows[0]["shape_n2"] == "17"
    assert rows[0]["shape_n1"] == "17"
    assert float(rows[0]["fv_max"]) > 0.0
    assert float(rows[0]["fv_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fv_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fv_distance_p95"]) >= 0.0
    assert math.isfinite(float(rows[0]["fv_edge_false_positive_fraction"]))
    assert math.isfinite(float(rows[0]["fv_strike_median_error"]))
    assert math.isfinite(float(rows[0]["fv_dip_median_error"]))
    assert float(rows[0]["fvt_max"]) > 0.0
    assert float(rows[0]["fvt_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fvt_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fvt_distance_p95"]) >= 0.0
    assert math.isfinite(float(rows[0]["fvt_edge_false_positive_fraction"]))
    assert float(rows[0]["fvt_strike_median_error"]) >= 0.0
    assert float(rows[0]["fvt_dip_median_error"]) >= 0.0
    assert rows[0]["skinning_enabled"] == "True"
    _assert_enabled_skin_summary_row(rows[0])
    assert int(rows[0]["skin_count"]) >= 0
    assert int(rows[0]["skin_cell_count"]) >= 0
    assert math.isfinite(float(rows[0]["skin_buffered_f1_r2"]))
    assert math.isfinite(float(rows[0]["skin_distance_p95"]))
    assert math.isfinite(float(rows[0]["skin_strike_median_error"]))
    assert math.isfinite(float(rows[0]["skin_dip_median_error"]))


def test_report_3d_synthetic_quality_geometry_case_set_writes_three_case_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["format_version"] == 1
    assert metrics["config"]["case_set"] == "geometry"
    assert metrics["config"]["shape"] == [17, 17, 17]
    assert [case["case_id"] for case in metrics["cases"]] == list(GEOMETRY_CASE_IDS)
    for case in metrics["cases"]:
        assert case["shape"] == [17, 17, 17]
        assert set(case["variants"]) == {"current_default"}
        assert case["truth"]["surface_voxel_count"] > 0
        variant = case["variants"]["current_default"]
        fvt = variant["pyosv"]["fvt"]
        assert fvt["shape"] == [17, 17, 17]
        assert fvt["finite_count"] == 17 * 17 * 17
        assert fvt["finite_fraction"] == 1.0
        assert math.isfinite(fvt["max"])
        assert fvt["max"] > 0.0

        fv_quality = variant["quality"]["fv_top_truth_count"]
        _assert_top_truth_quality_has_orientation(fv_quality)
        fvt_quality = variant["quality"]["fvt_top_truth_count"]
        _assert_top_truth_quality_has_orientation(fvt_quality)
        skin_quality = variant["quality"]["skin"]
        assert skin_quality is not None
        assert "topology" in skin_quality
        assert variant["pyosv"]["skins"]["skin_count"] >= 0

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        (case_id, "current_default") for case_id in GEOMETRY_CASE_IDS
    ]
    for row in rows:
        assert math.isfinite(float(row["fvt_max"]))
        assert float(row["fvt_max"]) > 0.0
        assert math.isfinite(float(row["fvt_buffered_f1_r2"]))
        assert float(row["fvt_buffered_f1_r2"]) >= 0.0
        assert math.isfinite(float(row["fvt_distance_p95"]))
        assert math.isfinite(float(row["fv_strike_median_error"]))
        assert math.isfinite(float(row["fv_dip_median_error"]))
        assert math.isfinite(float(row["fvt_strike_median_error"]))
        assert math.isfinite(float(row["fvt_dip_median_error"]))
        _assert_enabled_skin_summary_row(row)


def test_report_3d_synthetic_quality_extended_case_set_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "extended",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--save-volumes",
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["case_set"] == "extended"
    assert [case["case_id"] for case in metrics["cases"]] == list(EXTENDED_CASE_IDS)
    for case in metrics["cases"]:
        quality = case["quality"]
        assert "buffered_overlap_radius2" in quality["fvt_top_truth_count"]
        assert "fvt_top_truth_count" in quality["edge_false_positive"]
        assert math.isfinite(
            float(
                quality["edge_false_positive"]["fvt_top_truth_count"][
                    "edge_false_positive_fraction_of_candidates"
                ]
            )
        )
        assert case["variants"]["current_default"]["quality"] == quality

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7
    assert [row["case_id"] for row in rows] == list(EXTENDED_CASE_IDS)
    assert all(row["variant"] == "current_default" for row in rows)
    finite_summary_fields = (
        "fvt_buffered_f1_r2",
        "fvt_distance_p95",
        "fvt_strike_median_error",
        "fvt_dip_median_error",
    )
    for row in rows:
        for field in finite_summary_fields:
            assert math.isfinite(float(row[field]))
        assert "fvt_edge_false_positive_fraction" in row
        assert "fv_edge_false_positive_fraction" in row
        assert math.isfinite(float(row["fvt_edge_false_positive_fraction"]))
        assert math.isfinite(float(row["fv_edge_false_positive_fraction"]))

    for case_id in EXTENDED_CASE_IDS:
        assert (output_dir / case_id).is_dir()
        assert (output_dir / case_id / "fvt_py.dat").is_file()
        assert (output_dir / case_id / "skins.json").is_file()

    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    for case_id in EXTENDED_CASE_IDS:
        assert f"## {case_id}" in markdown


def test_report_3d_synthetic_quality_variants_write_metrics_and_summary_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--variants",
        (
            "current_default,no_surface_orientation_smoothing,"
            "final_norm_smoothing_1,voter_thin_normal"
        ),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    case = metrics["cases"][0]
    assert set(case["variants"]) == set(DIAGNOSTIC_VARIANTS)
    for variant in DIAGNOSTIC_VARIANTS:
        variant_report = case["variants"][variant]
        assert variant_report["pyosv"]["fvt"]["max"] > 0.0
        assert variant_report["skinning"] == {"enabled": True}
        assert variant_report["quality"]["skin"] is not None
        assert variant_report["pyosv"]["skins"] == variant_report["quality"]["skin"]["topology"]

    comparison = case["variant_comparison"]
    assert comparison["baseline_variant"] == "current_default"
    assert set(comparison["variants"]) == set(DIAGNOSTIC_VARIANTS)
    comparison_fields = (
        "fvt_buffered_f1_r2_delta_vs_current",
        "fvt_candidate_to_truth_p95_delta_vs_current",
        "fvt_strike_median_error_delta_vs_current",
        "fvt_dip_median_error_delta_vs_current",
        "fv_buffered_f1_r2_delta_vs_current",
        "skin_buffered_f1_r2_delta_vs_current",
        "skin_candidate_to_truth_p95_delta_vs_current",
        "skin_strike_median_error_delta_vs_current",
        "skin_dip_median_error_delta_vs_current",
        "skin_count_delta_vs_current",
    )
    assert all(
        comparison["variants"]["current_default"][field] == 0.0 for field in comparison_fields
    )
    for variant in DIAGNOSTIC_VARIANTS[1:]:
        for field in comparison_fields:
            assert math.isfinite(comparison["variants"][variant][field])

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        ("single_vertical_plane", variant) for variant in DIAGNOSTIC_VARIANTS
    ]
    csv_delta_fields = (
        "fvt_buffered_f1_delta_vs_baseline",
        "fvt_distance_p95_delta_vs_baseline",
        "fvt_strike_median_error_delta_vs_baseline",
        "fvt_dip_median_error_delta_vs_baseline",
        "skin_buffered_f1_delta_vs_baseline",
        "skin_distance_p95_delta_vs_baseline",
        "skin_strike_median_error_delta_vs_baseline",
        "skin_dip_median_error_delta_vs_baseline",
        "skin_count_delta_vs_baseline",
    )
    assert all(row["baseline_variant"] == "current_default" for row in rows)
    assert all(float(rows[0][field]) == 0.0 for field in csv_delta_fields)
    for row in rows[1:]:
        _assert_enabled_skin_summary_row(row)
        for field in csv_delta_fields:
            assert math.isfinite(float(row[field]))


def test_report_3d_synthetic_quality_diagnostic_variants_pass(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--variants",
        "no_surface_orientation_smoothing,final_norm_smoothing_1",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == [
        "no_surface_orientation_smoothing",
        "final_norm_smoothing_1",
    ]
    assert set(metrics["cases"][0]["variants"]) == {
        "no_surface_orientation_smoothing",
        "final_norm_smoothing_1",
    }
    assert metrics["cases"][0]["variant_comparison"] == {
        "baseline_variant": None,
        "variants": {},
    }

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["baseline_variant"] for row in rows] == ["", ""]
    assert [row["fvt_buffered_f1_delta_vs_baseline"] for row in rows] == ["", ""]
    assert [row["skin_buffered_f1_delta_vs_baseline"] for row in rows] == ["", ""]
    assert [row["skin_count_delta_vs_baseline"] for row in rows] == ["", ""]


def test_report_3d_synthetic_quality_thinning_diagnostic_is_opt_in(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert "thinning_diagnostic" not in variant


def test_report_3d_synthetic_quality_thinning_diagnostic_curved_surface(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
        "--thinning-diagnostics",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["thinning_diagnostic"] == {"enabled": True}
    for case in metrics["cases"]:
        variant = case["variants"]["current_default"]
        if case["case_id"] == "curved_surface":
            assert "thinning_diagnostic" in variant
            assert "thinning_diagnostic" in case
        else:
            assert "thinning_diagnostic" not in variant
            assert "thinning_diagnostic" not in case
    curved = next(case for case in metrics["cases"] if case["case_id"] == "curved_surface")
    diagnostic = curved["variants"]["current_default"]["thinning_diagnostic"]

    assert "reference" in diagnostic
    assert "normal" in diagnostic
    for mode in ("reference", "normal"):
        mode_report = diagnostic[mode]
        assert "fvt" in mode_report["pyosv"]
        _assert_top_truth_quality_has_orientation(mode_report["quality"]["fvt_top_truth_count"])

    keep_mask = diagnostic["keep_mask"]
    for count_key in (
        "reference_count",
        "normal_count",
        "intersection_count",
        "union_count",
        "reference_only_count",
        "normal_only_count",
    ):
        assert count_key in keep_mask
        assert keep_mask[count_key] >= 0
    assert math.isfinite(float(keep_mask["jaccard"]))
    assert "reference_only_buffered_overlap_radius2" in keep_mask
    assert "normal_only_buffered_overlap_radius2" in keep_mask

    delta = diagnostic["delta"]["normal_minus_reference"]
    for key in (
        "fvt_buffered_f1_r2",
        "fvt_candidate_to_truth_p95",
        "fvt_strike_median_error",
        "fvt_dip_median_error",
    ):
        assert math.isfinite(float(delta[key]))


def test_report_3d_synthetic_quality_thinning_diagnostic_summary_columns(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
        "--thinning-diagnostics",
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    for field in EXPECTED_THINNING_DIAGNOSTIC_SUMMARY_FIELDS:
        assert field in rows[0]

    curved_row = next(row for row in rows if row["case_id"] == "curved_surface")
    for field in EXPECTED_THINNING_DIAGNOSTIC_SUMMARY_FIELDS:
        assert curved_row[field] != ""
    assert math.isfinite(float(curved_row["thinning_diag_jaccard"]))

    non_diagnostic_rows = [row for row in rows if row["case_id"] != "curved_surface"]
    assert non_diagnostic_rows
    for row in non_diagnostic_rows:
        for field in EXPECTED_THINNING_DIAGNOSTIC_SUMMARY_FIELDS:
            assert row[field] == ""


def test_report_3d_synthetic_quality_thinning_diagnostic_explicit_cases(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
        "--thinning-diagnostics",
        "--thinning-diagnostic-cases",
        "single_dipping_plane,curved_surface",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    diagnostic_cases = {
        case["case_id"]
        for case in metrics["cases"]
        if "thinning_diagnostic" in case["variants"]["current_default"]
    }
    assert diagnostic_cases == {"single_dipping_plane", "curved_surface"}


def test_report_3d_synthetic_quality_unknown_thinning_diagnostic_case_fails(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(tmp_path / "synthetic_quality"),
        "--thinning-diagnostics",
        "--thinning-diagnostic-cases",
        "unknown_case",
    )

    assert result.returncode != 0
    assert "unknown thinning diagnostic case ID" in result.stderr
    assert "unknown_case" in result.stderr


def test_report_3d_synthetic_quality_thinning_diagnostic_input_mode_both(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
        "--input-mode",
        "both",
        "--scanner-backend",
        "reference-like",
        "--scanner-thin-mode",
        "reference",
        "--variants",
        "current_default",
        "--thinning-diagnostics",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    curved = next(case for case in metrics["cases"] if case["case_id"] == "curved_surface")
    for pipeline in ("oracle", "scanner"):
        variant = curved["pipelines"][pipeline]["variants"]["current_default"]
        assert "thinning_diagnostic" in variant

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    curved_rows = [row for row in rows if row["case_id"] == "curved_surface"]
    assert {(row["pipeline"], row["variant"]) for row in curved_rows} == {
        ("oracle", "current_default"),
        ("scanner", "current_default"),
    }
    for row in curved_rows:
        assert row["thinning_diag_reference_count"] != ""


def test_report_3d_synthetic_quality_thinning_diagnostic_writes_visual_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        ",".join(str(size) for size in shape),
        "--input-mode",
        "oracle",
        "--variants",
        "current_default",
        "--thinning-diagnostics",
        "--save-volumes",
        "--save-figures",
        "--write-markdown-index",
        "--output-dir",
        str(output_dir),
        "--pretty",
    )

    assert result.returncode == 0, result.stderr
    expected_size = shape[0] * shape[1] * shape[2] * 4
    diagnostic_dir = output_dir / "curved_surface" / "thinning_diagnostic"
    assert diagnostic_dir.is_dir()
    for name in EXPECTED_THINNING_DIAGNOSTIC_VOLUME_FILES:
        path = diagnostic_dir / name
        assert path.is_file()
        assert path.stat().st_size == expected_size
    for name in EXPECTED_THINNING_DIAGNOSTIC_I3_FIGURES:
        path = diagnostic_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0
    for axis in ("i1", "i2", "i3"):
        path = diagnostic_dir / f"fvt_reference_vs_normal_{axis}_center.png"
        assert path.is_file()
        assert path.stat().st_size > 0

    keep_reference = np.fromfile(diagnostic_dir / "keep_reference.dat", dtype=">f4")
    keep_normal_only = np.fromfile(diagnostic_dir / "keep_normal_only.dat", dtype=">f4")
    assert set(np.unique(keep_reference)).issubset({0.0, 1.0})
    assert set(np.unique(keep_normal_only)).issubset({0.0, 1.0})

    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "##### thinning diagnostic" in markdown
    assert "reference buffered F1" in markdown
    assert "normal buffered F1" in markdown
    assert "normal-minus-reference delta" in markdown
    assert "keep-mask Jaccard" in markdown
    for name in EXPECTED_THINNING_DIAGNOSTIC_I3_FIGURES:
        assert f"curved_surface/thinning_diagnostic/{name}" in markdown

    assert not (output_dir / "single_vertical_plane" / "thinning_diagnostic").exists()
    assert not (output_dir / "single_dipping_plane" / "thinning_diagnostic").exists()


def test_report_3d_synthetic_quality_unknown_variant_fails(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--variants",
        "current_default,unknown_variant",
    )

    assert result.returncode != 0
    assert "unknown variant" in result.stderr


def test_report_3d_synthetic_quality_normal_thin_mode_passes(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--voter-thin-mode",
        "normal",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["voting"]["voter_thin_mode"] == "normal"
    assert metrics["cases"][0]["pyosv"]["fvt"]["max"] > 0.0


def test_report_3d_synthetic_quality_records_voting_options(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--ru",
        "2",
        "--rv",
        "3",
        "--rw",
        "4",
        "--seed-distance",
        "2",
        "--seed-threshold",
        "0.6",
        "--attribute-smoothing",
        "1",
        "--reference-thin-sigma",
        "1.5",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["voting"] == {
        "ru": 2,
        "rv": 3,
        "rw": 4,
        "seed_distance": 2,
        "seed_threshold": 0.6,
        "attribute_smoothing": 1,
        "voter_thin_mode": "reference",
        "reference_thin_sigma": 1.5,
    }


def test_report_3d_synthetic_quality_records_skinner_options(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skinner-min-likelihood",
        "0.4",
        "--skinner-min-skin-size",
        "2",
        "--skinner-d",
        "2",
        "--skinner-ru",
        "6",
        "--skinner-rv",
        "7",
        "--skinner-rw",
        "8",
        "--skinner-max-steps",
        "3",
        "--skinner-du",
        "4.5",
        "--skinner-max-delta-strike",
        "20",
        "--no-skinner-reskin",
        "--small-skin-size",
        "5",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["skinning"] == {
        "enabled": True,
        "min_likelihood": 0.4,
        "min_skin_size": 2,
        "d": 2,
        "ru": 6,
        "rv": 7,
        "rw": 8,
        "max_steps": 3,
        "du": 4.5,
        "max_delta_strike": 20.0,
        "reskin": False,
        "small_skin_size": 5,
    }


def test_report_synthetic_quality_accepts_input_mode_scanner(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["input_mode"] == "scanner"
    assert variant["active_pipeline"] == "scanner"
    assert set(variant["pipelines"]) == {"scanner"}
    assert variant["pyosv"] == variant["pipelines"]["scanner"]["pyosv"]
    assert variant["quality"] == variant["pipelines"]["scanner"]["quality"]


def test_report_synthetic_quality_accepts_input_mode_both(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["input_mode"] == "both"
    assert variant["active_pipeline"] == "oracle"
    assert set(variant["pipelines"]) == {"oracle", "scanner"}
    assert variant["pyosv"] == variant["pipelines"]["oracle"]["pyosv"]
    assert variant["pipelines"]["scanner"]["pyosv"]["fvt"]["finite_fraction"] == 1.0


def test_input_mode_both_metrics_json_uses_canonical_pipeline_schema(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--variants",
        "current_default",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    case = metrics["cases"][0]
    assert set(case["pipelines"]) == {"oracle", "scanner"}
    for pipeline_name in ("oracle", "scanner"):
        pipeline = case["pipelines"][pipeline_name]
        assert set(pipeline) == {"variants", "variant_comparison"}
        assert "current_default" in pipeline["variants"]
        variant = pipeline["variants"]["current_default"]
        assert "pyosv" in variant
        assert "quality" in variant
        if pipeline_name == "scanner":
            assert "scanner_quality" in variant


def test_input_mode_oracle_summary_csv_uses_pipeline_oracle(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "oracle",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["pipeline"] == "oracle"
    assert {row["pipeline"] for row in rows} == {"oracle"}


def test_input_mode_scanner_summary_csv_uses_pipeline_scanner(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["pipeline"] == "scanner"
    assert {row["pipeline"] for row in rows} == {"scanner"}


def test_input_mode_both_summary_csv_has_oracle_and_scanner_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--variants",
        "current_default",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert "pipeline" in rows[0]
    assert len(rows) == 2
    assert {(row["case_id"], row["pipeline"], row["variant"]) for row in rows} == {
        ("single_vertical_plane", "oracle", "current_default"),
        ("single_vertical_plane", "scanner", "current_default"),
    }
    for row in rows:
        assert math.isfinite(float(row["fv_strike_median_error"]))
        assert math.isfinite(float(row["fv_dip_median_error"]))


def test_report_synthetic_quality_scanner_mode_records_scanner_config(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--scanner-backend",
        "fast",
        "--scanner-phi-min",
        "10",
        "--scanner-phi-max",
        "80",
        "--scanner-theta-min",
        "30",
        "--scanner-theta-max",
        "70",
        "--scanner-sigma1",
        "1.5",
        "--scanner-sigma2",
        "2.5",
        "--scanner-thin-mode",
        "normal",
        "--keep-scanner-edge-effects",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    expected_config = {
        "backend": "fast",
        "phi_min": 10.0,
        "phi_max": 80.0,
        "theta_min": 30.0,
        "theta_max": 70.0,
        "sigma1": 1.5,
        "sigma2": 2.5,
        "scanner_thin_mode": "normal",
        "remove_edge_effects": False,
        "input": {
            "background": 1.0,
            "fault_contrast": 0.85,
            "noise_sigma": 0.0,
            "seed": 20260706,
            "clip_min": 0.0,
            "clip_max": 1.0,
        },
    }
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["scanner"] == expected_config
    assert variant["scanner"]["config"] == expected_config


def test_report_synthetic_quality_scanner_mode_writes_finite_scanner_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    scanner = metrics["cases"][0]["variants"]["current_default"]["scanner"]
    for name in ("input", "ft", "fet", "pt", "fpt", "tt", "ftt"):
        summary = scanner[name]
        assert summary["shape"] == [17, 17, 17]
        assert summary["finite_count"] == 17 * 17 * 17
        assert summary["finite_fraction"] == 1.0
        assert math.isfinite(float(summary["min"]))
        assert math.isfinite(float(summary["max"]))
        assert math.isfinite(float(summary["mean"]))


def test_report_synthetic_quality_scanner_backend_fast_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--scanner-backend",
        "fast",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert variant["scanner"]["config"]["backend"] == "fast"
    assert variant["pyosv"]["fvt"]["finite_fraction"] == 1.0


def test_report_synthetic_quality_scanner_thin_mode_none_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--scanner-thin-mode",
        "none",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    scanner = metrics["cases"][0]["variants"]["current_default"]["scanner"]
    assert scanner["config"]["scanner_thin_mode"] == "none"
    assert scanner["fet"]["finite_fraction"] == 1.0
    assert scanner["fet"]["max"] == scanner["ft"]["max"]


def test_scanner_mode_reports_scanner_quality_metrics(tmp_path: Path) -> None:
    scanner_output_dir = tmp_path / "scanner_quality"
    scanner_result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(scanner_output_dir),
    )

    assert scanner_result.returncode == 0, scanner_result.stderr
    scanner_metrics = json.loads((scanner_output_dir / "metrics.json").read_text(encoding="utf-8"))
    scanner_variant = scanner_metrics["cases"][0]["variants"]["current_default"]
    assert "scanner_quality" in scanner_variant
    _assert_scanner_quality_contract(scanner_variant["scanner_quality"])

    both_output_dir = tmp_path / "both_quality"
    both_result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--output-dir",
        str(both_output_dir),
    )

    assert both_result.returncode == 0, both_result.stderr
    both_metrics = json.loads((both_output_dir / "metrics.json").read_text(encoding="utf-8"))
    both_variant = both_metrics["cases"][0]["variants"]["current_default"]
    assert "scanner_quality" not in both_variant
    _assert_scanner_quality_contract(both_variant["pipelines"]["scanner"]["scanner_quality"])


def test_scanner_input_association_has_positive_contrast_on_minimal_case(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    input_association = metrics["cases"][0]["variants"]["current_default"]["scanner_quality"][
        "input_association"
    ]
    assert input_association["truth_surface_mean"] < input_association["far_from_truth_mean"]
    assert input_association["contrast"] > 0.0


def test_report_scanner_mode_minimal_case_meets_loose_smoke_thresholds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    scanner_quality = variant["scanner_quality"]
    scanner_ft_overlap = scanner_quality["ft_top_truth_count"]["buffered_overlap_radius2"]
    fv_quality = variant["quality"]["fv_top_truth_count"]
    fvt_overlap = variant["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"]
    input_association = scanner_quality["input_association"]

    _assert_top_truth_quality_has_orientation(fv_quality)
    assert scanner_ft_overlap["buffered_f1"] >= 0.20
    assert fvt_overlap["buffered_f1"] >= 0.20
    assert input_association["contrast"] > 0.0


def test_report_input_mode_both_keeps_oracle_and_scanner_metrics_separate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    pipelines = variant["pipelines"]
    oracle_pipeline = pipelines["oracle"]
    scanner_pipeline = pipelines["scanner"]

    assert variant["active_pipeline"] == "oracle"
    assert variant["quality"] == oracle_pipeline["quality"]
    assert "scanner_quality" not in variant
    assert "scanner_quality" in scanner_pipeline

    oracle_fv = oracle_pipeline["quality"]["fv_top_truth_count"]
    scanner_fv = scanner_pipeline["quality"]["fv_top_truth_count"]
    oracle_fvt = oracle_pipeline["quality"]["fvt_top_truth_count"]
    scanner_fvt = scanner_pipeline["quality"]["fvt_top_truth_count"]
    scanner_ft = scanner_pipeline["scanner_quality"]["ft_top_truth_count"]
    for quality in (oracle_fvt, scanner_fvt, scanner_ft):
        assert math.isfinite(float(quality["buffered_overlap_radius2"]["buffered_f1"]))
        assert math.isfinite(float(quality["surface_distance"]["candidate_to_truth_p95"]))
    for quality in (oracle_fv, scanner_fv):
        _assert_top_truth_quality_has_orientation(quality)

    scanner_input = scanner_pipeline["scanner_quality"]["input_association"]
    assert math.isfinite(float(scanner_input["contrast"]))


def test_scanner_mode_summary_csv_contains_scanner_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    row = rows[0]
    for field in EXPECTED_SCANNER_SUMMARY_FIELDS:
        assert field in row
    assert row["input_mode"] == "scanner"
    assert row["scanner_backend"] == "reference-like"
    assert row["scanner_thin_mode"] == "reference"
    for field in (
        "scanner_ft_buffered_f1_r2",
        "scanner_ft_distance_p95",
        "scanner_strike_median_error",
        "scanner_dip_median_error",
        "scanner_input_contrast",
    ):
        assert math.isfinite(float(row[field]))


def test_input_mode_oracle_leaves_scanner_columns_empty_or_absent_consistently(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "oracle",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert "scanner_quality" not in variant

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    row = rows[0]
    assert row["input_mode"] == "oracle"
    for field in EXPECTED_SCANNER_SUMMARY_FIELDS:
        assert field in row
    for field in EXPECTED_SCANNER_SUMMARY_FIELDS[1:]:
        assert row[field] == ""


def test_report_synthetic_quality_rejects_invalid_input_mode_or_scanner_backend(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    bad_input_mode = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "invalid",
        "--output-dir",
        str(output_dir / "bad_input_mode"),
    )
    bad_backend = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--scanner-backend",
        "invalid",
        "--output-dir",
        str(output_dir / "bad_backend"),
    )

    assert bad_input_mode.returncode != 0
    assert "invalid choice" in bad_input_mode.stderr
    assert bad_backend.returncode != 0
    assert "invalid choice" in bad_backend.stderr


def test_report_3d_synthetic_quality_skinning_uses_stable_buffer_key(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--buffer-radius",
        "1",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    skin_quality = metrics["cases"][0]["quality"]["skin"]
    assert "buffered_overlap_radius2" in skin_quality

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert math.isfinite(float(rows[0]["skin_buffered_f1_r2"]))


def test_report_3d_synthetic_quality_skip_skinning_writes_disabled_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["skinning"]["enabled"] is False
    assert variant["skinning"] == {"enabled": False}
    assert variant["quality"]["skin"] is None
    assert variant["pyosv"]["skins"]["skin_count"] == 0
    assert variant["pyosv"]["skins"]["cell_count"] == 0

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["skinning_enabled"] == "False"
    assert rows[0]["skin_enabled"] == "False"
    assert rows[0]["skin_count"] == "0"
    assert rows[0]["skin_cell_count"] == "0"
    assert rows[0]["skin_unique_cell_count"] == "0"
    assert rows[0]["skin_duplicate_cell_count"] == "0"
    assert rows[0]["skin_largest_size"] == "0"
    assert rows[0]["skin_largest_fraction"] == "0.0"
    assert rows[0]["skin_small_count"] == "0"
    assert rows[0]["skin_small_cell_fraction"] == "0.0"
    for field in SKIN_EMPTY_WHEN_DISABLED_FIELDS:
        assert rows[0][field] == ""


def test_report_3d_synthetic_quality_skip_skinning_writes_disabled_skins_json(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
        "--save-volumes",
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    skins = json.loads(
        (output_dir / "single_vertical_plane" / "skins.json").read_text(encoding="utf-8")
    )
    assert skins == {
        "format_version": 1,
        "skinning_enabled": False,
        "skin_count": 0,
        "skins": [],
    }
    mask = np.fromfile(
        output_dir / "single_vertical_plane" / "skin_mask_py.dat",
        dtype=">f4",
    )
    assert np.count_nonzero(mask) == 0
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "skinning disabled" in markdown


def test_report_3d_synthetic_quality_invalid_skinner_options_fail(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--skinner-ru",
        "1",
    )

    assert result.returncode != 0
    assert "skinner_ru must be at least 2" in result.stderr


def test_report_3d_synthetic_quality_save_volumes_writes_expected_dat_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "single_vertical_plane"
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for name in EXPECTED_VOLUME_FILES:
        path = case_dir / name
        assert path.is_file()
        assert path.stat().st_size == expected_size
    skin_mask = np.fromfile(case_dir / "skin_mask_py.dat", dtype=">f4").astype(np.float32)
    assert set(np.unique(skin_mask)).issubset({0.0, 1.0})
    skins = json.loads((case_dir / "skins.json").read_text(encoding="utf-8"))
    assert skins["format_version"] == 1
    assert skins["skinning_enabled"] is True
    assert isinstance(skins["skin_count"], int)
    assert isinstance(skins["skins"], list)
    if skins["skins"]:
        first_skin = skins["skins"][0]
        assert first_skin["skin_index"] == 0
        assert first_skin["cell_count"] == len(first_skin["cells"])
        if first_skin["cells"]:
            assert set(first_skin["cells"][0]) == {
                "x1",
                "x2",
                "x3",
                "i1",
                "i2",
                "i3",
                "fl",
                "fp",
                "ft",
            }


def test_scanner_mode_save_volumes_writes_scanner_dat_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "single_vertical_plane"
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for name in EXPECTED_SCANNER_VOLUME_FILES:
        path = case_dir / name
        assert path.is_file()
        assert path.stat().st_size == expected_size

    ft_scan = np.fromfile(case_dir / "ft_scan.dat", dtype=">f4").astype(np.float32)
    ft_used = np.fromfile(case_dir / "ft_used.dat", dtype=">f4").astype(np.float32)
    assert np.count_nonzero(ft_scan) >= np.count_nonzero(ft_used)


def test_report_3d_synthetic_quality_geometry_save_volumes_splits_case_directories(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        ",".join(str(size) for size in shape),
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for case_id in GEOMETRY_CASE_IDS:
        case_dir = output_dir / case_id
        for name in EXPECTED_VOLUME_FILES:
            path = case_dir / name
            assert path.is_file()
            assert path.stat().st_size == expected_size
        assert (case_dir / "skins.json").is_file()


def test_report_3d_synthetic_quality_variants_save_volumes_split_skin_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--output-dir",
        str(output_dir),
        "--variants",
        "current_default,no_surface_orientation_smoothing",
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for variant in ("current_default", "no_surface_orientation_smoothing"):
        variant_dir = output_dir / "single_vertical_plane" / variant
        assert (variant_dir / "skins.json").is_file()
        mask_path = variant_dir / "skin_mask_py.dat"
        assert mask_path.is_file()
        assert mask_path.stat().st_size == expected_size


def test_input_mode_both_splits_oracle_and_scanner_outputs_without_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--input-mode",
        "both",
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "single_vertical_plane"
    oracle_dir = case_dir / "oracle"
    scanner_dir = case_dir / "scanner"
    expected_size = shape[0] * shape[1] * shape[2] * 4
    assert oracle_dir.is_dir()
    assert scanner_dir.is_dir()
    for name in EXPECTED_VOLUME_FILES:
        oracle_path = oracle_dir / name
        scanner_path = scanner_dir / name
        assert oracle_path.is_file()
        assert scanner_path.is_file()
        assert oracle_path.stat().st_size == expected_size
        assert scanner_path.stat().st_size == expected_size
    for name in EXPECTED_SCANNER_VOLUME_FILES:
        path = scanner_dir / name
        assert path.is_file()
        assert path.stat().st_size == expected_size
        assert not (oracle_dir / name).exists()
    assert (oracle_dir / "skins.json").is_file()
    assert (scanner_dir / "skins.json").is_file()


def test_report_3d_synthetic_quality_write_markdown_index_includes_case_and_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "# Controlled Synthetic Quality Report" in markdown
    assert "## single_vertical_plane" in markdown
    assert "buffered_f1_r2" in markdown
    assert "distance_p95" in markdown
    assert "strike_median_error" in markdown
    assert "dip_median_error" in markdown
    assert "skin_count" in markdown
    assert "skin_cell_count" in markdown
    assert "skin_buffered_f1_r2" in markdown
    assert "skin_distance_p95" in markdown
    assert "skin_strike_median_error" in markdown
    assert "skin_dip_median_error" in markdown
    assert "single_vertical_plane/figures/truth_vs_fvt_overlay_i3_center.png" in markdown
    assert "single_vertical_plane/figures/truth_vs_skin_overlay_i3_center.png" in markdown


def test_scanner_mode_markdown_links_scanner_figures(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "scanner input contrast" in markdown
    assert "scanner ft buffered_f1" in markdown
    assert "scanner ft distance_p95" in markdown
    assert "scanner strike median error" in markdown
    assert "scanner dip median error" in markdown
    assert "single_vertical_plane/figures/truth_vs_ft_scan_overlay_i3_center.png" in markdown
    assert "single_vertical_plane/figures/truth_vs_ft_used_overlay_i3_center.png" in markdown
    assert "single_vertical_plane/figures/truth_vs_fvt_overlay_i3_center.png" in markdown


def test_input_mode_both_markdown_links_pipeline_figures(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--output-dir",
        str(output_dir),
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "#### oracle pipeline" in markdown
    assert "#### scanner pipeline" in markdown
    assert "single_vertical_plane/oracle/figures/truth_vs_fvt_overlay_i3_center.png" in markdown
    assert "single_vertical_plane/oracle/figures/truth_vs_skin_overlay_i3_center.png" in markdown
    assert (
        "single_vertical_plane/scanner/figures/truth_vs_ft_scan_overlay_i3_center.png" in markdown
    )
    assert (
        "single_vertical_plane/scanner/figures/truth_vs_ft_used_overlay_i3_center.png" in markdown
    )
    assert "single_vertical_plane/scanner/figures/truth_vs_fvt_overlay_i3_center.png" in markdown


def test_report_3d_synthetic_quality_geometry_markdown_index_includes_each_case(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    for case_id in GEOMETRY_CASE_IDS:
        assert f"## {case_id}" in markdown
        assert f"{case_id}/figures/truth_vs_fvt_overlay_i3_center.png" in markdown
        assert f"{case_id}/figures/truth_vs_skin_overlay_i3_center.png" in markdown


def test_report_3d_synthetic_quality_save_figures_writes_expected_pngs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--save-figures",
    )

    assert result.returncode == 0, result.stderr
    figures_dir = output_dir / "single_vertical_plane" / "figures"
    for name in EXPECTED_I3_FIGURES:
        path = figures_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_scanner_mode_save_figures_writes_scanner_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--output-dir",
        str(output_dir),
        "--save-figures",
    )

    assert result.returncode == 0, result.stderr
    figures_dir = output_dir / "single_vertical_plane" / "figures"
    for name in EXPECTED_SCANNER_I3_FIGURES:
        path = figures_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_report_3d_synthetic_quality_geometry_save_figures_splits_case_directories(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--save-figures",
    )

    assert result.returncode == 0, result.stderr
    for case_id in GEOMETRY_CASE_IDS:
        figures_dir = output_dir / case_id / "figures"
        for name in EXPECTED_I3_FIGURES:
            path = figures_dir / name
            assert path.is_file()
            assert path.stat().st_size > 0
        for axis in ("i2", "i1"):
            path = figures_dir / f"truth_vs_skin_overlay_{axis}_center.png"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_report_3d_synthetic_quality_default_case_meets_smoke_thresholds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    quality = metrics["cases"][0]["quality"]["fvt_top_truth_count"]
    variant_quality = metrics["cases"][0]["variants"]["current_default"]["quality"][
        "fvt_top_truth_count"
    ]

    assert quality["buffered_overlap_radius2"]["buffered_f1"] >= 0.80
    assert quality["surface_distance"]["candidate_to_truth_p95"] <= 3.0
    assert quality["orientation_error"]["strike_median"] <= 10.0
    assert quality["orientation_error"]["dip_median"] <= 10.0
    assert variant_quality["buffered_overlap_radius2"]["buffered_f1"] >= 0.80
    assert variant_quality["surface_distance"]["candidate_to_truth_p95"] <= 3.0
    assert variant_quality["orientation_error"]["strike_median"] <= 10.0
    assert variant_quality["orientation_error"]["dip_median"] <= 10.0
