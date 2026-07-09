from __future__ import annotations

import csv
import importlib.util
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
COMPARISON_SCRIPT = REPO_ROOT / "examples" / "print_synthetic_quality_comparison.py"
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
    "voter_thin_hybrid",
    "voter_thin_hybrid_v2",
    "voter_thin_normal_plateau",
    "surface_support_weighted",
    "quality_skinner_v2",
    "quality_boundary_skinner_fallback",
    "quality_boundary_skinner_fallback_v2",
    "quality_boundary_skinner_fallback_v3",
    "quality_boundary_skinner_fallback_v4",
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
    "skin_seed_candidate_count_before_spacing",
    "skin_seed_count_after_spacing",
    "skin_seed_rejected_by_occupied",
    "skin_grow_attempt_count",
    "skin_discarded_empty_count",
    "skin_discarded_small_count",
    "skin_accepted_count",
    "skin_fallback_enabled",
    "skin_fallback_policy",
    "skin_fallback_used",
    "skin_fallback_reason",
    "skin_fallback_method",
    "skin_fallback_input",
    "skin_fallback_skin_count",
    "skin_fallback_cell_count",
    "skin_fallback_triggered_by_degraded_primary",
    "skin_fallback_degraded_reasons",
    "skin_fallback_replaced_primary",
    "skin_fallback_primary_skin_count",
    "skin_fallback_primary_cell_count",
    "skin_fallback_candidate_count",
    "skin_fallback_component_count",
    "skin_fallback_candidate_cell_count",
    "skin_fallback_largest_component_size",
    "skin_fallback_largest_component_fraction",
    "skin_fallback_top3_component_cell_count",
    "skin_fallback_top3_component_fraction",
    "skin_fallback_small_component_count",
    "skin_fallback_component_policy",
    "skin_fallback_accepted_component_count",
    "skin_fallback_discarded_component_count",
    "skin_fallback_accepted_component_cell_count",
    "skin_fallback_filter_min_component_size",
    "skin_fallback_filter_min_component_fraction_of_largest",
    "skin_fallback_filter_max_components",
    "skin_fallback_pruning_method",
    "skin_fallback_raw_component_cell_count",
    "skin_fallback_pruned_component_cell_count",
    "skin_fallback_pruned_fraction",
    "skin_fallback_largest_component_size_before_pruning",
    "skin_fallback_largest_component_size_after_pruning",
    "skin_fallback_pruning_removed_cell_count",
    "skin_fallback_skeletonization_axis_mode",
    "skin_fallback_coverage_before",
    "skin_fallback_coverage_after",
    "skin_fallback_v5_guardrail_enabled",
    "skin_fallback_v5_guardrail_passed",
    "skin_fallback_v5_guardrail_reasons",
    "skin_fallback_v5_guardrail_fallback_skin_count",
    "skin_fallback_v5_guardrail_coverage_of_fvt_positive",
    "skin_fallback_v5_guardrail_largest_skin_fraction",
    "skin_fallback_v5_guardrail_small_skin_cell_fraction",
    "skin_fallback_v5_guardrail_pruned_fraction",
    "skin_primary_count",
    "skin_primary_cell_count",
    "skin_primary_unique_cell_count",
    "skin_primary_largest_size",
    "skin_primary_largest_fraction",
    "skin_primary_small_count",
    "skin_primary_small_cell_fraction",
    "skin_primary_cell_coverage_of_fvt_positive",
    "skin_primary_largest_coverage_of_fvt_positive",
    "skin_primary_edge_shell_fraction",
    "skin_fvt_positive_edge_shell_fraction",
    "skin_scanner_target_positive_edge_shell_fraction",
    "skin_fvt_to_scanner_target_distance_p95",
    "skin_primary_degraded_candidate",
    "skin_primary_degraded_reasons",
    "skin_primary_boundary_degraded_candidate",
    "skin_primary_boundary_degraded_reasons",
    "skin_buffered_f1_r2",
    "skin_buffered_precision_r2",
    "skin_buffered_recall_r2",
    "skin_distance_candidate_to_truth_p95",
    "skin_distance_truth_to_candidate_p95",
    "skin_distance_hausdorff_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "skin_truth_component_count",
    "skin_covered_truth_component_count",
    "skin_uncovered_truth_component_count",
    "skin_over_merge_count",
    "skin_over_split_count",
    "skin_max_truth_components_per_skin",
    "skin_max_skins_per_truth_component",
    "skin_mean_purity",
    "skin_min_purity",
    "skin_mean_truth_component_recall",
    "skin_min_truth_component_recall",
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
    "skin_seed_candidate_count_before_spacing",
    "skin_seed_count_after_spacing",
    "skin_seed_rejected_by_occupied",
    "skin_grow_attempt_count",
    "skin_discarded_empty_count",
    "skin_discarded_small_count",
    "skin_accepted_count",
    "skin_fallback_skin_count",
    "skin_fallback_cell_count",
    "skin_fallback_primary_skin_count",
    "skin_fallback_primary_cell_count",
    "skin_fallback_candidate_count",
    "skin_fallback_component_count",
    "skin_fallback_candidate_cell_count",
    "skin_fallback_largest_component_size",
    "skin_fallback_largest_component_fraction",
    "skin_fallback_top3_component_cell_count",
    "skin_fallback_top3_component_fraction",
    "skin_fallback_small_component_count",
    "skin_fallback_accepted_component_count",
    "skin_fallback_discarded_component_count",
    "skin_fallback_accepted_component_cell_count",
    "skin_fallback_filter_min_component_size",
    "skin_fallback_filter_min_component_fraction_of_largest",
    "skin_fallback_filter_max_components",
    "skin_fallback_raw_component_cell_count",
    "skin_fallback_pruned_component_cell_count",
    "skin_fallback_pruned_fraction",
    "skin_fallback_largest_component_size_before_pruning",
    "skin_fallback_largest_component_size_after_pruning",
    "skin_fallback_pruning_removed_cell_count",
    "skin_fallback_coverage_before",
    "skin_fallback_coverage_after",
    "skin_fallback_v5_guardrail_fallback_skin_count",
    "skin_fallback_v5_guardrail_coverage_of_fvt_positive",
    "skin_fallback_v5_guardrail_largest_skin_fraction",
    "skin_fallback_v5_guardrail_small_skin_cell_fraction",
    "skin_fallback_v5_guardrail_pruned_fraction",
    "skin_primary_count",
    "skin_primary_cell_count",
    "skin_primary_unique_cell_count",
    "skin_primary_largest_size",
    "skin_primary_largest_fraction",
    "skin_primary_small_count",
    "skin_primary_small_cell_fraction",
    "skin_primary_cell_coverage_of_fvt_positive",
    "skin_primary_largest_coverage_of_fvt_positive",
    "skin_primary_edge_shell_fraction",
    "skin_fvt_positive_edge_shell_fraction",
    "skin_buffered_f1_r2",
    "skin_buffered_precision_r2",
    "skin_buffered_recall_r2",
    "skin_distance_candidate_to_truth_p95",
    "skin_distance_truth_to_candidate_p95",
    "skin_distance_hausdorff_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "skin_truth_component_count",
    "skin_covered_truth_component_count",
    "skin_uncovered_truth_component_count",
    "skin_over_merge_count",
    "skin_over_split_count",
    "skin_max_truth_components_per_skin",
    "skin_max_skins_per_truth_component",
    "skin_mean_purity",
    "skin_min_purity",
    "skin_mean_truth_component_recall",
    "skin_min_truth_component_recall",
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
    "skin_truth_component_count",
    "skin_covered_truth_component_count",
    "skin_uncovered_truth_component_count",
    "skin_over_merge_count",
    "skin_over_split_count",
    "skin_max_truth_components_per_skin",
    "skin_max_skins_per_truth_component",
    "skin_mean_purity",
    "skin_min_purity",
    "skin_mean_truth_component_recall",
    "skin_min_truth_component_recall",
    "skin_buffered_f1_delta_vs_baseline",
    "skin_distance_p95_delta_vs_baseline",
    "skin_strike_median_error_delta_vs_baseline",
    "skin_dip_median_error_delta_vs_baseline",
    "skin_count_delta_vs_baseline",
)
EXPECTED_SCANNER_SUMMARY_FIELDS = (
    "input_mode",
    "scanner_backend",
    "scanner_refinement_factor",
    "scanner_ensemble_reference_like_fraction",
    "scanner_ensemble_quality_fraction",
    "scanner_ensemble_fast_fraction",
    "scanner_thin_mode",
    "scanner_ft_buffered_f1_r2",
    "scanner_ft_distance_p95",
    "scanner_strike_median_error",
    "scanner_dip_median_error",
    "scanner_input_contrast",
    "scanner_matrix_best_fvt_positive_buffered_f1_backend",
    "scanner_matrix_best_skin_buffered_f1_backend",
    "scanner_matrix_best_boundary_edge_fp_backend",
)
EXPECTED_SCANNER_BACKEND_MATRIX_BACKENDS = ("reference-like", "quality", "fast")
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


def _load_report_module() -> object:
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _fault_skin(cell_indices: list[tuple[int, int, int]]) -> object:
    from pyosv.cells import FaultCell
    from pyosv.skin import FaultSkin

    return FaultSkin.from_cells(
        FaultCell(i1, i2, i3, 1.0, 0.0, 90.0) for i1, i2, i3 in cell_indices
    )


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


def _assert_scanner_backend_matrix_contract(
    matrix: dict[str, object],
    *,
    expected_selected_backend: str | None = None,
) -> None:
    backends = matrix["backends"]
    comparison = matrix["comparison"]
    selected_backend = comparison["selected_backend"]
    if expected_selected_backend is not None:
        assert selected_backend == expected_selected_backend
    expected_backends = set(EXPECTED_SCANNER_BACKEND_MATRIX_BACKENDS)
    if selected_backend not in expected_backends:
        expected_backends.add(selected_backend)
    assert set(backends) == expected_backends

    for backend in expected_backends:
        backend_report = backends[backend]
        assert backend_report["scanner"]["config"]["backend"] == backend
        assert backend_report["pyosv"]["fvt"]["finite_fraction"] == 1.0
        assert "quality" in backend_report
        _assert_scanner_quality_contract(backend_report["scanner_quality"])

    assert selected_backend in expected_backends
    assert comparison["best_fvt_positive_buffered_f1_backend"] in expected_backends
    assert comparison["best_skin_buffered_f1_backend"] in expected_backends
    assert comparison["best_boundary_edge_fp_backend"] in expected_backends
    deltas = comparison["deltas_vs_selected_backend"]
    for backend in expected_backends:
        values = comparison["metric_values"][backend]
        assert math.isfinite(float(values["fvt_positive_buffered_f1"]))
        assert math.isfinite(float(values["skin_buffered_f1"]))
        assert math.isfinite(float(values["fvt_positive_edge_false_positive_fraction"]))
        for metric, delta in deltas[backend].items():
            assert metric in values
            assert delta is not None
            assert math.isfinite(float(delta))
    for delta in deltas[selected_backend].values():
        assert delta == 0.0


def _assert_top_truth_quality_has_orientation(quality: dict[str, object]) -> None:
    assert "buffered_f1" in quality["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in quality["surface_distance"]
    orientation = quality["orientation_error"]
    assert orientation["count"] > 0
    assert math.isfinite(float(orientation["strike_median"]))
    assert math.isfinite(float(orientation["dip_median"]))


def _assert_positive_top_truth_quality_is_finite(quality: dict[str, object]) -> None:
    assert "buffered_f1" in quality["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in quality["surface_distance"]
    orientation = quality["orientation_error"]
    assert orientation["count"] >= 0
    assert math.isfinite(float(orientation["strike_median"]))
    assert math.isfinite(float(orientation["dip_median"]))


def _assert_finite_thinning_diagnostic_quality(diagnostic: dict[str, object]) -> None:
    for mode in ("reference", "normal"):
        mode_report = diagnostic[mode]
        quality = mode_report["quality"]["fvt_top_truth_count"]
        _assert_top_truth_quality_has_orientation(quality)
        assert math.isfinite(float(quality["buffered_overlap_radius2"]["buffered_f1"]))
        assert math.isfinite(float(quality["surface_distance"]["candidate_to_truth_p95"]))

    delta = diagnostic["delta"]["normal_minus_reference"]
    for key in (
        "fvt_buffered_f1_r2",
        "fvt_candidate_to_truth_p95",
        "fvt_strike_median_error",
        "fvt_dip_median_error",
    ):
        assert math.isfinite(float(delta[key]))


def _assert_scanner_downstream_contract(diagnostic: dict[str, object]) -> None:
    for key in (
        "scanner_ft_positive_candidate_count",
        "scanner_fet_positive_candidate_count",
        "fv_positive_candidate_count",
        "fvt_positive_candidate_count",
    ):
        assert int(diagnostic[key]) >= 0
    for key in (
        "scanner_ft_to_fet_retention_fraction",
        "scanner_ft_to_fv_positive_candidate_count_ratio",
        "scanner_ft_to_fvt_positive_candidate_count_ratio",
        "fv_to_fvt_positive_candidate_count_ratio",
        "fvt_candidate_to_scanner_ft_distance_p50",
        "fvt_candidate_to_scanner_ft_distance_p95",
        "fvt_candidate_to_fv_distance_p50",
        "fvt_candidate_to_fv_distance_p95",
        "scanner_ft_positive_edge_shell_fraction",
        "scanner_fet_positive_edge_shell_fraction",
        "fv_positive_edge_shell_fraction",
        "fvt_positive_edge_shell_fraction",
        "fvt_to_fv_positive_fraction",
        "fvt_positive_edge_candidate_fraction",
        "fvt_positive_edge_false_positive_fraction",
    ):
        assert math.isfinite(float(diagnostic[key]))
    for key, candidate_name, reference_name in (
        ("scanner_ft_vs_fv_positive_buffered_overlap_radius2", "scanner_ft", "fv"),
        ("scanner_ft_vs_fvt_positive_buffered_overlap_radius2", "scanner_ft", "fvt"),
        ("fv_vs_fvt_positive_buffered_overlap_radius2", "fv", "fvt"),
    ):
        overlap = diagnostic[key]
        assert isinstance(overlap, dict)
        assert overlap["candidate_mask"] == candidate_name
        assert overlap["reference_mask"] == reference_name
        assert math.isfinite(float(overlap["buffered_f1"]))
        assert math.isfinite(float(overlap["buffered_precision"]))
        assert math.isfinite(float(overlap["buffered_recall"]))
    assert diagnostic["voter_thin_mode"] in {
        "reference",
        "normal",
        "hybrid",
        "hybrid_v2",
        "normal_plateau",
    }
    assert diagnostic["scanner_thin_mode"] in {"none", "reference", "normal"}

    thinning_modes = diagnostic["thinning_modes"]
    assert set(thinning_modes) == {"reference", "hybrid", "hybrid_v2", "normal_plateau"}
    for mode in ("reference", "hybrid", "hybrid_v2", "normal_plateau"):
        mode_report = thinning_modes[mode]
        assert math.isfinite(float(mode_report["fvt_positive_buffered_f1_r2"]))
        assert math.isfinite(float(mode_report["fvt_positive_distance_p95"]))

    for key in (
        "hybrid_v2_tiebreaker_fet",
        "hybrid_v2_tiebreaker_fv",
        "hybrid_v2_tiebreaker_scanner_ft",
    ):
        assert math.isfinite(float(diagnostic[key]["fvt_positive_buffered_f1_r2"]))


def _assert_scanner_stage_loss_contract(diagnostic: dict[str, object]) -> None:
    expected_stages = (
        "scanner_ft_positive",
        "scanner_fet_positive",
        "seed_candidate",
        "seed_selected",
        "fv_positive",
        "fvt_positive",
        "skin",
    )
    expected_transitions = (
        "scanner_ft_positive_to_scanner_fet_positive",
        "scanner_fet_positive_to_seed_candidate",
        "seed_candidate_to_seed_selected",
        "seed_selected_to_fv_positive",
        "fv_positive_to_fvt_positive",
        "fvt_positive_to_skin",
    )

    stages = diagnostic["stages"]
    assert isinstance(stages, dict)
    assert set(expected_stages).issubset(stages)
    for stage_name in expected_stages:
        stage = stages[stage_name]
        assert isinstance(stage["candidate_count"], int)
        assert stage["candidate_count"] >= 0
        for key in (
            "edge_shell_fraction",
            "truth_buffered_f1_r2",
            "candidate_to_truth_p95",
            "truth_to_candidate_p95",
            "edge_false_positive_fraction_of_candidates",
        ):
            assert math.isfinite(float(stage[key]))

    transitions = diagnostic["transitions"]
    assert isinstance(transitions, dict)
    assert set(expected_transitions).issubset(transitions)
    for transition_name in expected_transitions:
        transition = transitions[transition_name]
        assert isinstance(transition["source_count"], int)
        assert isinstance(transition["target_count"], int)
        assert transition["source_count"] >= 0
        assert transition["target_count"] >= 0
        for key in (
            "target_to_source_count_ratio",
            "buffered_f1_r2",
            "target_to_source_distance_p95",
        ):
            assert math.isfinite(float(transition[key]))


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
    assert "--variant-preset" in result.stdout
    assert "--input-mode" in result.stdout
    assert "--workflow-mode" in result.stdout
    assert "reference" in result.stdout
    assert "quality" in result.stdout
    assert "diagnostic" in result.stdout
    assert "--scanner-backend" in result.stdout
    assert "--scanner-backend-matrix" in result.stdout
    assert "--scanner-downstream-diagnostics" in result.stdout
    assert "--scanner-thin-mode" in result.stdout
    assert "--save-volumes" in result.stdout
    assert "--save-figures" in result.stdout
    assert "--write-markdown-index" in result.stdout
    assert "--voter-thin-mode" in result.stdout
    assert "--surface-support-min-fraction" in result.stdout
    assert "--surface-support-exponent" in result.stdout
    assert "--thinning-diagnostics" in result.stdout
    assert "--include-thinning-diagnostic" in result.stdout
    assert "--thinning-diagnostic-cases" in result.stdout
    assert "--truth-surface-half-width" in result.stdout
    assert "--buffer-radius" in result.stdout
    assert "--skip-skinning" in result.stdout
    assert "--scanner-refinement-factor" in result.stdout
    assert "--scanner-backend {reference-like,fast,quality,ensemble}" in result.stdout
    assert "--skinner-min-likelihood" in result.stdout
    assert "--skinner-method" in result.stdout
    assert "--skinner-growth-source" in result.stdout
    assert "--skinner-ru" in result.stdout
    assert "--skinner-accepted-occupancy-radius" in result.stdout
    assert "--no-skinner-reskin" in result.stdout
    assert "--small-skin-size" in result.stdout


def test_report_3d_synthetic_quality_default_parse_resolves_default_variants(
    tmp_path: Path,
) -> None:
    module = _load_report_module()

    args = module.build_parser().parse_args(["--output-dir", str(tmp_path)])
    variants = module._resolve_variants(
        variants=args.variants,
        variant_preset=args.variant_preset,
    )

    assert args.variants is None
    assert args.variant_preset == "default"
    assert variants == ("current_default",)


def test_report_3d_synthetic_quality_parse_variants_accepts_voter_thin_hybrid() -> None:
    module = _load_report_module()

    assert module.parse_variants("voter_thin_hybrid") == ("voter_thin_hybrid",)


def test_report_3d_synthetic_quality_parse_variants_accepts_normal_plateau() -> None:
    module = _load_report_module()

    assert module.parse_variants("voter_thin_normal_plateau") == ("voter_thin_normal_plateau",)


def test_report_3d_synthetic_quality_parse_variants_accepts_hybrid_v2() -> None:
    module = _load_report_module()

    assert module.parse_variants("voter_thin_hybrid_v2") == ("voter_thin_hybrid_v2",)


def test_report_3d_synthetic_quality_parse_variants_accepts_hybrid_v2_recenter() -> None:
    module = _load_report_module()

    assert module.parse_variants("voter_thin_hybrid_v2_recenter_scanner_target") == (
        "voter_thin_hybrid_v2_recenter_scanner_target",
    )


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_edge_thin() -> None:
    module = _load_report_module()

    assert module.parse_variants("boundary_edge_thin_v1") == ("boundary_edge_thin_v1",)
    assert "boundary_edge_thin_v1" not in module.QUALITY_MATRIX_VARIANTS


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_seed_retention() -> None:
    module = _load_report_module()

    assert module.parse_variants("boundary_seed_retention_v1") == ("boundary_seed_retention_v1",)
    assert "boundary_seed_retention_v1" not in module.QUALITY_MATRIX_VARIANTS


def test_report_3d_synthetic_quality_parse_variants_accepts_surface_support_weighted() -> None:
    module = _load_report_module()

    assert module.parse_variants("surface_support_weighted") == ("surface_support_weighted",)


def test_report_3d_synthetic_quality_parse_variants_accepts_quality_skinner_v2() -> None:
    module = _load_report_module()

    assert module.parse_variants("quality_skinner_v2") == ("quality_skinner_v2",)


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_skinner_fallback() -> None:
    module = _load_report_module()

    assert module.parse_variants("quality_boundary_skinner_fallback") == (
        "quality_boundary_skinner_fallback",
    )


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_skinner_fallback_v2() -> None:
    module = _load_report_module()

    assert module.parse_variants("quality_boundary_skinner_fallback_v2") == (
        "quality_boundary_skinner_fallback_v2",
    )


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_skinner_fallback_v4() -> None:
    module = _load_report_module()

    assert module.parse_variants("quality_boundary_skinner_fallback_v4") == (
        "quality_boundary_skinner_fallback_v4",
    )


def test_report_3d_synthetic_quality_parse_variants_accepts_boundary_skinner_fallback_v5() -> None:
    module = _load_report_module()

    assert module.parse_variants("quality_boundary_skinner_fallback_v5") == (
        "quality_boundary_skinner_fallback_v5",
    )
    assert "quality_boundary_skinner_fallback_v5" not in module.QUALITY_MATRIX_VARIANTS


def test_find_synthetic_skins_thinned_uses_fvt_for_growth_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_report_module()
    fv = np.full((3, 3, 3), 1.0, dtype=np.float32)
    fvt = np.full((3, 3, 3), 2.0, dtype=np.float32)
    vp = np.zeros((3, 3, 3), dtype=np.float32)
    vt = np.zeros((3, 3, 3), dtype=np.float32)
    captured: dict[str, object] = {}

    class FakeSkinner:
        def __init__(self, **kwargs: object) -> None:
            captured["init_kwargs"] = kwargs

        def find_skins(
            self,
            volume: np.ndarray,
            p: np.ndarray,
            t: np.ndarray,
            **kwargs: object,
        ) -> list[object]:
            captured["volume"] = volume
            captured["p"] = p
            captured["t"] = t
            captured["kwargs"] = kwargs
            return []

    monkeypatch.setattr(module, "FaultSkinner", FakeSkinner)

    skins = module._find_synthetic_skins(
        fv,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(growth_source="thinned"),
    )

    assert skins == []
    assert captured["volume"] is fvt
    assert captured["p"] is vp
    assert captured["t"] is vt
    kwargs = captured["kwargs"]
    assert kwargs["ep"] is fvt
    assert kwargs["ft"] is fvt
    assert kwargs["pt"] is vp
    assert kwargs["tt"] is vt


def test_find_synthetic_skins_pre_thin_uses_fv_for_growth_and_fvt_for_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_report_module()
    fv = np.full((3, 3, 3), 1.0, dtype=np.float32)
    fvt = np.full((3, 3, 3), 2.0, dtype=np.float32)
    vp = np.zeros((3, 3, 3), dtype=np.float32)
    vt = np.zeros((3, 3, 3), dtype=np.float32)
    captured: dict[str, object] = {}

    class FakeSkinner:
        def __init__(self, **kwargs: object) -> None:
            captured["init_kwargs"] = kwargs

        def find_skins(
            self,
            volume: np.ndarray,
            p: np.ndarray,
            t: np.ndarray,
            **kwargs: object,
        ) -> list[object]:
            captured["volume"] = volume
            captured["p"] = p
            captured["t"] = t
            captured["kwargs"] = kwargs
            return []

    monkeypatch.setattr(module, "FaultSkinner", FakeSkinner)

    skins = module._find_synthetic_skins(
        fv,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(growth_source="pre_thin"),
    )

    assert skins == []
    assert captured["volume"] is fv
    assert captured["p"] is vp
    assert captured["t"] is vt
    kwargs = captured["kwargs"]
    assert kwargs["ep"] is fvt
    assert kwargs["ft"] is fv
    assert kwargs["pt"] is vp
    assert kwargs["tt"] is vt


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
    assert metrics["config"]["workflow_mode"] == "reference"
    assert metrics["config"]["variant_preset"] == "default"
    assert metrics["config"]["variants"] == ["current_default"]
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
    fv_positive_quality = quality["fv_positive_top_truth_count"]
    _assert_positive_top_truth_quality_is_finite(fv_positive_quality)
    fvt_positive_quality = quality["fvt_positive_top_truth_count"]
    _assert_positive_top_truth_quality_is_finite(fvt_positive_quality)
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
    assert rows[0]["workflow_mode"] == "reference"
    assert rows[0]["shape_n3"] == "17"
    assert rows[0]["shape_n2"] == "17"
    assert rows[0]["shape_n1"] == "17"
    assert float(rows[0]["fv_max"]) > 0.0
    assert float(rows[0]["fv_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fv_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fv_distance_p95"]) >= 0.0
    assert math.isfinite(float(rows[0]["fv_edge_false_positive_fraction"]))
    assert int(rows[0]["fv_positive_candidate_count"]) > 0
    assert math.isfinite(float(rows[0]["fv_strike_median_error"]))
    assert math.isfinite(float(rows[0]["fv_dip_median_error"]))
    assert float(rows[0]["fvt_max"]) > 0.0
    assert float(rows[0]["fvt_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fvt_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fvt_distance_p95"]) >= 0.0
    assert math.isfinite(float(rows[0]["fvt_edge_false_positive_fraction"]))
    assert int(rows[0]["fvt_positive_candidate_count"]) > 0
    assert math.isfinite(float(rows[0]["fvt_positive_buffered_f1_r2"]))
    assert float(rows[0]["fvt_positive_distance_p95"]) >= 0.0
    assert math.isfinite(float(rows[0]["fvt_positive_edge_false_positive_fraction"]))
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
        assert "fvt_top_truth_count" in quality
        assert "fvt_positive_top_truth_count" in quality
        assert "buffered_overlap_radius2" in quality["fvt_positive_top_truth_count"]
        assert "fvt_top_truth_count" in quality["edge_false_positive"]
        assert "fvt_positive_top_truth_count" in quality["edge_false_positive"]
        skin_quality = quality["skin"]
        assert skin_quality is not None
        assert "component_topology" in skin_quality
        component_topology = skin_quality["component_topology"]
        assert "truth_components" in component_topology
        assert "skins" in component_topology
        if case["case_id"] in {"parallel_planes", "crossing_planes"}:
            assert component_topology["truth_component_count"] == 2
            assert component_topology["skin_count"] >= 0
            assert component_topology["over_merge_skin_count"] >= 0
            assert component_topology["over_split_truth_component_count"] >= 0
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
        assert "fv_positive_candidate_count" in row
        assert "fvt_positive_candidate_count" in row
        assert "fvt_positive_buffered_f1_r2" in row
        assert "fvt_positive_distance_p95" in row
        assert "fvt_positive_edge_false_positive_fraction" in row
        assert "skin_truth_component_count" in row
        assert "skin_over_merge_count" in row
        assert "skin_over_split_count" in row
        assert math.isfinite(float(row["fvt_edge_false_positive_fraction"]))
        assert math.isfinite(float(row["fv_edge_false_positive_fraction"]))
        assert int(row["fv_positive_candidate_count"]) >= 0
        assert int(row["fvt_positive_candidate_count"]) >= 0
        assert math.isfinite(float(row["fvt_positive_buffered_f1_r2"]))
        assert math.isfinite(float(row["skin_mean_purity"]))
        assert math.isfinite(float(row["skin_mean_truth_component_recall"]))
        if row["case_id"] in {"parallel_planes", "crossing_planes"}:
            assert int(row["skin_truth_component_count"]) == 2
            assert int(row["skin_over_merge_count"]) >= 0
            assert int(row["skin_over_split_count"]) >= 0
        assert math.isfinite(float(row["fvt_positive_distance_p95"]))
        assert math.isfinite(float(row["fvt_positive_edge_false_positive_fraction"]))

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
            "final_norm_smoothing_1,voter_thin_normal,voter_thin_hybrid,"
            "voter_thin_hybrid_v2,voter_thin_normal_plateau,"
            "surface_support_weighted,quality_skinner_v2,"
            "quality_boundary_skinner_fallback,"
            "quality_boundary_skinner_fallback_v2,"
            "quality_boundary_skinner_fallback_v3,"
            "quality_boundary_skinner_fallback_v4"
        ),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    case = metrics["cases"][0]
    assert set(case["variants"]) == set(DIAGNOSTIC_VARIANTS)
    for variant in DIAGNOSTIC_VARIANTS:
        variant_report = case["variants"][variant]
        assert variant_report["pyosv"]["fvt"]["max"] > 0.0
        assert variant_report["skinning"]["enabled"] is True
        assert "diagnostics" in variant_report["skinning"]
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


def test_report_3d_synthetic_quality_quality_matrix_preset_runs_diagnostic_variants(
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
        "--variant-preset",
        "quality-matrix",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variant_preset"] == "quality-matrix"
    assert metrics["config"]["variants"] == list(DIAGNOSTIC_VARIANTS)
    assert set(metrics["cases"][0]["variants"]) == set(DIAGNOSTIC_VARIANTS)

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        ("single_vertical_plane", variant) for variant in DIAGNOSTIC_VARIANTS
    ]


def test_report_3d_synthetic_quality_explicit_variants_override_preset(
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
        "--variant-preset",
        "quality-matrix",
        "--variants",
        "current_default",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variant_preset"] == "quality-matrix"
    assert metrics["config"]["variants"] == ["current_default"]
    assert set(metrics["cases"][0]["variants"]) == {"current_default"}

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        ("single_vertical_plane", "current_default")
    ]


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
    _assert_finite_thinning_diagnostic_quality(diagnostic)

    reference_quality = diagnostic["reference"]["quality"]["fvt_top_truth_count"]
    normal_quality = diagnostic["normal"]["quality"]["fvt_top_truth_count"]
    reference_f1 = reference_quality["buffered_overlap_radius2"]["buffered_f1"]
    normal_f1 = normal_quality["buffered_overlap_radius2"]["buffered_f1"]
    reference_distance_p95 = reference_quality["surface_distance"]["candidate_to_truth_p95"]
    normal_distance_p95 = normal_quality["surface_distance"]["candidate_to_truth_p95"]
    delta = diagnostic["delta"]["normal_minus_reference"]

    assert normal_f1 > reference_f1
    assert normal_distance_p95 < reference_distance_p95
    assert delta["fvt_buffered_f1_r2"] > 0.05
    assert delta["fvt_candidate_to_truth_p95"] < -1.0

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


def test_report_3d_synthetic_quality_thinning_diagnostic_vertical_plane_is_finite(
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
        "single_vertical_plane",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    diagnostic_cases = {
        case["case_id"]
        for case in metrics["cases"]
        if "thinning_diagnostic" in case["variants"]["current_default"]
    }
    assert diagnostic_cases == {"single_vertical_plane"}

    vertical = next(case for case in metrics["cases"] if case["case_id"] == "single_vertical_plane")
    diagnostic = vertical["variants"]["current_default"]["thinning_diagnostic"]
    _assert_finite_thinning_diagnostic_quality(diagnostic)


def test_report_3d_synthetic_quality_thinning_diagnostic_does_not_change_primary_output(
    tmp_path: Path,
) -> None:
    plain_output_dir = tmp_path / "synthetic_quality_plain"
    diagnostic_output_dir = tmp_path / "synthetic_quality_diagnostic"
    common_args = (
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--input-mode",
        "oracle",
        "--variants",
        "current_default",
        "--skip-skinning",
    )

    plain_result = _run_script(
        *common_args,
        "--output-dir",
        str(plain_output_dir),
    )
    diagnostic_result = _run_script(
        *common_args,
        "--thinning-diagnostics",
        "--output-dir",
        str(diagnostic_output_dir),
    )

    assert plain_result.returncode == 0, plain_result.stderr
    assert diagnostic_result.returncode == 0, diagnostic_result.stderr
    plain_metrics = json.loads((plain_output_dir / "metrics.json").read_text(encoding="utf-8"))
    diagnostic_metrics = json.loads(
        (diagnostic_output_dir / "metrics.json").read_text(encoding="utf-8")
    )

    diagnostic_cases = {
        case["case_id"]: case["variants"]["current_default"] for case in diagnostic_metrics["cases"]
    }
    for plain_case in plain_metrics["cases"]:
        case_id = plain_case["case_id"]
        plain_variant = plain_case["variants"]["current_default"]
        diagnostic_variant = diagnostic_cases[case_id]
        assert math.isclose(
            float(plain_variant["pyosv"]["fvt"]["max"]),
            float(diagnostic_variant["pyosv"]["fvt"]["max"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        assert math.isclose(
            float(
                plain_variant["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"][
                    "buffered_f1"
                ]
            ),
            float(
                diagnostic_variant["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"][
                    "buffered_f1"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )


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
        _assert_finite_thinning_diagnostic_quality(variant["thinning_diagnostic"])

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


def test_report_3d_synthetic_quality_voter_thin_hybrid_variant_passes(
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
        "current_default,voter_thin_hybrid",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["cases"][0]["variants"]) == {
        "current_default",
        "voter_thin_hybrid",
    }
    hybrid = metrics["cases"][0]["variants"]["voter_thin_hybrid"]
    assert hybrid["pyosv"]["fvt"]["max"] > 0.0


def test_report_3d_synthetic_quality_voter_thin_normal_plateau_variant_passes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "extended",
        "--shape",
        "21,21,21",
        "--output-dir",
        str(output_dir),
        "--variants",
        "voter_thin_normal_plateau",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    boundary_case = next(case for case in metrics["cases"] if case["case_id"] == "boundary_plane")
    variant = boundary_case["variants"]["voter_thin_normal_plateau"]
    assert variant["pyosv"]["fvt"]["finite_fraction"] == 1.0
    assert (
        variant["quality"]["fvt_positive_top_truth_count"]["buffered_overlap_radius2"][
            "candidate_count"
        ]
        > 0
    )


def test_report_3d_synthetic_quality_voter_thin_hybrid_v2_variant_passes(
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
        "voter_thin_hybrid_v2",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["voter_thin_hybrid_v2"]
    assert variant["pyosv"]["fvt"]["max"] > 0.0


def test_fvt_recenter_moves_edge_candidate_toward_stronger_target() -> None:
    module = _load_report_module()
    fvt = np.zeros((5, 5, 5), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    target = np.zeros_like(fvt)
    fvt[2, 1, 2] = 4.0
    target[2, 1, 2] = 0.2
    target[2, 3, 2] = 0.9

    recentered, diagnostic = module._recenter_edge_fvt_to_target(
        fvt,
        vp,
        vt,
        target=target,
        target_source="scanner_fet",
        max_shift=3,
        edge_margin=2,
    )

    assert recentered[2, 1, 2] == 0.0
    assert recentered[2, 3, 2] == pytest.approx(4.0)
    assert diagnostic["fvt_recenter_candidate_count"] == 1
    assert diagnostic["fvt_recenter_moved_count"] == 1
    assert diagnostic["fvt_recenter_value_source"] == "original_fvt"


def test_fvt_recenter_collision_keeps_higher_original_fvt_deterministically() -> None:
    module = _load_report_module()
    fvt = np.zeros((5, 7, 5), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    target = np.zeros_like(fvt)
    fvt[2, 0, 2] = 2.0
    fvt[2, 2, 2] = 5.0
    target[2, 1, 2] = 1.0

    first, first_diagnostic = module._recenter_edge_fvt_to_target(
        fvt,
        vp,
        vt,
        target=target,
        target_source="scanner_fet",
        max_shift=1,
        edge_margin=3,
    )
    second, second_diagnostic = module._recenter_edge_fvt_to_target(
        fvt,
        vp,
        vt,
        target=target,
        target_source="scanner_fet",
        max_shift=1,
        edge_margin=3,
    )

    np.testing.assert_array_equal(first, second)
    assert first[2, 1, 2] == pytest.approx(5.0)
    assert np.count_nonzero(first) == 1
    assert first_diagnostic["fvt_recenter_collision_count"] == 1
    assert second_diagnostic["fvt_recenter_collision_count"] == 1


def test_boundary_edge_thin_v1_preserves_non_edge_hybrid_v2_mask() -> None:
    module = _load_report_module()
    voter = module.OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 7, 5), dtype=np.float32)
    fv[2, 0:3, 2] = 1.0
    fv[2, 3, 2] = 0.8
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    target = np.zeros_like(fv)
    target[2, 1, 2] = 1.0

    base = voter.thin(
        fv,
        vp,
        vt,
        mode="hybrid_v2",
        reference_sigma=0.0,
        plateau_tie_breaker=fv,
    )
    result, diagnostic = module._apply_boundary_edge_thin_v1(
        base,
        fv,
        vp,
        vt,
        voter=voter,
        target=target,
        target_source="scanner_fet",
        edge_margin=2,
    )

    non_edge = ~module._edge_mask(fv.shape, 2)
    np.testing.assert_array_equal(result[non_edge] > 0.0, base[non_edge] > 0.0)
    assert diagnostic["enabled"] is True
    assert diagnostic["target_source"] == "scanner_fet"


def test_boundary_edge_thin_v1_adopts_high_target_plateau_candidate() -> None:
    module = _load_report_module()
    voter = module.OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fvt = np.zeros((5, 5, 5), dtype=np.float32)
    fv = np.zeros_like(fvt)
    fv[2, 2:4, 2] = 1.0
    fvt[2, 4, 2] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    target = np.zeros_like(fvt)
    target[2, 4, 2] = 0.1
    target[2, 3, 2] = 0.9

    result, diagnostic = module._apply_boundary_edge_thin_v1(
        fvt,
        fv,
        vp,
        vt,
        voter=voter,
        target=target,
        target_source="scanner_fet",
        edge_margin=2,
    )

    assert result[2, 4, 2] == 0.0
    assert result[2, 3, 2] == pytest.approx(1.0)
    assert diagnostic["adopted_candidate_count"] == 1
    assert diagnostic["replaced_candidate_count"] == 1
    assert diagnostic["edge_positive_count_before"] == 1
    assert diagnostic["edge_positive_count_after"] == 1


def test_report_3d_synthetic_quality_recenter_oracle_records_target_source() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        variants=("voter_thin_hybrid_v2_recenter_scanner_target",),
        workflow_mode="quality",
        input_mode="oracle",
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    variant = report["cases"][0]["variants"]["voter_thin_hybrid_v2_recenter_scanner_target"]
    diagnostic = variant["fvt_recenter"]
    assert diagnostic["fvt_recenter_enabled"] is True
    assert diagnostic["fvt_recenter_target_source"] == "oracle_ft"
    assert diagnostic["fvt_recenter_edge_shell_only"] is True


def test_report_3d_synthetic_quality_recenter_scanner_records_diagnostics_and_csv(
    tmp_path: Path,
) -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        variants=("voter_thin_hybrid_v2_recenter_scanner_target",),
        workflow_mode="quality",
        input_mode="scanner",
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    variant = report["cases"][0]["variants"]["voter_thin_hybrid_v2_recenter_scanner_target"]
    diagnostic = variant["fvt_recenter"]
    assert diagnostic["fvt_recenter_target_source"] == "scanner_fet"
    for key in (
        "fvt_recenter_candidate_count",
        "fvt_recenter_moved_count",
        "fvt_recenter_collision_count",
        "fvt_recenter_positive_count_before",
        "fvt_recenter_positive_count_after",
    ):
        assert int(diagnostic[key]) >= 0
    for key in (
        "fvt_recenter_mean_shift",
        "fvt_recenter_p95_shift",
        "fvt_recenter_max_shift",
        "fvt_recenter_to_target_distance_p95_before",
        "fvt_recenter_to_target_distance_p95_after",
    ):
        assert math.isfinite(float(diagnostic[key]))

    module.write_summary_csv(report, tmp_path)
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    assert row["fvt_recenter_enabled"] == "True"
    assert row["fvt_recenter_target_source"] == "scanner_fet"
    assert row["fvt_recenter_candidate_count"] != ""
    assert row["fvt_recenter_to_target_distance_p95_after"] != ""


def test_report_3d_synthetic_quality_boundary_edge_thin_scanner_diagnostics_and_csv(
    tmp_path: Path,
) -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        variants=("boundary_edge_thin_v1",),
        workflow_mode="quality",
        input_mode="scanner",
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    variant = report["cases"][0]["variants"]["boundary_edge_thin_v1"]
    diagnostic = variant["boundary_edge_thin"]
    assert diagnostic["enabled"] is True
    assert diagnostic["target_source"] == "scanner_fet"
    assert diagnostic["edge_margin"] == module.EDGE_FALSE_POSITIVE_MARGIN
    for key in (
        "positive_count_before",
        "positive_count_after",
        "edge_positive_count_before",
        "edge_positive_count_after",
        "adopted_candidate_count",
        "replaced_candidate_count",
        "collision_count",
    ):
        assert int(diagnostic[key]) >= 0
    assert math.isfinite(float(diagnostic["to_target_distance_p95_before"]))
    assert math.isfinite(float(diagnostic["to_target_distance_p95_after"]))

    module.write_summary_csv(report, tmp_path)
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    assert row["boundary_edge_thin_enabled"] == "True"
    assert row["boundary_edge_thin_target_source"] == "scanner_fet"
    assert row["boundary_edge_thin_adopted_candidate_count"] != ""
    assert row["boundary_edge_thin_to_target_distance_p95_after"] != ""


def test_report_3d_synthetic_quality_hybrid_v2_output_unchanged_without_recenter() -> None:
    module = _load_report_module()
    kwargs = {
        "case_set": "minimal",
        "shape": (17, 17, 17),
        "workflow_mode": "quality",
        "input_mode": "oracle",
        "skinning_config": module.SyntheticSkinningConfig(enabled=False),
    }

    plain = module.build_report(variants=("voter_thin_hybrid_v2",), **kwargs)
    with_recenter_variant = module.build_report(
        variants=(
            "voter_thin_hybrid_v2",
            "voter_thin_hybrid_v2_recenter_scanner_target",
        ),
        **kwargs,
    )

    plain_variant = plain["cases"][0]["variants"]["voter_thin_hybrid_v2"]
    repeated_plain_variant = with_recenter_variant["cases"][0]["variants"]["voter_thin_hybrid_v2"]
    assert "fvt_recenter" not in plain_variant
    assert "fvt_recenter" not in repeated_plain_variant
    assert plain_variant["pyosv"]["fvt"] == repeated_plain_variant["pyosv"]["fvt"]
    assert (
        plain_variant["quality"]["fvt_positive_top_truth_count"]
        == repeated_plain_variant["quality"]["fvt_positive_top_truth_count"]
    )


def test_report_3d_synthetic_quality_surface_support_weighted_variant_passes(
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
        "surface_support_weighted",
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["surface_support_weighted"]
    variant = metrics["cases"][0]["variants"]["surface_support_weighted"]
    assert variant["pyosv"]["fvt"]["finite_fraction"] == 1.0
    assert variant["pyosv"]["voting"] == {
        "surface_support_min_fraction": 0.5,
        "surface_support_exponent": 1.0,
    }


def test_report_3d_synthetic_quality_quality_skinner_v2_records_effective_config(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--variants",
        "current_default,quality_skinner_v2",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["current_default", "quality_skinner_v2"]

    case = metrics["cases"][0]
    current_default = case["variants"]["current_default"]["config"]["skinning"]
    assert current_default == metrics["config"]["skinning"]
    assert current_default["method"] == "reference"
    assert current_default["growth_source"] == "thinned"
    assert current_default["accepted_occupancy_radius"] is None
    assert current_default["effective_accepted_occupancy_radius"] == 5

    quality_skinner = case["variants"]["quality_skinner_v2"]["config"]["skinning"]
    assert quality_skinner["method"] == "quality"
    assert quality_skinner["min_likelihood"] is None
    assert quality_skinner["adaptive_min_likelihood"] is True
    assert quality_skinner["seed_min_ep"] == 0.5
    assert quality_skinner["seed_planarity_source"] == "fvt"
    assert quality_skinner["growth_source"] == "pre_thin"
    assert quality_skinner["accepted_occupancy_radius"] == 1
    assert quality_skinner["effective_accepted_occupancy_radius"] == 1

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        ("single_vertical_plane", "current_default"),
        ("single_vertical_plane", "quality_skinner_v2"),
    ]
    assert rows[1]["baseline_variant"] == "current_default"
    assert math.isfinite(float(rows[1]["skin_buffered_f1_delta_vs_baseline"]))
    assert math.isfinite(float(rows[1]["skin_count_delta_vs_baseline"]))


def test_report_3d_synthetic_quality_boundary_fallback_variant_cli_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "quality_boundary_skinner_fallback",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["quality_boundary_skinner_fallback"]
    variant = metrics["cases"][0]["variants"]["quality_boundary_skinner_fallback"]
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_enabled"] is True
    assert diagnostics["fallback_policy"] == "empty_primary"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_nonempty"
    assert diagnostics["fallback_method"] == "connected_component_on_fvt"
    assert diagnostics["fallback_input"] == "fvt"
    assert diagnostics["fallback_skin_count"] == 0
    assert diagnostics["fallback_cell_count"] == 0
    for field in (
        "skin_fallback_component_count",
        "skin_fallback_candidate_cell_count",
        "skin_fallback_largest_component_size",
        "skin_fallback_largest_component_fraction",
        "skin_fallback_top3_component_cell_count",
        "skin_fallback_top3_component_fraction",
        "skin_fallback_small_component_count",
        "skin_fallback_component_policy",
        "skin_fallback_accepted_component_count",
        "skin_fallback_discarded_component_count",
        "skin_fallback_accepted_component_cell_count",
    ):
        assert field in diagnostics
    assert (
        diagnostics["skin_fallback_candidate_cell_count"] == diagnostics["fallback_candidate_count"]
    )
    assert diagnostics["skin_fallback_component_count"] > 0
    assert diagnostics["skin_fallback_largest_component_size"] > 0
    assert diagnostics["skin_fallback_largest_component_fraction"] > 0.0
    assert diagnostics["skin_fallback_component_policy"] == "all"
    for field, value in diagnostics.items():
        if field.startswith("skin_fallback_") and field not in {
            "skin_fallback_component_policy",
            "skin_fallback_pruning_method",
            "skin_fallback_skeletonization_axis_mode",
        }:
            assert math.isfinite(float(value))

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["variant"] == "quality_boundary_skinner_fallback"
    assert rows[0]["skin_fallback_enabled"] == "True"
    assert rows[0]["skin_fallback_policy"] == "empty_primary"
    assert rows[0]["skin_fallback_used"] == "False"
    assert rows[0]["skin_fallback_reason"] == "primary_skin_nonempty"
    assert rows[0]["skin_fallback_method"] == "connected_component_on_fvt"
    assert rows[0]["skin_fallback_input"] == "fvt"
    assert rows[0]["skin_fallback_skin_count"] == "0"
    assert rows[0]["skin_fallback_cell_count"] == "0"
    assert rows[0]["skin_fallback_component_policy"] == "all"
    for field in (
        "skin_fallback_component_count",
        "skin_fallback_candidate_cell_count",
        "skin_fallback_largest_component_size",
        "skin_fallback_largest_component_fraction",
        "skin_fallback_top3_component_cell_count",
        "skin_fallback_top3_component_fraction",
        "skin_fallback_small_component_count",
        "skin_fallback_accepted_component_count",
        "skin_fallback_discarded_component_count",
        "skin_fallback_accepted_component_cell_count",
    ):
        assert math.isfinite(float(rows[0][field]))


def test_report_3d_synthetic_quality_fallback_component_diagnostics_zero_without_fvt() -> None:
    module = _load_report_module()
    fvt = np.zeros((2, 3, 4), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    diagnostics: dict[str, object] = {}

    module._apply_boundary_skinner_fallback(
        [],
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(boundary_skinner_fallback=False),
        variant="current_default",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_enabled"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["skin_fallback_component_policy"] == "all"
    for field in (
        "skin_fallback_component_count",
        "skin_fallback_candidate_cell_count",
        "skin_fallback_largest_component_size",
        "skin_fallback_largest_component_fraction",
        "skin_fallback_top3_component_cell_count",
        "skin_fallback_top3_component_fraction",
        "skin_fallback_small_component_count",
        "skin_fallback_accepted_component_count",
        "skin_fallback_discarded_component_count",
        "skin_fallback_accepted_component_cell_count",
    ):
        assert diagnostics[field] == 0


def test_report_3d_synthetic_quality_fallback_component_ordering_is_deterministic() -> None:
    module = _load_report_module()
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 2, 0] = True
    mask[0, 2, 1] = True
    mask[0, 0, 2] = True
    mask[1, 0, 2] = True

    components = module._positive_mask_components(mask, connectivity="edge")

    assert components == [
        [(0, 0, 2), (1, 0, 2)],
        [(0, 2, 0), (0, 2, 1)],
    ]

    diagnostics = module._fallback_component_diagnostics(
        mask.astype(np.float32),
        min_skin_size=3,
        small_component_size=3,
        connectivity="edge",
    )
    assert diagnostics["skin_fallback_component_count"] == 2
    assert diagnostics["skin_fallback_candidate_cell_count"] == 4
    assert diagnostics["skin_fallback_largest_component_size"] == 2
    assert diagnostics["skin_fallback_largest_component_fraction"] == pytest.approx(0.5)
    assert diagnostics["skin_fallback_top3_component_cell_count"] == 4
    assert diagnostics["skin_fallback_top3_component_fraction"] == pytest.approx(1.0)
    assert diagnostics["skin_fallback_small_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_count"] == 0
    assert diagnostics["skin_fallback_discarded_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 0
    assert diagnostics["skin_fallback_filter_min_component_size"] == 0
    assert diagnostics["skin_fallback_filter_min_component_fraction_of_largest"] == 0.0
    assert diagnostics["skin_fallback_filter_max_components"] == 0


def test_report_3d_synthetic_quality_filtered_fallback_component_diagnostics() -> None:
    module = _load_report_module()
    mask = np.zeros((1, 1, 20), dtype=bool)
    mask[0, 0, 0:10] = True
    mask[0, 0, 12:14] = True
    mask[0, 0, 16] = True

    diagnostics = module._fallback_component_diagnostics(
        mask.astype(np.float32),
        min_skin_size=1,
        small_component_size=3,
        connectivity="edge",
        component_policy="degraded_primary_filtered",
    )

    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert diagnostics["skin_fallback_component_count"] == 3
    assert diagnostics["skin_fallback_candidate_cell_count"] == 13
    assert diagnostics["skin_fallback_largest_component_size"] == 10
    assert diagnostics["skin_fallback_top3_component_cell_count"] == 13
    assert diagnostics["skin_fallback_small_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_count"] == 1
    assert diagnostics["skin_fallback_discarded_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 10
    assert diagnostics["skin_fallback_filter_min_component_size"] == 8
    assert diagnostics["skin_fallback_filter_min_component_fraction_of_largest"] == pytest.approx(
        0.10
    )
    assert diagnostics["skin_fallback_filter_max_components"] == 3


def test_report_3d_synthetic_quality_filtered_fallback_keeps_largest_when_all_small() -> None:
    module = _load_report_module()
    mask = np.zeros((1, 1, 10), dtype=bool)
    mask[0, 0, 0:4] = True
    mask[0, 0, 7:10] = True

    diagnostics = module._fallback_component_diagnostics(
        mask.astype(np.float32),
        min_skin_size=1,
        small_component_size=3,
        connectivity="edge",
        component_policy="degraded_primary_filtered",
    )

    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert diagnostics["skin_fallback_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_count"] == 1
    assert diagnostics["skin_fallback_discarded_component_count"] == 1
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 4


def test_report_3d_synthetic_quality_skeletonized_fallback_prunes_subset() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 3, 4), dtype=np.float32)
    fvt[0, 2, 3] = 2.0
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    component = [(0, i2, i1) for i2 in range(3) for i1 in range(4)]

    mask, diagnostics = module._skeletonize_fallback_components(
        fvt,
        vp,
        vt,
        [component],
    )

    retained = {tuple(index) for index in np.argwhere(mask)}
    assert retained.issubset(set(component))
    assert retained == {
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 3),
    }
    assert diagnostics["skin_fallback_pruning_method"] == "fault_normal_line_collapse"
    assert diagnostics["skin_fallback_raw_component_cell_count"] == 12
    assert diagnostics["skin_fallback_pruned_component_cell_count"] == 4
    assert diagnostics["skin_fallback_pruned_fraction"] == pytest.approx(4 / 12)
    assert diagnostics["skin_fallback_largest_component_size_before_pruning"] == 12
    assert diagnostics["skin_fallback_largest_component_size_after_pruning"] == 4
    assert diagnostics["skin_fallback_pruning_removed_cell_count"] == 8
    assert diagnostics["skin_fallback_skeletonization_axis_mode"] == "i2"


def test_report_3d_synthetic_quality_skeletonized_fallback_ties_are_deterministic() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 4, 1), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    component = [(0, i2, 0) for i2 in range(4)]

    mask_a, diagnostics_a = module._skeletonize_fallback_components(fvt, vp, vt, [component])
    mask_b, diagnostics_b = module._skeletonize_fallback_components(fvt, vp, vt, [component])

    assert np.array_equal(mask_a, mask_b)
    assert diagnostics_a == diagnostics_b
    assert {tuple(index) for index in np.argwhere(mask_a)} == {(0, 1, 0)}


def test_report_3d_synthetic_quality_fallback_uses_fvt_candidate_mask_not_vt() -> None:
    module = _load_report_module()
    fvt = np.zeros((2, 2, 3), dtype=np.float32)
    fvt[0, 0, 0] = 1.0
    fvt[0, 0, 1] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.ones_like(fvt)
    skins: list[object] = []
    diagnostics: dict[str, object] = {}

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            min_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_used"] is True
    assert diagnostics["skin_fallback_candidate_cell_count"] == 2
    assert diagnostics["skin_fallback_component_count"] == 1
    assert diagnostics["fallback_cell_count"] == 2
    assert len(skins) == 1
    assert len(skins[0]) == 2
    assert {(cell.i1, cell.i2, cell.i3) for cell in skins[0]} == {
        (0, 0, 0),
        (1, 0, 0),
    }


def test_report_3d_synthetic_quality_fallback_invokes_original_threshold_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_report_module()
    fvt = np.zeros((2, 2, 3), dtype=np.float32)
    fvt[0, 0, 0] = 1.0
    fvt[0, 0, 1] = np.float32(module.NONZERO_EPSILON / 2.0)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    diagnostics: dict[str, object] = {}
    calls: list[dict[str, object]] = []

    def fake_find_connected_component_skins(
        fv: np.ndarray,
        vp_arg: np.ndarray,
        vt_arg: np.ndarray,
        *,
        min_likelihood: float | None = None,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
    ) -> list[object]:
        calls.append(
            {
                "fv": fv,
                "vp": vp_arg,
                "vt": vt_arg,
                "min_likelihood": min_likelihood,
                "min_skin_size": min_skin_size,
                "connectivity": connectivity,
            }
        )
        return []

    monkeypatch.setattr(
        module,
        "find_connected_component_skins",
        fake_find_connected_component_skins,
    )

    module._apply_boundary_skinner_fallback(
        [],
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            min_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_reason"] == "connected_component_fallback_empty"
    assert len(calls) == 1
    assert calls[0]["fv"] is fvt
    assert calls[0]["vp"] is vp
    assert calls[0]["vt"] is vt
    assert calls[0]["min_likelihood"] == pytest.approx(module.NONZERO_EPSILON)
    assert calls[0]["min_skin_size"] == 1
    assert calls[0]["connectivity"] == "edge"


def test_report_3d_synthetic_quality_boundary_fallback_v2_cli_records_policy(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "quality_boundary_skinner_fallback_v2",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["quality_boundary_skinner_fallback_v2"]
    variant = metrics["cases"][0]["variants"]["quality_boundary_skinner_fallback_v2"]
    assert variant["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert variant["config"]["skinning"]["boundary_skinner_fallback_policy"] == "degraded_primary"
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_policy"] == "degraded_primary"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_healthy"

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["variant"] == "quality_boundary_skinner_fallback_v2"
    assert rows[0]["skin_fallback_policy"] == "degraded_primary"
    assert rows[0]["skin_fallback_triggered_by_degraded_primary"] == "False"
    assert rows[0]["skin_fallback_replaced_primary"] == "False"
    assert rows[0]["skin_fallback_degraded_reasons"] == ""


def test_report_3d_synthetic_quality_boundary_fallback_v3_cli_records_filtered_policy(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_dir = tmp_path / "synthetic_quality"

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        variants=("quality_boundary_skinner_fallback_v3",),
    )
    assert report["config"]["variants"] == ["quality_boundary_skinner_fallback_v3"]
    build_variant = report["cases"][0]["variants"]["quality_boundary_skinner_fallback_v3"]
    assert (
        build_variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_filtered"
    )

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "quality_boundary_skinner_fallback_v3",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["quality_boundary_skinner_fallback_v3"]
    variant = metrics["cases"][0]["variants"]["quality_boundary_skinner_fallback_v3"]
    assert variant["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert (
        variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_filtered"
    )
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_healthy"
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["variant"] == "quality_boundary_skinner_fallback_v3"
    assert rows[0]["skin_fallback_policy"] == "degraded_primary_filtered"
    assert rows[0]["skin_fallback_triggered_by_degraded_primary"] == "False"
    assert rows[0]["skin_fallback_replaced_primary"] == "False"
    assert rows[0]["skin_fallback_component_policy"] == "degraded_primary_filtered"
    expected_min_component_size = max(
        8,
        math.ceil(0.05 * int(rows[0]["skin_fallback_candidate_cell_count"])),
    )
    assert int(rows[0]["skin_fallback_filter_min_component_size"]) == expected_min_component_size
    assert rows[0]["skin_fallback_filter_min_component_fraction_of_largest"] == "0.1"
    assert rows[0]["skin_fallback_filter_max_components"] == "3"


def test_report_3d_synthetic_quality_boundary_fallback_v4_cli_records_pruning_policy(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_dir = tmp_path / "synthetic_quality"

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        variants=("quality_boundary_skinner_fallback_v4",),
    )
    assert report["config"]["variants"] == ["quality_boundary_skinner_fallback_v4"]
    build_variant = report["cases"][0]["variants"]["quality_boundary_skinner_fallback_v4"]
    assert (
        build_variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_skeletonized"
    )

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "quality_boundary_skinner_fallback_v4",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["quality_boundary_skinner_fallback_v4"]
    variant = metrics["cases"][0]["variants"]["quality_boundary_skinner_fallback_v4"]
    assert variant["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert (
        variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_skeletonized"
    )
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_policy"] == "degraded_primary_skeletonized"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_healthy"
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_skeletonized"
    for field in (
        "skin_fallback_pruning_method",
        "skin_fallback_raw_component_cell_count",
        "skin_fallback_pruned_component_cell_count",
        "skin_fallback_pruned_fraction",
        "skin_fallback_largest_component_size_before_pruning",
        "skin_fallback_largest_component_size_after_pruning",
        "skin_fallback_pruning_removed_cell_count",
        "skin_fallback_skeletonization_axis_mode",
    ):
        assert field in diagnostics

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["variant"] == "quality_boundary_skinner_fallback_v4"
    assert rows[0]["skin_fallback_policy"] == "degraded_primary_skeletonized"
    assert rows[0]["skin_fallback_triggered_by_degraded_primary"] == "False"
    assert rows[0]["skin_fallback_replaced_primary"] == "False"
    assert rows[0]["skin_fallback_component_policy"] == "degraded_primary_skeletonized"
    assert rows[0]["skin_fallback_pruning_method"] == ""
    assert rows[0]["skin_fallback_raw_component_cell_count"] == "0"
    assert rows[0]["skin_fallback_pruned_component_cell_count"] == "0"
    assert rows[0]["skin_fallback_pruned_fraction"] == "0.0"
    assert rows[0]["skin_fallback_largest_component_size_before_pruning"] == "0"
    assert rows[0]["skin_fallback_largest_component_size_after_pruning"] == "0"
    assert rows[0]["skin_fallback_pruning_removed_cell_count"] == "0"
    assert rows[0]["skin_fallback_skeletonization_axis_mode"] == ""


def test_report_3d_synthetic_quality_boundary_fallback_v5_cli_records_guardrail_policy(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_dir = tmp_path / "synthetic_quality"

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        variants=("quality_boundary_skinner_fallback_v5",),
    )
    assert report["config"]["variants"] == ["quality_boundary_skinner_fallback_v5"]
    build_variant = report["cases"][0]["variants"]["quality_boundary_skinner_fallback_v5"]
    assert build_variant["config"]["skinning"]["method"] == "quality"
    assert build_variant["config"]["skinning"]["min_likelihood"] is None
    assert build_variant["config"]["skinning"]["accepted_occupancy_radius"] == 1
    assert build_variant["config"]["skinning"]["growth_source"] == "pre_thin"
    assert (
        build_variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_topology_guarded"
    )

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "quality_boundary_skinner_fallback_v5",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == ["quality_boundary_skinner_fallback_v5"]
    variant = metrics["cases"][0]["variants"]["quality_boundary_skinner_fallback_v5"]
    assert variant["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert (
        variant["config"]["skinning"]["boundary_skinner_fallback_policy"]
        == "degraded_primary_topology_guarded"
    )
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_policy"] == "degraded_primary_topology_guarded"
    assert "fallback_v5_guardrail" in diagnostics
    assert diagnostics["fallback_v5_guardrail"]["enabled"] is True
    assert diagnostics["fallback_used"] is False

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["variant"] == "quality_boundary_skinner_fallback_v5"
    assert rows[0]["skin_fallback_policy"] == "degraded_primary_topology_guarded"
    assert rows[0]["skin_fallback_v5_guardrail_enabled"] == "True"
    for column in (
        "skin_fallback_v5_guardrail_passed",
        "skin_fallback_v5_guardrail_reasons",
        "skin_fallback_v5_guardrail_fallback_skin_count",
        "skin_fallback_v5_guardrail_coverage_of_fvt_positive",
        "skin_fallback_v5_guardrail_largest_skin_fraction",
        "skin_fallback_v5_guardrail_small_skin_cell_fraction",
        "skin_fallback_v5_guardrail_pruned_fraction",
    ):
        assert column in rows[0]


def test_report_3d_synthetic_quality_boundary_fallback_v4_keeps_simple_scanner_primary() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        input_mode="scanner",
        workflow_mode="quality",
        variants=("quality_boundary_skinner_fallback_v4",),
        scanner_config=module.SyntheticScannerConfig(backend="quality"),
    )

    variant = report["cases"][0]["variants"]["quality_boundary_skinner_fallback_v4"]
    diagnostics = variant["skinning"]["diagnostics"]
    assert diagnostics["fallback_policy"] == "degraded_primary_skeletonized"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_replaced_primary"] is False
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False


@pytest.mark.parametrize(
    (
        "workflow_mode",
        "override",
        "expected_voter_thin_mode",
        "expected_support_min_fraction",
        "expected_support_exponent",
    ),
    [
        ("reference", None, "reference", 0.0, 0.0),
        ("quality", None, "hybrid_v2", 0.0, 0.0),
        ("quality", "reference", "reference", 0.0, 0.0),
        ("reference", "normal", "normal", 0.0, 0.0),
        ("reference", "hybrid", "hybrid", 0.0, 0.0),
        ("diagnostic", None, "reference", 0.0, 0.0),
        ("diagnostic", "normal", "normal", 0.0, 0.0),
    ],
)
def test_report_3d_synthetic_quality_workflow_mode_resolves_voter_thin_mode(
    tmp_path: Path,
    workflow_mode: str,
    override: str | None,
    expected_voter_thin_mode: str,
    expected_support_min_fraction: float,
    expected_support_exponent: float,
) -> None:
    output_dir = tmp_path / f"synthetic_quality_{workflow_mode}_{override or 'default'}"
    args = [
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        workflow_mode,
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    ]
    if override is not None:
        args.extend(["--voter-thin-mode", override])

    result = _run_script(*args)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == workflow_mode
    assert metrics["config"]["voting"]["voter_thin_mode"] == expected_voter_thin_mode
    assert (
        metrics["config"]["voting"]["surface_support_min_fraction"] == expected_support_min_fraction
    )
    assert metrics["config"]["voting"]["surface_support_exponent"] == expected_support_exponent


def test_report_3d_synthetic_quality_build_report_reference_workflow_defaults() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="reference",
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    assert report["config"]["workflow_mode"] == "reference"
    assert report["config"]["voting"]["voter_thin_mode"] == "reference"
    assert report["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert report["config"]["voting"]["surface_support_exponent"] == 0.0
    assert report["config"]["skinning"]["boundary_skinner_fallback"] is False
    assert report["config"]["skinning"]["boundary_skinner_fallback_policy"] == "empty_primary"


def test_report_3d_synthetic_quality_build_report_quality_workflow_defaults() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    assert report["config"]["workflow_mode"] == "quality"
    assert report["config"]["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert report["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert report["config"]["voting"]["surface_support_exponent"] == 0.0
    assert report["config"]["skinning"]["enabled"] is False
    assert report["config"]["skinning"]["method"] == "quality"
    assert report["config"]["skinning"]["growth_source"] == "pre_thin"
    assert report["config"]["skinning"]["min_likelihood"] is None
    assert report["config"]["skinning"]["adaptive_min_likelihood"] is True
    assert report["config"]["skinning"]["accepted_occupancy_radius"] == 1
    assert report["config"]["skinning"]["effective_accepted_occupancy_radius"] == 1
    assert report["config"]["skinning"]["boundary_skinner_fallback"] is True
    assert report["config"]["skinning"]["boundary_skinner_fallback_policy"] == "empty_primary"


def test_report_3d_synthetic_quality_build_report_explicit_skinner_method_wins() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        skinner_method_explicit=True,
        skinning_config=module.SyntheticSkinningConfig(enabled=False, method="reference"),
    )

    assert report["config"]["workflow_mode"] == "quality"
    assert report["config"]["skinning"]["method"] == "reference"
    assert report["config"]["skinning"]["min_likelihood"] == 0.5
    assert report["config"]["skinning"]["adaptive_min_likelihood"] is False
    assert report["config"]["skinning"]["boundary_skinner_fallback"] is False
    assert report["config"]["skinning"]["boundary_skinner_fallback_policy"] == "empty_primary"


def test_report_3d_synthetic_quality_build_report_explicit_voting_config_wins() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        workflow_mode="quality",
        voting_config=module.SyntheticVotingConfig(
            voter_thin_mode="reference",
            surface_support_min_fraction=0.0,
            surface_support_exponent=0.0,
        ),
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    assert report["config"]["workflow_mode"] == "quality"
    assert report["config"]["voting"]["voter_thin_mode"] == "reference"
    assert report["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert report["config"]["voting"]["surface_support_exponent"] == 0.0


def test_quality_workflow_current_default_uses_hybrid_v2_without_support(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "synthetic_quality_reference"
    quality_dir = tmp_path / "synthetic_quality_quality"

    reference_result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "reference",
        "--output-dir",
        str(reference_dir),
        "--skip-skinning",
    )
    quality_result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "current_default",
        "--output-dir",
        str(quality_dir),
        "--skip-skinning",
    )

    assert reference_result.returncode == 0, reference_result.stderr
    assert quality_result.returncode == 0, quality_result.stderr

    reference_metrics = json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))
    quality_metrics = json.loads((quality_dir / "metrics.json").read_text(encoding="utf-8"))
    assert reference_metrics["config"]["workflow_mode"] == "reference"
    assert reference_metrics["config"]["voting"]["voter_thin_mode"] == "reference"
    assert reference_metrics["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert reference_metrics["config"]["voting"]["surface_support_exponent"] == 0.0
    assert quality_metrics["config"]["workflow_mode"] == "quality"
    assert quality_metrics["config"]["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert quality_metrics["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert quality_metrics["config"]["voting"]["surface_support_exponent"] == 0.0
    assert quality_metrics["config"]["skinning"]["boundary_skinner_fallback"] is True

    with (reference_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        reference_rows = list(csv.DictReader(file))
    with (quality_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        quality_rows = list(csv.DictReader(file))

    def curved_oracle_row(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
        return next(
            row
            for row in rows
            if row["case_id"] == "curved_surface"
            and row["pipeline"] == "oracle"
            and row["variant"] == variant
        )

    reference_current = curved_oracle_row(reference_rows, "current_default")
    quality_current = curved_oracle_row(quality_rows, "current_default")

    quality_fields = (
        "fvt_buffered_f1_r2",
        "fvt_distance_p95",
        "fvt_strike_median_error",
        "fvt_dip_median_error",
    )
    for row in (reference_current, quality_current):
        for field in quality_fields:
            assert math.isfinite(float(row[field]))
        assert float(row["fvt_nonzero_fraction"]) > 0.0


@pytest.mark.parametrize("override", [None, "normal"])
def test_report_3d_synthetic_quality_workflow_mode_diagnostic_enables_thinning_diagnostic(
    tmp_path: Path,
    override: str | None,
) -> None:
    output_dir = tmp_path / f"synthetic_quality_diagnostic_{override or 'default'}"
    args = [
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "diagnostic",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    ]
    if override is not None:
        args.extend(["--voter-thin-mode", override])

    result = _run_script(*args)

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "diagnostic"
    assert metrics["config"]["thinning_diagnostic"] == {"enabled": True}
    assert metrics["config"]["voting"]["voter_thin_mode"] == (override or "reference")
    curved = next(case for case in metrics["cases"] if case["case_id"] == "curved_surface")
    assert "thinning_diagnostic" in curved["variants"]["current_default"]


def test_report_3d_synthetic_quality_quality_workflow_records_mode_and_defaults(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "quality"
    assert metrics["config"]["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert metrics["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert metrics["config"]["voting"]["surface_support_exponent"] == 0.0
    assert metrics["config"]["skinning"]["method"] == "quality"
    assert metrics["config"]["skinning"]["growth_source"] == "pre_thin"
    assert metrics["config"]["skinning"]["min_likelihood"] is None
    assert metrics["config"]["skinning"]["adaptive_min_likelihood"] is True
    assert metrics["config"]["skinning"]["accepted_occupancy_radius"] == 1
    assert metrics["config"]["skinning"]["effective_accepted_occupancy_radius"] == 1

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["workflow_mode"] == "quality"


def test_report_3d_synthetic_quality_quality_workflow_explicit_skinner_knobs_win(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--skinner-min-likelihood",
        "0.5",
        "--skinner-growth-source",
        "thinned",
        "--skinner-accepted-occupancy-radius",
        "none",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "quality"
    assert metrics["config"]["skinning"]["method"] == "quality"
    assert metrics["config"]["skinning"]["growth_source"] == "thinned"
    assert metrics["config"]["skinning"]["min_likelihood"] == 0.5
    assert metrics["config"]["skinning"]["adaptive_min_likelihood"] is False
    assert metrics["config"]["skinning"]["accepted_occupancy_radius"] is None
    assert metrics["config"]["skinning"]["effective_accepted_occupancy_radius"] == 5


def test_report_3d_synthetic_quality_quality_workflow_explicit_skinner_equals_form_wins(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--skinner-min-likelihood=0.5",
        "--skinner-growth-source=thinned",
        "--skinner-accepted-occupancy-radius=none",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "quality"
    assert metrics["config"]["skinning"]["method"] == "quality"
    assert metrics["config"]["skinning"]["growth_source"] == "thinned"
    assert metrics["config"]["skinning"]["min_likelihood"] == 0.5
    assert metrics["config"]["skinning"]["adaptive_min_likelihood"] is False
    assert metrics["config"]["skinning"]["accepted_occupancy_radius"] is None
    assert metrics["config"]["skinning"]["effective_accepted_occupancy_radius"] == 5


def test_report_3d_synthetic_quality_quality_workflow_explicit_skinner_reference_wins(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--skinner-method",
        "reference",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "quality"
    assert metrics["config"]["skinning"]["method"] == "reference"
    assert metrics["config"]["skinning"]["min_likelihood"] == 0.5
    assert metrics["config"]["skinning"]["adaptive_min_likelihood"] is False


def test_report_3d_synthetic_quality_skinner_method_quality_smoke(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--skinner-method",
        "quality",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["skinning"]["method"] == "quality"
    assert metrics["cases"][0]["quality"]["skin"] is not None


def test_report_3d_synthetic_quality_quality_workflow_support_cli_override(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--surface-support-min-fraction",
        "0.25",
        "--surface-support-exponent",
        "2.0",
        "--output-dir",
        str(output_dir),
        "--skip-skinning",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "quality"
    assert metrics["config"]["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert metrics["config"]["voting"]["surface_support_min_fraction"] == 0.25
    assert metrics["config"]["voting"]["surface_support_exponent"] == 2.0


def test_report_3d_synthetic_quality_diagnostic_workflow_records_mode_and_defaults(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "diagnostic",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["workflow_mode"] == "diagnostic"
    assert metrics["config"]["voting"]["voter_thin_mode"] == "reference"
    assert metrics["config"]["voting"]["surface_support_min_fraction"] == 0.0
    assert metrics["config"]["voting"]["surface_support_exponent"] == 0.0
    assert metrics["config"]["thinning_diagnostic"] == {"enabled": True}


def test_report_3d_synthetic_quality_invalid_workflow_mode_fails(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "invalid",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode != 0
    assert "workflow-mode" in result.stderr or "workflow_mode" in result.stderr


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
        "--surface-support-min-fraction",
        "0.25",
        "--surface-support-exponent",
        "2.0",
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
        "surface_support_min_fraction": 0.25,
        "surface_support_exponent": 2.0,
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
        "--skinner-accepted-occupancy-radius",
        "1",
        "--skinner-growth-source",
        "pre_thin",
        "--small-skin-size",
        "5",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["skinning"] == {
        "enabled": True,
        "method": "reference",
        "growth_source": "pre_thin",
        "min_likelihood": 0.4,
        "adaptive_min_likelihood": False,
        "seed_min_ep": 0.8,
        "seed_planarity_source": "fvt",
        "min_skin_size": 2,
        "d": 2,
        "ru": 6,
        "rv": 7,
        "rw": 8,
        "max_steps": 3,
        "du": 4.5,
        "max_delta_strike": 20.0,
        "reskin": False,
        "accepted_occupancy_radius": 1,
        "effective_accepted_occupancy_radius": 1,
        "small_skin_size": 5,
        "boundary_skinner_fallback": False,
        "boundary_skinner_fallback_policy": "empty_primary",
    }


def test_report_3d_synthetic_quality_invalid_growth_source_fails() -> None:
    module = _load_report_module()

    with pytest.raises(ValueError, match="skinner_growth_source must be one of"):
        module.SyntheticSkinningConfig(growth_source="dense")


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
    assert metrics["config"]["workflow_mode"] == "reference"
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
    assert metrics["config"]["workflow_mode"] == "reference"
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
        "refinement_factor": 2,
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


def test_synthetic_scanner_config_accepts_quality_backend() -> None:
    module = _load_report_module()

    config = module.SyntheticScannerConfig(backend="quality", refinement_factor=2)

    assert config.backend == "quality"
    assert config.as_report_dict()["refinement_factor"] == 2


def test_synthetic_scanner_config_accepts_ensemble_backend() -> None:
    module = _load_report_module()

    config = module.SyntheticScannerConfig(backend="ensemble", refinement_factor=2)

    assert config.backend == "ensemble"
    assert config.as_report_dict()["backend"] == "ensemble"


def test_report_synthetic_quality_scanner_backend_quality_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--scanner-backend",
        "quality",
        "--scanner-refinement-factor",
        "2",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert variant["scanner"]["config"]["backend"] == "quality"
    assert variant["scanner"]["config"]["refinement_factor"] == 2
    assert variant["scanner"]["confidence"]["shape"] == [17, 17, 17]
    assert variant["scanner"]["confidence"]["finite_fraction"] == 1.0
    assert variant["pyosv"]["fvt"]["finite_fraction"] == 1.0


def test_report_synthetic_quality_scanner_backend_ensemble_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--workflow-mode",
        "quality",
        "--scanner-backend",
        "ensemble",
        "--scanner-refinement-factor",
        "2",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    scanner = variant["scanner"]
    assert scanner["config"]["backend"] == "ensemble"
    assert scanner["ft"]["finite_fraction"] == 1.0
    assert scanner["pt"]["finite_fraction"] == 1.0
    assert scanner["tt"]["finite_fraction"] == 1.0
    assert scanner["ft"]["shape"] == [17, 17, 17]
    assert scanner["pt"]["shape"] == [17, 17, 17]
    assert scanner["tt"]["shape"] == [17, 17, 17]

    fractions = scanner["selection_fraction_by_backend"]
    assert set(fractions) == {"reference-like", "quality", "fast"}
    assert math.isclose(sum(float(value) for value in fractions.values()), 1.0, abs_tol=1e-6)
    assert scanner["ensemble"]["selection_fraction_by_backend"] == fractions
    assert set(scanner["ensemble"]["components"]) == {"reference-like", "quality", "fast"}
    assert "confidence" in scanner["ensemble"]["components"]["quality"]
    assert variant["pyosv"]["fvt"]["finite_fraction"] == 1.0

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    assert row["scanner_backend"] == "ensemble"
    csv_fraction_sum = sum(
        float(row[field])
        for field in (
            "scanner_ensemble_reference_like_fraction",
            "scanner_ensemble_quality_fraction",
            "scanner_ensemble_fast_fraction",
        )
    )
    assert math.isclose(csv_fraction_sum, 1.0, abs_tol=1e-6)


def test_scanner_backend_matrix_scanner_mode_reports_all_backends(
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
        "--scanner-backend-matrix",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    matrix = variant["scanner_backend_matrix"]
    assert metrics["config"]["scanner_backend_matrix"] is True
    assert variant["scanner"]["config"]["backend"] == "fast"
    _assert_scanner_backend_matrix_contract(matrix, expected_selected_backend="fast")


def test_scanner_backend_matrix_ensemble_selected_has_baseline(
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
        "ensemble",
        "--scanner-backend-matrix",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    matrix = variant["scanner_backend_matrix"]
    assert variant["scanner"]["config"]["backend"] == "ensemble"
    _assert_scanner_backend_matrix_contract(matrix, expected_selected_backend="ensemble")


def test_scanner_backend_matrix_both_mode_lives_on_scanner_pipeline(
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
        "--scanner-backend-matrix",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    scanner_pipeline = variant["pipelines"]["scanner"]
    assert metrics["config"]["scanner_backend_matrix"] is True
    assert "scanner_backend_matrix" not in variant
    _assert_scanner_backend_matrix_contract(scanner_pipeline["scanner_backend_matrix"])


def test_scanner_backend_matrix_oracle_mode_is_noop(
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
        "--scanner-backend-matrix",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["scanner_backend_matrix"] is False
    assert "scanner_backend_matrix" not in variant
    assert "scanner_quality" not in variant


def test_scanner_backend_matrix_summary_csv_columns(
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
        "--scanner-backend-matrix",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    row = rows[0]
    for field in (
        "scanner_matrix_best_fvt_positive_buffered_f1_backend",
        "scanner_matrix_best_skin_buffered_f1_backend",
        "scanner_matrix_best_boundary_edge_fp_backend",
    ):
        assert field in row
        assert row[field] in EXPECTED_SCANNER_BACKEND_MATRIX_BACKENDS


def test_scanner_backend_matrix_default_off_keeps_report_without_matrix(
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
    assert metrics["config"]["scanner_backend_matrix"] is False
    assert "scanner_backend_matrix" not in variant

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["scanner_matrix_best_fvt_positive_buffered_f1_backend"] == ""
    assert rows[0]["scanner_matrix_best_skin_buffered_f1_backend"] == ""
    assert rows[0]["scanner_matrix_best_boundary_edge_fp_backend"] == ""


def test_scanner_downstream_diagnostics_are_opt_in_and_do_not_change_outputs(
    tmp_path: Path,
) -> None:
    plain_output_dir = tmp_path / "scanner_plain"
    diagnostic_output_dir = tmp_path / "scanner_downstream"
    common_args = (
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "scanner",
        "--workflow-mode",
        "quality",
        "--variants",
        "current_default",
    )

    plain_result = _run_script(
        *common_args,
        "--output-dir",
        str(plain_output_dir),
    )
    diagnostic_result = _run_script(
        *common_args,
        "--scanner-downstream-diagnostics",
        "--output-dir",
        str(diagnostic_output_dir),
    )

    assert plain_result.returncode == 0, plain_result.stderr
    assert diagnostic_result.returncode == 0, diagnostic_result.stderr
    plain_metrics = json.loads((plain_output_dir / "metrics.json").read_text(encoding="utf-8"))
    diagnostic_metrics = json.loads(
        (diagnostic_output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    plain_variant = plain_metrics["cases"][0]["variants"]["current_default"]
    diagnostic_variant = diagnostic_metrics["cases"][0]["variants"]["current_default"]

    assert plain_metrics["config"]["scanner_downstream_diagnostics"] is False
    assert diagnostic_metrics["config"]["scanner_downstream_diagnostics"] is True
    assert "scanner_downstream" not in plain_variant
    assert "scanner_stage_loss" not in plain_variant
    assert "scanner_downstream" in diagnostic_variant
    assert "scanner_stage_loss" in diagnostic_variant
    _assert_scanner_downstream_contract(diagnostic_variant["scanner_downstream"])
    _assert_scanner_stage_loss_contract(diagnostic_variant["scanner_stage_loss"])
    assert diagnostic_variant["scanner_downstream"]["voter_thin_mode"] == "hybrid_v2"
    assert diagnostic_variant["scanner_downstream"]["plateau_tie_breaker_source"] == "scanner_fet"

    assert math.isclose(
        float(plain_variant["pyosv"]["fvt"]["max"]),
        float(diagnostic_variant["pyosv"]["fvt"]["max"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert plain_variant["pyosv"]["skins"] == diagnostic_variant["pyosv"]["skins"]
    assert math.isclose(
        float(
            plain_variant["quality"]["fvt_positive_top_truth_count"]["buffered_overlap_radius2"][
                "buffered_f1"
            ]
        ),
        float(
            diagnostic_variant["quality"]["fvt_positive_top_truth_count"][
                "buffered_overlap_radius2"
            ]["buffered_f1"]
        ),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    with (diagnostic_output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    for field in (
        "scanner_downstream_ft_positive_candidate_count",
        "scanner_downstream_fet_positive_candidate_count",
        "scanner_downstream_ft_to_fet_retention_fraction",
        "scanner_downstream_fv_positive_candidate_count",
        "scanner_downstream_fvt_positive_candidate_count",
        "scanner_downstream_fvt_to_fv_positive_fraction",
        "scanner_downstream_fvt_positive_edge_candidate_fraction",
        "scanner_downstream_fvt_positive_edge_false_positive_fraction",
        "scanner_downstream_scanner_ft_positive_count",
        "scanner_downstream_scanner_fet_positive_count",
        "scanner_downstream_fv_positive_count",
        "scanner_downstream_fvt_positive_count",
        "scanner_downstream_ft_to_fvt_overlap_f1",
        "scanner_downstream_fvt_to_ft_distance_p95",
        "scanner_downstream_fvt_edge_shell_fraction",
        "scanner_downstream_fv_to_fvt_positive_ratio",
        "scanner_downstream_reference_fvt_positive_buffered_f1_r2",
        "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2",
        "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2",
        "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2",
    ):
        assert row[field] != ""
        assert math.isfinite(float(row[field]))
    assert row["scanner_downstream_voter_thin_mode"] == "hybrid_v2"
    assert row["scanner_downstream_plateau_tie_breaker_source"] == "scanner_fet"
    assert row["scanner_downstream_scanner_thin_mode"] == "reference"


def test_scanner_stage_loss_diagnostics_json_and_summary_contract(tmp_path: Path) -> None:
    module = _load_report_module()
    scanner_config = module.SyntheticScannerConfig(backend="quality", refinement_factor=2)
    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        input_mode="scanner",
        workflow_mode="quality",
        variants=("current_default",),
        include_scanner_downstream_diagnostics=True,
        scanner_config=scanner_config,
    )
    variant = report["cases"][0]["variants"]["current_default"]

    assert "scanner_stage_loss" in variant
    _assert_scanner_stage_loss_contract(variant["scanner_stage_loss"])

    summary_path = module.write_summary_csv(report, tmp_path)
    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    scanner_stage_fields = (
        "scanner_stage_ft_positive_count",
        "scanner_stage_fet_positive_count",
        "scanner_stage_seed_candidate_count",
        "scanner_stage_seed_selected_count",
        "scanner_stage_fv_positive_count",
        "scanner_stage_fvt_positive_count",
        "scanner_stage_skin_count",
        "scanner_stage_ft_truth_f1_r2",
        "scanner_stage_fet_truth_f1_r2",
        "scanner_stage_seed_selected_truth_f1_r2",
        "scanner_stage_fv_truth_f1_r2",
        "scanner_stage_fvt_truth_f1_r2",
        "scanner_stage_skin_truth_f1_r2",
        "scanner_stage_ft_to_fet_ratio",
        "scanner_stage_fet_to_seed_selected_f1_r2",
        "scanner_stage_seed_selected_to_fv_f1_r2",
        "scanner_stage_fv_to_fvt_ratio",
        "scanner_stage_fvt_to_skin_f1_r2",
        "scanner_stage_fvt_to_skin_distance_p95",
    )
    for field in scanner_stage_fields:
        assert field in row
        assert row[field] != ""
        assert math.isfinite(float(row[field]))

    plain_report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        input_mode="scanner",
        workflow_mode="quality",
        variants=("current_default",),
        include_scanner_downstream_diagnostics=False,
        scanner_config=scanner_config,
    )
    plain_variant = plain_report["cases"][0]["variants"]["current_default"]
    assert "scanner_stage_loss" not in plain_variant
    plain_summary_path = module.write_summary_csv(plain_report, tmp_path / "plain")
    with plain_summary_path.open(encoding="utf-8", newline="") as file:
        plain_row = next(csv.DictReader(file))
    for field in scanner_stage_fields:
        assert plain_row[field] == ""

    oracle_report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        input_mode="oracle",
        workflow_mode="quality",
        variants=("current_default",),
        include_scanner_downstream_diagnostics=True,
        scanner_config=scanner_config,
    )
    oracle_variant = oracle_report["cases"][0]["variants"]["current_default"]
    assert "scanner_stage_loss" not in oracle_variant
    oracle_summary_path = module.write_summary_csv(oracle_report, tmp_path / "oracle")
    with oracle_summary_path.open(encoding="utf-8", newline="") as file:
        oracle_row = next(csv.DictReader(file))
    for field in scanner_stage_fields:
        assert oracle_row[field] == ""


def test_boundary_seed_retention_adds_edge_target_seed_not_selected_by_default() -> None:
    module = _load_report_module()
    voting_config = module.SyntheticVotingConfig(seed_distance=2, seed_threshold=0.5)
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    target = np.zeros_like(ft)
    ft[2, 2, 2] = 0.9
    ft[0, 2, 2] = 0.4
    target[0, 2, 2] = 1.0

    default_seeds, retained_seeds, diagnostic = module._boundary_seed_retention_v1_seeds(
        voting_config=voting_config,
        ft=ft,
        pt=pt,
        tt=tt,
        target=target,
        target_source="scanner_fet",
        edge_margin=2,
    )

    assert [(seed.i1, seed.i2, seed.i3) for seed in default_seeds] == [(2, 2, 2)]
    assert [(seed.i1, seed.i2, seed.i3) for seed in retained_seeds] == [
        (2, 2, 2),
        (2, 2, 0),
    ]
    assert diagnostic["default_seed_count"] == 1
    assert diagnostic["boundary_candidate_count"] == 1
    assert diagnostic["added_seed_count"] == 1
    assert diagnostic["total_seed_count"] == 2
    assert diagnostic["added_seed_edge_shell_fraction"] == pytest.approx(1.0)
    assert diagnostic["added_seed_target_mean"] == pytest.approx(1.0)


def test_boundary_seed_retention_report_diagnostic_and_summary_columns(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    scanner_config = module.SyntheticScannerConfig(backend="quality", refinement_factor=2)
    report = module.build_report(
        case_set="minimal",
        shape=(17, 17, 17),
        input_mode="scanner",
        workflow_mode="quality",
        variants=("boundary_seed_retention_v1",),
        include_scanner_downstream_diagnostics=True,
        scanner_config=scanner_config,
    )
    variant = report["cases"][0]["variants"]["boundary_seed_retention_v1"]
    diagnostic = variant["boundary_seed_retention"]

    assert diagnostic["enabled"] is True
    assert diagnostic["target_source"] == "scanner_fet"
    assert diagnostic["edge_margin"] == 2
    assert diagnostic["total_seed_count"] == (
        diagnostic["default_seed_count"] + diagnostic["added_seed_count"]
    )
    seed_stage = variant["scanner_stage_loss"]["stages"]["seed_selected"]
    assert seed_stage["candidate_count"] == diagnostic["total_seed_count"]
    assert seed_stage["default_candidate_count"] == diagnostic["default_seed_count"]
    assert seed_stage["added_candidate_count"] == diagnostic["added_seed_count"]

    summary_path = module.write_summary_csv(report, tmp_path)
    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    assert row["boundary_seed_retention_enabled"] == "True"
    assert row["boundary_seed_retention_target_source"] == "scanner_fet"
    assert (
        int(row["boundary_seed_retention_default_seed_count"]) == diagnostic["default_seed_count"]
    )
    assert (
        int(row["boundary_seed_retention_boundary_candidate_count"])
        == diagnostic["boundary_candidate_count"]
    )
    assert int(row["boundary_seed_retention_added_seed_count"]) == diagnostic["added_seed_count"]
    assert int(row["boundary_seed_retention_total_seed_count"]) == diagnostic["total_seed_count"]


def test_scanner_downstream_diagnostics_both_mode_lives_on_scanner_pipeline(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "both_scanner_downstream"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--input-mode",
        "both",
        "--scanner-downstream-diagnostics",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    variant = metrics["cases"][0]["variants"]["current_default"]
    assert metrics["config"]["scanner_downstream_diagnostics"] is True
    assert "scanner_downstream" not in variant
    assert "scanner_stage_loss" not in variant
    assert "scanner_downstream" not in variant["pipelines"]["oracle"]
    assert "scanner_stage_loss" not in variant["pipelines"]["oracle"]
    scanner_downstream = variant["pipelines"]["scanner"]["scanner_downstream"]
    _assert_scanner_downstream_contract(scanner_downstream)
    scanner_stage_loss = variant["pipelines"]["scanner"]["scanner_stage_loss"]
    _assert_scanner_stage_loss_contract(scanner_stage_loss)

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    rows_by_pipeline = {row["pipeline"]: row for row in rows}
    assert rows_by_pipeline["oracle"]["scanner_downstream_scanner_ft_positive_count"] == ""
    assert rows_by_pipeline["oracle"]["scanner_stage_ft_positive_count"] == ""
    scanner_row = rows_by_pipeline["scanner"]
    assert scanner_row["scanner_downstream_scanner_ft_positive_count"] != ""
    assert math.isfinite(float(scanner_row["scanner_downstream_ft_to_fvt_overlap_f1"]))
    assert scanner_row["scanner_stage_ft_positive_count"] != ""
    assert math.isfinite(float(scanner_row["scanner_stage_fvt_to_skin_distance_p95"]))


def test_synthetic_quality_comparison_helper_prints_selected_columns(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    comparison_dir = tmp_path / "comparison"
    fieldnames = (
        "case_id",
        "pipeline",
        "variant",
        "fvt_positive_buffered_f1_r2",
        "skin_buffered_f1_r2",
        "scanner_ft_buffered_f1_r2",
        "scanner_downstream_fvt_to_ft_distance_p95",
        "skin_fallback_used",
    )
    rows = (
        {
            "case_id": "boundary_plane",
            "pipeline": "oracle",
            "variant": "current_default",
            "fvt_positive_buffered_f1_r2": "0.993",
            "skin_buffered_f1_r2": "0.993",
        },
        {
            "case_id": "boundary_plane",
            "pipeline": "scanner",
            "variant": "current_default",
            "fvt_positive_buffered_f1_r2": "0.739",
            "skin_buffered_f1_r2": "0.454",
            "scanner_ft_buffered_f1_r2": "1.0",
            "scanner_downstream_fvt_to_ft_distance_p95": "0.0",
            "skin_fallback_used": "False",
        },
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARISON_SCRIPT.relative_to(REPO_ROOT)),
            str(summary_csv),
            "--variant",
            "current_default",
            "--output-dir",
            str(comparison_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_rows = list(csv.DictReader(result.stdout.splitlines()))
    assert output_rows == [
        {
            "case_id": "boundary_plane",
            "variant": "current_default",
            "oracle_fvt_positive_f1": "0.993",
            "scanner_fvt_positive_f1": "0.739",
            "delta_fvt": "-0.254",
            "oracle_skin_f1": "0.993",
            "scanner_skin_f1": "0.454",
            "delta_skin": "-0.539",
            "scanner_ft_f1": "1.0",
            "scanner_downstream_fvt_to_ft_distance_p95": "0.0",
            "fallback_used": "False",
        }
    ]
    with (comparison_dir / "synthetic_quality_comparison.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        assert list(csv.DictReader(file)) == output_rows


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
    assert row["scanner_refinement_factor"] == "2"
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
    variant = metrics["cases"][0]["variants"]["current_default"]
    skin_quality = variant["quality"]["skin"]
    diagnostics = variant["skinning"]["diagnostics"]
    assert "buffered_overlap_radius2" in skin_quality
    assert diagnostics["accepted_skin_count"] == skin_quality["topology"]["skin_count"]
    for field in (
        "seed_candidate_count_before_spacing",
        "seed_count_after_spacing",
        "seed_count_rejected_by_occupied",
        "grow_attempt_count",
        "grown_skin_count_before_min_size",
        "discarded_empty_skin_count",
        "discarded_small_skin_count",
        "accepted_skin_count",
        "accepted_cell_count",
        "accepted_occupancy_radius",
        "seed_min_ep",
        "seed_threshold",
        "grow_threshold",
    ):
        assert math.isfinite(float(diagnostics[field]))

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert math.isfinite(float(rows[0]["skin_buffered_f1_r2"]))
    assert rows[0]["skin_accepted_count"] == str(diagnostics["accepted_skin_count"])
    for field in (
        "skin_primary_count",
        "skin_primary_cell_count",
        "skin_primary_unique_cell_count",
        "skin_primary_largest_size",
        "skin_primary_largest_fraction",
        "skin_primary_small_count",
        "skin_primary_small_cell_fraction",
        "skin_primary_cell_coverage_of_fvt_positive",
        "skin_primary_largest_coverage_of_fvt_positive",
        "skin_primary_edge_shell_fraction",
        "skin_fvt_positive_edge_shell_fraction",
    ):
        assert math.isfinite(float(diagnostics[field]))
        assert math.isfinite(float(rows[0][field]))
    assert isinstance(diagnostics["skin_primary_degraded_candidate"], bool)
    assert isinstance(diagnostics["skin_primary_degraded_reasons"], list)
    assert isinstance(diagnostics["skin_primary_boundary_degraded_candidate"], bool)
    assert isinstance(diagnostics["skin_primary_boundary_degraded_reasons"], list)
    assert rows[0]["skin_primary_degraded_candidate"] in {"True", "False"}
    assert rows[0]["skin_primary_degraded_reasons"] == ",".join(
        diagnostics["skin_primary_degraded_reasons"]
    )
    assert rows[0]["skin_primary_boundary_degraded_candidate"] in {"True", "False"}
    assert rows[0]["skin_primary_boundary_degraded_reasons"] == ",".join(
        diagnostics["skin_primary_boundary_degraded_reasons"]
    )


def test_report_3d_synthetic_quality_primary_skin_diagnostics_mark_degraded() -> None:
    module = _load_report_module()
    diagnostics: dict[str, object] = {}
    skins = [
        _fault_skin([(index, 0, 0)])
        for index in range(module.SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT)
    ]

    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=(3, 3, 20),
        fvt_positive_candidate_count=40,
        small_skin_size=10,
    )

    assert diagnostics["skin_primary_degraded_candidate"] is True
    assert diagnostics["skin_primary_count"] == 8
    assert diagnostics["skin_primary_unique_cell_count"] == 8
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == pytest.approx(0.2)
    assert diagnostics["skin_primary_largest_coverage_of_fvt_positive"] == pytest.approx(0.025)
    assert diagnostics["skin_primary_degraded_reasons"] == [
        "low_fvt_positive_coverage",
        "fragmented_primary_skins",
        "high_small_skin_cell_fraction",
    ]


def test_report_3d_synthetic_quality_primary_skin_diagnostics_keep_healthy() -> None:
    module = _load_report_module()
    diagnostics: dict[str, object] = {}
    skins = [_fault_skin([(index, 0, 0) for index in range(8)])]

    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=(3, 3, 20),
        fvt_positive_candidate_count=8,
        small_skin_size=5,
    )

    assert diagnostics["skin_primary_degraded_candidate"] is False
    assert diagnostics["skin_primary_degraded_reasons"] == []
    assert diagnostics["skin_primary_count"] == 1
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == pytest.approx(1.0)
    assert diagnostics["skin_primary_largest_fraction"] == pytest.approx(1.0)


def test_report_3d_synthetic_quality_degraded_primary_policy_triggers_fallback() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 5), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [_fault_skin([(0, 0, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=5,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v2",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert "fvt_positive_edge_shell" in diagnostics["skin_primary_boundary_degraded_reasons"]
    assert diagnostics["fallback_replaced_primary"] is True
    assert diagnostics["fallback_primary_skin_count"] == 1
    assert diagnostics["fallback_primary_cell_count"] == 1
    assert diagnostics["fallback_candidate_count"] == 5
    assert diagnostics["fallback_coverage_before"] == pytest.approx(0.2)
    assert diagnostics["fallback_coverage_after"] == pytest.approx(1.0)
    assert diagnostics["fallback_degraded_reasons"] == ["low_fvt_positive_coverage"]
    assert diagnostics["fallback_reason"] == "degraded_primary:undercovered"
    assert len(skins) == 1
    assert len(skins[0]) == 5


def test_report_3d_synthetic_quality_degraded_primary_policy_blocks_non_boundary() -> None:
    module = _load_report_module()
    fvt = np.zeros((7, 7, 7), dtype=np.float32)
    fvt[3, 3, 2:5] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [_fault_skin([(3, 3, 3)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=3,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v2",
        diagnostics=diagnostics,
    )

    assert diagnostics["skin_primary_degraded_candidate"] is True
    assert diagnostics["skin_primary_degraded_reasons"] == ["low_fvt_positive_coverage"]
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is False
    assert diagnostics["skin_primary_boundary_degraded_reasons"] == []
    assert diagnostics["skin_fvt_positive_edge_shell_fraction"] == pytest.approx(0.0)
    assert diagnostics["skin_primary_edge_shell_fraction"] == pytest.approx(0.0)
    assert diagnostics["fallback_policy"] == "degraded_primary"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_boundary_degraded_not_detected"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_degraded_reasons"] == ["low_fvt_positive_coverage"]
    assert len(skins) == 1
    assert len(skins[0]) == 1


def test_report_3d_synthetic_quality_filtered_degraded_primary_policy_filters_fallback() -> None:
    module = _load_report_module()
    fvt = np.zeros((1, 1, 20), dtype=np.float32)
    fvt[0, 0, 0:10] = 1.0
    fvt[0, 0, 12:14] = 1.0
    fvt[0, 0, 16] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [_fault_skin([(0, 0, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=13,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v3",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert diagnostics["fallback_replaced_primary"] is True
    assert diagnostics["fallback_candidate_count"] == 13
    assert diagnostics["fallback_coverage_before"] == pytest.approx(1 / 13)
    assert diagnostics["fallback_coverage_after"] == pytest.approx(10 / 13)
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert diagnostics["skin_fallback_component_count"] == 3
    assert diagnostics["skin_fallback_accepted_component_count"] == 1
    assert diagnostics["skin_fallback_discarded_component_count"] == 2
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 10
    assert diagnostics["skin_fallback_filter_min_component_size"] == 8
    assert diagnostics["skin_fallback_filter_min_component_fraction_of_largest"] == pytest.approx(
        0.10
    )
    assert diagnostics["skin_fallback_filter_max_components"] == 3
    assert len(skins) == 1
    assert len(skins[0]) == 10
    assert {(cell.i1, cell.i2, cell.i3) for cell in skins[0]} == {
        (index, 0, 0) for index in range(10)
    }


def test_report_3d_synthetic_quality_skeletonized_degraded_primary_policy_prunes_fallback() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 3, 20), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    scanner_target_positive_mask = np.zeros_like(fvt, dtype=bool)
    scanner_target_positive_mask[:, 1, :] = True
    skins = [_fault_skin([(0, 1, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=60,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_skeletonized",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v4",
        diagnostics=diagnostics,
        scanner_target_positive_mask=scanner_target_positive_mask,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary_skeletonized"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_skeletonized"
    assert diagnostics["skin_fallback_accepted_component_count"] == 1
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 60
    assert diagnostics["skin_fallback_pruning_method"] == "fault_normal_line_collapse"
    assert diagnostics["skin_fallback_raw_component_cell_count"] == 60
    assert diagnostics["skin_fallback_pruned_component_cell_count"] == 20
    assert diagnostics["skin_fallback_pruned_fraction"] == pytest.approx(20 / 60)
    assert diagnostics["skin_fallback_largest_component_size_before_pruning"] == 60
    assert diagnostics["skin_fallback_largest_component_size_after_pruning"] == 20
    assert diagnostics["skin_fallback_pruning_removed_cell_count"] == 40
    assert diagnostics["skin_fallback_skeletonization_axis_mode"] == "i2"
    assert diagnostics["fallback_coverage_after"] == pytest.approx(20 / 60)
    assert len(skins) == 1
    assert len(skins[0]) == 20
    assert {(cell.i1, cell.i2, cell.i3) for cell in skins[0]} == {
        (index, 1, 0) for index in range(20)
    }


def test_report_3d_synthetic_quality_skeletonized_policy_requires_scanner_target() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 3, 20), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    skins = [_fault_skin([(0, 1, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=60,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_skeletonized",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v4",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary_skeletonized"
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_boundary_degraded_not_sufficient"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_replaced_primary"] is False
    assert len(skins) == 1
    assert len(skins[0]) == 1


def test_report_3d_synthetic_quality_v5_guardrail_blocks_fragmented_fallback() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 3, 20), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    scanner_target_positive_mask = np.zeros_like(fvt, dtype=bool)
    scanner_target_positive_mask[:, 1, :] = True
    skins = [_fault_skin([(0, 1, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=60,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_topology_guarded",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v5",
        diagnostics=diagnostics,
        scanner_target_positive_mask=scanner_target_positive_mask,
    )

    guardrail = diagnostics["fallback_v5_guardrail"]
    assert diagnostics["fallback_policy"] == "degraded_primary_topology_guarded"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "fallback_v5_guardrail_failed"
    assert diagnostics["fallback_replaced_primary"] is False
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert guardrail["enabled"] is True
    assert guardrail["passed"] is False
    assert "coverage_of_fvt_positive_below_min" in guardrail["reasons"]
    assert "pruned_fraction_exceeds_max" in guardrail["reasons"]
    assert guardrail["fallback_skin_count"] == 1
    assert guardrail["coverage_of_fvt_positive"] == pytest.approx(20 / 60)
    assert guardrail["pruned_fraction"] == pytest.approx(40 / 60)
    assert len(skins) == 1
    assert len(skins[0]) == 1


def test_report_3d_synthetic_quality_v5_rejects_empty_primary_fallback() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 20), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    scanner_target_positive_mask = np.ones_like(fvt, dtype=bool)
    skins: list[object] = []
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=20,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_topology_guarded",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v5",
        diagnostics=diagnostics,
        scanner_target_positive_mask=scanner_target_positive_mask,
    )

    guardrail = diagnostics["fallback_v5_guardrail"]
    assert diagnostics["fallback_policy"] == "degraded_primary_topology_guarded"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "empty_primary_not_supported_by_topology_guarded"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_replaced_primary"] is False
    assert diagnostics["fallback_skin_count"] == 0
    assert diagnostics["fallback_degraded_reasons"] == [
        "empty_primary_skin",
        "low_fvt_positive_coverage",
    ]
    assert guardrail["enabled"] is True
    assert guardrail["passed"] is False
    assert guardrail["reasons"] == ["empty_primary_not_supported"]
    assert skins == []


def test_report_3d_synthetic_quality_v5_guardrail_passes_and_replaces_primary() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 20), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    scanner_target_positive_mask = np.ones_like(fvt, dtype=bool)
    skins = [_fault_skin([(0, 0, 0)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=20,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_topology_guarded",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v5",
        diagnostics=diagnostics,
        scanner_target_positive_mask=scanner_target_positive_mask,
    )

    guardrail = diagnostics["fallback_v5_guardrail"]
    assert diagnostics["fallback_policy"] == "degraded_primary_topology_guarded"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_reason"] == "degraded_primary:undercovered"
    assert diagnostics["fallback_replaced_primary"] is True
    assert guardrail["enabled"] is True
    assert guardrail["passed"] is True
    assert guardrail["reasons"] == []
    assert guardrail["fallback_skin_count"] == 1
    assert guardrail["coverage_of_fvt_positive"] == pytest.approx(1.0)
    assert guardrail["largest_skin_fraction"] == pytest.approx(1.0)
    assert guardrail["small_skin_cell_fraction"] == pytest.approx(0.0)
    assert guardrail["pruned_fraction"] == pytest.approx(0.0)
    assert len(skins) == 1
    assert len(skins[0]) == 20


def test_report_3d_synthetic_quality_filtered_degraded_primary_triggers_on_skin_count() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 100), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [_fault_skin([(index, 0, 0) for index in range(93)])]
    skins.extend(_fault_skin([(index, 0, 0)]) for index in range(93, 100))
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=100,
        small_skin_size=1,
    )
    assert (
        diagnostics["skin_primary_count"] == module.SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT
    )
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == pytest.approx(1.0)
    assert (
        diagnostics["skin_primary_largest_fraction"]
        > module.SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION
    )
    assert diagnostics["skin_primary_small_cell_fraction"] == pytest.approx(0.0)

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v3",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert diagnostics["fallback_degraded_reasons"] == ["fragmented_primary_skins"]
    assert diagnostics["fallback_reason"] == "degraded_primary:fragmented"
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert len(skins) == 1
    assert len(skins[0]) == 100


def test_report_3d_synthetic_quality_filtered_degraded_primary_triggers_on_largest() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 10), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [
        _fault_skin([(index, 0, 0) for index in range(5)]),
        _fault_skin([(index, 0, 0) for index in range(5, 10)]),
    ]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=10,
        small_skin_size=1,
    )
    assert (
        diagnostics["skin_primary_count"] < module.SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT
    )
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == pytest.approx(1.0)
    assert (
        diagnostics["skin_primary_largest_fraction"]
        < module.SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION
    )
    assert diagnostics["skin_primary_small_cell_fraction"] == pytest.approx(0.0)

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v3",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert diagnostics["fallback_used"] is True
    assert diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert diagnostics["fallback_degraded_reasons"] == ["fragmented_primary_skins"]
    assert diagnostics["fallback_reason"] == "degraded_primary:fragmented"
    assert diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert len(skins) == 1
    assert len(skins[0]) == 10


def test_report_3d_synthetic_quality_scanner_boundary_v3_improves_skin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Manual 49^3 benchmark:
    # PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
    #   --case-set extended \
    #   --shape 49,49,49 \
    #   --workflow-mode quality \
    #   --variants current_default,quality_boundary_skinner_fallback_v2,quality_boundary_skinner_fallback_v3 \
    #   --input-mode scanner \
    #   --scanner-backend quality \
    #   --scanner-refinement-factor 2 \
    #   --output-dir outputs/3d/synthetic_quality/scanner_boundary_v3_49 \
    #   --pretty
    #
    # Lightweight boundary summary:
    # python - <<'PY'
    # import csv
    # from pathlib import Path
    # summary = Path("outputs/3d/synthetic_quality/scanner_boundary_v3_49/summary.csv")
    # rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    # cols = [
    #     "case_id", "variant",
    #     "fvt_positive_buffered_f1_r2",
    #     "skin_buffered_f1_r2",
    #     "skin_count",
    #     "skin_cell_count",
    #     "skin_primary_cell_coverage_of_fvt_positive",
    #     "skin_fallback_used",
    #     "skin_fallback_policy",
    #     "skin_fallback_component_policy",
    #     "skin_fallback_accepted_component_count",
    #     "skin_fallback_accepted_component_cell_count",
    # ]
    # print(",".join(cols))
    # for row in rows:
    #     if row["case_id"] == "boundary_plane":
    #         print(",".join(str(row.get(c, "")) for c in cols))
    # PY
    module = _load_report_module()
    boundary_definition = next(
        definition for definition in module.EXTENDED_CASES if definition.case_id == "boundary_plane"
    )
    monkeypatch.setitem(module.CASE_SETS, "boundary_only", (boundary_definition,))

    report = module.build_report(
        case_set="boundary_only",
        shape=(33, 33, 33),
        workflow_mode="quality",
        input_mode="scanner",
        scanner_config=module.SyntheticScannerConfig(
            backend="quality",
            refinement_factor=2,
        ),
        variants=(
            "current_default",
            "quality_boundary_skinner_fallback_v2",
            "quality_boundary_skinner_fallback_v3",
        ),
    )
    variants = report["cases"][0]["variants"]
    current = variants["current_default"]
    v2 = variants["quality_boundary_skinner_fallback_v2"]
    v3 = variants["quality_boundary_skinner_fallback_v3"]

    current_f1 = current["quality"]["skin"]["buffered_overlap_radius2"]["buffered_f1"]
    v3_f1 = v3["quality"]["skin"]["buffered_overlap_radius2"]["buffered_f1"]
    current_diagnostics = current["skinning"]["diagnostics"]
    v2_diagnostics = v2["skinning"]["diagnostics"]
    v3_diagnostics = v3["skinning"]["diagnostics"]

    assert current["config"]["skinning"]["boundary_skinner_fallback_policy"] == "empty_primary"
    assert current_diagnostics["skin_primary_count"] > 0
    assert current_diagnostics["skin_primary_degraded_candidate"] is True
    assert "low_fvt_positive_coverage" in current_diagnostics["skin_primary_degraded_reasons"]
    assert current_diagnostics["skin_primary_cell_coverage_of_fvt_positive"] < 0.50
    assert current_diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert (
        "low_primary_coverage_with_edge_local_candidates"
        in current_diagnostics["skin_primary_boundary_degraded_reasons"]
    )
    assert current_diagnostics["skin_fvt_positive_edge_shell_fraction"] > 0.0
    assert current_diagnostics["skin_primary_edge_shell_fraction"] > 0.0
    assert math.isfinite(
        float(current_diagnostics["skin_scanner_target_positive_edge_shell_fraction"])
    )
    assert math.isfinite(float(current_diagnostics["skin_fvt_to_scanner_target_distance_p95"]))
    assert current_diagnostics["fallback_used"] is False
    assert current_diagnostics["fallback_reason"] == "primary_skin_nonempty"
    assert current_diagnostics["fallback_triggered_by_degraded_primary"] is False

    assert v2_diagnostics["fallback_policy"] == "degraded_primary"
    assert v2_diagnostics["fallback_used"] is True
    assert v2_diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert v2_diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert v2_diagnostics["skin_fallback_component_policy"] == "all"

    assert v3_diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert v3_diagnostics["fallback_used"] is True
    assert v3_diagnostics["fallback_triggered_by_degraded_primary"] is True
    assert v3_diagnostics["skin_primary_boundary_degraded_candidate"] is True
    assert v3_diagnostics["skin_fallback_component_policy"] == "degraded_primary_filtered"
    assert v3_diagnostics["skin_fallback_component_count"] > 0
    assert v3_diagnostics["skin_fallback_accepted_component_count"] > 0
    assert v3_diagnostics["skin_fallback_discarded_component_count"] >= 0
    assert v3_diagnostics["skin_fallback_accepted_component_cell_count"] > 0
    assert v3_diagnostics["fallback_coverage_after"] >= v3_diagnostics["fallback_coverage_before"]
    assert v3_f1 >= current_f1 + 0.25
    assert v3_f1 >= 0.85
    assert v3["quality"]["skin"]["topology"]["skin_count"] <= 3
    assert (
        v3["quality"]["skin"]["topology"]["skin_count"]
        <= v2["quality"]["skin"]["topology"]["skin_count"]
    )

    module.write_summary_csv(report, tmp_path)
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as file:
        summary_rows = {
            row["variant"]: row
            for row in csv.DictReader(file)
            if row["case_id"] == "boundary_plane"
        }

    required_summary_columns = (
        "skin_primary_count",
        "skin_primary_cell_count",
        "skin_primary_largest_fraction",
        "skin_primary_cell_coverage_of_fvt_positive",
        "skin_primary_largest_coverage_of_fvt_positive",
        "skin_primary_edge_shell_fraction",
        "skin_fvt_positive_edge_shell_fraction",
        "skin_scanner_target_positive_edge_shell_fraction",
        "skin_fvt_to_scanner_target_distance_p95",
        "skin_primary_degraded_candidate",
        "skin_primary_degraded_reasons",
        "skin_primary_boundary_degraded_candidate",
        "skin_primary_boundary_degraded_reasons",
        "skin_fallback_policy",
        "skin_fallback_component_policy",
        "skin_fallback_component_count",
        "skin_fallback_largest_component_size",
        "skin_fallback_largest_component_fraction",
        "skin_fallback_accepted_component_count",
        "skin_fallback_discarded_component_count",
        "skin_fallback_accepted_component_cell_count",
        "skin_fallback_filter_min_component_size",
        "skin_fallback_filter_max_components",
        "skin_fallback_coverage_before",
        "skin_fallback_coverage_after",
    )
    assert set(summary_rows) == {
        "current_default",
        "quality_boundary_skinner_fallback_v2",
        "quality_boundary_skinner_fallback_v3",
    }
    for row in summary_rows.values():
        for column in required_summary_columns:
            assert column in row

    assert summary_rows["current_default"]["input_mode"] == "scanner"
    assert summary_rows["current_default"]["scanner_backend"] == "quality"
    assert summary_rows["current_default"]["skin_primary_degraded_candidate"] == "True"
    assert (
        "low_fvt_positive_coverage"
        in summary_rows["current_default"]["skin_primary_degraded_reasons"]
    )
    assert summary_rows["current_default"]["skin_primary_boundary_degraded_candidate"] == "True"
    assert (
        "low_primary_coverage_with_edge_local_candidates"
        in summary_rows["current_default"]["skin_primary_boundary_degraded_reasons"]
    )
    for column in (
        "skin_primary_edge_shell_fraction",
        "skin_fvt_positive_edge_shell_fraction",
        "skin_scanner_target_positive_edge_shell_fraction",
        "skin_fvt_to_scanner_target_distance_p95",
    ):
        assert math.isfinite(float(summary_rows["current_default"][column]))
    assert summary_rows["current_default"]["skin_fallback_policy"] == "empty_primary"
    assert summary_rows["current_default"]["skin_fallback_used"] == "False"
    assert (
        summary_rows["quality_boundary_skinner_fallback_v3"]["skin_fallback_policy"]
        == "degraded_primary_filtered"
    )
    assert (
        summary_rows["quality_boundary_skinner_fallback_v3"]["skin_fallback_component_policy"]
        == "degraded_primary_filtered"
    )
    assert (
        summary_rows["quality_boundary_skinner_fallback_v3"][
            "skin_fallback_triggered_by_degraded_primary"
        ]
        == "True"
    )
    assert (
        float(summary_rows["quality_boundary_skinner_fallback_v3"]["skin_buffered_f1_r2"])
        >= float(summary_rows["current_default"]["skin_buffered_f1_r2"]) + 0.25
    )


def test_report_3d_synthetic_quality_scanner_vertical_v3_blocks_boundary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_report_module()
    vertical_definition = next(
        definition
        for definition in module.EXTENDED_CASES
        if definition.case_id == "single_vertical_plane"
    )
    monkeypatch.setitem(module.CASE_SETS, "vertical_only", (vertical_definition,))

    report = module.build_report(
        case_set="vertical_only",
        shape=(49, 49, 49),
        workflow_mode="quality",
        input_mode="scanner",
        scanner_config=module.SyntheticScannerConfig(
            backend="quality",
            refinement_factor=2,
        ),
        variants=("quality_boundary_skinner_fallback_v3",),
    )
    variant = report["cases"][0]["variants"]["quality_boundary_skinner_fallback_v3"]
    diagnostics = variant["skinning"]["diagnostics"]

    assert diagnostics["skin_primary_degraded_candidate"] is True
    assert diagnostics["skin_primary_degraded_reasons"] == [
        "low_fvt_positive_coverage",
        "fragmented_primary_skins",
    ]
    assert diagnostics["skin_primary_boundary_degraded_candidate"] is False
    assert diagnostics["skin_primary_boundary_degraded_reasons"] == []
    assert diagnostics["skin_primary_edge_shell_fraction"] == pytest.approx(0.0)
    assert diagnostics["skin_fvt_positive_edge_shell_fraction"] < 0.20
    assert diagnostics["fallback_policy"] == "degraded_primary_filtered"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_boundary_degraded_not_detected"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_replaced_primary"] is False


def test_report_3d_synthetic_quality_degraded_primary_policy_keeps_healthy_primary() -> None:
    module = _load_report_module()
    fvt = np.ones((1, 1, 5), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.zeros_like(fvt)
    skins = [_fault_skin([(index, 0, 0) for index in range(5)])]
    diagnostics: dict[str, object] = {}
    module._add_primary_skin_diagnostics(
        diagnostics,
        skins,
        shape=fvt.shape,
        fvt_positive_candidate_count=5,
        small_skin_size=1,
    )

    module._apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=module.SyntheticSkinningConfig(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary",
            small_skin_size=1,
        ),
        variant="quality_boundary_skinner_fallback_v2",
        diagnostics=diagnostics,
    )

    assert diagnostics["fallback_policy"] == "degraded_primary"
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_reason"] == "primary_skin_healthy"
    assert diagnostics["fallback_triggered_by_degraded_primary"] is False
    assert diagnostics["fallback_replaced_primary"] is False
    assert diagnostics["fallback_degraded_reasons"] == []
    assert diagnostics["fallback_coverage_before"] == pytest.approx(1.0)
    assert diagnostics["fallback_coverage_after"] == pytest.approx(1.0)
    assert len(skins) == 1
    assert len(skins[0]) == 5


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


def test_report_3d_synthetic_quality_invalid_skinner_options_fail(
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


def test_scanner_quality_save_volumes_writes_confidence_dat_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--input-mode",
        "scanner",
        "--scanner-backend",
        "quality",
        "--scanner-refinement-factor",
        "2",
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    path = output_dir / "single_vertical_plane" / "scanner_confidence.dat"
    assert path.is_file()
    assert path.stat().st_size == shape[0] * shape[1] * shape[2] * 4


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


def test_report_3d_synthetic_quality_boundary_plane_zero_fvt_has_no_positive_candidates() -> None:
    module = _load_report_module()

    report = module.build_report(
        case_set="extended",
        shape=(17, 17, 17),
        variants=("current_default",),
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
    )

    boundary_case = next(case for case in report["cases"] if case["case_id"] == "boundary_plane")
    variant = boundary_case["variants"]["current_default"]

    assert variant["pyosv"]["fvt"]["max"] == 0.0
    assert (
        variant["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"]["candidate_count"] > 0
    )
    assert (
        variant["quality"]["fvt_positive_top_truth_count"]["buffered_overlap_radius2"][
            "candidate_count"
        ]
        == 0
    )
    assert (
        variant["quality"]["edge_false_positive"]["fvt_positive_top_truth_count"]["candidate_count"]
        == 0
    )
