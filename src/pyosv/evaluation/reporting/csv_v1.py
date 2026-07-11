"""Legacy summary CSV v1 serializer and row extraction helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from pyosv.evaluation.reporting.json_v1 import LegacyReportV1Adapter
from pyosv.evaluation.reporting.models import Report

from .summary_schema_v1 import SUMMARY_CSV_V1_FIELDS

PIPELINE_NAMES = ("oracle", "scanner")

CSV_VARIANT_COMPARISON_FIELDS = (
    (
        "fvt_buffered_f1_delta_vs_baseline",
        "fvt_buffered_f1_r2_delta_vs_current",
    ),
    (
        "fvt_distance_p95_delta_vs_baseline",
        "fvt_candidate_to_truth_p95_delta_vs_current",
    ),
    (
        "fvt_strike_median_error_delta_vs_baseline",
        "fvt_strike_median_error_delta_vs_current",
    ),
    (
        "fvt_dip_median_error_delta_vs_baseline",
        "fvt_dip_median_error_delta_vs_current",
    ),
    (
        "skin_buffered_f1_delta_vs_baseline",
        "skin_buffered_f1_r2_delta_vs_current",
    ),
    (
        "skin_distance_p95_delta_vs_baseline",
        "skin_candidate_to_truth_p95_delta_vs_current",
    ),
    (
        "skin_strike_median_error_delta_vs_baseline",
        "skin_strike_median_error_delta_vs_current",
    ),
    (
        "skin_dip_median_error_delta_vs_baseline",
        "skin_dip_median_error_delta_vs_current",
    ),
    (
        "skin_count_delta_vs_baseline",
        "skin_count_delta_vs_current",
    ),
)


def write_summary_csv(report: Report | Mapping[str, Any], output_dir: str | PathLike[str]) -> Path:
    """Write the legacy summary CSV v1 representation."""

    if isinstance(report, Report):
        report = LegacyReportV1Adapter().to_dict(report)
    output_path = Path(output_dir) / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_CSV_V1_FIELDS,
        )
        writer.writeheader()
        config = report.get("config", {})
        input_mode = str(config.get("input_mode", "oracle"))
        workflow_mode = str(config.get("workflow_mode", "reference"))
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            for pipeline, pipeline_report in _iter_pipeline_reports(case["pipelines"]):
                variant_comparison = pipeline_report["variant_comparison"]
                baseline_variant = variant_comparison["baseline_variant"]
                comparison_variants = variant_comparison["variants"]
                for variant, variant_report in pipeline_report["variants"].items():
                    pyosv = variant_report["pyosv"]
                    fv = pyosv["fv"]
                    fvt = pyosv["fvt"]
                    quality = variant_report["quality"]
                    fv_quality = quality["fv_top_truth_count"]
                    fvt_quality = quality["fvt_top_truth_count"]
                    fv_positive_quality = quality["fv_positive_top_truth_count"]
                    fvt_positive_quality = quality["fvt_positive_top_truth_count"]
                    edge_false_positive = quality["edge_false_positive"]
                    skinning = variant_report["skinning"]
                    skin_quality = quality["skin"]
                    skin_summary = _summary_csv_skin_row(
                        enabled=bool(skinning["enabled"]),
                        quality=skin_quality,
                        diagnostics=skinning.get("diagnostics"),
                    )
                    comparison_row = _summary_csv_comparison_row(
                        comparison_variants.get(variant, {}),
                    )
                    scanner_row = _summary_csv_scanner_row(
                        variant_report=variant_report,
                        input_mode=input_mode,
                    )
                    scanner_matrix_row = _summary_csv_scanner_backend_matrix_row(
                        variant_report=variant_report,
                    )
                    scanner_downstream_row = _summary_csv_scanner_downstream_row(
                        variant_report=variant_report,
                    )
                    scanner_stage_loss_row = _summary_csv_scanner_stage_loss_row(
                        variant_report=variant_report,
                    )
                    fvt_recenter_row = _summary_csv_fvt_recenter_row(
                        variant_report=variant_report,
                    )
                    boundary_edge_thin_row = _summary_csv_boundary_edge_thin_row(
                        variant_report=variant_report,
                    )
                    boundary_seed_retention_row = _summary_csv_boundary_seed_retention_row(
                        variant_report=variant_report,
                    )
                    thinning_diagnostic_row = _summary_csv_thinning_diagnostic_row(
                        variant_report.get("thinning_diagnostic"),
                    )
                    voting_row = _summary_csv_voting_row(variant_report=variant_report)
                    writer.writerow(
                        {
                            "case_id": case["case_id"],
                            "pipeline": pipeline,
                            "variant": variant,
                            "baseline_variant": baseline_variant,
                            "input_mode": input_mode,
                            "workflow_mode": workflow_mode,
                            "shape_n3": n3,
                            "shape_n2": n2,
                            "shape_n1": n1,
                            **voting_row,
                            "fv_max": fv["max"],
                            "fv_mean": fv["mean"],
                            "fv_nonzero_fraction": fv["nonzero_fraction"],
                            "fv_buffered_f1_r2": fv_quality["buffered_overlap_radius2"][
                                "buffered_f1"
                            ],
                            "fv_distance_p95": fv_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fv_edge_false_positive_fraction": edge_false_positive[
                                "fv_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fv_positive_candidate_count": fv_positive_quality[
                                "buffered_overlap_radius2"
                            ]["candidate_count"],
                            "fv_strike_median_error": fv_quality["orientation_error"][
                                "strike_median"
                            ],
                            "fv_dip_median_error": fv_quality["orientation_error"]["dip_median"],
                            "fvt_max": fvt["max"],
                            "fvt_mean": fvt["mean"],
                            "fvt_nonzero_fraction": fvt["nonzero_fraction"],
                            "fvt_buffered_f1_r2": fvt_quality["buffered_overlap_radius2"][
                                "buffered_f1"
                            ],
                            "fvt_distance_p95": fvt_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fvt_edge_false_positive_fraction": edge_false_positive[
                                "fvt_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fvt_positive_candidate_count": fvt_positive_quality[
                                "buffered_overlap_radius2"
                            ]["candidate_count"],
                            "fvt_positive_buffered_f1_r2": fvt_positive_quality[
                                "buffered_overlap_radius2"
                            ]["buffered_f1"],
                            "fvt_positive_distance_p95": fvt_positive_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fvt_positive_edge_false_positive_fraction": edge_false_positive[
                                "fvt_positive_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fvt_strike_median_error": fvt_quality["orientation_error"][
                                "strike_median"
                            ],
                            "fvt_dip_median_error": fvt_quality["orientation_error"]["dip_median"],
                            **skin_summary,
                            **scanner_row,
                            **scanner_matrix_row,
                            **scanner_downstream_row,
                            **scanner_stage_loss_row,
                            **fvt_recenter_row,
                            **boundary_edge_thin_row,
                            **boundary_seed_retention_row,
                            **thinning_diagnostic_row,
                            **comparison_row,
                        }
                    )
    return output_path


def _summary_csv_voting_row(*, variant_report: Mapping[str, Any]) -> dict[str, str | float | int]:
    """Return stable scalar voter diagnostics, including for pre-diagnostic reports."""

    pyosv = variant_report.get("pyosv")
    voting = pyosv.get("voting") if isinstance(pyosv, Mapping) else None
    if not isinstance(voting, Mapping):
        voting = {}
    diagnostic = voting.get("diagnostic_summary")
    if not isinstance(diagnostic, Mapping):
        diagnostic = {}

    projection_count = diagnostic.get(
        "surface_projection_count",
        diagnostic.get("projection_count", 0),
    )
    return {
        "surface_voting_boundary_policy": str(
            voting.get("surface_voting_boundary_policy", "reference")
        ),
        "voter_boundary_affected_seed_count": int(
            diagnostic.get("boundary_affected_seed_count", 0)
        ),
        "voter_skipped_seed_count": int(diagnostic.get("skipped_seed_count", 0)),
        "voter_support_fraction_mean": float(diagnostic.get("support_fraction_mean", 1.0)),
        "voter_support_fraction_min": float(diagnostic.get("support_fraction_min", 1.0)),
        "voter_surface_projection_count": int(projection_count),
        "voter_selected_invalid_sample_count": int(
            diagnostic.get("selected_invalid_sample_count", 0)
        ),
        "voter_face_center_vote_count": int(diagnostic.get("face_center_vote_count", 0)),
    }


def _summary_csv_scanner_row(
    *,
    variant_report: Mapping[str, Any],
    input_mode: str,
) -> dict[str, str | float | None]:
    empty_row: dict[str, str | float | None] = {
        "scanner_backend": None,
        "scanner_refinement_factor": None,
        "scanner_ensemble_reference_like_fraction": None,
        "scanner_ensemble_quality_fraction": None,
        "scanner_ensemble_fast_fraction": None,
        "scanner_thin_mode": None,
        "scanner_ft_buffered_f1_r2": None,
        "scanner_ft_distance_p95": None,
        "scanner_strike_median_error": None,
        "scanner_dip_median_error": None,
        "scanner_input_contrast": None,
    }
    if input_mode == "oracle":
        return empty_row

    scanner_report = variant_report
    if "pipelines" in variant_report:
        scanner_report = variant_report["pipelines"].get("scanner", {})
    scanner = scanner_report.get("scanner")
    scanner_quality = scanner_report.get("scanner_quality")
    if not scanner or not scanner_quality:
        return empty_row

    ft_quality = scanner_quality["ft_top_truth_count"]
    orientation_error = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
    input_association = scanner_quality["input_association"]
    selection_fraction = scanner.get("selection_fraction_by_backend")
    if not isinstance(selection_fraction, Mapping):
        selection_fraction = {}
    return {
        "scanner_backend": scanner["config"]["backend"],
        "scanner_refinement_factor": scanner["config"]["refinement_factor"],
        "scanner_ensemble_reference_like_fraction": selection_fraction.get("reference-like"),
        "scanner_ensemble_quality_fraction": selection_fraction.get("quality"),
        "scanner_ensemble_fast_fraction": selection_fraction.get("fast"),
        "scanner_thin_mode": scanner["config"]["scanner_thin_mode"],
        "scanner_ft_buffered_f1_r2": ft_quality["buffered_overlap_radius2"]["buffered_f1"],
        "scanner_ft_distance_p95": ft_quality["surface_distance"]["candidate_to_truth_p95"],
        "scanner_strike_median_error": orientation_error["strike_median"],
        "scanner_dip_median_error": orientation_error["dip_median"],
        "scanner_input_contrast": input_association["contrast"],
    }


def _summary_csv_scanner_backend_matrix_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | None]:
    empty_row = {
        "scanner_matrix_best_fvt_positive_buffered_f1_backend": None,
        "scanner_matrix_best_skin_buffered_f1_backend": None,
        "scanner_matrix_best_boundary_edge_fp_backend": None,
    }
    matrix = variant_report.get("scanner_backend_matrix")
    if not isinstance(matrix, Mapping):
        return empty_row
    comparison = matrix.get("comparison")
    if not isinstance(comparison, Mapping):
        return empty_row
    return {
        "scanner_matrix_best_fvt_positive_buffered_f1_backend": comparison.get(
            "best_fvt_positive_buffered_f1_backend"
        ),
        "scanner_matrix_best_skin_buffered_f1_backend": comparison.get(
            "best_skin_buffered_f1_backend"
        ),
        "scanner_matrix_best_boundary_edge_fp_backend": comparison.get(
            "best_boundary_edge_fp_backend"
        ),
    }


def _summary_csv_scanner_downstream_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | int | float | None]:
    empty_row: dict[str, str | int | float | None] = {
        "scanner_downstream_ft_positive_candidate_count": None,
        "scanner_downstream_fet_positive_candidate_count": None,
        "scanner_downstream_ft_to_fet_retention_fraction": None,
        "scanner_downstream_fv_positive_candidate_count": None,
        "scanner_downstream_fvt_positive_candidate_count": None,
        "scanner_downstream_fvt_to_fv_positive_fraction": None,
        "scanner_downstream_fvt_positive_edge_candidate_fraction": None,
        "scanner_downstream_fvt_positive_edge_false_positive_fraction": None,
        "scanner_downstream_scanner_ft_positive_count": None,
        "scanner_downstream_scanner_fet_positive_count": None,
        "scanner_downstream_fv_positive_count": None,
        "scanner_downstream_fvt_positive_count": None,
        "scanner_downstream_ft_to_fvt_overlap_f1": None,
        "scanner_downstream_fvt_to_ft_distance_p95": None,
        "scanner_downstream_fvt_edge_shell_fraction": None,
        "scanner_downstream_fv_to_fvt_positive_ratio": None,
        "scanner_downstream_voter_thin_mode": None,
        "scanner_downstream_plateau_tie_breaker_source": None,
        "scanner_downstream_scanner_thin_mode": None,
        "scanner_downstream_reference_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2": None,
    }
    diagnostic = variant_report.get("scanner_downstream")
    if not isinstance(diagnostic, Mapping):
        return empty_row
    thinning_modes = diagnostic.get("thinning_modes")
    if not isinstance(thinning_modes, Mapping):
        thinning_modes = {}
    ft_to_fvt_overlap = diagnostic.get("scanner_ft_vs_fvt_positive_buffered_overlap_radius2")
    if not isinstance(ft_to_fvt_overlap, Mapping):
        ft_to_fvt_overlap = {}

    def thinning_f1(mode: str) -> float | None:
        mode_report = thinning_modes.get(mode)
        if not isinstance(mode_report, Mapping):
            return None
        value = mode_report.get("fvt_positive_buffered_f1_r2")
        return None if value is None else float(value)

    return {
        "scanner_downstream_ft_positive_candidate_count": diagnostic.get(
            "scanner_ft_positive_candidate_count"
        ),
        "scanner_downstream_fet_positive_candidate_count": diagnostic.get(
            "scanner_fet_positive_candidate_count"
        ),
        "scanner_downstream_ft_to_fet_retention_fraction": diagnostic.get(
            "scanner_ft_to_fet_retention_fraction"
        ),
        "scanner_downstream_fv_positive_candidate_count": diagnostic.get(
            "fv_positive_candidate_count"
        ),
        "scanner_downstream_fvt_positive_candidate_count": diagnostic.get(
            "fvt_positive_candidate_count"
        ),
        "scanner_downstream_fvt_to_fv_positive_fraction": diagnostic.get(
            "fvt_to_fv_positive_fraction"
        ),
        "scanner_downstream_fvt_positive_edge_candidate_fraction": diagnostic.get(
            "fvt_positive_edge_candidate_fraction"
        ),
        "scanner_downstream_fvt_positive_edge_false_positive_fraction": diagnostic.get(
            "fvt_positive_edge_false_positive_fraction"
        ),
        "scanner_downstream_scanner_ft_positive_count": diagnostic.get(
            "scanner_ft_positive_candidate_count"
        ),
        "scanner_downstream_scanner_fet_positive_count": diagnostic.get(
            "scanner_fet_positive_candidate_count"
        ),
        "scanner_downstream_fv_positive_count": diagnostic.get("fv_positive_candidate_count"),
        "scanner_downstream_fvt_positive_count": diagnostic.get("fvt_positive_candidate_count"),
        "scanner_downstream_ft_to_fvt_overlap_f1": ft_to_fvt_overlap.get("buffered_f1"),
        "scanner_downstream_fvt_to_ft_distance_p95": diagnostic.get(
            "fvt_candidate_to_scanner_ft_distance_p95"
        ),
        "scanner_downstream_fvt_edge_shell_fraction": diagnostic.get(
            "fvt_positive_edge_shell_fraction"
        ),
        "scanner_downstream_fv_to_fvt_positive_ratio": diagnostic.get(
            "fv_to_fvt_positive_candidate_count_ratio"
        ),
        "scanner_downstream_voter_thin_mode": diagnostic.get("voter_thin_mode"),
        "scanner_downstream_plateau_tie_breaker_source": diagnostic.get(
            "plateau_tie_breaker_source"
        ),
        "scanner_downstream_scanner_thin_mode": diagnostic.get("scanner_thin_mode"),
        "scanner_downstream_reference_fvt_positive_buffered_f1_r2": thinning_f1("reference"),
        "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2": thinning_f1("hybrid"),
        "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2": thinning_f1("hybrid_v2"),
        "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2": thinning_f1(
            "normal_plateau"
        ),
    }


def _summary_csv_scanner_stage_loss_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, int | float | None]:
    empty_row: dict[str, int | float | None] = {
        "scanner_stage_ft_positive_count": None,
        "scanner_stage_fet_positive_count": None,
        "scanner_stage_seed_candidate_count": None,
        "scanner_stage_seed_selected_count": None,
        "scanner_stage_fv_positive_count": None,
        "scanner_stage_fvt_positive_count": None,
        "scanner_stage_skin_count": None,
        "scanner_stage_ft_truth_f1_r2": None,
        "scanner_stage_fet_truth_f1_r2": None,
        "scanner_stage_seed_selected_truth_f1_r2": None,
        "scanner_stage_fv_truth_f1_r2": None,
        "scanner_stage_fvt_truth_f1_r2": None,
        "scanner_stage_skin_truth_f1_r2": None,
        "scanner_stage_ft_to_fet_ratio": None,
        "scanner_stage_fet_to_seed_selected_f1_r2": None,
        "scanner_stage_seed_selected_to_fv_f1_r2": None,
        "scanner_stage_fv_to_fvt_ratio": None,
        "scanner_stage_fvt_to_skin_f1_r2": None,
        "scanner_stage_fvt_to_skin_distance_p95": None,
    }
    diagnostic = variant_report.get("scanner_stage_loss")
    if not isinstance(diagnostic, Mapping):
        return empty_row
    stages = diagnostic.get("stages")
    transitions = diagnostic.get("transitions")
    if not isinstance(stages, Mapping) or not isinstance(transitions, Mapping):
        return empty_row

    def stage_value(stage: str, key: str) -> int | float | None:
        stage_report = stages.get(stage)
        if not isinstance(stage_report, Mapping):
            return None
        value = stage_report.get(key)
        return None if value is None else value

    def transition_value(transition: str, key: str) -> int | float | None:
        transition_report = transitions.get(transition)
        if not isinstance(transition_report, Mapping):
            return None
        value = transition_report.get(key)
        return None if value is None else value

    return {
        "scanner_stage_ft_positive_count": stage_value("scanner_ft_positive", "candidate_count"),
        "scanner_stage_fet_positive_count": stage_value("scanner_fet_positive", "candidate_count"),
        "scanner_stage_seed_candidate_count": stage_value("seed_candidate", "candidate_count"),
        "scanner_stage_seed_selected_count": stage_value("seed_selected", "candidate_count"),
        "scanner_stage_fv_positive_count": stage_value("fv_positive", "candidate_count"),
        "scanner_stage_fvt_positive_count": stage_value("fvt_positive", "candidate_count"),
        "scanner_stage_skin_count": stage_value("skin", "candidate_count"),
        "scanner_stage_ft_truth_f1_r2": stage_value("scanner_ft_positive", "truth_buffered_f1_r2"),
        "scanner_stage_fet_truth_f1_r2": stage_value(
            "scanner_fet_positive", "truth_buffered_f1_r2"
        ),
        "scanner_stage_seed_selected_truth_f1_r2": stage_value(
            "seed_selected", "truth_buffered_f1_r2"
        ),
        "scanner_stage_fv_truth_f1_r2": stage_value("fv_positive", "truth_buffered_f1_r2"),
        "scanner_stage_fvt_truth_f1_r2": stage_value("fvt_positive", "truth_buffered_f1_r2"),
        "scanner_stage_skin_truth_f1_r2": stage_value("skin", "truth_buffered_f1_r2"),
        "scanner_stage_ft_to_fet_ratio": transition_value(
            "scanner_ft_positive_to_scanner_fet_positive",
            "target_to_source_count_ratio",
        ),
        "scanner_stage_fet_to_seed_selected_f1_r2": transition_value(
            "scanner_fet_positive_to_seed_selected",
            "buffered_f1_r2",
        ),
        "scanner_stage_seed_selected_to_fv_f1_r2": transition_value(
            "seed_selected_to_fv_positive",
            "buffered_f1_r2",
        ),
        "scanner_stage_fv_to_fvt_ratio": transition_value(
            "fv_positive_to_fvt_positive",
            "target_to_source_count_ratio",
        ),
        "scanner_stage_fvt_to_skin_f1_r2": transition_value(
            "fvt_positive_to_skin",
            "buffered_f1_r2",
        ),
        "scanner_stage_fvt_to_skin_distance_p95": transition_value(
            "fvt_positive_to_skin",
            "target_to_source_distance_p95",
        ),
    }


def _summary_csv_fvt_recenter_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | float | None]:
    keys = (
        "fvt_recenter_enabled",
        "fvt_recenter_target_source",
        "fvt_recenter_candidate_count",
        "fvt_recenter_moved_count",
        "fvt_recenter_collision_count",
        "fvt_recenter_mean_shift",
        "fvt_recenter_p95_shift",
        "fvt_recenter_max_shift",
        "fvt_recenter_edge_shell_only",
        "fvt_recenter_positive_count_before",
        "fvt_recenter_positive_count_after",
        "fvt_recenter_to_target_distance_p95_before",
        "fvt_recenter_to_target_distance_p95_after",
    )
    diagnostic = variant_report.get("fvt_recenter")
    if not isinstance(diagnostic, Mapping):
        return {key: None for key in keys}
    return {key: diagnostic.get(key) for key in keys}


def _summary_csv_boundary_edge_thin_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | float | None]:
    keys = {
        "boundary_edge_thin_enabled": "enabled",
        "boundary_edge_thin_target_source": "target_source",
        "boundary_edge_thin_adopted_candidate_count": "adopted_candidate_count",
        "boundary_edge_thin_replaced_candidate_count": "replaced_candidate_count",
        "boundary_edge_thin_positive_count_before": "positive_count_before",
        "boundary_edge_thin_positive_count_after": "positive_count_after",
        "boundary_edge_thin_to_target_distance_p95_before": "to_target_distance_p95_before",
        "boundary_edge_thin_to_target_distance_p95_after": "to_target_distance_p95_after",
    }
    diagnostic = variant_report.get("boundary_edge_thin")
    if not isinstance(diagnostic, Mapping):
        return {key: False if key == "boundary_edge_thin_enabled" else None for key in keys}
    return {key: diagnostic.get(source_key) for key, source_key in keys.items()}


def _summary_csv_boundary_seed_retention_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | None]:
    keys = {
        "boundary_seed_retention_enabled": "enabled",
        "boundary_seed_retention_target_source": "target_source",
        "boundary_seed_retention_default_seed_count": "default_seed_count",
        "boundary_seed_retention_boundary_candidate_count": "boundary_candidate_count",
        "boundary_seed_retention_added_seed_count": "added_seed_count",
        "boundary_seed_retention_total_seed_count": "total_seed_count",
    }
    diagnostic = variant_report.get("boundary_seed_retention")
    if not isinstance(diagnostic, Mapping):
        return {key: False if key == "boundary_seed_retention_enabled" else None for key in keys}
    return {key: diagnostic.get(source_key) for key, source_key in keys.items()}


def _iter_pipeline_reports(
    pipelines: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    unknown = sorted(set(pipelines).difference(PIPELINE_NAMES))
    if unknown:
        raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
    return tuple(
        (pipeline, pipelines[pipeline]) for pipeline in PIPELINE_NAMES if pipeline in pipelines
    )


def _summary_csv_thinning_diagnostic_row(
    diagnostic: Mapping[str, Any] | None,
) -> dict[str, float | int | None]:
    empty_row: dict[str, float | int | None] = {
        "thinning_diag_reference_fvt_buffered_f1_r2": None,
        "thinning_diag_normal_fvt_buffered_f1_r2": None,
        "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2": None,
        "thinning_diag_reference_fvt_distance_p95": None,
        "thinning_diag_normal_fvt_distance_p95": None,
        "thinning_diag_normal_minus_reference_fvt_distance_p95": None,
        "thinning_diag_reference_count": None,
        "thinning_diag_normal_count": None,
        "thinning_diag_intersection_count": None,
        "thinning_diag_reference_only_count": None,
        "thinning_diag_normal_only_count": None,
        "thinning_diag_jaccard": None,
    }
    if diagnostic is None:
        return empty_row

    reference_quality = diagnostic["reference"]["quality"]["fvt_top_truth_count"]
    normal_quality = diagnostic["normal"]["quality"]["fvt_top_truth_count"]
    delta = diagnostic["delta"]["normal_minus_reference"]
    keep_mask = diagnostic["keep_mask"]
    return {
        "thinning_diag_reference_fvt_buffered_f1_r2": reference_quality["buffered_overlap_radius2"][
            "buffered_f1"
        ],
        "thinning_diag_normal_fvt_buffered_f1_r2": normal_quality["buffered_overlap_radius2"][
            "buffered_f1"
        ],
        "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2": delta["fvt_buffered_f1_r2"],
        "thinning_diag_reference_fvt_distance_p95": reference_quality["surface_distance"][
            "candidate_to_truth_p95"
        ],
        "thinning_diag_normal_fvt_distance_p95": normal_quality["surface_distance"][
            "candidate_to_truth_p95"
        ],
        "thinning_diag_normal_minus_reference_fvt_distance_p95": delta[
            "fvt_candidate_to_truth_p95"
        ],
        "thinning_diag_reference_count": keep_mask["reference_count"],
        "thinning_diag_normal_count": keep_mask["normal_count"],
        "thinning_diag_intersection_count": keep_mask["intersection_count"],
        "thinning_diag_reference_only_count": keep_mask["reference_only_count"],
        "thinning_diag_normal_only_count": keep_mask["normal_only_count"],
        "thinning_diag_jaccard": keep_mask["jaccard"],
    }


def _summary_csv_skin_row(
    *,
    enabled: bool,
    quality: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | None]:
    diagnostic_row = _summary_csv_skin_diagnostics_row(enabled=enabled, diagnostics=diagnostics)
    component_topology_row = _summary_csv_skin_component_topology_row(None)
    if quality is None:
        return {
            "skinning_enabled": enabled,
            "skin_enabled": enabled,
            "skin_count": 0,
            "skin_cell_count": 0,
            "skin_unique_cell_count": 0,
            "skin_duplicate_cell_count": 0,
            "skin_largest_size": 0,
            "skin_largest_fraction": 0.0,
            "skin_small_count": 0,
            "skin_small_cell_fraction": 0.0,
            **diagnostic_row,
            "skin_buffered_f1_r2": None,
            "skin_buffered_precision_r2": None,
            "skin_buffered_recall_r2": None,
            "skin_distance_p95": None,
            "skin_distance_candidate_to_truth_p95": None,
            "skin_distance_truth_to_candidate_p95": None,
            "skin_distance_hausdorff_p95": None,
            "skin_strike_median_error": None,
            "skin_dip_median_error": None,
            **component_topology_row,
        }

    topology = quality["topology"]
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
    component_topology_row = _summary_csv_skin_component_topology_row(
        quality.get("component_topology")
    )
    return {
        "skinning_enabled": enabled,
        "skin_enabled": enabled,
        "skin_count": topology["skin_count"],
        "skin_cell_count": topology["cell_count"],
        "skin_unique_cell_count": topology["unique_cell_count"],
        "skin_duplicate_cell_count": topology["duplicate_cell_count"],
        "skin_largest_size": topology["largest_skin_size"],
        "skin_largest_fraction": topology["largest_skin_fraction"],
        "skin_small_count": topology["small_skin_count"],
        "skin_small_cell_fraction": topology["small_skin_cell_fraction"],
        **diagnostic_row,
        "skin_buffered_f1_r2": overlap["buffered_f1"],
        "skin_buffered_precision_r2": overlap["buffered_precision"],
        "skin_buffered_recall_r2": overlap["buffered_recall"],
        "skin_distance_p95": distance["candidate_to_truth_p95"],
        "skin_distance_candidate_to_truth_p95": distance["candidate_to_truth_p95"],
        "skin_distance_truth_to_candidate_p95": distance["truth_to_candidate_p95"],
        "skin_distance_hausdorff_p95": distance["hausdorff_p95"],
        "skin_strike_median_error": orientation["strike_median"],
        "skin_dip_median_error": orientation["dip_median"],
        **component_topology_row,
    }


def _summary_csv_skin_component_topology_row(
    component_topology: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    if component_topology is None:
        return {
            "skin_truth_component_count": None,
            "skin_covered_truth_component_count": None,
            "skin_uncovered_truth_component_count": None,
            "skin_over_merge_count": None,
            "skin_over_split_count": None,
            "skin_max_truth_components_per_skin": None,
            "skin_max_skins_per_truth_component": None,
            "skin_mean_purity": None,
            "skin_min_purity": None,
            "skin_mean_truth_component_recall": None,
            "skin_min_truth_component_recall": None,
        }
    return {
        "skin_truth_component_count": component_topology["truth_component_count"],
        "skin_covered_truth_component_count": component_topology["covered_truth_component_count"],
        "skin_uncovered_truth_component_count": component_topology[
            "uncovered_truth_component_count"
        ],
        "skin_over_merge_count": component_topology["over_merge_skin_count"],
        "skin_over_split_count": component_topology["over_split_truth_component_count"],
        "skin_max_truth_components_per_skin": component_topology["max_truth_components_per_skin"],
        "skin_max_skins_per_truth_component": component_topology["max_skins_per_truth_component"],
        "skin_mean_purity": component_topology["mean_skin_purity"],
        "skin_min_purity": component_topology["min_skin_purity"],
        "skin_mean_truth_component_recall": component_topology["mean_truth_component_recall"],
        "skin_min_truth_component_recall": component_topology["min_truth_component_recall"],
    }


def _empty_summary_csv_skin_fallback_v5_guardrail_row(
    *,
    enabled: bool | None,
) -> dict[str, bool | int | float | str | None]:
    return {
        "skin_fallback_v5_guardrail_enabled": enabled,
        "skin_fallback_v5_guardrail_passed": None,
        "skin_fallback_v5_guardrail_reasons": None,
        "skin_fallback_v5_guardrail_fallback_skin_count": None,
        "skin_fallback_v5_guardrail_coverage_of_fvt_positive": None,
        "skin_fallback_v5_guardrail_largest_skin_fraction": None,
        "skin_fallback_v5_guardrail_small_skin_cell_fraction": None,
        "skin_fallback_v5_guardrail_pruned_fraction": None,
    }


def _summary_csv_skin_fallback_v5_guardrail_row(
    diagnostics: Mapping[str, Any],
) -> dict[str, bool | int | float | str | None]:
    guardrail = diagnostics.get("fallback_v5_guardrail")
    if not isinstance(guardrail, Mapping):
        return _empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=False)
    reasons = guardrail.get("reasons")
    if isinstance(reasons, list):
        reasons = ",".join(str(reason) for reason in reasons)
    return {
        "skin_fallback_v5_guardrail_enabled": guardrail.get("enabled"),
        "skin_fallback_v5_guardrail_passed": guardrail.get("passed"),
        "skin_fallback_v5_guardrail_reasons": reasons,
        "skin_fallback_v5_guardrail_fallback_skin_count": guardrail.get("fallback_skin_count"),
        "skin_fallback_v5_guardrail_coverage_of_fvt_positive": guardrail.get(
            "coverage_of_fvt_positive"
        ),
        "skin_fallback_v5_guardrail_largest_skin_fraction": guardrail.get("largest_skin_fraction"),
        "skin_fallback_v5_guardrail_small_skin_cell_fraction": guardrail.get(
            "small_skin_cell_fraction"
        ),
        "skin_fallback_v5_guardrail_pruned_fraction": guardrail.get("pruned_fraction"),
    }


def _summary_csv_skin_diagnostics_row(
    *,
    enabled: bool,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | str | None]:
    if not enabled:
        return {
            "skin_seed_candidate_count_before_spacing": 0,
            "skin_seed_count_after_spacing": 0,
            "skin_seed_rejected_by_occupied": 0,
            "skin_grow_attempt_count": 0,
            "skin_discarded_empty_count": 0,
            "skin_discarded_small_count": 0,
            "skin_accepted_count": 0,
            "skin_fallback_enabled": False,
            "skin_fallback_policy": None,
            "skin_fallback_used": False,
            "skin_fallback_reason": None,
            "skin_fallback_method": None,
            "skin_fallback_input": None,
            "skin_fallback_skin_count": 0,
            "skin_fallback_cell_count": 0,
            "skin_fallback_triggered_by_degraded_primary": False,
            "skin_fallback_degraded_reasons": "",
            "skin_fallback_replaced_primary": False,
            "skin_fallback_primary_skin_count": 0,
            "skin_fallback_primary_cell_count": 0,
            "skin_fallback_candidate_count": 0,
            "skin_fallback_component_count": 0,
            "skin_fallback_candidate_cell_count": 0,
            "skin_fallback_largest_component_size": 0,
            "skin_fallback_largest_component_fraction": 0.0,
            "skin_fallback_top3_component_cell_count": 0,
            "skin_fallback_top3_component_fraction": 0.0,
            "skin_fallback_small_component_count": 0,
            "skin_fallback_component_policy": "all",
            "skin_fallback_accepted_component_count": 0,
            "skin_fallback_discarded_component_count": 0,
            "skin_fallback_accepted_component_cell_count": 0,
            "skin_fallback_filter_min_component_size": 0,
            "skin_fallback_filter_min_component_fraction_of_largest": 0.0,
            "skin_fallback_filter_max_components": 0,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": 0,
            "skin_fallback_pruned_component_cell_count": 0,
            "skin_fallback_pruned_fraction": 0.0,
            "skin_fallback_largest_component_size_before_pruning": 0,
            "skin_fallback_largest_component_size_after_pruning": 0,
            "skin_fallback_pruning_removed_cell_count": 0,
            "skin_fallback_skeletonization_axis_mode": None,
            "skin_fallback_coverage_before": 0.0,
            "skin_fallback_coverage_after": 0.0,
            **_empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=False),
            "skin_primary_count": 0,
            "skin_primary_cell_count": 0,
            "skin_primary_unique_cell_count": 0,
            "skin_primary_largest_size": 0,
            "skin_primary_largest_fraction": 0.0,
            "skin_primary_small_count": 0,
            "skin_primary_small_cell_fraction": 0.0,
            "skin_primary_cell_coverage_of_fvt_positive": 0.0,
            "skin_primary_largest_coverage_of_fvt_positive": 0.0,
            "skin_primary_edge_shell_fraction": 0.0,
            "skin_fvt_positive_edge_shell_fraction": 0.0,
            "skin_scanner_target_positive_edge_shell_fraction": None,
            "skin_fvt_to_scanner_target_distance_p95": None,
            "skin_primary_degraded_candidate": False,
            "skin_primary_degraded_reasons": "",
            "skin_primary_boundary_degraded_candidate": False,
            "skin_primary_boundary_degraded_reasons": "",
        }
    if diagnostics is None:
        return {
            "skin_seed_candidate_count_before_spacing": None,
            "skin_seed_count_after_spacing": None,
            "skin_seed_rejected_by_occupied": None,
            "skin_grow_attempt_count": None,
            "skin_discarded_empty_count": None,
            "skin_discarded_small_count": None,
            "skin_accepted_count": None,
            "skin_fallback_enabled": None,
            "skin_fallback_policy": None,
            "skin_fallback_used": None,
            "skin_fallback_reason": None,
            "skin_fallback_method": None,
            "skin_fallback_input": None,
            "skin_fallback_skin_count": None,
            "skin_fallback_cell_count": None,
            "skin_fallback_triggered_by_degraded_primary": None,
            "skin_fallback_degraded_reasons": None,
            "skin_fallback_replaced_primary": None,
            "skin_fallback_primary_skin_count": None,
            "skin_fallback_primary_cell_count": None,
            "skin_fallback_candidate_count": None,
            "skin_fallback_component_count": None,
            "skin_fallback_candidate_cell_count": None,
            "skin_fallback_largest_component_size": None,
            "skin_fallback_largest_component_fraction": None,
            "skin_fallback_top3_component_cell_count": None,
            "skin_fallback_top3_component_fraction": None,
            "skin_fallback_small_component_count": None,
            "skin_fallback_component_policy": None,
            "skin_fallback_accepted_component_count": None,
            "skin_fallback_discarded_component_count": None,
            "skin_fallback_accepted_component_cell_count": None,
            "skin_fallback_filter_min_component_size": None,
            "skin_fallback_filter_min_component_fraction_of_largest": None,
            "skin_fallback_filter_max_components": None,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": None,
            "skin_fallback_pruned_component_cell_count": None,
            "skin_fallback_pruned_fraction": None,
            "skin_fallback_largest_component_size_before_pruning": None,
            "skin_fallback_largest_component_size_after_pruning": None,
            "skin_fallback_pruning_removed_cell_count": None,
            "skin_fallback_skeletonization_axis_mode": None,
            "skin_fallback_coverage_before": None,
            "skin_fallback_coverage_after": None,
            **_empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=None),
            "skin_primary_count": None,
            "skin_primary_cell_count": None,
            "skin_primary_unique_cell_count": None,
            "skin_primary_largest_size": None,
            "skin_primary_largest_fraction": None,
            "skin_primary_small_count": None,
            "skin_primary_small_cell_fraction": None,
            "skin_primary_cell_coverage_of_fvt_positive": None,
            "skin_primary_largest_coverage_of_fvt_positive": None,
            "skin_primary_edge_shell_fraction": None,
            "skin_fvt_positive_edge_shell_fraction": None,
            "skin_scanner_target_positive_edge_shell_fraction": None,
            "skin_fvt_to_scanner_target_distance_p95": None,
            "skin_primary_degraded_candidate": None,
            "skin_primary_degraded_reasons": None,
            "skin_primary_boundary_degraded_candidate": None,
            "skin_primary_boundary_degraded_reasons": None,
        }
    degraded_reasons = diagnostics.get("skin_primary_degraded_reasons")
    if isinstance(degraded_reasons, list):
        degraded_reasons = ",".join(str(reason) for reason in degraded_reasons)
    boundary_degraded_reasons = diagnostics.get("skin_primary_boundary_degraded_reasons")
    if isinstance(boundary_degraded_reasons, list):
        boundary_degraded_reasons = ",".join(str(reason) for reason in boundary_degraded_reasons)
    fallback_degraded_reasons = diagnostics.get("fallback_degraded_reasons")
    if isinstance(fallback_degraded_reasons, list):
        fallback_degraded_reasons = ",".join(str(reason) for reason in fallback_degraded_reasons)
    return {
        "skin_seed_candidate_count_before_spacing": diagnostics.get(
            "seed_candidate_count_before_spacing"
        ),
        "skin_seed_count_after_spacing": diagnostics.get("seed_count_after_spacing"),
        "skin_seed_rejected_by_occupied": diagnostics.get("seed_count_rejected_by_occupied"),
        "skin_grow_attempt_count": diagnostics.get("grow_attempt_count"),
        "skin_discarded_empty_count": diagnostics.get("discarded_empty_skin_count"),
        "skin_discarded_small_count": diagnostics.get("discarded_small_skin_count"),
        "skin_accepted_count": diagnostics.get("accepted_skin_count"),
        "skin_fallback_enabled": diagnostics.get("fallback_enabled"),
        "skin_fallback_policy": diagnostics.get("fallback_policy"),
        "skin_fallback_used": diagnostics.get("fallback_used"),
        "skin_fallback_reason": diagnostics.get("fallback_reason"),
        "skin_fallback_method": diagnostics.get("fallback_method"),
        "skin_fallback_input": diagnostics.get("fallback_input"),
        "skin_fallback_skin_count": diagnostics.get("fallback_skin_count"),
        "skin_fallback_cell_count": diagnostics.get("fallback_cell_count"),
        "skin_fallback_triggered_by_degraded_primary": diagnostics.get(
            "fallback_triggered_by_degraded_primary"
        ),
        "skin_fallback_degraded_reasons": fallback_degraded_reasons,
        "skin_fallback_replaced_primary": diagnostics.get("fallback_replaced_primary"),
        "skin_fallback_primary_skin_count": diagnostics.get("fallback_primary_skin_count"),
        "skin_fallback_primary_cell_count": diagnostics.get("fallback_primary_cell_count"),
        "skin_fallback_candidate_count": diagnostics.get("fallback_candidate_count"),
        "skin_fallback_component_count": diagnostics.get("skin_fallback_component_count"),
        "skin_fallback_candidate_cell_count": diagnostics.get("skin_fallback_candidate_cell_count"),
        "skin_fallback_largest_component_size": diagnostics.get(
            "skin_fallback_largest_component_size"
        ),
        "skin_fallback_largest_component_fraction": diagnostics.get(
            "skin_fallback_largest_component_fraction"
        ),
        "skin_fallback_top3_component_cell_count": diagnostics.get(
            "skin_fallback_top3_component_cell_count"
        ),
        "skin_fallback_top3_component_fraction": diagnostics.get(
            "skin_fallback_top3_component_fraction"
        ),
        "skin_fallback_small_component_count": diagnostics.get(
            "skin_fallback_small_component_count"
        ),
        "skin_fallback_component_policy": diagnostics.get("skin_fallback_component_policy"),
        "skin_fallback_accepted_component_count": diagnostics.get(
            "skin_fallback_accepted_component_count"
        ),
        "skin_fallback_discarded_component_count": diagnostics.get(
            "skin_fallback_discarded_component_count"
        ),
        "skin_fallback_accepted_component_cell_count": diagnostics.get(
            "skin_fallback_accepted_component_cell_count"
        ),
        "skin_fallback_filter_min_component_size": diagnostics.get(
            "skin_fallback_filter_min_component_size"
        ),
        "skin_fallback_filter_min_component_fraction_of_largest": diagnostics.get(
            "skin_fallback_filter_min_component_fraction_of_largest"
        ),
        "skin_fallback_filter_max_components": diagnostics.get(
            "skin_fallback_filter_max_components"
        ),
        "skin_fallback_pruning_method": diagnostics.get("skin_fallback_pruning_method"),
        "skin_fallback_raw_component_cell_count": diagnostics.get(
            "skin_fallback_raw_component_cell_count"
        ),
        "skin_fallback_pruned_component_cell_count": diagnostics.get(
            "skin_fallback_pruned_component_cell_count"
        ),
        "skin_fallback_pruned_fraction": diagnostics.get("skin_fallback_pruned_fraction"),
        "skin_fallback_largest_component_size_before_pruning": diagnostics.get(
            "skin_fallback_largest_component_size_before_pruning"
        ),
        "skin_fallback_largest_component_size_after_pruning": diagnostics.get(
            "skin_fallback_largest_component_size_after_pruning"
        ),
        "skin_fallback_pruning_removed_cell_count": diagnostics.get(
            "skin_fallback_pruning_removed_cell_count"
        ),
        "skin_fallback_skeletonization_axis_mode": diagnostics.get(
            "skin_fallback_skeletonization_axis_mode"
        ),
        "skin_fallback_coverage_before": diagnostics.get("fallback_coverage_before"),
        "skin_fallback_coverage_after": diagnostics.get("fallback_coverage_after"),
        **_summary_csv_skin_fallback_v5_guardrail_row(diagnostics),
        "skin_primary_count": diagnostics.get("skin_primary_count"),
        "skin_primary_cell_count": diagnostics.get("skin_primary_cell_count"),
        "skin_primary_unique_cell_count": diagnostics.get("skin_primary_unique_cell_count"),
        "skin_primary_largest_size": diagnostics.get("skin_primary_largest_size"),
        "skin_primary_largest_fraction": diagnostics.get("skin_primary_largest_fraction"),
        "skin_primary_small_count": diagnostics.get("skin_primary_small_count"),
        "skin_primary_small_cell_fraction": diagnostics.get("skin_primary_small_cell_fraction"),
        "skin_primary_cell_coverage_of_fvt_positive": diagnostics.get(
            "skin_primary_cell_coverage_of_fvt_positive"
        ),
        "skin_primary_largest_coverage_of_fvt_positive": diagnostics.get(
            "skin_primary_largest_coverage_of_fvt_positive"
        ),
        "skin_primary_edge_shell_fraction": diagnostics.get("skin_primary_edge_shell_fraction"),
        "skin_fvt_positive_edge_shell_fraction": diagnostics.get(
            "skin_fvt_positive_edge_shell_fraction"
        ),
        "skin_scanner_target_positive_edge_shell_fraction": diagnostics.get(
            "skin_scanner_target_positive_edge_shell_fraction"
        ),
        "skin_fvt_to_scanner_target_distance_p95": diagnostics.get(
            "skin_fvt_to_scanner_target_distance_p95"
        ),
        "skin_primary_degraded_candidate": diagnostics.get("skin_primary_degraded_candidate"),
        "skin_primary_degraded_reasons": degraded_reasons,
        "skin_primary_boundary_degraded_candidate": diagnostics.get(
            "skin_primary_boundary_degraded_candidate"
        ),
        "skin_primary_boundary_degraded_reasons": boundary_degraded_reasons,
    }


def _summary_csv_comparison_row(comparison: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        csv_field: comparison.get(json_field)
        for csv_field, json_field in CSV_VARIANT_COMPARISON_FIELDS
    }
