"""Declarative specifications for quality-report comparison and promotion."""

from __future__ import annotations

from dataclasses import dataclass


MATCH_KEY_FIELDS = (
    "case_id",
    "pipeline",
    "input_mode",
    "workflow_mode",
    "scanner_backend",
    "scanner_refinement_factor",
    "shape_n3",
    "shape_n2",
    "shape_n1",
)

METRIC_COLUMNS = (
    "fv_buffered_f1_r2",
    "fv_distance_p95",
    "fv_edge_false_positive_fraction",
    "fvt_buffered_f1_r2",
    "fvt_distance_p95",
    "fvt_positive_candidate_count",
    "fvt_positive_buffered_f1_r2",
    "fvt_positive_distance_p95",
    "fvt_positive_edge_false_positive_fraction",
    "skin_count",
    "skin_cell_count",
    "skin_buffered_f1_r2",
    "skin_distance_p95",
    "skin_distance_candidate_to_truth_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "skin_over_merge_count",
    "skin_over_split_count",
    "skin_mean_purity",
    "skin_min_purity",
    "skin_mean_truth_component_recall",
    "skin_min_truth_component_recall",
    "scanner_ft_buffered_f1_r2",
    "scanner_ft_distance_p95",
    "scanner_downstream_fvt_to_ft_distance_p95",
    "scanner_downstream_fvt_positive_edge_false_positive_fraction",
    "skin_fallback_used",
    "skin_fallback_replaced_primary",
)

HIGHER_IS_BETTER = frozenset(
    {
        "fv_buffered_f1_r2",
        "fvt_buffered_f1_r2",
        "fvt_positive_buffered_f1_r2",
        "skin_buffered_f1_r2",
        "scanner_ft_buffered_f1_r2",
    }
)
LOWER_IS_BETTER = frozenset(
    {
        "fv_distance_p95",
        "fv_edge_false_positive_fraction",
        "fvt_distance_p95",
        "fvt_positive_distance_p95",
        "fvt_positive_edge_false_positive_fraction",
        "skin_distance_p95",
        "skin_distance_candidate_to_truth_p95",
        "skin_strike_median_error",
        "skin_dip_median_error",
        "scanner_ft_distance_p95",
        "scanner_downstream_fvt_to_ft_distance_p95",
        "scanner_downstream_fvt_positive_edge_false_positive_fraction",
    }
)
MATERIAL_REGRESSION_THRESHOLDS = {
    "skin_buffered_f1_r2": ("lt", -0.02),
    "fvt_positive_buffered_f1_r2": ("lt", -0.02),
    "skin_distance_p95": ("gt", 2.0),
    "fvt_positive_distance_p95": ("gt", 2.0),
}

EXTENDED_CASES = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)
TOPOLOGY_CASES = ("parallel_planes", "crossing_planes")
FALSE_FALLBACK_CASES = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "weak_noisy_plane",
)


@dataclass(frozen=True)
class CoverageSpec:
    name: str
    pipeline: str
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class GateSpec:
    name: str
    required_shape: tuple[str, str, str]
    scanner_backend: str
    scanner_refinement_factor: str
    boundary_skin_f1_min: float
    boundary_skin_count_max: float
    boundary_ratio_range: tuple[float, float]
    changed_fvt_f1_min: float
    changed_fvt_distance_max: float
    coverage: tuple[CoverageSpec, ...]
    always_check_boundary_fvt: bool = False
    require_unchanged_oracle_metrics: bool = False
    allow_materially_improved_false_fallback: bool = True
    required_comparison_profile: str | None = None


SCANNER_BOUNDARY_GATE = GateSpec(
    name="scanner-boundary",
    required_shape=("49", "49", "49"),
    scanner_backend="quality",
    scanner_refinement_factor="2",
    boundary_skin_f1_min=0.90,
    boundary_skin_count_max=3,
    boundary_ratio_range=(0.75, 1.25),
    changed_fvt_f1_min=0.90,
    changed_fvt_distance_max=2.0,
    coverage=(
        CoverageSpec("boundary_plane_scanner_quality_ref2_49", "scanner", ("boundary_plane",)),
        CoverageSpec(
            "non_boundary_scanner_quality_ref2_49",
            "scanner",
            tuple(case_id for case_id in EXTENDED_CASES if case_id != "boundary_plane"),
        ),
        CoverageSpec("oracle_49", "oracle", EXTENDED_CASES),
        CoverageSpec(
            "false_fallback_replacement_scanner_quality_ref2_49",
            "scanner",
            FALSE_FALLBACK_CASES,
        ),
        CoverageSpec("topology_scanner_quality_ref2_49", "scanner", TOPOLOGY_CASES),
    ),
)

SCANNER_BOUNDARY_REFERENCE_LIKE_GATE = GateSpec(
    name="scanner-boundary-reference-like",
    required_shape=("49", "49", "49"),
    scanner_backend="reference-like",
    scanner_refinement_factor="2",
    boundary_skin_f1_min=0.90,
    boundary_skin_count_max=3,
    boundary_ratio_range=(0.75, 1.25),
    changed_fvt_f1_min=0.90,
    changed_fvt_distance_max=2.0,
    coverage=(
        CoverageSpec(
            "boundary_plane_scanner_reference_like_ref2_49",
            "scanner",
            ("boundary_plane",),
        ),
        CoverageSpec(
            "non_boundary_scanner_reference_like_ref2_49",
            "scanner",
            tuple(case_id for case_id in EXTENDED_CASES if case_id != "boundary_plane"),
        ),
        CoverageSpec("oracle_49", "oracle", EXTENDED_CASES),
        CoverageSpec(
            "false_fallback_replacement_scanner_reference_like_ref2_49",
            "scanner",
            FALSE_FALLBACK_CASES,
        ),
        CoverageSpec(
            "topology_scanner_reference_like_ref2_49",
            "scanner",
            TOPOLOGY_CASES,
        ),
    ),
    always_check_boundary_fvt=True,
    require_unchanged_oracle_metrics=True,
    allow_materially_improved_false_fallback=False,
    required_comparison_profile="quality-workflow-scanner-thinning-v1",
)

PROMOTION_GATES = {
    SCANNER_BOUNDARY_GATE.name: SCANNER_BOUNDARY_GATE,
    SCANNER_BOUNDARY_REFERENCE_LIKE_GATE.name: SCANNER_BOUNDARY_REFERENCE_LIKE_GATE,
}

DEFAULT_CANDIDATES = (
    "boundary_edge_thin_v1",
    "boundary_seed_retention_v1",
    "quality_boundary_skinner_fallback_v5",
)
