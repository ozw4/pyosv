"""Diagnostics for experimental synthetic skinning policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.synthetic_metrics import skin_mask_from_skins, skin_topology_metrics

SKIN_PRIMARY_DEGRADED_MIN_CELL_COVERAGE = 0.50
SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT = 8
SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION = 0.75
SKIN_PRIMARY_DEGRADED_MAX_SMALL_CELL_FRACTION = 0.25
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_FVT_EDGE_SHELL_FRACTION = 0.25
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_SCANNER_TARGET_DISTANCE_P95 = 2.0
SKIN_PRIMARY_BOUNDARY_DEGRADED_MAX_CELL_COVERAGE = 0.50
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_FVT_FRACTION = 0.15
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_PRIMARY_FRACTION = 0.05


def add_primary_skin_diagnostics(
    diagnostics: dict[str, Any],
    skins: Sequence[Any],
    *,
    shape: tuple[int, int, int],
    fvt_positive_candidate_count: int,
    small_skin_size: int,
) -> None:
    """Add primary-skin topology diagnostics without mutating ``skins``."""

    topology = skin_topology_metrics(skins, shape, small_skin_size=small_skin_size)
    positive_count = int(fvt_positive_candidate_count)
    if positive_count < 0:
        raise ValueError("fvt_positive_candidate_count must be non-negative")
    unique_cell_count = int(topology["unique_cell_count"])
    largest_size = int(topology["largest_skin_size"])
    cell_coverage = float(unique_cell_count / positive_count) if positive_count else 0.0
    largest_coverage = float(largest_size / positive_count) if positive_count else 0.0
    primary_edge_shell_fraction = quality_metrics.edge_candidate_fraction(
        skin_mask_from_skins(skins, shape), edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
    )
    reasons = primary_skin_degraded_reasons(
        fvt_positive_candidate_count=positive_count,
        skin_count=int(topology["skin_count"]),
        cell_coverage_of_fvt_positive=cell_coverage,
        largest_fraction=float(topology["largest_skin_fraction"]),
        small_skin_cell_fraction=float(topology["small_skin_cell_fraction"]),
    )
    diagnostics.update(
        {
            "skin_primary_count": int(topology["skin_count"]),
            "skin_primary_cell_count": int(topology["cell_count"]),
            "skin_primary_unique_cell_count": unique_cell_count,
            "skin_primary_largest_size": largest_size,
            "skin_primary_largest_fraction": float(topology["largest_skin_fraction"]),
            "skin_primary_small_count": int(topology["small_skin_count"]),
            "skin_primary_small_cell_fraction": float(topology["small_skin_cell_fraction"]),
            "skin_primary_cell_coverage_of_fvt_positive": cell_coverage,
            "skin_primary_largest_coverage_of_fvt_positive": largest_coverage,
            "skin_primary_edge_shell_fraction": primary_edge_shell_fraction,
            "skin_primary_degraded_candidate": bool(reasons),
            "skin_primary_degraded_reasons": reasons,
        }
    )


def primary_skin_degraded_reasons(
    *,
    fvt_positive_candidate_count: int,
    skin_count: int,
    cell_coverage_of_fvt_positive: float,
    largest_fraction: float,
    small_skin_cell_fraction: float,
) -> list[str]:
    """Return stable reason labels for a degraded primary result."""

    if int(fvt_positive_candidate_count) <= 0:
        return []
    reasons = []
    if int(skin_count) == 0:
        reasons.append("empty_primary_skin")
    if float(cell_coverage_of_fvt_positive) < SKIN_PRIMARY_DEGRADED_MIN_CELL_COVERAGE:
        reasons.append("low_fvt_positive_coverage")
    if int(skin_count) > 0 and (
        int(skin_count) >= SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT
        or float(largest_fraction) < SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION
    ):
        reasons.append("fragmented_primary_skins")
    if float(small_skin_cell_fraction) > SKIN_PRIMARY_DEGRADED_MAX_SMALL_CELL_FRACTION:
        reasons.append("high_small_skin_cell_fraction")
    return reasons


def primary_boundary_degraded_reasons(
    *,
    generic_degraded: bool,
    fvt_positive_candidate_count: int,
    cell_coverage_of_fvt_positive: float,
    fvt_positive_edge_shell_fraction: float,
    primary_edge_shell_fraction: float,
    fvt_to_scanner_target_distance_p95: float | None,
) -> list[str]:
    """Return stable boundary-local degradation reason labels."""

    if not generic_degraded or int(fvt_positive_candidate_count) <= 0:
        return []
    reasons = []
    if (
        float(fvt_positive_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_FVT_EDGE_SHELL_FRACTION
    ):
        reasons.append("fvt_positive_edge_shell")
    if (
        fvt_to_scanner_target_distance_p95 is not None
        and float(fvt_to_scanner_target_distance_p95)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_SCANNER_TARGET_DISTANCE_P95
    ):
        reasons.append("fvt_far_from_scanner_target")
    if (
        float(cell_coverage_of_fvt_positive) < SKIN_PRIMARY_BOUNDARY_DEGRADED_MAX_CELL_COVERAGE
        and float(fvt_positive_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_FVT_FRACTION
        and float(primary_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_PRIMARY_FRACTION
    ):
        reasons.append("low_primary_coverage_with_edge_local_candidates")
    return reasons
