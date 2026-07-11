"""Spatial correspondence metrics between successive 3D pipeline stages."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from .quality_metrics import edge_mask

__all__ = ["stage_transition_correspondence_metrics"]


def stage_transition_correspondence_metrics(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    match_radius: float,
    edge_margin: int,
) -> dict[str, Any]:
    """Measure retained source and introduced target voxels between stages.

    A voxel is matched when the nearest voxel in the opposite mask is no
    farther than ``match_radius`` in Euclidean voxel-center distance.
    """

    source, target = _validate_masks(source_mask, target_mask)
    radius = _validate_match_radius(match_radius)
    margin = _validate_edge_margin(edge_margin)

    source_count = int(np.count_nonzero(source))
    target_count = int(np.count_nonzero(target))
    source_distances = _distances_to_mask(target) if target_count else None
    target_distances = _distances_to_mask(source) if source_count else None

    result = _region_metrics(
        source,
        target,
        source_distances=source_distances,
        target_distances=target_distances,
        match_radius=radius,
    )

    boundary_shell = edge_mask(source.shape, margin)
    result["match_radius"] = radius
    result["regions"] = {
        "boundary_shell": _region_metrics(
            source & boundary_shell,
            target & boundary_shell,
            source_distances=source_distances,
            target_distances=target_distances,
            match_radius=radius,
        ),
        "interior": _region_metrics(
            source & ~boundary_shell,
            target & ~boundary_shell,
            source_distances=source_distances,
            target_distances=target_distances,
            match_radius=radius,
        ),
    }
    return result


def _region_metrics(
    source_population: np.ndarray,
    target_population: np.ndarray,
    *,
    source_distances: np.ndarray | None,
    target_distances: np.ndarray | None,
    match_radius: float,
) -> dict[str, Any]:
    source_count = int(np.count_nonzero(source_population))
    target_count = int(np.count_nonzero(target_population))

    retained_source_count = _matched_count(source_population, source_distances, match_radius)
    matched_target_count = _matched_count(target_population, target_distances, match_radius)
    lost_source_count = source_count - retained_source_count
    introduced_target_count = target_count - matched_target_count

    source_distance_median, source_distance_p95 = _distance_statistics(
        source_population, source_distances
    )
    target_distance_median, target_distance_p95 = _distance_statistics(
        target_population, target_distances
    )
    return {
        "source_count": source_count,
        "target_count": target_count,
        "retained_source_count": retained_source_count,
        "lost_source_count": lost_source_count,
        "retained_source_fraction": _fraction_or_none(retained_source_count, source_count),
        "lost_source_fraction": _fraction_or_none(lost_source_count, source_count),
        "matched_target_count": matched_target_count,
        "introduced_target_count": introduced_target_count,
        "introduced_target_fraction": _fraction_or_none(introduced_target_count, target_count),
        "source_to_target_distance_median": source_distance_median,
        "source_to_target_distance_p95": source_distance_p95,
        "target_to_source_distance_median": target_distance_median,
        "target_to_source_distance_p95": target_distance_p95,
    }


def _validate_masks(
    source_mask: np.ndarray, target_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"source_mask and target_mask must be 3D, got {source.ndim}D and {target.ndim}D"
        )
    if source.shape != target.shape:
        raise ValueError(f"mask shapes must match, got {source.shape} and {target.shape}")
    return source, target


def _validate_match_radius(value: float) -> float:
    if not np.isscalar(value):
        raise ValueError("match_radius must be a finite non-negative scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("match_radius must be a finite non-negative scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("match_radius must be a finite non-negative scalar")
    return result


def _validate_edge_margin(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("edge_margin must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("edge_margin must be a non-negative integer")
    return result


def _distances_to_mask(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(~mask)


def _matched_count(
    population: np.ndarray, distances: np.ndarray | None, match_radius: float
) -> int:
    if distances is None:
        return 0
    return int(np.count_nonzero(population & (distances <= match_radius)))


def _distance_statistics(
    population: np.ndarray, distances: np.ndarray | None
) -> tuple[float | None, float | None]:
    if distances is None or not np.any(population):
        return None, None
    values = distances[population]
    return float(np.median(values)), float(np.percentile(values, 95.0))


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)
