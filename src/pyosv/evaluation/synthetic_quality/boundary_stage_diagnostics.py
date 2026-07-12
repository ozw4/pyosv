"""Spatial correspondence metrics between successive 3D pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

from pyosv.synthetic3d import Synthetic3DCase

from .config import SyntheticTruthMetricConfig, _validate_nonnegative_finite_scalar
from .models import PipelineStageTrace3D
from .quality_metrics import edge_mask
from . import quality_metrics

__all__ = [
    "build_scanner_boundary_stage_diagnostics",
    "stage_mask_profile",
    "stage_transition_correspondence_metrics",
    "transition_centroid_shift_metrics",
    "volume_edge_distance_map",
]

STAGE_ORDER = (
    "scanner_ft_positive",
    "scanner_fet_positive",
    "seed_candidate",
    "seed_selected",
    "fv_positive",
    "fvt_positive",
    "primary_skin",
    "fallback_skin",
    "final_skin",
)
TRANSITION_PAIRS = (
    ("scanner_ft_positive", "scanner_fet_positive"),
    ("scanner_fet_positive", "seed_candidate"),
    ("seed_candidate", "seed_selected"),
    ("seed_selected", "fv_positive"),
    ("fv_positive", "fvt_positive"),
    ("fvt_positive", "primary_skin"),
    ("primary_skin", "final_skin"),
    ("fvt_positive", "final_skin"),
    ("fvt_positive", "fallback_skin"),
    ("primary_skin", "fallback_skin"),
)


def build_scanner_boundary_stage_diagnostics(
    *,
    case: Synthetic3DCase,
    scanner_volumes: Mapping[str, np.ndarray],
    stage_trace: PipelineStageTrace3D,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_diagnostics: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build detailed diagnostics from the masks captured by one scanner run."""

    match_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius, "buffer_radius"
    )
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width, "truth_surface_half_width"
    )
    edge_margin = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
    masks = {
        "scanner_ft_positive": quality_metrics.positive_candidate_mask(
            scanner_volumes["scanner_ft"]
        ),
        "scanner_fet_positive": quality_metrics.positive_candidate_mask(
            scanner_volumes["scanner_fet"]
        ),
        "seed_candidate": stage_trace.seed_candidate_mask,
        "seed_selected": stage_trace.seed_selected_mask,
        "fv_positive": stage_trace.fv_positive_mask,
        "fvt_positive": stage_trace.fvt_positive_mask,
        "primary_skin": stage_trace.primary_skin_mask,
        "fallback_skin": stage_trace.fallback_skin_mask,
        "final_skin": stage_trace.final_skin_mask,
    }
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    stages = {
        name: stage_mask_profile(
            masks[name],
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            match_radius=match_radius,
            edge_margin=edge_margin,
            max_exact_edge_distance=3,
        )
        for name in STAGE_ORDER
    }
    transitions = {}
    for source_name, target_name in TRANSITION_PAIRS:
        key = f"{source_name}_to_{target_name}"
        correspondence = stage_transition_correspondence_metrics(
            masks[source_name],
            masks[target_name],
            match_radius=match_radius,
            edge_margin=edge_margin,
        )
        correspondence.update(
            transition_centroid_shift_metrics(
                masks[source_name],
                masks[target_name],
                truth_strike=case.truth_strike,
                truth_dip=case.truth_dip,
                truth_reference_mask=truth_surface_mask,
            )
        )
        transitions[key] = correspondence

    skinning = {} if skinning_diagnostics is None else skinning_diagnostics
    report = {
        "config": {
            "match_radius": float(match_radius),
            "edge_margin": edge_margin,
            "edge_distance_exact_max": 3,
            "component_connectivity": "edge",
        },
        "stage_order": list(STAGE_ORDER),
        "transition_order": [f"{source}_to_{target}" for source, target in TRANSITION_PAIRS],
        "stages": stages,
        "transitions": transitions,
        "skinning": {
            "enabled": bool(stage_trace.skinning_enabled),
            "fallback_enabled": bool(skinning.get("fallback_enabled", False)),
            "fallback_used": bool(stage_trace.fallback_used),
            "fallback_reason": skinning.get("fallback_reason"),
        },
    }
    volumes = {
        f"scanner_boundary_stage_{name}": np.asarray(masks[name], dtype=np.float32)
        for name in STAGE_ORDER
    }
    volumes["scanner_boundary_stage_boundary_shell"] = edge_mask(case.shape, edge_margin).astype(
        np.float32
    )
    return report, volumes


def volume_edge_distance_map(shape: tuple[int, int, int]) -> np.ndarray:
    """Return each voxel's discrete distance from the nearest volume face."""

    if len(shape) != 3 or any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or int(size) < 0
        for size in shape
    ):
        raise ValueError("shape must contain three non-negative integers")
    shape = tuple(int(size) for size in shape)
    distances = np.empty(shape, dtype=np.intp)
    if distances.size == 0:
        return distances
    axis_distances = [np.minimum(np.arange(size), np.arange(size)[::-1]) for size in shape]
    distances[...] = np.minimum(
        np.minimum(axis_distances[0][:, None, None], axis_distances[1][None, :, None]),
        axis_distances[2][None, None, :],
    )
    return distances


def stage_mask_profile(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    match_radius: float,
    edge_margin: int,
    max_exact_edge_distance: int = 3,
) -> dict[str, Any]:
    """Summarize one stage relative to synthetic fault truth."""

    candidate, truth_fault = _validate_masks(candidate_mask, truth_fault_mask)
    truth_surface = np.asarray(truth_surface_mask, dtype=bool)
    if truth_surface.ndim != 3 or truth_surface.shape != candidate.shape:
        raise ValueError(
            "truth_surface_mask must be 3D and match candidate_mask shape, "
            f"got {truth_surface.shape} and {candidate.shape}"
        )
    radius = _validate_match_radius(match_radius)
    margin = _validate_edge_margin(edge_margin)
    max_distance = _validate_nonnegative_integer(max_exact_edge_distance, "max_exact_edge_distance")

    candidate_count = int(np.count_nonzero(candidate))
    truth_count = int(np.count_nonzero(truth_fault))
    candidate_to_fault = _distances_to_mask(truth_fault) if truth_count else None
    truth_to_candidate = _distances_to_mask(candidate) if candidate_count else None
    candidate_to_surface = _distances_to_mask(truth_surface) if np.any(truth_surface) else None

    shell = edge_mask(candidate.shape, margin)
    edge_distances = volume_edge_distance_map(candidate.shape)
    edge_profile: dict[str, Any] = {}
    for distance in range(max_distance + 1):
        population = edge_distances == distance
        edge_profile[str(distance)] = _truth_region_metrics(
            candidate & population,
            truth_fault & population,
            truth_surface & population,
            candidate_to_fault=candidate_to_fault,
            truth_to_candidate=truth_to_candidate,
            candidate_to_surface=candidate_to_surface,
            match_radius=radius,
        )
    population = edge_distances > max_distance
    edge_profile[f"{max_distance + 1}_plus"] = _truth_region_metrics(
        candidate & population,
        truth_fault & population,
        truth_surface & population,
        candidate_to_fault=candidate_to_fault,
        truth_to_candidate=truth_to_candidate,
        candidate_to_surface=candidate_to_surface,
        match_radius=radius,
    )

    return {
        "candidate_count": candidate_count,
        "truth": _truth_region_metrics(
            candidate,
            truth_fault,
            truth_surface,
            candidate_to_fault=candidate_to_fault,
            truth_to_candidate=truth_to_candidate,
            candidate_to_surface=candidate_to_surface,
            match_radius=radius,
        ),
        "regions": {
            "boundary_shell": _truth_region_metrics(
                candidate & shell,
                truth_fault & shell,
                truth_surface & shell,
                candidate_to_fault=candidate_to_fault,
                truth_to_candidate=truth_to_candidate,
                candidate_to_surface=candidate_to_surface,
                match_radius=radius,
            ),
            "interior": _truth_region_metrics(
                candidate & ~shell,
                truth_fault & ~shell,
                truth_surface & ~shell,
                candidate_to_fault=candidate_to_fault,
                truth_to_candidate=truth_to_candidate,
                candidate_to_surface=candidate_to_surface,
                match_radius=radius,
            ),
        },
        "components": _component_summary(candidate),
        "edge_distance_profile": edge_profile,
    }


def transition_centroid_shift_metrics(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    truth_reference_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure a stage shift in OSV coordinates and along the mean truth normal."""

    source, target = _validate_masks(source_mask, target_mask)
    strike = np.asarray(truth_strike)
    dip = np.asarray(truth_dip)
    reference = np.asarray(truth_reference_mask, dtype=bool)
    for name, array in (("truth_strike", strike), ("truth_dip", dip)):
        if array.ndim != 3 or array.shape != source.shape:
            raise ValueError(f"{name} must be 3D and match mask shape")
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite numeric values")
    if reference.ndim != 3 or reference.shape != source.shape:
        raise ValueError("truth_reference_mask must be 3D and match mask shape")

    source_centroid = _centroid_x1_x2_x3(source)
    target_centroid = _centroid_x1_x2_x3(target)
    shift = None
    magnitude = None
    if source_centroid is not None and target_centroid is not None:
        shift_array = np.asarray(target_centroid) - np.asarray(source_centroid)
        shift = [float(value) for value in shift_array]
        magnitude = float(np.linalg.norm(shift_array))

    normal = None
    resultant_length = None
    if np.any(reference):
        p = np.deg2rad(strike[reference].astype(np.float64, copy=False))
        t = np.deg2rad(dip[reference].astype(np.float64, copy=False))
        normals = np.column_stack((-np.cos(t), np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p)))
        mean_normal = np.mean(normals, axis=0)
        resultant_length = float(np.linalg.norm(mean_normal))
        if resultant_length > 1.0e-12:
            normal = [float(value) for value in mean_normal / resultant_length]

    normal_shift = None
    tangential_magnitude = None
    if shift is not None and normal is not None and magnitude is not None:
        normal_shift = float(np.dot(shift, normal))
        tangential_magnitude = float(
            np.sqrt(max(0.0, magnitude * magnitude - normal_shift * normal_shift))
        )
    return {
        "source_centroid_x1_x2_x3": source_centroid,
        "target_centroid_x1_x2_x3": target_centroid,
        "shift_x1_x2_x3": shift,
        "shift_magnitude": magnitude,
        "representative_truth_normal_x1_x2_x3": normal,
        "truth_normal_resultant_length": resultant_length,
        "normal_shift": normal_shift,
        "tangential_shift_magnitude": tangential_magnitude,
    }


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


def _truth_region_metrics(
    candidate_population: np.ndarray,
    truth_population: np.ndarray,
    truth_surface_population: np.ndarray,
    *,
    candidate_to_fault: np.ndarray | None,
    truth_to_candidate: np.ndarray | None,
    candidate_to_surface: np.ndarray | None,
    match_radius: float,
) -> dict[str, Any]:
    candidate_count = int(np.count_nonzero(candidate_population))
    truth_count = int(np.count_nonzero(truth_population))
    matched_truth_count = _matched_count(truth_population, truth_to_candidate, match_radius)
    matched_candidate_count = _matched_count(candidate_population, candidate_to_fault, match_radius)
    recall = _fraction_or_none(matched_truth_count, truth_count)
    precision = _fraction_or_none(matched_candidate_count, candidate_count)
    buffered_f1 = None
    if recall is not None and precision is not None:
        buffered_f1 = (
            0.0
            if recall + precision == 0.0
            else float(2.0 * recall * precision / (recall + precision))
        )
    candidate_median, candidate_p95 = _distance_statistics(
        candidate_population, candidate_to_surface
    )
    truth_median, truth_p95 = _distance_statistics(truth_surface_population, truth_to_candidate)
    return {
        "truth_count": truth_count,
        "matched_truth_count": matched_truth_count,
        "truth_recall": recall,
        "matched_candidate_count": matched_candidate_count,
        "candidate_count": candidate_count,
        "candidate_precision": precision,
        "buffered_f1": buffered_f1,
        "candidate_to_truth_distance_median": candidate_median,
        "candidate_to_truth_distance_p95": candidate_p95,
        "truth_to_candidate_distance_median": truth_median,
        "truth_to_candidate_distance_p95": truth_p95,
    }


def _component_summary(mask: np.ndarray) -> dict[str, Any]:
    labels, component_count = label(mask, generate_binary_structure(3, 2))
    if component_count == 0:
        largest_size = 0
        largest_fraction = 0.0
    else:
        sizes = np.bincount(labels.ravel())[1:]
        largest_size = int(np.max(sizes))
        largest_fraction = float(largest_size / np.count_nonzero(mask))
    return {
        "connectivity": "edge",
        "component_count": int(component_count),
        "largest_component_size": largest_size,
        "largest_component_fraction": largest_fraction,
    }


def _centroid_x1_x2_x3(mask: np.ndarray) -> list[float] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    centroid_i3_i2_i1 = np.mean(indices, axis=0)
    return [float(value) for value in centroid_i3_i2_i1[::-1]]


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


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
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
