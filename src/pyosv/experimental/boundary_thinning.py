"""Experimental boundary post-thinning transforms for quality studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.synthetic_metrics import surface_distance_metrics
from pyosv.voting3d import OptimalSurfaceVoter

FVT_RECENTER_MAX_SHIFT = 3


@dataclass(frozen=True, slots=True)
class ArrayTransformResult:
    """Array output and JSON-safe diagnostics from an experimental transform."""

    output: np.ndarray
    diagnostics: dict[str, Any]


def recenter_edge_fvt_to_target(
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    target: np.ndarray,
    target_source: str,
    max_shift: int = FVT_RECENTER_MAX_SHIFT,
    edge_margin: int,
) -> ArrayTransformResult:
    """Recenter edge-shell FVT samples toward a stronger target sample."""

    fvt_array = np.asarray(fvt, dtype=np.float32)
    vp_array = np.asarray(vp, dtype=np.float32)
    vt_array = np.asarray(vt, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if fvt_array.ndim != 3:
        raise ValueError("fvt must be a 3D array")
    if vp_array.shape != fvt_array.shape or vt_array.shape != fvt_array.shape:
        raise ValueError("fvt, vp, and vt shapes must match")
    if target_array.shape != fvt_array.shape:
        raise ValueError("fvt and target shapes must match")
    if not all(
        np.all(np.isfinite(array)) for array in (fvt_array, vp_array, vt_array, target_array)
    ):
        raise ValueError("fvt recenter inputs must contain only finite values")
    shift_limit = _validate_nonnegative_int(max_shift, "max_shift")
    before_positive = quality_metrics.positive_candidate_mask(fvt_array)
    edge_candidates = before_positive & quality_metrics.edge_mask(fvt_array.shape, edge_margin)
    candidate_count = int(np.count_nonzero(edge_candidates))
    recentered = np.zeros_like(fvt_array, dtype=np.float32)
    destination_to_candidate: dict[
        tuple[int, int, int], tuple[int, float, float, tuple[int, int, int]]
    ] = {}
    shifts: list[float] = []
    moved_count = 0
    collision_count = 0
    for stable_index, (i3, i2, i1) in enumerate(np.argwhere(before_positive)):
        source = (int(i3), int(i2), int(i1))
        if bool(edge_candidates[source]):
            destination, shift = _fvt_recenter_destination(
                source, vp_array, vt_array, target_array, max_shift=shift_limit
            )
            shifts.append(float(shift))
            if shift > 0:
                moved_count += 1
        else:
            destination, shift = source, 0
        value = float(fvt_array[source])
        existing = destination_to_candidate.get(destination)
        candidate_record = (stable_index, value, float(shift), source)
        if existing is None:
            destination_to_candidate[destination] = candidate_record
        else:
            collision_count += 1
            if value > existing[1] or (value == existing[1] and stable_index < existing[0]):
                destination_to_candidate[destination] = candidate_record
    for destination, (_, value, _, _) in destination_to_candidate.items():
        recentered[destination] = np.float32(value)

    shift_values = np.asarray(shifts, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "fvt_recenter_enabled": True,
        "fvt_recenter_target_source": target_source,
        "fvt_recenter_candidate_count": candidate_count,
        "fvt_recenter_moved_count": int(moved_count),
        "fvt_recenter_collision_count": int(collision_count),
        "fvt_recenter_mean_shift": float(np.mean(shift_values)) if shift_values.size else 0.0,
        "fvt_recenter_p95_shift": (
            float(np.percentile(shift_values, 95)) if shift_values.size else 0.0
        ),
        "fvt_recenter_max_shift": float(np.max(shift_values)) if shift_values.size else 0.0,
        "fvt_recenter_edge_shell_only": True,
        "fvt_recenter_positive_count_before": int(np.count_nonzero(before_positive)),
        "fvt_recenter_positive_count_after": int(
            np.count_nonzero(quality_metrics.positive_candidate_mask(recentered))
        ),
        "fvt_recenter_value_source": "original_fvt",
    }
    diagnostics.update(
        fvt_recenter_target_distance_diagnostics(
            before=before_positive,
            after=quality_metrics.positive_candidate_mask(recentered),
            target=quality_metrics.positive_candidate_mask(target_array),
        )
    )
    return ArrayTransformResult(recentered, diagnostics)


def fvt_recenter_target_distance_diagnostics(
    *, before: np.ndarray | None, after: np.ndarray | None, target: np.ndarray | None
) -> dict[str, float | None]:
    """Build JSON-safe before/after distance diagnostics."""

    if before is None or after is None or target is None:
        before_p95 = after_p95 = None
    else:
        before_distance = surface_distance_metrics(
            np.asarray(before, dtype=bool), np.asarray(target, dtype=bool)
        )
        after_distance = surface_distance_metrics(
            np.asarray(after, dtype=bool), np.asarray(target, dtype=bool)
        )
        before_p95 = before_distance["candidate_to_truth_p95"]
        after_p95 = after_distance["candidate_to_truth_p95"]
    return {
        "fvt_recenter_to_target_distance_p95_before": before_p95,
        "fvt_recenter_to_target_distance_p95_after": after_p95,
    }


def apply_boundary_edge_thin_v1(
    fvt: np.ndarray,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    voter: OptimalSurfaceVoter,
    target: np.ndarray,
    target_source: str,
    edge_margin: int,
) -> ArrayTransformResult:
    """Replace edge-line samples with deterministic target-guided candidates."""

    arrays = tuple(np.asarray(array, dtype=np.float32) for array in (fvt, fv, vp, vt, target))
    fvt_array, fv_array, vp_array, vt_array, target_array = arrays
    if fvt_array.ndim != 3:
        raise ValueError("fvt must be a 3D array")
    if any(array.shape != fvt_array.shape for array in arrays[1:]):
        raise ValueError("boundary_edge_thin_v1 input shapes must match")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("boundary_edge_thin_v1 inputs must contain only finite values")
    edge_mask = quality_metrics.edge_mask(fvt_array.shape, edge_margin)
    before_positive = quality_metrics.positive_candidate_mask(fvt_array)
    edge_positive_before = before_positive & edge_mask
    target_positive = quality_metrics.positive_candidate_mask(target_array)
    target_plateau = voter.thin(
        fv_array, vp_array, vt_array, mode="normal_plateau", plateau_tie_breaker=target_array
    )
    candidate_mask = (
        quality_metrics.positive_candidate_mask(target_plateau) & edge_mask & target_positive
    )
    result = fvt_array.copy()
    collision_count = adopted_count = replaced_count = 0
    line_candidates: dict[tuple[int, int, int], tuple[float, float, int, tuple[int, int, int]]] = {}
    for index in np.argwhere(candidate_mask):
        i3, i2, i1 = map(int, index)
        axis = dominant_fault_normal_array_axis(
            float(vp_array[i3, i2, i1]), float(vt_array[i3, i2, i1])
        )
        key = _boundary_edge_line_key(axis, i3, i2, i1)
        record = (
            float(target_array[i3, i2, i1]),
            float(fv_array[i3, i2, i1]),
            int(np.ravel_multi_index((i3, i2, i1), fvt_array.shape)),
            (i3, i2, i1),
        )
        existing = line_candidates.get(key)
        if existing is None or _boundary_edge_candidate_precedes(record, existing):
            line_candidates[key] = record
    for key, (_, _, _, destination) in line_candidates.items():
        selector = _boundary_edge_line_selector(key)
        base_line = np.zeros(fvt_array.shape, dtype=bool)
        base_line[selector] = edge_positive_before[selector]
        base_count = int(np.count_nonzero(base_line))
        collision_count += base_count
        destination_was_positive = bool(before_positive[destination])
        if base_count > 0:
            base_records = []
            for base_index in np.argwhere(base_line):
                b3, b2, b1 = map(int, base_index)
                base_records.append(
                    (
                        float(target_array[b3, b2, b1]),
                        float(fv_array[b3, b2, b1]),
                        int(np.ravel_multi_index((b3, b2, b1), fvt_array.shape)),
                        (b3, b2, b1),
                    )
                )
            best_base = min(base_records, key=lambda item: (-item[0], -item[1], item[2]))
            candidate_record = (
                float(target_array[destination]),
                float(fv_array[destination]),
                int(np.ravel_multi_index(destination, fvt_array.shape)),
                destination,
            )
            if not _boundary_edge_candidate_precedes(candidate_record, best_base):
                continue
            result[base_line] = np.float32(0.0)
            replaced_count += int(
                np.count_nonzero(base_line & ~_single_index_mask(fvt_array.shape, destination))
            )
        if not destination_was_positive:
            adopted_count += 1
        result[destination] = np.float32(fv_array[destination])
    after_positive = quality_metrics.positive_candidate_mask(result)
    distance_before = surface_distance_metrics(before_positive, target_positive)
    distance_after = surface_distance_metrics(after_positive, target_positive)
    diagnostics = {
        "enabled": True,
        "target_source": target_source,
        "edge_margin": int(edge_margin),
        "positive_count_before": int(np.count_nonzero(before_positive)),
        "positive_count_after": int(np.count_nonzero(after_positive)),
        "edge_positive_count_before": int(np.count_nonzero(edge_positive_before)),
        "edge_positive_count_after": int(np.count_nonzero(after_positive & edge_mask)),
        "adopted_candidate_count": int(adopted_count),
        "replaced_candidate_count": int(replaced_count),
        "collision_count": int(collision_count),
        "to_target_distance_p95_before": float(distance_before["candidate_to_truth_p95"]),
        "to_target_distance_p95_after": float(distance_after["candidate_to_truth_p95"]),
    }
    return ArrayTransformResult(result.astype(np.float32, copy=False), diagnostics)


def dominant_fault_normal_array_axis(strike: float, dip: float) -> int:
    p, t = math.radians(strike), math.radians(dip)
    components = (
        abs(-math.sin(t) * math.sin(p)),
        abs(math.sin(t) * math.cos(p)),
        abs(-math.cos(t)),
    )
    return int(np.argmax(np.asarray(components, dtype=np.float32)))


def _fvt_recenter_destination(source, vp, vt, target, *, max_shift):
    if max_shift <= 0:
        return source, 0
    i3, i2, i1 = source
    axis = dominant_fault_normal_array_axis(float(vp[source]), float(vt[source]))
    current_target = best_target = float(target[source])
    best_destination, best_abs_shift = source, 0
    for offset in range(-max_shift, max_shift + 1):
        if offset == 0:
            continue
        destination = [i3, i2, i1]
        destination[axis] += offset
        if not all(
            0 <= value < size for value, size in zip(destination, target.shape, strict=True)
        ):
            continue
        destination_tuple = tuple(map(int, destination))
        target_value, abs_shift = float(target[destination_tuple]), abs(offset)
        if target_value > best_target or (
            target_value == best_target and best_abs_shift > 0 and abs_shift < best_abs_shift
        ):
            best_destination, best_target, best_abs_shift = (
                destination_tuple,
                target_value,
                abs_shift,
            )
    return (source, 0) if best_target <= current_target else (best_destination, best_abs_shift)


def _boundary_edge_line_key(axis, i3, i2, i1):
    return (axis, i2, i1) if axis == 0 else ((axis, i3, i1) if axis == 1 else (axis, i3, i2))


def _boundary_edge_line_selector(key):
    axis, first, second = key
    return (
        (slice(None), first, second)
        if axis == 0
        else ((first, slice(None), second) if axis == 1 else (first, second, slice(None)))
    )


def _boundary_edge_candidate_precedes(candidate, existing):
    if candidate[0] != existing[0]:
        return candidate[0] > existing[0]
    if candidate[1] != existing[1]:
        return candidate[1] > existing[1]
    return candidate[2] < existing[2]


def _single_index_mask(shape, index):
    mask = np.zeros(shape, dtype=bool)
    mask[index] = True
    return mask


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
