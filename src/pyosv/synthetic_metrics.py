"""Metrics for controlled synthetic truth evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from pyosv.skin import FaultSkin

__all__ = [
    "buffered_surface_overlap",
    "masked_orientation_error",
    "skin_mask_from_skins",
    "skin_orientation_error",
    "skin_topology_metrics",
    "skin_truth_metrics",
    "surface_distance_metrics",
    "top_k_mask",
    "top_truth_count_mask",
]


def top_k_mask(values: np.ndarray, k: int) -> np.ndarray:
    """Return a deterministic mask selecting the largest ``k`` samples."""

    value_array = _validate_finite_array(values, "values")
    count = _validate_k(k)
    if count == 0:
        return np.zeros(value_array.shape, dtype=bool)
    if count >= value_array.size:
        return np.ones(value_array.shape, dtype=bool)

    flat_values = value_array.astype(np.float64, copy=False).ravel()
    flat_indices = np.arange(flat_values.size)
    selected_indices = np.lexsort((flat_indices, -flat_values))[:count]

    mask = np.zeros(flat_values.shape, dtype=bool)
    mask[selected_indices] = True
    return mask.reshape(value_array.shape)


def top_truth_count_mask(values: np.ndarray, truth_mask: np.ndarray) -> np.ndarray:
    """Return a top-k mask where k is the number of truth voxels."""

    value_array = np.asarray(values)
    truth = np.asarray(truth_mask, dtype=bool)
    if value_array.shape != truth.shape:
        raise ValueError(f"array shapes must match, got {value_array.shape} and {truth.shape}")
    return top_k_mask(value_array, int(np.count_nonzero(truth)))


def buffered_surface_overlap(
    candidate_mask: np.ndarray,
    truth_mask: np.ndarray,
    *,
    radius: float,
) -> dict[str, float | int]:
    """Return exact and buffered overlap metrics for candidate and truth masks."""

    candidate, truth = _validate_mask_pair(candidate_mask, truth_mask)
    buffer_radius = _validate_nonnegative_finite_scalar(radius, "radius")

    candidate_count = int(np.count_nonzero(candidate))
    truth_count = int(np.count_nonzero(truth))
    intersection_count = int(np.count_nonzero(candidate & truth))
    union_count = int(np.count_nonzero(candidate | truth))

    truth_buffer = _distance_buffer(truth, buffer_radius)
    candidate_buffer = _distance_buffer(candidate, buffer_radius)
    candidate_in_truth_buffer_count = int(np.count_nonzero(candidate & truth_buffer))
    truth_in_candidate_buffer_count = int(np.count_nonzero(truth & candidate_buffer))

    precision = _precision(intersection_count, candidate_count)
    recall = _recall(intersection_count, truth_count)
    buffered_precision = _precision(candidate_in_truth_buffer_count, candidate_count)
    buffered_recall = _recall(truth_in_candidate_buffer_count, truth_count)

    return {
        "candidate_count": candidate_count,
        "truth_count": truth_count,
        "intersection_count": intersection_count,
        "union_count": union_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "jaccard": _jaccard(intersection_count, union_count),
        "buffered_precision": buffered_precision,
        "buffered_recall": buffered_recall,
        "buffered_f1": _f1(buffered_precision, buffered_recall),
        "radius": float(buffer_radius),
    }


def surface_distance_metrics(
    candidate_mask: np.ndarray,
    truth_mask: np.ndarray,
) -> dict[str, float | int]:
    """Return directional and symmetric surface distance metrics."""

    candidate, truth = _validate_mask_pair(candidate_mask, truth_mask)

    candidate_count = int(np.count_nonzero(candidate))
    truth_count = int(np.count_nonzero(truth))
    if candidate_count == 0 and truth_count == 0:
        return _distance_result(candidate_count, truth_count, 0.0, 0.0)
    if candidate_count == 0 or truth_count == 0:
        penalty = _volume_diagonal(candidate.shape)
        return _distance_result(candidate_count, truth_count, penalty, penalty)

    distance_to_truth = distance_transform_edt(~truth)
    candidate_to_truth = _distance_summary(distance_to_truth[candidate])

    distance_to_candidate = distance_transform_edt(~candidate)
    truth_to_candidate = _distance_summary(distance_to_candidate[truth])

    result = {
        "candidate_count": candidate_count,
        "truth_count": truth_count,
        **_prefixed_summary("candidate_to_truth", candidate_to_truth),
        **_prefixed_summary("truth_to_candidate", truth_to_candidate),
    }
    result["symmetric_chamfer_mean"] = float(
        0.5 * (candidate_to_truth["mean"] + truth_to_candidate["mean"])
    )
    result["hausdorff_p95"] = float(max(candidate_to_truth["p95"], truth_to_candidate["p95"]))
    return result


def masked_orientation_error(
    predicted_strike: np.ndarray,
    predicted_dip: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Return strike and dip error summaries over ``mask``."""

    pred_strike = np.asarray(predicted_strike)
    pred_dip = np.asarray(predicted_dip)
    true_strike = np.asarray(truth_strike)
    true_dip = np.asarray(truth_dip)
    sample_mask = np.asarray(mask, dtype=bool)
    _validate_same_shape(
        pred_strike,
        pred_dip,
        true_strike,
        true_dip,
        sample_mask,
    )

    count = int(np.count_nonzero(sample_mask))
    if count == 0:
        return {
            "count": 0,
            "strike_mean": 0.0,
            "strike_median": 0.0,
            "strike_p90": 0.0,
            "strike_p95": 0.0,
            "dip_mean": 0.0,
            "dip_median": 0.0,
            "dip_p90": 0.0,
            "dip_p95": 0.0,
        }

    strike_error = _periodic_abs_error(pred_strike[sample_mask], true_strike[sample_mask], 180.0)
    dip_error = np.abs(
        pred_dip[sample_mask].astype(np.float64, copy=False)
        - true_dip[sample_mask].astype(np.float64, copy=False)
    )
    if not np.all(np.isfinite(strike_error)):
        raise ValueError("masked strike values must contain only finite values")
    if not np.all(np.isfinite(dip_error)):
        raise ValueError("masked dip values must contain only finite values")

    return {
        "count": count,
        **_prefixed_summary("strike", _distance_summary(strike_error)),
        **_prefixed_summary("dip", _distance_summary(dip_error)),
    }


def skin_mask_from_skins(
    skins: Sequence[FaultSkin],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a ``(n3, n2, n1)`` boolean mask for skin cell indices."""

    volume_shape = _validate_3d_shape(shape)
    mask = np.zeros(volume_shape, dtype=bool)
    for skin in skins:
        for cell in skin:
            i1, i2, i3 = _cell_index_i123(cell)
            _validate_cell_bounds((i1, i2, i3), volume_shape)
            mask[i3, i2, i1] = True
    return mask


def skin_topology_metrics(
    skins: Sequence[FaultSkin],
    shape: tuple[int, int, int],
    *,
    small_skin_size: int = 10,
) -> dict[str, float | int]:
    """Return size and duplicate-index summaries for fault skins."""

    volume_shape = _validate_3d_shape(shape)
    small_size = _validate_nonnegative_int(small_skin_size, "small_skin_size")
    skin_list = list(skins)

    cell_count = 0
    unique_indices: set[tuple[int, int, int]] = set()
    largest_skin_size = 0
    small_skin_count = 0
    small_skin_cell_count = 0

    for skin in skin_list:
        skin_size = len(skin)
        cell_count += skin_size
        largest_skin_size = max(largest_skin_size, skin_size)
        if skin_size < small_size:
            small_skin_count += 1
            small_skin_cell_count += skin_size
        for cell in skin:
            index = _cell_index_i123(cell)
            _validate_cell_bounds(index, volume_shape)
            unique_indices.add(index)

    unique_cell_count = len(unique_indices)
    return {
        "skin_count": len(skin_list),
        "cell_count": cell_count,
        "unique_cell_count": unique_cell_count,
        "duplicate_cell_count": cell_count - unique_cell_count,
        "largest_skin_size": largest_skin_size,
        "largest_skin_fraction": float(largest_skin_size / cell_count) if cell_count else 0.0,
        "small_skin_size": small_size,
        "small_skin_count": small_skin_count,
        "small_skin_cell_count": small_skin_cell_count,
        "small_skin_cell_fraction": (
            float(small_skin_cell_count / cell_count) if cell_count else 0.0
        ),
    }


def skin_orientation_error(
    skins: Sequence[FaultSkin],
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
) -> dict[str, float | int]:
    """Return strike and dip error summaries for skin cells against truth volumes."""

    true_strike, true_dip = _validate_orientation_truth_arrays(truth_strike, truth_dip)

    predicted_strikes: list[float] = []
    predicted_dips: list[float] = []
    sampled_strikes: list[float] = []
    sampled_dips: list[float] = []

    for skin in skins:
        for cell in skin:
            i1, i2, i3 = _cell_index_i123(cell)
            _validate_cell_bounds((i1, i2, i3), true_strike.shape)
            fp = _finite_cell_orientation(getattr(cell, "fp"), "fp")
            ft = _finite_cell_orientation(getattr(cell, "ft"), "ft")
            predicted_strikes.append(fp)
            predicted_dips.append(ft)
            sampled_strikes.append(float(true_strike[i3, i2, i1]))
            sampled_dips.append(float(true_dip[i3, i2, i1]))

    if not predicted_strikes:
        empty_mask = np.zeros(true_strike.shape, dtype=bool)
        return masked_orientation_error(true_strike, true_dip, true_strike, true_dip, empty_mask)

    mask = np.ones(len(predicted_strikes), dtype=bool)
    return masked_orientation_error(
        np.asarray(predicted_strikes, dtype=np.float32),
        np.asarray(predicted_dips, dtype=np.float32),
        np.asarray(sampled_strikes, dtype=np.float32),
        np.asarray(sampled_dips, dtype=np.float32),
        mask,
    )


def skin_truth_metrics(
    skins: Sequence[FaultSkin],
    *,
    shape: tuple[int, int, int],
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
    small_skin_size: int = 10,
) -> dict[str, Any]:
    """Return topology, overlap, distance, and orientation metrics for skins."""

    volume_shape = _validate_3d_shape(shape)
    skin_list = list(skins)
    candidate_mask = skin_mask_from_skins(skin_list, volume_shape)
    fault_mask = _validate_mask_shape(truth_fault_mask, volume_shape, "truth_fault_mask")
    surface_mask = _validate_mask_shape(truth_surface_mask, volume_shape, "truth_surface_mask")
    true_strike, true_dip = _validate_orientation_truth_arrays(truth_strike, truth_dip)
    _validate_same_shape(candidate_mask, true_strike, true_dip)
    buffer_key = _buffered_overlap_key(buffer_radius)

    return {
        "topology": skin_topology_metrics(
            skin_list,
            volume_shape,
            small_skin_size=small_skin_size,
        ),
        buffer_key: buffered_surface_overlap(
            candidate_mask,
            fault_mask,
            radius=buffer_radius,
        ),
        "surface_distance": surface_distance_metrics(candidate_mask, surface_mask),
        "orientation_error": skin_orientation_error(skin_list, true_strike, true_dip),
    }


def _validate_finite_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    try:
        finite = np.isfinite(array)
    except TypeError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(finite):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    count = int(k)
    if count < 0:
        raise ValueError("k must be non-negative")
    return count


def _validate_mask_pair(
    candidate_mask: np.ndarray,
    truth_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate_mask, dtype=bool)
    truth = np.asarray(truth_mask, dtype=bool)
    if candidate.shape != truth.shape:
        raise ValueError(f"mask shapes must match, got {candidate.shape} and {truth.shape}")
    return candidate, truth


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shape = arrays[0].shape
    for array in arrays[1:]:
        if array.shape != shape:
            raise ValueError(f"array shapes must match, got {shape} and {array.shape}")


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_3d_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("shape must be a positive 3D tuple")
    return tuple(_validate_positive_int(value, "shape") for value in shape)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} entries must be positive integers")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} entries must be positive integers")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} entries must be positive integers")
    return result


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _cell_index_i123(cell: object) -> tuple[int, int, int]:
    index = getattr(cell, "index", None)
    if index is None:
        index = (getattr(cell, "i1"), getattr(cell, "i2"), getattr(cell, "i3"))
    try:
        i1, i2, i3 = tuple(index)
    except (TypeError, ValueError) as error:
        raise ValueError("cell index must contain three entries in (i1, i2, i3) order") from error
    return (
        _validate_index_component(i1, "i1"),
        _validate_index_component(i2, "i2"),
        _validate_index_component(i3, "i3"),
    )


def _validate_index_component(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cell {name} index must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"cell {name} index must be an integer")
    return int(value)


def _validate_cell_bounds(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> None:
    i1, i2, i3 = index
    n3, n2, n1 = shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        raise ValueError(f"cell index {index} is outside volume shape {shape}")


def _validate_orientation_truth_arrays(
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_strike = _validate_finite_array(truth_strike, "truth_strike")
    true_dip = _validate_finite_array(truth_dip, "truth_dip")
    _validate_same_shape(true_strike, true_dip)
    if true_strike.ndim != 3:
        raise ValueError(f"truth arrays must be 3D, got shape {true_strike.shape}")
    return true_strike, true_dip


def _validate_mask_shape(
    mask: np.ndarray,
    shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} shape must match {shape}, got {result.shape}")
    return result


def _finite_cell_orientation(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"cell {name} orientation must be finite")
    return result


def _buffered_overlap_key(radius: float) -> str:
    value = _validate_nonnegative_finite_scalar(radius, "buffer_radius")
    if value.is_integer():
        suffix = str(int(value))
    else:
        suffix = str(value).replace(".", "p")
    return f"buffered_overlap_radius{suffix}"


def _distance_buffer(mask: np.ndarray, radius: float) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    return distance_transform_edt(~mask) <= radius


def _precision(intersection_count: int, candidate_count: int) -> float:
    if candidate_count == 0:
        return 1.0
    return float(intersection_count / candidate_count)


def _recall(intersection_count: int, truth_count: int) -> float:
    if truth_count == 0:
        return 1.0
    return float(intersection_count / truth_count)


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return float(2.0 * precision * recall / denominator) if denominator else 0.0


def _jaccard(intersection_count: int, union_count: int) -> float:
    if union_count == 0:
        return 1.0
    return float(intersection_count / union_count)


def _distance_summary(distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(distances, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": summary["mean"],
        f"{prefix}_median": summary["median"],
        f"{prefix}_p90": summary["p90"],
        f"{prefix}_p95": summary["p95"],
    }


def _distance_result(
    candidate_count: int,
    truth_count: int,
    candidate_to_truth_value: float,
    truth_to_candidate_value: float,
) -> dict[str, float | int]:
    candidate_to_truth = _constant_summary(candidate_to_truth_value)
    truth_to_candidate = _constant_summary(truth_to_candidate_value)
    return {
        "candidate_count": candidate_count,
        "truth_count": truth_count,
        **_prefixed_summary("candidate_to_truth", candidate_to_truth),
        **_prefixed_summary("truth_to_candidate", truth_to_candidate),
        "symmetric_chamfer_mean": float(
            0.5 * (candidate_to_truth["mean"] + truth_to_candidate["mean"])
        ),
        "hausdorff_p95": float(max(candidate_to_truth["p95"], truth_to_candidate["p95"])),
    }


def _constant_summary(value: float) -> dict[str, float]:
    result = float(value)
    return {
        "mean": result,
        "median": result,
        "p90": result,
        "p95": result,
    }


def _volume_diagonal(shape: tuple[int, ...]) -> float:
    return float(np.sqrt(np.sum((np.asarray(shape, dtype=np.float64) - 1.0) ** 2)))


def _periodic_abs_error(
    actual: np.ndarray,
    expected: np.ndarray,
    period: float,
) -> np.ndarray:
    actual_values = actual.astype(np.float64, copy=False)
    expected_values = expected.astype(np.float64, copy=False)
    half_period = 0.5 * period
    return np.abs((actual_values - expected_values + half_period) % period - half_period)
