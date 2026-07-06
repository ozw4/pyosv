"""Metrics for controlled synthetic truth evaluations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "buffered_surface_overlap",
    "masked_orientation_error",
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
