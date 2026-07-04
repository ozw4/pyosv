"""Practical-equivalence metrics for OSV output comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def finite_value_report(x: np.ndarray) -> dict[str, float | int | tuple[int, ...]]:
    """Return finite and non-finite value counts for an array.

    Non-finite values are reported explicitly. Summary statistics are computed
    over finite values only; if there are no finite values, the finite summary
    fields are ``nan``.
    """

    values = np.asarray(x)
    finite = np.isfinite(values)
    nan = np.isnan(values)
    posinf = np.isposinf(values)
    neginf = np.isneginf(values)

    size = values.size
    finite_count = int(np.count_nonzero(finite))
    if finite_count:
        finite_values = values[finite].astype(np.float64, copy=False)
        finite_min = float(np.min(finite_values))
        finite_max = float(np.max(finite_values))
        finite_mean = float(np.mean(finite_values))
    else:
        finite_min = float("nan")
        finite_max = float("nan")
        finite_mean = float("nan")

    return {
        "shape": values.shape,
        "size": int(size),
        "finite_count": finite_count,
        "nan_count": int(np.count_nonzero(nan)),
        "posinf_count": int(np.count_nonzero(posinf)),
        "neginf_count": int(np.count_nonzero(neginf)),
        "finite_fraction": float(finite_count / size) if size else 0.0,
        "finite_min": finite_min,
        "finite_max": finite_max,
        "finite_mean": finite_mean,
    }


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return zero-mean normalized correlation for two finite arrays.

    Constant inputs have undefined correlation and carry no localization signal,
    so this function returns ``0.0`` when either centered input has zero norm.
    """

    av, bv = _validate_comparable_finite_arrays(a, b)
    if av.size == 0:
        raise ValueError("arrays must not be empty")

    avec = av.astype(np.float64, copy=False).ravel()
    bvec = bv.astype(np.float64, copy=False).ravel()
    avec = avec - float(np.mean(avec))
    bvec = bvec - float(np.mean(bvec))

    anorm = float(np.linalg.norm(avec))
    bnorm = float(np.linalg.norm(bvec))
    if anorm == 0.0 or bnorm == 0.0:
        return 0.0
    return float(np.dot(avec, bvec) / (anorm * bnorm))


def top_percentile_mask(
    x: np.ndarray, percentile: float, *, positive_only: bool = True
) -> np.ndarray:
    """Return a boolean mask for values at or above a percentile threshold.

    When ``positive_only`` is true, values less than or equal to zero are never
    selected. Arrays with no positive samples return an empty mask.
    """

    values = np.asarray(x)
    if values.size == 0:
        raise ValueError("array must not be empty")
    _validate_percentile(percentile)
    if not np.all(np.isfinite(values)):
        raise ValueError("array must contain only finite values")

    if positive_only:
        selectable = values > 0
        if not np.any(selectable):
            return np.zeros(values.shape, dtype=bool)
        percentile_values = values[selectable]
    else:
        percentile_values = values

    threshold = float(np.percentile(percentile_values.astype(np.float64, copy=False), percentile))
    mask = values >= threshold
    if positive_only:
        mask &= selectable
    return mask


def buffered_ridge_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    percentile: float = 99.0,
    radius: float = 2.0,
    positive_only: bool = True,
) -> dict[str, float | int]:
    """Compare sparse ridge masks with exact and buffered overlap metrics.

    Empty masks produce zero counts and zero-valued overlap ratios. Buffered
    precision counts candidate ridge samples inside the dilated reference mask;
    buffered recall counts reference ridge samples inside the dilated candidate
    mask.
    """

    reference_values, candidate_values = _validate_comparable_finite_arrays(reference, candidate)
    _validate_radius(radius)

    reference_mask = top_percentile_mask(reference_values, percentile, positive_only=positive_only)
    candidate_mask = top_percentile_mask(candidate_values, percentile, positive_only=positive_only)

    reference_count = int(np.count_nonzero(reference_mask))
    candidate_count = int(np.count_nonzero(candidate_mask))
    intersection_count = int(np.count_nonzero(reference_mask & candidate_mask))
    union_count = int(np.count_nonzero(reference_mask | candidate_mask))

    reference_buffer = _dilate_mask(reference_mask, radius)
    candidate_buffer = _dilate_mask(candidate_mask, radius)
    candidate_in_reference_buffer_count = int(np.count_nonzero(candidate_mask & reference_buffer))
    reference_in_candidate_buffer_count = int(np.count_nonzero(reference_mask & candidate_buffer))

    precision = _ratio(intersection_count, candidate_count)
    recall = _ratio(intersection_count, reference_count)
    buffered_precision = _ratio(candidate_in_reference_buffer_count, candidate_count)
    buffered_recall = _ratio(reference_in_candidate_buffer_count, reference_count)

    return {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "intersection_count": intersection_count,
        "union_count": union_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "jaccard": _ratio(intersection_count, union_count),
        "candidate_in_reference_buffer_count": candidate_in_reference_buffer_count,
        "reference_in_candidate_buffer_count": reference_in_candidate_buffer_count,
        "buffered_precision": buffered_precision,
        "buffered_recall": buffered_recall,
        "buffered_f1": _f1(buffered_precision, buffered_recall),
        "radius": float(radius),
        "percentile": float(percentile),
    }


def sparse_ridge_distance_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    percentile: float = 99.0,
    positive_only: bool = True,
) -> dict[str, float | int | None]:
    """Return symmetric distance-transform metrics between sparse ridge masks.

    If either ridge mask is empty, all distance values are ``None``. This avoids
    reporting misleading infinite or volume-size-dependent distances when there
    is no ridge target on one side.
    """

    reference_values, candidate_values = _validate_comparable_finite_arrays(reference, candidate)
    reference_mask = top_percentile_mask(reference_values, percentile, positive_only=positive_only)
    candidate_mask = top_percentile_mask(candidate_values, percentile, positive_only=positive_only)

    reference_count = int(np.count_nonzero(reference_mask))
    candidate_count = int(np.count_nonzero(candidate_mask))
    result: dict[str, float | int | None] = {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "candidate_to_reference_mean": None,
        "candidate_to_reference_median": None,
        "candidate_to_reference_p90": None,
        "candidate_to_reference_p95": None,
        "reference_to_candidate_mean": None,
        "reference_to_candidate_median": None,
        "reference_to_candidate_p90": None,
        "reference_to_candidate_p95": None,
        "percentile": float(percentile),
    }
    if reference_count == 0 or candidate_count == 0:
        return result

    distance_to_reference = distance_transform_edt(~reference_mask)
    candidate_to_reference = distance_to_reference[candidate_mask]
    result.update(_distance_summary("candidate_to_reference", candidate_to_reference))

    distance_to_candidate = distance_transform_edt(~candidate_mask)
    reference_to_candidate = distance_to_candidate[reference_mask]
    result.update(_distance_summary("reference_to_candidate", reference_to_candidate))
    return result


def orientation_field_report(
    likelihood: np.ndarray,
    strike: np.ndarray,
    dip: np.ndarray,
    *,
    percentile: float = 99.0,
    nonzero_epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    """Summarize voting orientation fields on a high-likelihood mask."""

    likelihood_values, strike_values = _validate_comparable_shape_arrays(likelihood, strike)
    _, dip_values = _validate_comparable_shape_arrays(likelihood_values, dip)
    _validate_percentile(percentile)
    epsilon = _validate_nonnegative_finite_scalar(nonzero_epsilon, "nonzero_epsilon")

    high_likelihood_mask = _finite_positive_percentile_mask(likelihood_values, percentile)
    finite_strike = np.isfinite(strike_values)
    finite_dip = np.isfinite(dip_values)
    paired_mask = high_likelihood_mask & finite_strike & finite_dip

    return {
        "likelihood_source": "fv",
        "likelihood_percentile": float(percentile),
        "likelihood_mask_count": int(np.count_nonzero(high_likelihood_mask)),
        "paired_sample_count": int(np.count_nonzero(paired_mask)),
        "strike": {
            **_orientation_value_counts(strike_values, epsilon),
            "high_likelihood": _distribution_summary(
                strike_values[high_likelihood_mask & finite_strike]
            ),
        },
        "dip": {
            **_orientation_value_counts(dip_values, epsilon),
            "high_likelihood": _distribution_summary(dip_values[high_likelihood_mask & finite_dip]),
        },
        "paired_high_likelihood": {
            "strike": _distribution_summary(strike_values[paired_mask]),
            "dip": _distribution_summary(dip_values[paired_mask]),
        },
    }


def orientation_angle_error(
    actual: np.ndarray,
    expected_degrees: float | np.ndarray,
    *,
    period: float = 180.0,
) -> np.ndarray:
    """Return wrapped absolute angular error in degrees.

    ``period`` controls angle equivalence. Use ``period=180`` for normal or
    strike axes where opposite directions are equivalent, and ``period=360``
    when full azimuth wraparound should be preserved.
    """

    values = np.asarray(actual)
    expected = np.asarray(expected_degrees)
    angle_period = _validate_positive_finite_scalar(period, "period")
    if not np.all(np.isfinite(values)):
        raise ValueError("actual angles must contain only finite values")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected_degrees must contain only finite values")

    try:
        actual_values, expected_values = np.broadcast_arrays(
            values.astype(np.float32, copy=False),
            expected.astype(np.float32, copy=False),
        )
    except ValueError as error:
        raise ValueError("actual and expected angles must be broadcast-compatible") from error

    half_period = np.float32(0.5 * angle_period)
    return np.abs(
        (actual_values - expected_values + half_period) % np.float32(angle_period) - half_period
    ).astype(np.float32, copy=False)


def strike_dip_angle_error(
    actual_strike: np.ndarray,
    actual_dip: np.ndarray,
    *,
    expected_strike: float,
    expected_dip: float,
    strike_period: float = 360.0,
) -> dict[str, np.ndarray]:
    """Return strike wraparound error and absolute dip error in degrees."""

    strike_values, dip_values = _validate_comparable_finite_arrays(actual_strike, actual_dip)
    dip_expected = _validate_finite_scalar(expected_dip, "expected_dip")
    return {
        "strike": orientation_angle_error(
            strike_values,
            expected_strike,
            period=strike_period,
        ),
        "dip": np.abs(dip_values.astype(np.float32, copy=False) - np.float32(dip_expected)).astype(
            np.float32,
            copy=False,
        ),
    }


def top_percentile_overlap(
    a: np.ndarray, b: np.ndarray, percentile: float = 95.0, *, positive_only: bool = False
) -> dict[str, float]:
    """Compare overlap of high-value masks from two finite arrays."""

    av, bv = _validate_comparable_finite_arrays(a, b)
    a_mask = top_percentile_mask(av, percentile, positive_only=positive_only)
    b_mask = top_percentile_mask(bv, percentile, positive_only=positive_only)

    size = float(a_mask.size)
    a_count = float(np.count_nonzero(a_mask))
    b_count = float(np.count_nonzero(b_mask))
    overlap_count = float(np.count_nonzero(a_mask & b_mask))
    union_count = float(np.count_nonzero(a_mask | b_mask))

    return {
        "percentile": float(percentile),
        "a_count": a_count,
        "b_count": b_count,
        "overlap_count": overlap_count,
        "union_count": union_count,
        "a_fraction": a_count / size,
        "b_fraction": b_count / size,
        "overlap_fraction": overlap_count / size,
        "overlap_over_a": overlap_count / a_count if a_count else 0.0,
        "overlap_over_b": overlap_count / b_count if b_count else 0.0,
        "jaccard": overlap_count / union_count if union_count else 0.0,
    }


def _validate_comparable_finite_arrays(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    av, bv = _validate_comparable_shape_arrays(a, b)
    if not np.all(np.isfinite(av)):
        raise ValueError("first array must contain only finite values")
    if not np.all(np.isfinite(bv)):
        raise ValueError("second array must contain only finite values")
    return av, bv


def _validate_comparable_shape_arrays(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    av = np.asarray(a)
    bv = np.asarray(b)
    if av.shape != bv.shape:
        raise ValueError(f"array shapes must match, got {av.shape} and {bv.shape}")
    return av, bv


def _validate_percentile(percentile: float) -> None:
    if not np.isfinite(percentile):
        raise ValueError("percentile must be finite")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")


def _validate_radius(radius: float) -> None:
    if not np.isfinite(radius):
        raise ValueError("radius must be finite")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")


def _validate_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_positive_percentile_mask(values: np.ndarray, percentile: float) -> np.ndarray:
    finite_positive = np.isfinite(values) & (values > 0)
    if not np.any(finite_positive):
        return np.zeros(values.shape, dtype=bool)
    threshold = float(
        np.percentile(values[finite_positive].astype(np.float64, copy=False), percentile)
    )
    return finite_positive & (values >= threshold)


def _orientation_value_counts(values: np.ndarray, nonzero_epsilon: float) -> dict[str, int | float]:
    finite = np.isfinite(values)
    finite_nonzero = finite & (np.abs(values) > nonzero_epsilon)
    return {
        "size": int(values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonzero_count": int(np.count_nonzero(finite_nonzero)),
        "finite_fraction": float(np.count_nonzero(finite) / values.size) if values.size else 0.0,
        "nonzero_fraction": (
            float(np.count_nonzero(finite_nonzero) / values.size) if values.size else 0.0
        ),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite_values = np.asarray(values)[np.isfinite(values)].astype(np.float64, copy=False)
    if finite_values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "median_absolute_deviation": None,
        }

    median = float(np.median(finite_values))
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(finite_values - median))),
    }


def _dilate_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius == 0.0 or not np.any(mask):
        return mask.copy()
    structure = _ball_structure(mask.ndim, radius)
    return binary_dilation(mask, structure=structure)


def _ball_structure(ndim: int, radius: float) -> np.ndarray:
    radius_samples = int(np.ceil(radius))
    axes = np.ogrid[(slice(-radius_samples, radius_samples + 1),) * ndim]
    distance_squared = np.zeros((2 * radius_samples + 1,) * ndim, dtype=np.float64)
    for axis in axes:
        distance_squared = distance_squared + axis.astype(np.float64) ** 2
    return distance_squared <= radius * radius


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return float(2.0 * precision * recall / denominator) if denominator else 0.0


def _distance_summary(prefix: str, distances: np.ndarray) -> dict[str, float]:
    values = distances.astype(np.float64, copy=False)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90.0)),
        f"{prefix}_p95": float(np.percentile(values, 95.0)),
    }
