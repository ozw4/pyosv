"""Practical-equivalence metrics for OSV output comparisons."""

from __future__ import annotations

import numpy as np


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


def top_percentile_mask(x: np.ndarray, percentile: float) -> np.ndarray:
    """Return a boolean mask for values at or above a percentile threshold."""

    values = np.asarray(x)
    if values.size == 0:
        raise ValueError("array must not be empty")
    _validate_percentile(percentile)
    if not np.all(np.isfinite(values)):
        raise ValueError("array must contain only finite values")

    threshold = float(np.percentile(values.astype(np.float64, copy=False), percentile))
    return values >= threshold


def top_percentile_overlap(
    a: np.ndarray, b: np.ndarray, percentile: float = 95.0
) -> dict[str, float]:
    """Compare overlap of high-value masks from two finite arrays."""

    av, bv = _validate_comparable_finite_arrays(a, b)
    a_mask = top_percentile_mask(av, percentile)
    b_mask = top_percentile_mask(bv, percentile)

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
    av = np.asarray(a)
    bv = np.asarray(b)
    if av.shape != bv.shape:
        raise ValueError(f"array shapes must match, got {av.shape} and {bv.shape}")
    if not np.all(np.isfinite(av)):
        raise ValueError("first array must contain only finite values")
    if not np.all(np.isfinite(bv)):
        raise ValueError("second array must contain only finite values")
    return av, bv


def _validate_percentile(percentile: float) -> None:
    if not np.isfinite(percentile):
        raise ValueError("percentile must be finite")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")
