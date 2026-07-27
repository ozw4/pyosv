"""Shared masks and counts for numeric candidate volumes."""

from __future__ import annotations

import numpy as np

NONZERO_EPSILON = 1.0e-6


def nonzero_mask(
    values: np.ndarray,
    *,
    epsilon: float = NONZERO_EPSILON,
) -> np.ndarray:
    """Return the canonical nonzero mask for finite numeric values.

    Floating-point and complex values use an epsilon-aware magnitude test.
    Integer and boolean values retain their exact nonzero semantics.
    """

    array = _finite_numeric_array(values)
    threshold = _epsilon(epsilon)
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer):
        return array != 0
    return np.abs(array) > threshold


def positive_candidate_mask(
    values: np.ndarray,
    *,
    epsilon: float = NONZERO_EPSILON,
) -> np.ndarray:
    """Return the canonical positive candidate mask for finite real values."""

    array = _finite_numeric_array(values)
    threshold = _epsilon(epsilon)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("values must be real")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer):
        return array > 0
    return array > threshold


def nonzero_count(
    values: np.ndarray,
    *,
    epsilon: float = NONZERO_EPSILON,
) -> int:
    """Count values selected by :func:`nonzero_mask`."""

    return int(np.count_nonzero(nonzero_mask(values, epsilon=epsilon)))


def nonzero_fraction(
    values: np.ndarray,
    *,
    epsilon: float = NONZERO_EPSILON,
) -> float:
    """Return the fraction selected by :func:`nonzero_mask`."""

    array = np.asarray(values)
    if array.size == 0:
        _finite_numeric_array(array)
        return 0.0
    return float(nonzero_count(array, epsilon=epsilon) / array.size)


def _finite_numeric_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    try:
        numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    except TypeError as error:
        raise ValueError("values must be numeric") from error
    if not numeric:
        raise ValueError("values must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")
    return array


def _epsilon(value: float) -> np.float32:
    epsilon = np.float32(value)
    if not np.isfinite(epsilon) or epsilon < np.float32(0.0):
        raise ValueError("epsilon must be finite and non-negative")
    return epsilon


__all__ = [
    "NONZERO_EPSILON",
    "nonzero_count",
    "nonzero_fraction",
    "nonzero_mask",
    "positive_candidate_mask",
]
