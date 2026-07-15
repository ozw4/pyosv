"""Fault-likelihood smoothing and unit-range normalization."""

from __future__ import annotations

import numpy as np

from pyosv._voting3d.validation import (
    _validate_finite_array3,
    _validate_nonnegative_float,
    _validate_positive_int,
)
from pyosv.filters import smooth3d


def _normalize_and_power_3d(
    x: np.ndarray,
    *,
    sigma: float = 0.0,
    power: int = 8,
) -> np.ndarray:
    """Normalize a final 3D vote map using Java-reference default semantics.

    By default this mirrors ``OptimalSurfaceVoter.normalization``: subtract the
    global minimum, divide by the global maximum when nonzero, then apply
    ``1 - (1 - x) ** power`` without additional smoothing. Set ``sigma > 0`` to
    opt in to the practical smoothed vote-map behavior.
    """

    x_array = _validate_finite_array3(x, "x")
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    power_int = _validate_positive_int(power, "power")
    return _normalize_and_power_3d_validated(
        x_array,
        sigma=sigma_float,
        power=power_int,
    )


def _normalize_and_power_3d_validated(
    x: np.ndarray,
    *,
    sigma: float,
    power: int,
) -> np.ndarray:
    """Normalize a validated finite float32 3D array with validated scalars.

    The caller must guarantee that ``x`` is a finite 3D ``float32`` array and
    that ``sigma`` and ``power`` satisfy the public helper's constraints.
    A copy is made before any operation, matching the public helper semantics.
    """

    x_array = x.astype(np.float32, copy=True)

    if x_array.size == 0:
        return x_array

    if sigma > 0.0:
        x_array = smooth3d(x_array, sigma).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(x_array)
    enhanced = np.float32(1.0) - np.power(np.float32(1.0) - x_array, power)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)


def _smooth_fault_likelihood_3d(
    ft: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    ft_array = _validate_finite_array3(ft, "ft")
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    return _smooth_fault_likelihood_3d_validated(ft_array, sigma=sigma_float)


def _smooth_fault_likelihood_3d_validated(
    ft: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Smooth a validated finite float32 3D array with validated ``sigma``.

    The caller must guarantee the same array and scalar preconditions enforced
    by :func:`_smooth_fault_likelihood_3d`. A copy is made before smoothing or
    normalization so the input remains unchanged.
    """

    ft_array = ft.astype(np.float32, copy=True)

    if ft_array.size == 0:
        return ft_array

    if sigma > 0.0:
        ft_array = smooth3d(ft_array, sigma).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(ft_array)
    return ft_array


def _normalize_unit_range_in_place(x: np.ndarray) -> None:
    x -= np.min(x)
    max_value = np.max(x)
    if max_value > 0.0:
        x /= max_value
    np.clip(x, 0.0, 1.0, out=x)
