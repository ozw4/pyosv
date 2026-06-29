"""Smoothing adapters for OSV-style array layouts."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["smooth1d", "smooth2d", "smooth3d"]


def smooth1d(x, sigma: float, *, axis: int = -1, mode: str = "nearest") -> np.ndarray:
    """Smooth along one axis using SciPy Gaussian filtering.

    This is a practical approximation of Mines JTK recursive exponential and
    recursive Gaussian smoothing, not a bit-exact reimplementation.
    """

    x_array = np.asarray(x)
    if sigma <= 0:
        return x_array.copy()

    smoothed = ndimage.gaussian_filter1d(x_array, sigma=sigma, axis=axis, mode=mode)
    return _restore_array_type(smoothed, x_array.dtype)


def smooth2d(x, sigma: float | tuple[float, float], *, mode: str = "nearest") -> np.ndarray:
    """Smooth a 2D ``(n2, n1)`` array using SciPy Gaussian filtering.

    This is a practical approximation of Mines JTK recursive exponential and
    recursive Gaussian smoothing, not a bit-exact reimplementation.
    """

    x_array = np.asarray(x)
    if x_array.ndim != 2:
        raise ValueError("x must have shape (n2, n1)")
    if _all_sigmas_nonpositive(sigma):
        return x_array.copy()

    smoothed = ndimage.gaussian_filter(x_array, sigma=sigma, mode=mode)
    return _restore_array_type(smoothed, x_array.dtype)


def smooth3d(
    x,
    sigma: float | tuple[float, float, float],
    *,
    mode: str = "nearest",
) -> np.ndarray:
    """Smooth a 3D ``(n3, n2, n1)`` array using SciPy Gaussian filtering.

    This is a practical approximation of Mines JTK recursive exponential and
    recursive Gaussian smoothing, not a bit-exact reimplementation.
    """

    x_array = np.asarray(x)
    if x_array.ndim != 3:
        raise ValueError("x must have shape (n3, n2, n1)")
    if _all_sigmas_nonpositive(sigma):
        return x_array.copy()

    smoothed = ndimage.gaussian_filter(x_array, sigma=sigma, mode=mode)
    return _restore_array_type(smoothed, x_array.dtype)


def _all_sigmas_nonpositive(sigma: float | tuple[float, ...]) -> bool:
    sigma_array = np.asarray(sigma)
    return bool(np.all(sigma_array <= 0))


def _restore_array_type(smoothed: np.ndarray, input_dtype: np.dtype) -> np.ndarray:
    if input_dtype == np.dtype("float32") and smoothed.dtype != np.float32:
        return smoothed.astype(np.float32, copy=False)
    return smoothed
