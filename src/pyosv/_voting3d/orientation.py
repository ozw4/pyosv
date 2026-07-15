"""Fault-surface orientation helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyosv._voting3d.validation import (
    _validate_finite_array2,
    _validate_finite_vector3,
    _validate_nonnegative_float,
)
from pyosv.dp import smooth_surface_2d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives


_SURFACE_ORIENTATION_BACKENDS = ("full_surface", "center_separable")
_GAUSSIAN_TRUNCATE = 4.0


def _surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    *,
    sigma: float | None = None,
    backend: str = "full_surface",
) -> tuple[float, float]:
    """Return orientation from center differences of a local ``u(w,v)`` surface.

    ``surface`` must be a finite 2D ``(nw, nv)`` array with at least three
    samples along both axes. If ``sigma`` is ``None`` or ``0.0``, derivatives
    are computed from the raw surface. If ``sigma`` is positive, the surface is
    smoothed before computing centered ``du/dv`` and ``du/dw``. The input
    surface is never modified. Strike/dip signs are delegated to
    ``strike_and_dip_from_local_surface_derivatives``.
    """

    normal_array = _validate_finite_vector3(normal, "normal")
    dip_array = _validate_finite_vector3(dip, "dip")
    strike_array = _validate_finite_vector3(strike, "strike")
    if backend == "full_surface":
        du_dv, du_dw = _surface_center_derivatives_full_surface(surface, sigma)
    elif backend == "center_separable":
        du_dv, du_dw = _surface_center_derivatives_center_separable(surface, sigma)
    else:
        raise ValueError(
            "surface orientation backend must be one of "
            f"{_SURFACE_ORIENTATION_BACKENDS}, got {backend!r}",
        )
    return strike_and_dip_from_local_surface_derivatives(
        normal_array,
        dip_array,
        strike_array,
        du_dv,
        du_dw,
    )


def _surface_center_derivatives_full_surface(
    surface: np.ndarray,
    sigma: float | None,
) -> tuple[float, float]:
    """Smooth the full surface before taking its center differences."""

    surface_array = _smooth_surface_for_orientation(surface, sigma)
    return _surface_center_derivatives(surface_array)


def _surface_center_derivatives_center_separable(
    surface: np.ndarray,
    sigma: float | None,
) -> tuple[float, float]:
    """Compute only the smoothed 3x3 patch around the surface center."""

    center_patch = _smooth_surface_center_patch_separable(surface, sigma)
    return _surface_center_derivatives(center_patch)


def _smooth_surface_center_patch_separable(
    surface: np.ndarray,
    sigma: float | None,
) -> np.ndarray:
    """Return the smoothed 3x3 center patch by separable contraction."""

    surface_array = _validated_surface_copy(surface)
    nw, nv = surface_array.shape
    iw = nw // 2
    iv = nv // 2
    if sigma is None:
        return surface_array[iw - 1 : iw + 2, iv - 1 : iv + 2].copy()

    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    if sigma_float == 0.0:
        return surface_array[iw - 1 : iw + 2, iv - 1 : iv + 2].copy()

    weights_w = _gaussian_weights_nearest(
        nw,
        sigma_float,
        _GAUSSIAN_TRUNCATE,
        (iw - 1, iw, iw + 1),
    )
    weights_v = _gaussian_weights_nearest(
        nv,
        sigma_float,
        _GAUSSIAN_TRUNCATE,
        (iv - 1, iv, iv + 1),
    )

    # scipy.ndimage.gaussian_filter processes axis 0 before axis 1 and stores
    # intermediates in the output dtype. Preserve that float32 rounding here.
    smoothed_w = (weights_w @ surface_array).astype(np.float32)
    return (smoothed_w @ weights_v.T).astype(np.float32)


@lru_cache(maxsize=128)
def _gaussian_weights_nearest(
    size: int,
    sigma: float,
    truncate: float,
    targets: tuple[int, ...],
) -> np.ndarray:
    """Return Gaussian weights with SciPy's nearest-edge folding."""

    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    scaled = offsets.astype(np.float64) / sigma
    kernel = np.exp(-0.5 * scaled * scaled)
    kernel /= np.sum(kernel)

    weights = np.zeros((len(targets), size), dtype=np.float64)
    for row, target in enumerate(targets):
        indices = np.clip(target + offsets, 0, size - 1)
        np.add.at(weights[row], indices, kernel)
    weights.flags.writeable = False
    return weights


def _smooth_surface_for_orientation(
    surface: np.ndarray,
    sigma: float | None,
) -> np.ndarray:
    surface_array = _validated_surface_copy(surface)

    if sigma is None:
        return surface_array

    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    if sigma_float == 0.0:
        return surface_array

    return smooth_surface_2d(
        surface_array,
        sigma1=sigma_float,
        sigma2=sigma_float,
    ).astype(np.float32, copy=False)


def _validated_surface_copy(surface: np.ndarray) -> np.ndarray:
    surface_array = _validate_finite_array2(surface, "surface").astype(
        np.float32,
        copy=True,
    )
    if surface_array.shape[0] < 3 or surface_array.shape[1] < 3:
        raise ValueError("surface must have at least three samples along w and v")
    return surface_array


def _surface_center_derivatives(surface: np.ndarray) -> tuple[float, float]:
    iw = surface.shape[0] // 2
    iv = surface.shape[1] // 2
    du_dv = float(0.5 * (surface[iw, iv + 1] - surface[iw, iv - 1]))
    du_dw = float(0.5 * (surface[iw + 1, iv] - surface[iw - 1, iv]))
    return du_dv, du_dw
