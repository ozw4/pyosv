"""Fault-surface orientation helpers."""

from __future__ import annotations

import numpy as np

from pyosv._voting3d.validation import (
    _validate_finite_array2,
    _validate_finite_vector3,
    _validate_nonnegative_float,
)
from pyosv.dp import smooth_surface_2d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives


def _surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    *,
    sigma: float | None = None,
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
    surface_array = _smooth_surface_for_orientation(surface, sigma)
    du_dv, du_dw = _surface_center_derivatives(surface_array)
    return strike_and_dip_from_local_surface_derivatives(
        normal_array,
        dip_array,
        strike_array,
        du_dv,
        du_dw,
    )


def _smooth_surface_for_orientation(
    surface: np.ndarray,
    sigma: float | None,
) -> np.ndarray:
    surface_array = _validate_finite_array2(surface, "surface").astype(
        np.float32,
        copy=True,
    )
    if surface_array.shape[0] < 3 or surface_array.shape[1] < 3:
        raise ValueError("surface must have at least three samples along w and v")

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


def _surface_center_derivatives(surface: np.ndarray) -> tuple[float, float]:
    iw = surface.shape[0] // 2
    iv = surface.shape[1] // 2
    du_dv = float(0.5 * (surface[iw, iv + 1] - surface[iw, iv - 1]))
    du_dw = float(0.5 * (surface[iw + 1, iv] - surface[iw - 1, iv]))
    return du_dv, du_dw
