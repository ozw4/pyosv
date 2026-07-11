"""Validated 3D dynamic-programming surface operations."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pyosv._dp.validation import (
    validate_cost_3d,
    validate_int,
    validate_nonnegative_float,
    validate_nonnegative_int,
    validate_positive_int,
)


def find_surface_3d(
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
    attribute_smoothing: int = 1,
    surface_smoothing1: float = 0.0,
    surface_smoothing2: float = 0.0,
    find_path: Callable[..., np.ndarray],
    smooth_attributes: Callable[..., np.ndarray],
    smooth_surface: Callable[..., np.ndarray],
) -> np.ndarray:
    """Find a 2D optimal lag surface through a 3D ``(nw, nv, nu)`` cost volume."""

    cost_array = validate_cost_3d(cost)
    lmin_int = validate_int(lmin, "lmin")
    bstrain1_int = validate_positive_int(bstrain1, "bstrain1")
    bstrain2_int = validate_positive_int(bstrain2, "bstrain2")
    attribute_smoothing_int = validate_nonnegative_int(attribute_smoothing, "attribute_smoothing")
    surface_smoothing1_float = validate_nonnegative_float(surface_smoothing1, "surface_smoothing1")
    surface_smoothing2_float = validate_nonnegative_float(surface_smoothing2, "surface_smoothing2")

    smoothed_cost = cost_array.copy()
    for _ in range(attribute_smoothing_int):
        smoothed_cost = smooth_attributes(
            smoothed_cost,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        )

    nw, nv, _ = smoothed_cost.shape
    surface = np.empty((nw, nv), dtype=np.float32)
    for iw in range(nw):
        surface[iw] = find_path(
            smoothed_cost[iw],
            lmin=lmin_int,
            bstrain=bstrain1_int,
            attribute_smoothing=0,
            path_smoothing=0.0,
        )

    if surface_smoothing1_float > 0.0 or surface_smoothing2_float > 0.0:
        surface = smooth_surface(
            surface,
            sigma1=surface_smoothing1_float,
            sigma2=surface_smoothing2_float,
        )

    return surface.astype(np.float32, copy=False)


def update_shift_ranges_3d(ru: int, rv: int, rw: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_lmins`` and ``_lmaxs`` arrays for 3D surface shift bounds."""

    ru_int = validate_nonnegative_int(ru, "ru")
    rv_int = validate_nonnegative_int(rv, "rv")
    rw_int = validate_nonnegative_int(rw, "rw")

    nv = 2 * rv_int + 1
    nw = 2 * rw_int + 1
    lmins = np.zeros((nw, nv), dtype=np.int32)
    lmaxs = np.zeros((nw, nv), dtype=np.int32)

    for iw in range(-rw_int, rw_int + 1):
        iw_index = iw + rw_int
        for iv in range(-rv_int, rv_int + 1):
            wv = math.sqrt(iw * iw + iv * iv)
            if wv > 2.0:
                shift = math.floor(float(wv) + 0.5)
                iv_index = iv + rv_int
                lmins[iw_index, iv_index] = max(-shift, -ru_int)
                lmaxs[iw_index, iv_index] = min(shift, ru_int)

    return lmins, lmaxs
