"""Validated 2D dynamic-programming path operations."""

from __future__ import annotations

import math

import numpy as np

from pyosv._dp.path2d_numba import accumulate_2d as accumulate_2d_numba
from pyosv._dp.path2d_numba import backtrack_2d as backtrack_2d_numba
from pyosv._dp.path2d_python import accumulate_2d as accumulate_2d_python
from pyosv._dp.path2d_python import backtrack_2d as backtrack_2d_python
from pyosv._dp.validation import (
    validate_cost_2d,
    validate_direction,
    validate_int,
    validate_nonnegative_float,
    validate_nonnegative_int,
    validate_positive_int,
)


def strain_to_bstrain(strain_max: float) -> int:
    """Convert a maximum strain bound to the integer DP step spacing."""

    try:
        strain = float(strain_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("strain_max must satisfy 0 < strain_max <= 1") from exc
    if not math.isfinite(strain) or strain <= 0.0 or strain > 1.0:
        raise ValueError("strain_max must satisfy 0 < strain_max <= 1")
    return int(math.ceil(1.0 / strain))


def shift_range(ru: int) -> tuple[int, int, int]:
    """Return ``(lmin, lmax, nl)`` for the fault-normal shift radius ``ru``."""

    ru_int = validate_nonnegative_int(ru, "ru")
    lmin = -ru_int
    lmax = ru_int
    return lmin, lmax, 1 + lmax - lmin


def update_shift_ranges(ru: int, rv: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_lmins`` and ``_lmaxs`` arrays for OSV shift bounds."""

    ru_int = validate_nonnegative_int(ru, "ru")
    rv_int = validate_nonnegative_int(rv, "rv")
    lmin, lmax, _ = shift_range(ru_int)
    nv = 2 * rv_int + 1
    lmins = np.zeros(nv, dtype=np.int32)
    lmaxs = np.zeros(nv, dtype=np.int32)
    for iv in range(-rv_int, rv_int + 1):
        if abs(iv) > 2:
            index = iv + rv_int
            lmins[index] = max(-abs(iv), lmin)
            lmaxs[index] = min(abs(iv), lmax)
    return lmins, lmaxs


def accumulate_2d(
    cost: np.ndarray,
    *,
    bstrain: int,
    direction: int = 1,
    use_numba: bool,
) -> np.ndarray:
    cost_array = validate_cost_2d(cost)
    bstrain_int = validate_positive_int(bstrain, "bstrain")
    direction_int = validate_direction(direction)
    kernel = accumulate_2d_numba if use_numba else accumulate_2d_python
    return kernel(cost_array, bstrain_int, direction_int)


def backtrack_reverse_2d(
    accumulated: np.ndarray,
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    use_numba: bool,
) -> np.ndarray:
    accumulated_array = validate_cost_2d(accumulated)
    cost_array = validate_cost_2d(cost)
    if accumulated_array.shape != cost_array.shape:
        raise ValueError("accumulated and cost must have the same shape")
    lmin_int = validate_int(lmin, "lmin")
    bstrain_int = validate_positive_int(bstrain, "bstrain")
    kernel = backtrack_2d_numba if use_numba else backtrack_2d_python
    return kernel(accumulated_array, cost_array, lmin_int, bstrain_int, -1)


def find_path_2d(
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    attribute_smoothing: int = 1,
    path_smoothing: float = 0.0,
    use_numba: bool,
) -> np.ndarray:
    from pyosv._dp.smoothing import smooth_fault_attributes_2d, smooth_path_1d

    cost_array = validate_cost_2d(cost)
    lmin_int = validate_int(lmin, "lmin")
    bstrain_int = validate_positive_int(bstrain, "bstrain")
    attribute_smoothing_int = validate_nonnegative_int(attribute_smoothing, "attribute_smoothing")
    path_smoothing_float = validate_nonnegative_float(path_smoothing, "path_smoothing")

    smoothed_cost = cost_array.copy()
    for _ in range(attribute_smoothing_int):
        smoothed_cost = smooth_fault_attributes_2d(
            smoothed_cost, bstrain=bstrain_int, use_numba=use_numba
        )
    accumulated = accumulate_2d(
        smoothed_cost, bstrain=bstrain_int, direction=1, use_numba=use_numba
    )
    path = backtrack_reverse_2d(
        accumulated,
        smoothed_cost,
        lmin=lmin_int,
        bstrain=bstrain_int,
        use_numba=use_numba,
    )
    if path_smoothing_float > 0.0:
        path = smooth_path_1d(path, path_smoothing_float, bstrain=bstrain_int)
    return path.astype(np.float32, copy=False)
