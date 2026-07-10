"""Helpers for OSV-style dynamic-programming path kernels."""

from __future__ import annotations

import math
import operator

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE, njit
from pyosv.filters import smooth1d, smooth2d

__all__ = [
    "accumulate_2d",
    "accumulate_forward_2d",
    "backtrack_reverse_2d",
    "find_surface_3d",
    "find_path_2d",
    "shift_range",
    "smooth_fault_attributes_2d",
    "smooth_fault_attributes_3d",
    "smooth_surface_2d",
    "smooth_path_1d",
    "strain_to_bstrain",
    "update_shift_ranges",
    "update_shift_ranges_3d",
    "validate_cost_2d",
    "validate_cost_3d",
]


def validate_cost_2d(cost: np.ndarray) -> np.ndarray:
    """Validate and normalize a 2D ``(ni, nl)`` cost array."""

    cost_array = np.asarray(cost)
    if cost_array.ndim != 2:
        raise ValueError("cost must have shape (ni, nl)")

    cost_float32 = cost_array.astype(np.float32, copy=False)
    if not np.isfinite(cost_float32).all():
        raise ValueError("cost must contain only finite values")

    return cost_float32


def validate_cost_3d(cost: np.ndarray) -> np.ndarray:
    """Validate and normalize a 3D local surface cost volume."""

    cost_array = np.asarray(cost)
    if cost_array.ndim != 3:
        raise ValueError("cost must have shape (nw, nv, nu)")

    cost_float32 = cost_array.astype(np.float32, copy=False)
    if not np.isfinite(cost_float32).all():
        raise ValueError("cost must contain only finite values")

    return cost_float32


def accumulate_forward_2d(cost: np.ndarray, *, bstrain: int) -> np.ndarray:
    """Accumulate 2D path costs in the forward path direction."""

    return accumulate_2d(cost, bstrain=bstrain, direction=1)


def accumulate_2d(cost: np.ndarray, *, bstrain: int, direction: int = 1) -> np.ndarray:
    """Accumulate 2D path costs with a lag-change spacing constraint.

    The input shape is ``(ni, nl)`` where ``ni`` is the path direction and
    ``nl`` is the lag axis. Lower costs are preferred.
    """

    cost_array = validate_cost_2d(cost)
    bstrain_int = _validate_positive_int(bstrain, "bstrain")
    direction_int = _validate_direction(direction)

    if NUMBA_AVAILABLE:
        return _accumulate_2d_numba(cost_array, bstrain_int, direction_int)
    return _accumulate_2d_python(cost_array, bstrain_int, direction_int)


def _accumulate_2d_python(cost: np.ndarray, bstrain: int, direction: int) -> np.ndarray:
    ni, nl = cost.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    stop = ni if direction > 0 else -1
    step = 1 if direction > 0 else -1

    accumulated = np.empty_like(cost, dtype=np.float32)
    accumulated[start, :] = cost[start, :]

    for ii in range(start + step, stop, step):
        ji = min(max(ii - step, 0), ni_last)
        jb = min(max(ii - step * bstrain, 0), ni_last)

        for il in range(nl):
            il_minus = max(il - 1, 0)
            il_plus = min(il + 1, nl_last)

            cost_minus = accumulated[jb, il_minus]
            cost_same = accumulated[ji, il]
            cost_plus = accumulated[jb, il_plus]

            for kb in range(ji, jb, -step):
                cost_minus += cost[kb, il_minus]
                cost_plus += cost[kb, il_plus]

            accumulated[ii, il] = (
                _min3_prefer_center(
                    cost_minus,
                    cost_same,
                    cost_plus,
                )
                + cost[ii, il]
            )

    return accumulated


@njit(cache=True)
def _accumulate_2d_numba(cost: np.ndarray, bstrain: int, direction: int) -> np.ndarray:
    ni, nl = cost.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    stop = ni if direction > 0 else -1
    step = 1 if direction > 0 else -1

    accumulated = np.empty_like(cost, dtype=np.float32)
    for il in range(nl):
        accumulated[start, il] = cost[start, il]

    for ii in range(start + step, stop, step):
        ji = min(max(ii - step, 0), ni_last)
        jb = min(max(ii - step * bstrain, 0), ni_last)

        for il in range(nl):
            il_minus = max(il - 1, 0)
            il_plus = min(il + 1, nl_last)

            cost_minus = accumulated[jb, il_minus]
            cost_same = accumulated[ji, il]
            cost_plus = accumulated[jb, il_plus]

            for kb in range(ji, jb, -step):
                cost_minus += cost[kb, il_minus]
                cost_plus += cost[kb, il_plus]

            accumulated[ii, il] = (
                _min3_prefer_center_numba(cost_minus, cost_same, cost_plus) + cost[ii, il]
            )

    return accumulated


def backtrack_reverse_2d(
    accumulated: np.ndarray,
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> np.ndarray:
    """Backtrack a path in reverse through forward-accumulated 2D costs."""

    accumulated_array = validate_cost_2d(accumulated)
    cost_array = validate_cost_2d(cost)
    if accumulated_array.shape != cost_array.shape:
        raise ValueError("accumulated and cost must have the same shape")

    lmin_int = _validate_int(lmin, "lmin")
    bstrain_int = _validate_positive_int(bstrain, "bstrain")
    return _backtrack_2d(
        accumulated_array,
        cost_array,
        lmin=lmin_int,
        bstrain=bstrain_int,
        direction=-1,
    )


def smooth_path_1d(path: np.ndarray, sigma: float, *, bstrain: int = 1) -> np.ndarray:
    """Smooth a 1D lag path with the package Gaussian smoothing adapter."""

    path_array = np.asarray(path, dtype=np.float32)
    if path_array.ndim != 1:
        raise ValueError("path must have shape (ni,)")
    if not np.isfinite(path_array).all():
        raise ValueError("path must contain only finite values")

    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    bstrain_int = _validate_positive_int(bstrain, "bstrain")
    return smooth1d(path_array, sigma_float * bstrain_int)


def smooth_surface_2d(
    surface: np.ndarray,
    *,
    sigma1: float = 0.0,
    sigma2: float = 0.0,
) -> np.ndarray:
    """Smooth a 2D ``(nw, nv)`` lag surface along ``v`` and ``w`` axes."""

    surface_array = np.asarray(surface, dtype=np.float32)
    if surface_array.ndim != 2:
        raise ValueError("surface must have shape (nw, nv)")
    if not np.isfinite(surface_array).all():
        raise ValueError("surface must contain only finite values")

    sigma1_float = _validate_nonnegative_float(sigma1, "sigma1")
    sigma2_float = _validate_nonnegative_float(sigma2, "sigma2")
    if sigma1_float == 0.0 and sigma2_float == 0.0:
        return surface_array.copy()

    return smooth2d(surface_array, (sigma2_float, sigma1_float))


def smooth_fault_attributes_2d(cost: np.ndarray, *, bstrain: int) -> np.ndarray:
    """Smooth 2D fault attributes with forward and reverse DP accumulation."""

    cost_array = validate_cost_2d(cost)
    bstrain_int = _validate_positive_int(bstrain, "bstrain")
    forward = accumulate_2d(cost_array, bstrain=bstrain_int, direction=1)
    reverse = accumulate_2d(cost_array, bstrain=bstrain_int, direction=-1)
    return (forward + reverse - cost_array).astype(np.float32, copy=False)


def smooth_fault_attributes_3d(
    cost: np.ndarray,
    *,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray:
    """Smooth 3D local surface costs in ``v`` and then ``w`` directions."""

    cost_array = validate_cost_3d(cost)
    bstrain1_int = _validate_positive_int(bstrain1, "bstrain1")
    bstrain2_int = _validate_positive_int(bstrain2, "bstrain2")

    nw, nv, nu = cost_array.shape
    smoothed_v = np.empty((nw, nv, nu), dtype=np.float32)
    for iw in range(nw):
        smoothed_v[iw] = smooth_fault_attributes_2d(
            cost_array[iw],
            bstrain=bstrain1_int,
        )

    smoothed_w = np.empty_like(smoothed_v, dtype=np.float32)
    for iv in range(nv):
        smoothed_w[:, iv, :] = smooth_fault_attributes_2d(
            smoothed_v[:, iv, :],
            bstrain=bstrain2_int,
        )

    return smoothed_w


def find_path_2d(
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    attribute_smoothing: int = 1,
    path_smoothing: float = 0.0,
) -> np.ndarray:
    """Find a 1D optimal path through a 2D ``(ni, nl)`` cost image."""

    cost_array = validate_cost_2d(cost)
    lmin_int = _validate_int(lmin, "lmin")
    bstrain_int = _validate_positive_int(bstrain, "bstrain")
    attribute_smoothing_int = _validate_nonnegative_int(
        attribute_smoothing,
        "attribute_smoothing",
    )
    path_smoothing_float = _validate_nonnegative_float(path_smoothing, "path_smoothing")

    smoothed_cost = cost_array.copy()
    for _ in range(attribute_smoothing_int):
        smoothed_cost = smooth_fault_attributes_2d(smoothed_cost, bstrain=bstrain_int)

    accumulated = accumulate_forward_2d(smoothed_cost, bstrain=bstrain_int)
    path = backtrack_reverse_2d(
        accumulated,
        smoothed_cost,
        lmin=lmin_int,
        bstrain=bstrain_int,
    )
    if path_smoothing_float > 0.0:
        path = smooth_path_1d(path, path_smoothing_float, bstrain=bstrain_int)

    return path.astype(np.float32, copy=False)


def find_surface_3d(
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
    attribute_smoothing: int = 1,
    surface_smoothing1: float = 0.0,
    surface_smoothing2: float = 0.0,
) -> np.ndarray:
    """Find a 2D optimal lag surface through a 3D ``(nw, nv, nu)`` cost volume."""

    cost_array = validate_cost_3d(cost)
    lmin_int = _validate_int(lmin, "lmin")
    bstrain1_int = _validate_positive_int(bstrain1, "bstrain1")
    bstrain2_int = _validate_positive_int(bstrain2, "bstrain2")
    attribute_smoothing_int = _validate_nonnegative_int(
        attribute_smoothing,
        "attribute_smoothing",
    )
    surface_smoothing1_float = _validate_nonnegative_float(
        surface_smoothing1,
        "surface_smoothing1",
    )
    surface_smoothing2_float = _validate_nonnegative_float(
        surface_smoothing2,
        "surface_smoothing2",
    )

    smoothed_cost = cost_array.copy()
    for _ in range(attribute_smoothing_int):
        smoothed_cost = smooth_fault_attributes_3d(
            smoothed_cost,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        )

    nw, nv, _ = smoothed_cost.shape
    surface = np.empty((nw, nv), dtype=np.float32)
    for iw in range(nw):
        surface[iw] = find_path_2d(
            smoothed_cost[iw],
            lmin=lmin_int,
            bstrain=bstrain1_int,
            attribute_smoothing=0,
            path_smoothing=0.0,
        )

    if surface_smoothing1_float > 0.0 or surface_smoothing2_float > 0.0:
        surface = smooth_surface_2d(
            surface,
            sigma1=surface_smoothing1_float,
            sigma2=surface_smoothing2_float,
        )

    return surface.astype(np.float32, copy=False)


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

    ru_int = _validate_nonnegative_int(ru, "ru")
    lmin = -ru_int
    lmax = ru_int
    nl = 1 + lmax - lmin
    return lmin, lmax, nl


def update_shift_ranges(ru: int, rv: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_lmins`` and ``_lmaxs`` arrays for OSV shift bounds."""

    ru_int = _validate_nonnegative_int(ru, "ru")
    rv_int = _validate_nonnegative_int(rv, "rv")
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


def update_shift_ranges_3d(ru: int, rv: int, rw: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_lmins`` and ``_lmaxs`` arrays for 3D surface shift bounds."""

    ru_int = _validate_nonnegative_int(ru, "ru")
    rv_int = _validate_nonnegative_int(rv, "rv")
    rw_int = _validate_nonnegative_int(rw, "rw")

    nv = 2 * rv_int + 1
    nw = 2 * rw_int + 1
    lmins = np.zeros((nw, nv), dtype=np.int32)
    lmaxs = np.zeros((nw, nv), dtype=np.int32)

    for iw in range(-rw_int, rw_int + 1):
        iw_index = iw + rw_int
        for iv in range(-rv_int, rv_int + 1):
            wv = math.sqrt(iw * iw + iv * iv)
            if wv > 2.0:
                shift = _java_round(wv)
                iv_index = iv + rv_int
                lmins[iw_index, iv_index] = max(-shift, -ru_int)
                lmaxs[iw_index, iv_index] = min(shift, ru_int)

    return lmins, lmaxs


def _backtrack_2d(
    accumulated: np.ndarray,
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _backtrack_2d_numba(accumulated, cost, lmin, bstrain, direction)
    return _backtrack_2d_python(accumulated, cost, lmin, bstrain, direction)


def _backtrack_2d_python(
    accumulated: np.ndarray,
    cost: np.ndarray,
    lmin: int,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    ni, nl = accumulated.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    end = ni_last if direction > 0 else 0
    step = 1 if direction > 0 else -1
    inverse_bstrain = 1.0 / bstrain

    path = np.empty(ni, dtype=np.float32)
    ii = start
    il = min(max(-lmin, 0), nl_last)
    best_cost = accumulated[ii, il]
    for lag_index in range(nl):
        if accumulated[ii, lag_index] < best_cost:
            il = lag_index
            best_cost = accumulated[ii, lag_index]

    path[ii] = il + lmin
    while ii != end:
        ji = min(max(ii + step, 0), ni_last)
        jb = min(max(ii + step * bstrain, 0), ni_last)
        il_minus = max(il - 1, 0)
        il_plus = min(il + 1, nl_last)

        cost_minus = accumulated[jb, il_minus]
        cost_same = accumulated[ji, il]
        cost_plus = accumulated[jb, il_plus]
        for kb in range(ji, jb, step):
            cost_minus += cost[kb, il_minus]
            cost_plus += cost[kb, il_plus]

        lag_changed = False
        best_cost = _min3_prefer_center(cost_minus, cost_same, cost_plus)
        if best_cost != cost_same:
            if best_cost == cost_minus:
                next_il = il_minus
            else:
                next_il = il_plus
            lag_changed = next_il != il
            il = next_il

        ii += step
        path[ii] = il + lmin
        if lag_changed:
            du = (path[ii] - path[ii - step]) * inverse_bstrain
            path[ii] = path[ii - step] + du
            for _ in range(ji, jb, step):
                ii += step
                path[ii] = path[ii - step] + du

    return path


@njit(cache=True)
def _backtrack_2d_numba(
    accumulated: np.ndarray,
    cost: np.ndarray,
    lmin: int,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    ni, nl = accumulated.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    end = ni_last if direction > 0 else 0
    step = 1 if direction > 0 else -1
    inverse_bstrain = 1.0 / bstrain

    path = np.empty(ni, dtype=np.float32)
    ii = start
    il = min(max(-lmin, 0), nl_last)
    best_cost = accumulated[ii, il]
    for lag_index in range(nl):
        if accumulated[ii, lag_index] < best_cost:
            il = lag_index
            best_cost = accumulated[ii, lag_index]

    path[ii] = il + lmin
    while ii != end:
        ji = min(max(ii + step, 0), ni_last)
        jb = min(max(ii + step * bstrain, 0), ni_last)
        il_minus = max(il - 1, 0)
        il_plus = min(il + 1, nl_last)

        cost_minus = accumulated[jb, il_minus]
        cost_same = accumulated[ji, il]
        cost_plus = accumulated[jb, il_plus]
        for kb in range(ji, jb, step):
            cost_minus += cost[kb, il_minus]
            cost_plus += cost[kb, il_plus]

        lag_changed = False
        best_cost = _min3_prefer_center_numba(cost_minus, cost_same, cost_plus)
        if best_cost != cost_same:
            if best_cost == cost_minus:
                next_il = il_minus
            else:
                next_il = il_plus
            lag_changed = next_il != il
            il = next_il

        ii += step
        path[ii] = il + lmin
        if lag_changed:
            du = (path[ii] - path[ii - step]) * inverse_bstrain
            path[ii] = path[ii - step] + du
            for _ in range(ji, jb, step):
                ii += step
                path[ii] = path[ii - step] + du

    return path


def _min3_prefer_center(a: float, b: float, c: float) -> float:
    if b <= a:
        if b <= c:
            return b
        return c
    if a <= c:
        return a
    return c


@njit(cache=True)
def _min3_prefer_center_numba(a: float, b: float, c: float) -> float:
    if b <= a:
        if b <= c:
            return b
        return c
    if a <= c:
        return a
    return c


def _validate_direction(direction: int) -> int:
    direction_int = _validate_int(direction, "direction")
    if direction_int not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return direction_int


def _java_round(value: float) -> int:
    return math.floor(float(value) + 0.5)


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_positive_int(value: int, name: str) -> int:
    value_int = _validate_int(value, name)
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _validate_nonnegative_float(value: float, name: str) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc

    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float


def _validate_nonnegative_int(value: int, name: str) -> int:
    try:
        value_int = _validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

    return value_int
