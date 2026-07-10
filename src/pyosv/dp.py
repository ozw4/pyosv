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


def _find_surface_3d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
    attribute_smoothing: int = 1,
    surface_smoothing1: float = 0.0,
    surface_smoothing2: float = 0.0,
) -> tuple[np.ndarray | None, int]:
    """Find a surface while excluding invalid lag states from all DP stages.

    This private path is used by boundary-aware surface voting. ``None`` means
    that no strain-feasible surface spans the supplied tangential rectangle.
    The integer return value counts columns whose lag was projected back to a
    valid state after optional surface smoothing.
    """

    cost_array = validate_cost_3d(cost)
    valid_mask_array = _validate_valid_mask_3d(valid_mask, cost_array.shape)
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

    if 0 in cost_array.shape:
        return None, 0
    if not valid_mask_array.any(axis=2).all():
        return None, 0

    smoothed_cost = cost_array.copy()
    for _ in range(attribute_smoothing_int):
        smoothed_cost = _smooth_fault_attributes_3d_masked(
            smoothed_cost,
            valid_mask_array,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        )

    effective_mask = valid_mask_array & np.isfinite(smoothed_cost)
    if not effective_mask.any(axis=2).all():
        return None, 0

    surface = _extract_masked_surface_rows(
        smoothed_cost,
        effective_mask,
        lmin=lmin_int,
        bstrain=bstrain1_int,
    )
    if surface is None:
        surface = _minimum_cost_masked_surface(
            smoothed_cost,
            effective_mask,
            lmin=lmin_int,
        )
    if not effective_mask.all() and not _surface_respects_masked_strain(
        surface,
        effective_mask,
        lmin=lmin_int,
        bstrain1=bstrain1_int,
        bstrain2=bstrain2_int,
    ):
        surface = _recover_bidirectionally_feasible_surface(
            surface,
            effective_mask,
            lmin=lmin_int,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        )
        if surface is None:
            return None, 0

    smoothing_applied = surface_smoothing1_float > 0.0 or surface_smoothing2_float > 0.0
    if smoothing_applied:
        surface = smooth_surface_2d(
            surface,
            sigma1=surface_smoothing1_float,
            sigma2=surface_smoothing2_float,
        )
        surface, projection_count, projection_ok = _project_surface_to_valid_mask(
            surface,
            effective_mask,
            lmin_int,
        )
        if not projection_ok:
            return None, 0
    else:
        projection_count = 0
    return surface.astype(np.float32, copy=False), projection_count


def _extract_masked_surface_rows(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> np.ndarray | None:
    nrow, npath, _ = cost.shape
    surface = np.empty((nrow, npath), dtype=np.float32)
    for row in range(nrow):
        path, feasible = _find_path_2d_masked(
            cost[row],
            valid_mask[row],
            lmin=lmin,
            bstrain=bstrain,
        )
        if not feasible:
            return None
        surface[row] = path
    return surface


def _minimum_cost_masked_surface(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
) -> np.ndarray:
    nw, nv, _ = cost.shape
    surface = np.empty((nw, nv), dtype=np.float32)
    for iw in range(nw):
        for iv in range(nv):
            valid_indices = np.flatnonzero(valid_mask[iw, iv])
            valid_costs = cost[iw, iv, valid_indices]
            best_offset = int(np.argmin(valid_costs))
            surface[iw, iv] = np.float32(lmin + int(valid_indices[best_offset]))
    return surface


def _surface_respects_masked_strain(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
) -> bool:
    nw, nv = surface.shape
    nu = valid_mask.shape[2]
    for iw in range(nw):
        for iv in range(nv):
            lag_index = math.floor(float(surface[iw, iv] - lmin) + 0.5)
            if lag_index < 0 or lag_index >= nu or not valid_mask[iw, iv, lag_index]:
                return False

    tolerance = np.float32(1.0e-6)
    strain1 = np.float32(1.0 / bstrain1)
    strain2 = np.float32(1.0 / bstrain2)
    if nv > 1 and np.any(np.abs(np.diff(surface, axis=1)) > strain1 + tolerance):
        return False
    return not (nw > 1 and np.any(np.abs(np.diff(surface, axis=0)) > strain2 + tolerance))


def _recover_bidirectionally_feasible_surface(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray | None:
    """Recover a deterministic feasible surface from column lag intervals."""

    nw, nv, nu = valid_mask.shape
    lower = np.empty((nw, nv), dtype=np.float64)
    upper = np.empty((nw, nv), dtype=np.float64)
    for iw in range(nw):
        for iv in range(nv):
            valid_indices = np.flatnonzero(valid_mask[iw, iv])
            if valid_indices.size == 0:
                return None
            first_index = int(valid_indices[0])
            last_index = int(valid_indices[-1])
            first_lag = lmin + first_index
            last_lag = lmin + last_index
            lower[iw, iv] = max(float(lmin), float(first_lag) - 0.5)
            if last_index == nu - 1:
                upper[iw, iv] = float(last_lag)
            else:
                upper_boundary = np.float32(float(last_lag) + 0.5)
                upper[iw, iv] = float(
                    np.nextafter(
                        upper_boundary,
                        np.float32(-np.inf),
                        dtype=np.float32,
                    )
                )

    strain1 = 1.0 / bstrain1
    strain2 = 1.0 / bstrain2
    for _ in range(max(1, 2 * nw * nv)):
        changed = False
        for iw in range(nw):
            for iv in range(nv - 1):
                changed |= _tighten_strain_bounds_pair(
                    lower,
                    upper,
                    (iw, iv),
                    (iw, iv + 1),
                    strain1,
                )
        for iw in range(nw - 1):
            for iv in range(nv):
                changed |= _tighten_strain_bounds_pair(
                    lower,
                    upper,
                    (iw, iv),
                    (iw + 1, iv),
                    strain2,
                )
        if np.any(lower > upper + 1.0e-9):
            return None
        if not changed:
            break

    width = upper - lower
    denominator = float(np.sum(width * width))
    if denominator > 0.0:
        alpha = float(np.sum((surface.astype(np.float64) - lower) * width) / denominator)
        alpha = min(max(alpha, 0.0), 1.0)
    else:
        alpha = 0.0
    candidates = (
        lower + alpha * width,
        lower,
        upper,
        0.5 * (lower + upper),
    )
    for candidate in candidates:
        candidate_float32 = candidate.astype(np.float32)
        if _surface_respects_masked_strain(
            candidate_float32,
            valid_mask,
            lmin=lmin,
            bstrain1=bstrain1,
            bstrain2=bstrain2,
        ):
            return candidate_float32
    return None


def _tighten_strain_bounds_pair(
    lower: np.ndarray,
    upper: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    strain: float,
) -> bool:
    first_lower = lower[first]
    second_lower = lower[second]
    first_upper = upper[first]
    second_upper = upper[second]
    new_first_lower = max(first_lower, second_lower - strain)
    new_second_lower = max(second_lower, first_lower - strain)
    new_first_upper = min(first_upper, second_upper + strain)
    new_second_upper = min(second_upper, first_upper + strain)
    changed = (
        new_first_lower != first_lower
        or new_second_lower != second_lower
        or new_first_upper != first_upper
        or new_second_upper != second_upper
    )
    lower[first] = new_first_lower
    lower[second] = new_second_lower
    upper[first] = new_first_upper
    upper[second] = new_second_upper
    return changed


def _smooth_fault_attributes_3d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray:
    """Apply staged attribute smoothing without admitting masked lag states."""

    nw, nv, nu = cost.shape
    smoothed_v = np.empty((nw, nv, nu), dtype=np.float32)
    for iw in range(nw):
        smoothed_v[iw] = _smooth_fault_attributes_2d_masked(
            cost[iw],
            valid_mask[iw],
            bstrain1,
        )

    smoothed_w = np.empty_like(smoothed_v, dtype=np.float32)
    for iv in range(nv):
        smoothed_w[:, iv, :] = _smooth_fault_attributes_2d_masked(
            smoothed_v[:, iv, :],
            valid_mask[:, iv, :],
            bstrain2,
        )
    return smoothed_w


def _smooth_fault_attributes_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _smooth_fault_attributes_2d_masked_numba(cost, valid_mask, bstrain)
    return _smooth_fault_attributes_2d_masked_python(cost, valid_mask, bstrain)


def _smooth_fault_attributes_2d_masked_python(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    forward = _accumulate_2d_masked_python(cost, valid_mask, bstrain, 1)
    reverse = _accumulate_2d_masked_python(cost, valid_mask, bstrain, -1)
    smoothed = np.full(cost.shape, np.inf, dtype=np.float32)
    finite = valid_mask & np.isfinite(cost) & np.isfinite(forward) & np.isfinite(reverse)
    smoothed[finite] = forward[finite] + reverse[finite] - cost[finite]
    return smoothed


@njit(cache=True)
def _smooth_fault_attributes_2d_masked_numba(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    forward = _accumulate_2d_masked_numba(cost, valid_mask, bstrain, 1)
    reverse = _accumulate_2d_masked_numba(cost, valid_mask, bstrain, -1)
    ni, nl = cost.shape
    smoothed = np.full(cost.shape, np.inf, dtype=np.float32)
    for ii in range(ni):
        for il in range(nl):
            if (
                valid_mask[ii, il]
                and np.isfinite(cost[ii, il])
                and np.isfinite(forward[ii, il])
                and np.isfinite(reverse[ii, il])
            ):
                smoothed[ii, il] = forward[ii, il] + reverse[ii, il] - cost[ii, il]
    return smoothed


def _find_path_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> tuple[np.ndarray, bool]:
    if NUMBA_AVAILABLE:
        accumulated = _accumulate_2d_masked_numba(cost, valid_mask, bstrain, 1)
        return _backtrack_2d_masked_numba(
            accumulated,
            cost,
            valid_mask,
            lmin,
            bstrain,
            -1,
        )

    accumulated = _accumulate_2d_masked_python(cost, valid_mask, bstrain, 1)
    return _backtrack_2d_masked_python(
        accumulated,
        cost,
        valid_mask,
        lmin,
        bstrain,
        -1,
    )


def _accumulate_2d_masked_python(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    ni, nl = cost.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    stop = ni if direction > 0 else -1
    step = 1 if direction > 0 else -1

    accumulated = np.full(cost.shape, np.inf, dtype=np.float32)
    for il in range(nl):
        if valid_mask[start, il] and np.isfinite(cost[start, il]):
            accumulated[start, il] = cost[start, il]

    for ii in range(start + step, stop, step):
        ji = min(max(ii - step, 0), ni_last)
        jb = min(max(ii - step * bstrain, 0), ni_last)

        for il in range(nl):
            if not valid_mask[ii, il] or not np.isfinite(cost[ii, il]):
                continue

            il_minus = max(il - 1, 0)
            il_plus = min(il + 1, nl_last)
            cost_minus = np.float32(np.inf)
            if _masked_interpolated_transition_is_valid_python(
                valid_mask,
                ii,
                il,
                jb,
                il_minus,
                bstrain,
            ):
                cost_minus = _masked_transition_cost_python(
                    accumulated,
                    cost,
                    valid_mask,
                    ji,
                    jb,
                    -step,
                    il_minus,
                )
            cost_same = accumulated[ji, il]
            cost_plus = np.float32(np.inf)
            if _masked_interpolated_transition_is_valid_python(
                valid_mask,
                ii,
                il,
                jb,
                il_plus,
                bstrain,
            ):
                cost_plus = _masked_transition_cost_python(
                    accumulated,
                    cost,
                    valid_mask,
                    ji,
                    jb,
                    -step,
                    il_plus,
                )
            best_cost = _min3_prefer_center(cost_minus, cost_same, cost_plus)
            if np.isfinite(best_cost):
                accumulated[ii, il] = best_cost + cost[ii, il]

    return accumulated


def _masked_transition_cost_python(
    accumulated: np.ndarray,
    cost: np.ndarray,
    valid_mask: np.ndarray,
    start: int,
    stop: int,
    step: int,
    lag_index: int,
) -> np.float32:
    transition_cost = accumulated[stop, lag_index]
    if not np.isfinite(transition_cost):
        return np.float32(np.inf)
    for ii in range(start, stop, step):
        if not valid_mask[ii, lag_index] or not np.isfinite(cost[ii, lag_index]):
            return np.float32(np.inf)
        transition_cost += cost[ii, lag_index]
    return np.float32(transition_cost)


def _masked_interpolated_transition_is_valid_python(
    valid_mask: np.ndarray,
    current_index: int,
    current_lag_index: int,
    previous_index: int,
    previous_lag_index: int,
    bstrain: int,
) -> bool:
    step = 1 if previous_index > current_index else -1
    inverse_bstrain = np.float32(1.0 / bstrain)
    lag = np.float32(current_lag_index)
    lag_step = np.float32((previous_lag_index - current_lag_index) * inverse_bstrain)
    sample_index = current_index
    while sample_index != previous_index:
        sample_index += step
        lag = np.float32(lag + lag_step)
        selected_lag_index = math.floor(float(lag) + 0.5)
        if (
            selected_lag_index < 0
            or selected_lag_index >= valid_mask.shape[1]
            or not valid_mask[sample_index, selected_lag_index]
        ):
            return False
    return True


@njit(cache=True)
def _accumulate_2d_masked_numba(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    ni, nl = cost.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    stop = ni if direction > 0 else -1
    step = 1 if direction > 0 else -1

    accumulated = np.full(cost.shape, np.inf, dtype=np.float32)
    for il in range(nl):
        if valid_mask[start, il] and np.isfinite(cost[start, il]):
            accumulated[start, il] = cost[start, il]

    for ii in range(start + step, stop, step):
        ji = min(max(ii - step, 0), ni_last)
        jb = min(max(ii - step * bstrain, 0), ni_last)

        for il in range(nl):
            if not valid_mask[ii, il] or not np.isfinite(cost[ii, il]):
                continue

            il_minus = max(il - 1, 0)
            il_plus = min(il + 1, nl_last)
            cost_minus = np.float32(np.inf)
            if _masked_interpolated_transition_is_valid_numba(
                valid_mask,
                ii,
                il,
                jb,
                il_minus,
                bstrain,
            ):
                cost_minus = _masked_transition_cost_numba(
                    accumulated,
                    cost,
                    valid_mask,
                    ji,
                    jb,
                    -step,
                    il_minus,
                )
            cost_same = accumulated[ji, il]
            cost_plus = np.float32(np.inf)
            if _masked_interpolated_transition_is_valid_numba(
                valid_mask,
                ii,
                il,
                jb,
                il_plus,
                bstrain,
            ):
                cost_plus = _masked_transition_cost_numba(
                    accumulated,
                    cost,
                    valid_mask,
                    ji,
                    jb,
                    -step,
                    il_plus,
                )
            best_cost = _min3_prefer_center_numba(cost_minus, cost_same, cost_plus)
            if np.isfinite(best_cost):
                accumulated[ii, il] = best_cost + cost[ii, il]

    return accumulated


@njit(cache=True)
def _masked_transition_cost_numba(
    accumulated: np.ndarray,
    cost: np.ndarray,
    valid_mask: np.ndarray,
    start: int,
    stop: int,
    step: int,
    lag_index: int,
) -> np.float32:
    transition_cost = accumulated[stop, lag_index]
    if not np.isfinite(transition_cost):
        return np.float32(np.inf)
    for ii in range(start, stop, step):
        if not valid_mask[ii, lag_index] or not np.isfinite(cost[ii, lag_index]):
            return np.float32(np.inf)
        transition_cost += cost[ii, lag_index]
    return np.float32(transition_cost)


@njit(cache=True)
def _masked_interpolated_transition_is_valid_numba(
    valid_mask: np.ndarray,
    current_index: int,
    current_lag_index: int,
    previous_index: int,
    previous_lag_index: int,
    bstrain: int,
) -> bool:
    step = 1 if previous_index > current_index else -1
    inverse_bstrain = np.float32(1.0 / bstrain)
    lag = np.float32(current_lag_index)
    lag_step = np.float32((previous_lag_index - current_lag_index) * inverse_bstrain)
    sample_index = current_index
    while sample_index != previous_index:
        sample_index += step
        lag = np.float32(lag + lag_step)
        selected_lag_index = math.floor(float(lag) + 0.5)
        if (
            selected_lag_index < 0
            or selected_lag_index >= valid_mask.shape[1]
            or not valid_mask[sample_index, selected_lag_index]
        ):
            return False
    return True


def _backtrack_2d_masked_python(
    accumulated: np.ndarray,
    cost: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
    bstrain: int,
    direction: int,
) -> tuple[np.ndarray, bool]:
    ni, nl = accumulated.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    end = ni_last if direction > 0 else 0
    step = 1 if direction > 0 else -1
    inverse_bstrain = np.float32(1.0 / bstrain)

    path = np.zeros(ni, dtype=np.float32)
    il, found = _masked_start_lag_python(accumulated, valid_mask, start, lmin)
    if not found:
        return path, False

    ii = start
    path[ii] = il + lmin
    while ii != end:
        ji = min(max(ii + step, 0), ni_last)
        jb = min(max(ii + step * bstrain, 0), ni_last)
        il_minus = max(il - 1, 0)
        il_plus = min(il + 1, nl_last)

        cost_minus = np.float32(np.inf)
        if _masked_interpolated_transition_is_valid_python(
            valid_mask,
            ii,
            il,
            jb,
            il_minus,
            bstrain,
        ):
            cost_minus = _masked_transition_cost_python(
                accumulated,
                cost,
                valid_mask,
                ji,
                jb,
                step,
                il_minus,
            )
        cost_same = accumulated[ji, il]
        cost_plus = np.float32(np.inf)
        if _masked_interpolated_transition_is_valid_python(
            valid_mask,
            ii,
            il,
            jb,
            il_plus,
            bstrain,
        ):
            cost_plus = _masked_transition_cost_python(
                accumulated,
                cost,
                valid_mask,
                ji,
                jb,
                step,
                il_plus,
            )
        best_cost = _min3_prefer_center(cost_minus, cost_same, cost_plus)
        if not np.isfinite(best_cost):
            return path, False

        next_il = il
        if best_cost != cost_same:
            next_il = il_minus if best_cost == cost_minus else il_plus
        lag_changed = next_il != il
        il = next_il

        ii += step
        path[ii] = il + lmin
        if lag_changed:
            du = np.float32((path[ii] - path[ii - step]) * inverse_bstrain)
            path[ii] = path[ii - step] + du
            for _ in range(ji, jb, step):
                ii += step
                path[ii] = path[ii - step] + du

    return path, _path_uses_valid_mask_python(path, valid_mask, lmin)


def _masked_start_lag_python(
    accumulated: np.ndarray,
    valid_mask: np.ndarray,
    start: int,
    lmin: int,
) -> tuple[int, bool]:
    nl = accumulated.shape[1]
    center = min(max(-lmin, 0), nl - 1)
    best_lag = center
    best_cost = np.float32(np.inf)
    found = False
    if valid_mask[start, center] and np.isfinite(accumulated[start, center]):
        best_cost = accumulated[start, center]
        found = True
    for lag_index in range(nl):
        candidate = accumulated[start, lag_index]
        if valid_mask[start, lag_index] and np.isfinite(candidate) and candidate < best_cost:
            best_lag = lag_index
            best_cost = candidate
            found = True
    return best_lag, found


def _path_uses_valid_mask_python(
    path: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
) -> bool:
    nl = valid_mask.shape[1]
    for ii in range(path.shape[0]):
        lag_index = math.floor(float(path[ii] - lmin) + 0.5)
        if lag_index < 0 or lag_index >= nl or not valid_mask[ii, lag_index]:
            return False
    return True


@njit(cache=True)
def _backtrack_2d_masked_numba(
    accumulated: np.ndarray,
    cost: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
    bstrain: int,
    direction: int,
) -> tuple[np.ndarray, bool]:
    ni, nl = accumulated.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    end = ni_last if direction > 0 else 0
    step = 1 if direction > 0 else -1
    inverse_bstrain = np.float32(1.0 / bstrain)

    path = np.zeros(ni, dtype=np.float32)
    center = min(max(-lmin, 0), nl_last)
    il = center
    best_cost = np.float32(np.inf)
    found = False
    if valid_mask[start, center] and np.isfinite(accumulated[start, center]):
        best_cost = accumulated[start, center]
        found = True
    for lag_index in range(nl):
        candidate = accumulated[start, lag_index]
        if valid_mask[start, lag_index] and np.isfinite(candidate) and candidate < best_cost:
            il = lag_index
            best_cost = candidate
            found = True
    if not found:
        return path, False

    ii = start
    path[ii] = il + lmin
    while ii != end:
        ji = min(max(ii + step, 0), ni_last)
        jb = min(max(ii + step * bstrain, 0), ni_last)
        il_minus = max(il - 1, 0)
        il_plus = min(il + 1, nl_last)

        cost_minus = np.float32(np.inf)
        if _masked_interpolated_transition_is_valid_numba(
            valid_mask,
            ii,
            il,
            jb,
            il_minus,
            bstrain,
        ):
            cost_minus = _masked_transition_cost_numba(
                accumulated,
                cost,
                valid_mask,
                ji,
                jb,
                step,
                il_minus,
            )
        cost_same = accumulated[ji, il]
        cost_plus = np.float32(np.inf)
        if _masked_interpolated_transition_is_valid_numba(
            valid_mask,
            ii,
            il,
            jb,
            il_plus,
            bstrain,
        ):
            cost_plus = _masked_transition_cost_numba(
                accumulated,
                cost,
                valid_mask,
                ji,
                jb,
                step,
                il_plus,
            )
        best_cost = _min3_prefer_center_numba(cost_minus, cost_same, cost_plus)
        if not np.isfinite(best_cost):
            return path, False

        next_il = il
        if best_cost != cost_same:
            next_il = il_minus if best_cost == cost_minus else il_plus
        lag_changed = next_il != il
        il = next_il

        ii += step
        path[ii] = il + lmin
        if lag_changed:
            du = np.float32((path[ii] - path[ii - step]) * inverse_bstrain)
            path[ii] = path[ii - step] + du
            for _ in range(ji, jb, step):
                ii += step
                path[ii] = path[ii - step] + du

    for path_index in range(ni):
        lag_index = math.floor(float(path[path_index] - lmin) + 0.5)
        if lag_index < 0 or lag_index >= nl or not valid_mask[path_index, lag_index]:
            return path, False
    return path, True


def _project_surface_to_valid_mask(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
) -> tuple[np.ndarray, int, bool]:
    if NUMBA_AVAILABLE:
        return _project_surface_to_valid_mask_numba(surface, valid_mask, lmin)
    return _project_surface_to_valid_mask_python(surface, valid_mask, lmin)


def _project_surface_to_valid_mask_python(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
) -> tuple[np.ndarray, int, bool]:
    projected = surface.astype(np.float32, copy=True)
    nw, nv = projected.shape
    nu = valid_mask.shape[2]
    projection_count = 0
    for iw in range(nw):
        for iv in range(nv):
            value = projected[iw, iv]
            first_valid_index = -1
            last_valid_index = -1
            for candidate_index in range(nu):
                if valid_mask[iw, iv, candidate_index]:
                    if first_valid_index < 0:
                        first_valid_index = candidate_index
                    last_valid_index = candidate_index
            if first_valid_index < 0:
                return projected, projection_count, False

            lower = np.float32(lmin + first_valid_index)
            upper = np.float32(lmin + last_valid_index)
            if value < lower:
                projected[iw, iv] = lower
                projection_count += 1
                continue
            if value > upper:
                projected[iw, iv] = upper
                projection_count += 1
                continue

            lag_index = math.floor(float(value - lmin) + 0.5)
            if 0 <= lag_index < nu and valid_mask[iw, iv, lag_index]:
                continue

            nearest_index = -1
            nearest_distance = np.inf
            for candidate_index in range(first_valid_index, last_valid_index + 1):
                if valid_mask[iw, iv, candidate_index]:
                    distance = abs(float(value - np.float32(lmin + candidate_index)))
                    if distance < nearest_distance:
                        nearest_index = candidate_index
                        nearest_distance = distance
            if nearest_index < 0:
                return projected, projection_count, False
            projected[iw, iv] = np.float32(lmin + nearest_index)
            projection_count += 1
    return projected, projection_count, True


@njit(cache=True)
def _project_surface_to_valid_mask_numba(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
) -> tuple[np.ndarray, int, bool]:
    projected = surface.astype(np.float32)
    nw, nv = projected.shape
    nu = valid_mask.shape[2]
    projection_count = 0
    for iw in range(nw):
        for iv in range(nv):
            value = projected[iw, iv]
            first_valid_index = -1
            last_valid_index = -1
            for candidate_index in range(nu):
                if valid_mask[iw, iv, candidate_index]:
                    if first_valid_index < 0:
                        first_valid_index = candidate_index
                    last_valid_index = candidate_index
            if first_valid_index < 0:
                return projected, projection_count, False

            lower = np.float32(lmin + first_valid_index)
            upper = np.float32(lmin + last_valid_index)
            if value < lower:
                projected[iw, iv] = lower
                projection_count += 1
                continue
            if value > upper:
                projected[iw, iv] = upper
                projection_count += 1
                continue

            lag_index = math.floor(float(value - lmin) + 0.5)
            if 0 <= lag_index < nu and valid_mask[iw, iv, lag_index]:
                continue

            nearest_index = -1
            nearest_distance = np.inf
            for candidate_index in range(first_valid_index, last_valid_index + 1):
                if valid_mask[iw, iv, candidate_index]:
                    distance = abs(float(value - np.float32(lmin + candidate_index)))
                    if distance < nearest_distance:
                        nearest_index = candidate_index
                        nearest_distance = distance
            if nearest_index < 0:
                return projected, projection_count, False
            projected[iw, iv] = np.float32(lmin + nearest_index)
            projection_count += 1
    return projected, projection_count, True


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


def _validate_valid_mask_3d(
    valid_mask: np.ndarray,
    cost_shape: tuple[int, ...],
) -> np.ndarray:
    valid_mask_array = np.asarray(valid_mask)
    if valid_mask_array.ndim != 3:
        raise ValueError("valid_mask must have shape (nw, nv, nu)")
    if valid_mask_array.shape != cost_shape:
        raise ValueError("valid_mask and cost must have the same shape")
    if valid_mask_array.dtype != np.bool_:
        raise ValueError("valid_mask must have boolean dtype")
    return valid_mask_array
