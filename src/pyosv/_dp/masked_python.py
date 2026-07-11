"""Python kernels for masked dynamic-programming surfaces."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pyosv._dp.path2d_python import min3_prefer_center as _min3_prefer_center
from pyosv._dp.rounding import (
    _java_rounding_cell_bounds,
    _project_value_to_java_rounding_cell,
)


def _smooth_fault_attributes_2d_masked_python(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
    *,
    accumulate: Callable[..., np.ndarray],
) -> np.ndarray:
    forward = accumulate(cost, valid_mask, bstrain, 1)
    reverse = accumulate(cost, valid_mask, bstrain, -1)
    smoothed = np.full(cost.shape, np.inf, dtype=np.float32)
    finite = valid_mask & np.isfinite(cost) & np.isfinite(forward) & np.isfinite(reverse)
    smoothed[finite] = forward[finite] + reverse[finite] - cost[finite]
    return smoothed


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
            lag_index = math.floor(float(value - lmin) + 0.5)
            if 0 <= lag_index < nu and valid_mask[iw, iv, lag_index]:
                lower, upper = _java_rounding_cell_bounds(lmin, lag_index, nu)
                if lower <= value <= upper:
                    continue

            nearest_value = np.float32(0.0)
            nearest_distance = np.inf
            found = False
            for candidate_index in range(nu):
                if not valid_mask[iw, iv, candidate_index]:
                    continue
                candidate_value, distance = _project_value_to_java_rounding_cell(
                    value,
                    lmin,
                    candidate_index,
                    nu,
                )
                if distance < nearest_distance:
                    nearest_value = candidate_value
                    nearest_distance = distance
                    found = True
            if not found:
                return projected, projection_count, False
            projected[iw, iv] = nearest_value
            projection_count += 1
    return projected, projection_count, True
