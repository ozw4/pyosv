"""Pure-Python 2D dynamic-programming kernels."""

from __future__ import annotations

import numpy as np


def min3_prefer_center(a: float, b: float, c: float) -> float:
    if b <= a:
        if b <= c:
            return b
        return c
    if a <= c:
        return a
    return c


def accumulate_2d(cost: np.ndarray, bstrain: int, direction: int) -> np.ndarray:
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
                min3_prefer_center(cost_minus, cost_same, cost_plus) + cost[ii, il]
            )
    return accumulated


def backtrack_2d(
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
        best_cost = min3_prefer_center(cost_minus, cost_same, cost_plus)
        if best_cost != cost_same:
            next_il = il_minus if best_cost == cost_minus else il_plus
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
