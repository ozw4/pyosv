"""Pure-Python batched dynamic-programming smoothing kernels."""

from __future__ import annotations

import numpy as np

from pyosv._dp.path2d_python import min3_prefer_center


def _accumulate_2d_into(
    cost: np.ndarray,
    bstrain: int,
    direction: int,
    accumulated: np.ndarray,
) -> None:
    ni, nl = cost.shape
    ni_last = ni - 1
    nl_last = nl - 1
    start = 0 if direction > 0 else ni_last
    stop = ni if direction > 0 else -1
    step = 1 if direction > 0 else -1
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


def accumulate_2d_batch(cost: np.ndarray, bstrain: int, direction: int) -> np.ndarray:
    """Accumulate validated ``(batch, ni, nl)`` costs independently."""

    accumulated = np.empty_like(cost, dtype=np.float32)
    for ib in range(cost.shape[0]):
        _accumulate_2d_into(cost[ib], bstrain, direction, accumulated[ib])
    return accumulated


def smooth_fault_attributes_batch(cost: np.ndarray, bstrain: int) -> np.ndarray:
    """Smooth validated ``(batch, ni, nl)`` costs with reusable scratch."""

    smoothed = np.empty(cost.shape, dtype=np.float32)
    smooth_fault_attributes_batch_into(cost, bstrain, smoothed)
    return smoothed


def smooth_fault_attributes_batch_into(
    cost: np.ndarray,
    bstrain: int,
    smoothed: np.ndarray,
) -> None:
    """Smooth a validated batch into caller-provided storage."""

    batch, ni, nl = cost.shape
    reverse = np.empty((ni, nl), dtype=np.float32)
    for ib in range(batch):
        _accumulate_2d_into(cost[ib], bstrain, 1, smoothed[ib])
        _accumulate_2d_into(cost[ib], bstrain, -1, reverse)
        np.add(smoothed[ib], reverse, out=smoothed[ib])
        np.subtract(smoothed[ib], cost[ib], out=smoothed[ib])
