"""Numba UVW sampling kernels."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit


@njit(cache=True)
def _samples_in_uvw_box_numba(
    c1: int,
    c2: int,
    c3: int,
    ru: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
    lmins: np.ndarray,
    lmaxs: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fx.shape
    costs = np.ones((2 * rw + 1, 2 * rv + 1, 2 * ru + 1), dtype=np.float32)
    for kw in range(costs.shape[0]):
        iw = kw - rw
        for kv in range(costs.shape[1]):
            iv = kv - rv
            ku_min = lmins[kw, kv] + ru
            ku_max = lmaxs[kw, kv] + ru
            for ku in range(ku_min, ku_max + 1):
                iu = ku - ru
                x1 = np.float32(
                    float(c1)
                    + float(iw) * float(strike[0])
                    + float(iv) * float(dip[0])
                    + float(iu) * float(normal[0])
                )
                x2 = np.float32(
                    float(c2)
                    + float(iw) * float(strike[1])
                    + float(iv) * float(dip[1])
                    + float(iu) * float(normal[1])
                )
                x3 = np.float32(
                    float(c3)
                    + float(iw) * float(strike[2])
                    + float(iv) * float(dip[2])
                    + float(iu) * float(normal[2])
                )
                j1 = math.floor(float(x1) + 0.5)
                j2 = math.floor(float(x2) + 0.5)
                j3 = math.floor(float(x3) + 0.5)
                j1 = min(max(j1, 0), n1 - 1)
                j2 = min(max(j2, 0), n2 - 1)
                j3 = min(max(j3, 0), n3 - 1)
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]
    return costs


@njit(cache=True)
def _samples_in_uvw_box_masked_numba(
    c1: int,
    c2: int,
    c3: int,
    ru: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
    lmins: np.ndarray,
    lmaxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n3, n2, n1 = fx.shape
    costs = np.ones((2 * rw + 1, 2 * rv + 1, 2 * ru + 1), dtype=np.float32)
    valid_lag_mask = np.zeros(costs.shape, dtype=np.bool_)
    admissible_count = 0
    in_bounds_count = 0
    for kw in range(costs.shape[0]):
        iw = kw - rw
        for kv in range(costs.shape[1]):
            iv = kv - rv
            ku_min = lmins[kw, kv] + ru
            ku_max = lmaxs[kw, kv] + ru
            for ku in range(ku_min, ku_max + 1):
                admissible_count += 1
                iu = ku - ru
                x1 = np.float32(
                    float(c1)
                    + float(iw) * float(strike[0])
                    + float(iv) * float(dip[0])
                    + float(iu) * float(normal[0])
                )
                x2 = np.float32(
                    float(c2)
                    + float(iw) * float(strike[1])
                    + float(iv) * float(dip[1])
                    + float(iu) * float(normal[1])
                )
                x3 = np.float32(
                    float(c3)
                    + float(iw) * float(strike[2])
                    + float(iv) * float(dip[2])
                    + float(iu) * float(normal[2])
                )
                j1 = math.floor(float(x1) + 0.5)
                j2 = math.floor(float(x2) + 0.5)
                j3 = math.floor(float(x3) + 0.5)
                if not (0 <= j1 < n1 and 0 <= j2 < n2 and 0 <= j3 < n3):
                    continue
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]
                valid_lag_mask[kw, kv, ku] = True
                in_bounds_count += 1
    return costs, valid_lag_mask, admissible_count, in_bounds_count
