"""Numba surface-score kernels."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit
from pyosv._voting3d.accumulation import _is_valid_surface_vote_sample


@njit(cache=True)
def _surface_vote_average_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    n3, n2, n1 = ft.shape
    fa = np.float32(0.0)
    valid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue

            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count


@njit(cache=True)
def _surface_vote_average_masked_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int, int]:
    n3, n2, n1 = ft.shape
    nu = valid_lag_mask.shape[2]
    fa = np.float32(0.0)
    valid_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
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
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count, invalid_count
