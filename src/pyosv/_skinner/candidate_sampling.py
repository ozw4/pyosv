"""Candidate-slice sampling kernels for reference-like skin growth."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit


def _candidate_slice_python(
    fv: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    ws: np.ndarray,
    o1: float,
    o2: float,
    o3: float,
    u_start: int,
    u_stop: int,
    v_center: int,
    w_center: int,
    step_sign: int,
    axis_code: int,
    row_count: int,
) -> np.ndarray:
    return _sample_candidate_slice(
        fv,
        us,
        vs,
        ws,
        o1,
        o2,
        o3,
        u_start,
        u_stop,
        v_center,
        w_center,
        step_sign,
        axis_code,
        row_count,
    )


def _sample_candidate_slice(
    fv: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    ws: np.ndarray,
    o1: float,
    o2: float,
    o3: float,
    u_start: int,
    u_stop: int,
    v_center: int,
    w_center: int,
    step_sign: int,
    axis_code: int,
    row_count: int,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    samples = np.zeros((row_count, u_stop - u_start + 1), dtype=np.float32)
    for row in range(row_count):
        iv = v_center + step_sign * row if axis_code == 0 else v_center
        iw = w_center + step_sign * row if axis_code == 1 else w_center
        for col in range(samples.shape[1]):
            iu = u_start + col
            x1 = np.float32(o1 + float(us[0, iu]) + float(vs[0, iv]) + float(ws[0, iw]))
            x2 = np.float32(o2 + float(us[1, iu]) + float(vs[1, iv]) + float(ws[1, iw]))
            x3 = np.float32(o3 + float(us[2, iu]) + float(vs[2, iv]) + float(ws[2, iw]))
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if 0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3:
                samples[row, col] = fv[i3, i2, i1]

    return samples


@njit(cache=True)
def _candidate_slice_numba(
    fv: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    ws: np.ndarray,
    o1: float,
    o2: float,
    o3: float,
    u_start: int,
    u_stop: int,
    v_center: int,
    w_center: int,
    step_sign: int,
    axis_code: int,
    row_count: int,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    samples = np.zeros((row_count, u_stop - u_start + 1), dtype=np.float32)
    for row in range(row_count):
        iv = v_center + step_sign * row if axis_code == 0 else v_center
        iw = w_center + step_sign * row if axis_code == 1 else w_center
        for col in range(samples.shape[1]):
            iu = u_start + col
            x1 = np.float32(o1 + float(us[0, iu]) + float(vs[0, iv]) + float(ws[0, iw]))
            x2 = np.float32(o2 + float(us[1, iu]) + float(vs[1, iv]) + float(ws[1, iw]))
            x3 = np.float32(o3 + float(us[2, iu]) + float(vs[2, iv]) + float(ws[2, iw]))
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if 0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3:
                samples[row, col] = fv[i3, i2, i1]

    return samples
