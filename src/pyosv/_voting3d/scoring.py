"""Surface-score dispatch helpers."""

from __future__ import annotations

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE
from pyosv._voting3d.scoring_numba import (
    _surface_vote_average_masked_numba,
    _surface_vote_average_numba,
)
from pyosv._voting3d.scoring_python import (
    _surface_vote_average_masked_python,
    _surface_vote_average_python,
)


def _surface_vote_average(
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
    if NUMBA_AVAILABLE:
        return _surface_vote_average_numba(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)
    return _surface_vote_average_python(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)


def _surface_vote_average_masked(
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
    if NUMBA_AVAILABLE:
        return _surface_vote_average_masked_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            w_offset,
            v_offset,
            lmin,
            normal,
            dip,
            strike,
            surface,
            valid_lag_mask,
            ft,
        )
    return _surface_vote_average_masked_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        w_offset,
        v_offset,
        lmin,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
        ft,
    )
