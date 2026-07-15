"""Smoothing operations shared by dynamic-programming paths and surfaces."""

from __future__ import annotations

import numpy as np

from pyosv._dp.batch_numba import (
    smooth_fault_attributes_batch_into as smooth_fault_attributes_batch_into_numba,
)
from pyosv._dp.batch_python import (
    smooth_fault_attributes_batch_into as smooth_fault_attributes_batch_into_python,
)
from pyosv._dp.path2d import accumulate_2d
from pyosv._dp.validation import (
    validate_cost_2d,
    validate_cost_3d,
    validate_nonnegative_float,
    validate_positive_int,
)
from pyosv.filters import smooth1d, smooth2d


def smooth_path_1d(path: np.ndarray, sigma: float, *, bstrain: int = 1) -> np.ndarray:
    path_array = np.asarray(path, dtype=np.float32)
    if path_array.ndim != 1:
        raise ValueError("path must have shape (ni,)")
    if not np.isfinite(path_array).all():
        raise ValueError("path must contain only finite values")
    sigma_float = validate_nonnegative_float(sigma, "sigma")
    bstrain_int = validate_positive_int(bstrain, "bstrain")
    return smooth1d(path_array, sigma_float * bstrain_int)


def smooth_surface_2d(
    surface: np.ndarray, *, sigma1: float = 0.0, sigma2: float = 0.0
) -> np.ndarray:
    surface_array = np.asarray(surface, dtype=np.float32)
    if surface_array.ndim != 2:
        raise ValueError("surface must have shape (nw, nv)")
    if not np.isfinite(surface_array).all():
        raise ValueError("surface must contain only finite values")
    sigma1_float = validate_nonnegative_float(sigma1, "sigma1")
    sigma2_float = validate_nonnegative_float(sigma2, "sigma2")
    if sigma1_float == 0.0 and sigma2_float == 0.0:
        return surface_array.copy()
    return smooth2d(surface_array, (sigma2_float, sigma1_float))


def smooth_fault_attributes_2d(cost: np.ndarray, *, bstrain: int, use_numba: bool) -> np.ndarray:
    cost_array = validate_cost_2d(cost)
    bstrain_int = validate_positive_int(bstrain, "bstrain")
    forward = accumulate_2d(cost_array, bstrain=bstrain_int, direction=1, use_numba=use_numba)
    reverse = accumulate_2d(cost_array, bstrain=bstrain_int, direction=-1, use_numba=use_numba)
    return (forward + reverse - cost_array).astype(np.float32, copy=False)


def smooth_fault_attributes_3d(
    cost: np.ndarray, *, bstrain1: int, bstrain2: int, use_numba: bool
) -> np.ndarray:
    cost_array = validate_cost_3d(cost)
    bstrain1_int = validate_positive_int(bstrain1, "bstrain1")
    bstrain2_int = validate_positive_int(bstrain2, "bstrain2")
    batch_kernel_into = (
        smooth_fault_attributes_batch_into_numba
        if use_numba
        else smooth_fault_attributes_batch_into_python
    )
    smoothed_v = np.empty(cost_array.shape, dtype=np.float32)
    batch_kernel_into(cost_array, bstrain1_int, smoothed_v)
    transposed_v = smoothed_v.transpose(1, 0, 2).copy(order="C")
    batch_kernel_into(transposed_v, bstrain2_int, smoothed_v.transpose(1, 0, 2))
    return smoothed_v
