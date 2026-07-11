"""Compatibility facade for OSV-style dynamic-programming operations."""

# ruff: noqa: F401 -- private imports intentionally preserve the legacy facade.

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE
from pyosv._dp.masked_numba import (
    _accumulate_2d_masked_numba,
    _backtrack_2d_masked_numba,
    _masked_interpolated_transition_is_valid_numba,
    _masked_transition_cost_numba,
    _project_surface_to_valid_mask_numba,
    _smooth_fault_attributes_2d_masked_numba as _smooth_fault_attributes_2d_masked_numba_impl,
)
from pyosv._dp.masked_python import (
    _accumulate_2d_masked_python,
    _backtrack_2d_masked_python,
    _masked_interpolated_transition_is_valid_python,
    _masked_start_lag_python,
    _masked_transition_cost_python,
    _path_uses_valid_mask_python,
    _project_surface_to_valid_mask_python,
    _smooth_fault_attributes_2d_masked_python as _smooth_fault_attributes_2d_masked_python_impl,
)
from pyosv._dp.masked_surface import (
    _extract_masked_surface_rows as _extract_masked_surface_rows_impl,
)
from pyosv._dp.masked_surface import (
    _find_path_2d_masked as _find_path_2d_masked_impl,
)
from pyosv._dp.masked_surface import (
    _find_surface_3d_masked as _find_surface_3d_masked_impl,
)
from pyosv._dp.masked_surface import (
    _minimum_cost_masked_surface,
    _project_surface_to_valid_mask as _project_surface_to_valid_mask_impl,
    _smooth_fault_attributes_2d_masked as _smooth_fault_attributes_2d_masked_impl,
    _smooth_fault_attributes_3d_masked as _smooth_fault_attributes_3d_masked_impl,
    _validate_valid_mask_3d,
)
from pyosv._dp.path2d import accumulate_2d as _accumulate_2d_impl
from pyosv._dp.path2d import backtrack_reverse_2d as _backtrack_reverse_2d_impl
from pyosv._dp.path2d import find_path_2d as _find_path_2d_impl
from pyosv._dp.path2d import shift_range, strain_to_bstrain, update_shift_ranges
from pyosv._dp.path2d_numba import accumulate_2d as _accumulate_2d_numba
from pyosv._dp.path2d_numba import backtrack_2d as _backtrack_2d_numba
from pyosv._dp.path2d_numba import min3_prefer_center as _min3_prefer_center_numba
from pyosv._dp.path2d_python import accumulate_2d as _accumulate_2d_python
from pyosv._dp.path2d_python import backtrack_2d as _backtrack_2d_python
from pyosv._dp.path2d_python import min3_prefer_center as _min3_prefer_center
from pyosv._dp.rounding import (
    _distance_to_closed_interval,
    _java_rounding_cell_bounds,
    _project_value_to_java_rounding_cell,
    _propagate_rounding_cell_run_domains,
    _recover_bidirectionally_feasible_surface,
    _search_feasible_rounding_cell_runs,
    _surface_from_single_rounding_cell_runs,
    _surface_respects_masked_strain,
    _tighten_strain_bounds_pair,
    _valid_rounding_cell_runs,
)
from pyosv._dp.smoothing import smooth_fault_attributes_2d as _smooth_fault_attributes_2d_impl
from pyosv._dp.smoothing import smooth_fault_attributes_3d as _smooth_fault_attributes_3d_impl
from pyosv._dp.smoothing import smooth_path_1d, smooth_surface_2d
from pyosv._dp.surface3d import find_surface_3d as _find_surface_3d_impl
from pyosv._dp.surface3d import update_shift_ranges_3d
from pyosv._dp.validation import (
    validate_cost_2d,
    validate_cost_3d,
    validate_direction as _validate_direction,
    validate_int as _validate_int,
    validate_nonnegative_float as _validate_nonnegative_float,
    validate_nonnegative_int as _validate_nonnegative_int,
    validate_positive_int as _validate_positive_int,
)

__all__ = [
    "accumulate_2d",
    "accumulate_forward_2d",
    "backtrack_reverse_2d",
    "find_path_2d",
    "find_surface_3d",
    "shift_range",
    "smooth_fault_attributes_2d",
    "smooth_fault_attributes_3d",
    "smooth_path_1d",
    "smooth_surface_2d",
    "strain_to_bstrain",
    "update_shift_ranges",
    "update_shift_ranges_3d",
    "validate_cost_2d",
    "validate_cost_3d",
]


def accumulate_forward_2d(cost: np.ndarray, *, bstrain: int) -> np.ndarray:
    """Accumulate 2D path costs in the forward path direction."""

    return accumulate_2d(cost, bstrain=bstrain, direction=1)


def accumulate_2d(cost: np.ndarray, *, bstrain: int, direction: int = 1) -> np.ndarray:
    """Accumulate 2D path costs with a lag-change spacing constraint."""

    return _accumulate_2d_impl(
        cost, bstrain=bstrain, direction=direction, use_numba=NUMBA_AVAILABLE
    )


def backtrack_reverse_2d(
    accumulated: np.ndarray,
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> np.ndarray:
    """Backtrack a path in reverse through forward-accumulated 2D costs."""

    return _backtrack_reverse_2d_impl(
        accumulated,
        cost,
        lmin=lmin,
        bstrain=bstrain,
        use_numba=NUMBA_AVAILABLE,
    )


def _backtrack_2d(
    accumulated: np.ndarray,
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    direction: int,
) -> np.ndarray:
    kernel = _backtrack_2d_numba if NUMBA_AVAILABLE else _backtrack_2d_python
    return kernel(accumulated, cost, lmin, bstrain, direction)


def smooth_fault_attributes_2d(cost: np.ndarray, *, bstrain: int) -> np.ndarray:
    """Smooth 2D fault attributes with forward and reverse DP accumulation."""

    return _smooth_fault_attributes_2d_impl(cost, bstrain=bstrain, use_numba=NUMBA_AVAILABLE)


def smooth_fault_attributes_3d(
    cost: np.ndarray,
    *,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray:
    """Smooth 3D local surface costs in ``v`` and then ``w`` directions."""

    return _smooth_fault_attributes_3d_impl(
        cost,
        bstrain1=bstrain1,
        bstrain2=bstrain2,
        use_numba=NUMBA_AVAILABLE,
    )


def find_path_2d(
    cost: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    attribute_smoothing: int = 1,
    path_smoothing: float = 0.0,
) -> np.ndarray:
    """Find a 1D optimal path through a 2D ``(ni, nl)`` cost image."""

    return _find_path_2d_impl(
        cost,
        lmin=lmin,
        bstrain=bstrain,
        attribute_smoothing=attribute_smoothing,
        path_smoothing=path_smoothing,
        use_numba=NUMBA_AVAILABLE,
    )


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

    return _find_surface_3d_impl(
        cost,
        lmin=lmin,
        bstrain1=bstrain1,
        bstrain2=bstrain2,
        attribute_smoothing=attribute_smoothing,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        find_path=find_path_2d,
        smooth_attributes=smooth_fault_attributes_3d,
        smooth_surface=smooth_surface_2d,
    )


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
    return _find_surface_3d_masked_impl(
        cost,
        valid_mask,
        lmin=lmin,
        bstrain1=bstrain1,
        bstrain2=bstrain2,
        attribute_smoothing=attribute_smoothing,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        smooth_attributes=_smooth_fault_attributes_3d_masked,
        extract_rows=_extract_masked_surface_rows,
        minimum_cost_surface=_minimum_cost_masked_surface,
        surface_respects_strain=_surface_respects_masked_strain,
        recover_feasible_surface=_recover_bidirectionally_feasible_surface,
        project_surface=_project_surface_to_valid_mask,
        smooth_surface=smooth_surface_2d,
    )


def _extract_masked_surface_rows(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> np.ndarray | None:
    return _extract_masked_surface_rows_impl(
        cost,
        valid_mask,
        lmin=lmin,
        bstrain=bstrain,
        find_path_masked=_find_path_2d_masked,
    )


def _smooth_fault_attributes_3d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bstrain1: int,
    bstrain2: int,
) -> np.ndarray:
    return _smooth_fault_attributes_3d_masked_impl(
        cost,
        valid_mask,
        bstrain1=bstrain1,
        bstrain2=bstrain2,
        smooth_attributes_2d=_smooth_fault_attributes_2d_masked,
    )


def _smooth_fault_attributes_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    return _smooth_fault_attributes_2d_masked_impl(
        cost,
        valid_mask,
        bstrain,
        use_numba=NUMBA_AVAILABLE,
        python_kernel=_smooth_fault_attributes_2d_masked_python,
        numba_kernel=_smooth_fault_attributes_2d_masked_numba,
    )


def _smooth_fault_attributes_2d_masked_python(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    return _smooth_fault_attributes_2d_masked_python_impl(
        cost,
        valid_mask,
        bstrain,
        accumulate=_accumulate_2d_masked_python,
    )


def _smooth_fault_attributes_2d_masked_numba(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
) -> np.ndarray:
    return _smooth_fault_attributes_2d_masked_numba_impl(
        cost,
        valid_mask,
        bstrain,
        accumulate=_accumulate_2d_masked_numba,
    )


def _find_path_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
) -> tuple[np.ndarray, bool]:
    return _find_path_2d_masked_impl(
        cost,
        valid_mask,
        lmin=lmin,
        bstrain=bstrain,
        use_numba=NUMBA_AVAILABLE,
        accumulate_python=_accumulate_2d_masked_python,
        backtrack_python=_backtrack_2d_masked_python,
        accumulate_numba=_accumulate_2d_masked_numba,
        backtrack_numba=_backtrack_2d_masked_numba,
    )


def _project_surface_to_valid_mask(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
) -> tuple[np.ndarray, int, bool]:
    return _project_surface_to_valid_mask_impl(
        surface,
        valid_mask,
        lmin,
        use_numba=NUMBA_AVAILABLE,
        python_projector=_project_surface_to_valid_mask_python,
        numba_projector=_project_surface_to_valid_mask_numba,
    )


def _java_round(value: float) -> int:
    return math.floor(float(value) + 0.5)
