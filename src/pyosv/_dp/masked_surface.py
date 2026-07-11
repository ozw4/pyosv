"""Masked 3D surface orchestration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pyosv._dp.validation import (
    validate_cost_3d,
    validate_int as _validate_int,
    validate_nonnegative_float as _validate_nonnegative_float,
    validate_nonnegative_int as _validate_nonnegative_int,
    validate_positive_int as _validate_positive_int,
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
    smooth_attributes: Callable[..., np.ndarray],
    extract_rows: Callable[..., np.ndarray | None],
    minimum_cost_surface: Callable[..., np.ndarray],
    surface_respects_strain: Callable[..., bool],
    recover_feasible_surface: Callable[..., np.ndarray | None],
    project_surface: Callable[..., tuple[np.ndarray, int, bool]],
    smooth_surface: Callable[..., np.ndarray],
) -> tuple[np.ndarray | None, int]:
    """Find a surface while excluding invalid lag states from all DP stages.

    This private path is used by boundary-aware surface voting. ``None`` means
    that no strain-feasible surface spans the supplied tangential rectangle.
    The integer return value counts columns whose value differs between the raw
    smoothed surface and the final mask-and-strain-feasible surface. Each
    column is counted at most once; the count is zero when smoothing is disabled.
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
        smoothed_cost = smooth_attributes(
            smoothed_cost,
            valid_mask_array,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        )

    effective_mask = valid_mask_array & np.isfinite(smoothed_cost)
    if not effective_mask.any(axis=2).all():
        return None, 0

    surface = extract_rows(
        smoothed_cost,
        effective_mask,
        lmin=lmin_int,
        bstrain=bstrain1_int,
    )
    if surface is None:
        surface = minimum_cost_surface(
            smoothed_cost,
            effective_mask,
            lmin=lmin_int,
        )
    if not surface_respects_strain(
        surface,
        effective_mask,
        lmin=lmin_int,
        bstrain1=bstrain1_int,
        bstrain2=bstrain2_int,
    ):
        surface = recover_feasible_surface(
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
        raw_smoothed_surface = smooth_surface(
            surface,
            sigma1=surface_smoothing1_float,
            sigma2=surface_smoothing2_float,
        )
        surface, _, projection_ok = project_surface(
            raw_smoothed_surface,
            effective_mask,
            lmin_int,
        )
        if not projection_ok:
            return None, 0
        if not surface_respects_strain(
            surface,
            effective_mask,
            lmin=lmin_int,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        ):
            surface = recover_feasible_surface(
                raw_smoothed_surface,
                effective_mask,
                lmin=lmin_int,
                bstrain1=bstrain1_int,
                bstrain2=bstrain2_int,
            )
            if surface is None:
                return None, 0
        if not surface_respects_strain(
            surface,
            effective_mask,
            lmin=lmin_int,
            bstrain1=bstrain1_int,
            bstrain2=bstrain2_int,
        ):
            return None, 0
        projection_count = int(np.count_nonzero(surface != raw_smoothed_surface))
    else:
        projection_count = 0
    final_surface = surface.astype(np.float32, copy=False)
    if not surface_respects_strain(
        final_surface,
        effective_mask,
        lmin=lmin_int,
        bstrain1=bstrain1_int,
        bstrain2=bstrain2_int,
    ):
        return None, 0
    return final_surface, projection_count


def _extract_masked_surface_rows(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    find_path_masked: Callable[..., tuple[np.ndarray, bool]],
) -> np.ndarray | None:
    nrow, npath, _ = cost.shape
    surface = np.empty((nrow, npath), dtype=np.float32)
    for row in range(nrow):
        path, feasible = find_path_masked(
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


def _smooth_fault_attributes_3d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bstrain1: int,
    bstrain2: int,
    smooth_attributes_2d: Callable[..., np.ndarray],
) -> np.ndarray:
    """Apply staged attribute smoothing without admitting masked lag states."""

    nw, nv, nu = cost.shape
    smoothed_v = np.empty((nw, nv, nu), dtype=np.float32)
    for iw in range(nw):
        smoothed_v[iw] = smooth_attributes_2d(
            cost[iw],
            valid_mask[iw],
            bstrain1,
        )

    smoothed_w = np.empty_like(smoothed_v, dtype=np.float32)
    for iv in range(nv):
        smoothed_w[:, iv, :] = smooth_attributes_2d(
            smoothed_v[:, iv, :],
            valid_mask[:, iv, :],
            bstrain2,
        )
    return smoothed_w


def _smooth_fault_attributes_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    bstrain: int,
    *,
    use_numba: bool,
    python_kernel: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    numba_kernel: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    if use_numba:
        return numba_kernel(cost, valid_mask, bstrain)
    return python_kernel(cost, valid_mask, bstrain)


def _find_path_2d_masked(
    cost: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lmin: int,
    bstrain: int,
    use_numba: bool,
    accumulate_python: Callable[..., np.ndarray],
    backtrack_python: Callable[..., tuple[np.ndarray, bool]],
    accumulate_numba: Callable[..., np.ndarray],
    backtrack_numba: Callable[..., tuple[np.ndarray, bool]],
) -> tuple[np.ndarray, bool]:
    if use_numba:
        accumulated = accumulate_numba(cost, valid_mask, bstrain, 1)
        return backtrack_numba(
            accumulated,
            cost,
            valid_mask,
            lmin,
            bstrain,
            -1,
        )

    accumulated = accumulate_python(cost, valid_mask, bstrain, 1)
    return backtrack_python(
        accumulated,
        cost,
        valid_mask,
        lmin,
        bstrain,
        -1,
    )


def _project_surface_to_valid_mask(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    lmin: int,
    *,
    use_numba: bool,
    python_projector: Callable[..., tuple[np.ndarray, int, bool]],
    numba_projector: Callable[..., tuple[np.ndarray, int, bool]],
) -> tuple[np.ndarray, int, bool]:
    """Project columns to valid rounding cells, counting each changed column once."""

    if use_numba:
        return numba_projector(surface, valid_mask, lmin)
    return python_projector(surface, valid_mask, lmin)


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
