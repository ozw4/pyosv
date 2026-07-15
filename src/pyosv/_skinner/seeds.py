"""Seed selection and accepted-skin occupancy helpers."""

from __future__ import annotations

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE
from pyosv._seed_selection import _select_skinner_seed_indices_3d
from pyosv._skinner.models import _SkinCell
from pyosv._skinner.occupancy import _SkinOccupancyMask
from pyosv._skinner.validation import (
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_finite_float,
    _validate_nonnegative_int,
    _validate_unit_interval_float,
)
from pyosv.skin import FaultSkin

_REFERENCE_SEED_MIN_EP = 0.8


def _adaptive_skin_likelihood_threshold(values: np.ndarray) -> float:
    """Choose a quality-mode skin likelihood threshold from positive samples."""

    value_array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(value_array).all():
        raise ValueError("values must contain only finite values")

    positive = value_array[value_array > np.float32(0.0)]
    if positive.size == 0:
        return 1.0

    return float(np.clip(np.percentile(positive, 70.0), 0.25, 0.75))


def _mark_occupied_skin(
    occupied: _SkinOccupancyMask,
    skin: FaultSkin,
    radius: int = 5,
) -> None:
    radius_int = _validate_nonnegative_int(radius, "accepted_occupancy_radius")
    _mark_occupied_skin_validated(occupied, skin, radius_int)


def _mark_occupied_skin_validated(
    occupied: _SkinOccupancyMask,
    skin: FaultSkin,
    radius: int,
) -> None:
    """Mark a skin using a previously validated nonnegative radius."""

    for cell in skin:
        occupied.mark_box(
            cell.i1,
            cell.i2,
            cell.i3,
            radius,
            radius,
            radius,
        )


def _find_reference_seeds(
    d: int,
    fm: float,
    ep: np.ndarray,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    *,
    min_ep: float = _REFERENCE_SEED_MIN_EP,
) -> list[_SkinCell]:
    """Select reference-like starting cells for future skin growth."""

    distance = _validate_nonnegative_int(d, "d")
    threshold = _validate_nonnegative_finite_float(fm, "fm")
    planarity_threshold = _validate_unit_interval_float(min_ep, "min_ep")
    ep_array, ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
        (ep, ft, pt, tt),
        ("ep", "ft", "pt", "tt"),
    )
    return _find_reference_seeds_validated(
        d=distance,
        fm=threshold,
        ep=ep_array,
        ft=ft_array,
        pt=pt_array,
        tt=tt_array,
        min_ep=planarity_threshold,
    )


def _find_reference_seeds_validated(
    d: int,
    fm: float,
    ep: np.ndarray,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    *,
    min_ep: float = _REFERENCE_SEED_MIN_EP,
) -> list[_SkinCell]:
    """Select seeds from validated native-float32, finite, matching 3D arrays."""

    distance = d
    threshold = fm
    planarity_threshold = min_ep
    ep_array, ft_array, pt_array, tt_array = ep, ft, pt, tt
    _, n2, n1 = ft_array.shape
    plane_size = n2 * n1
    accepted_indices = _select_skinner_seed_indices_3d(
        ep_array,
        ft_array,
        np.float32(planarity_threshold),
        np.float32(threshold),
        distance,
        use_numba=NUMBA_AVAILABLE,
    )
    seeds: list[_SkinCell] = []
    for flat_index in accepted_indices:
        i3, remainder = divmod(int(flat_index), plane_size)
        i2, i1 = divmod(remainder, n1)
        seeds.append(
            _SkinCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            )
        )
    return seeds
