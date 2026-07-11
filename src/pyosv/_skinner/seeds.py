"""Seed selection and accepted-skin occupancy helpers."""

from __future__ import annotations

import operator

import numpy as np

from pyosv._skinner.grid import _SkinCellGrid
from pyosv._skinner.models import _SkinCell
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


def _mark_occupied_skin(occupied: _SkinCellGrid, skin: FaultSkin, radius: int = 5) -> None:
    radius_int = _validate_nonnegative_int(radius, "accepted_occupancy_radius")
    for cell in skin:
        occupied.set_cells_in_box(
            _SkinCell(cell.x1, cell.x2, cell.x3, cell.fl, cell.fp, cell.ft),
            radius_int,
            radius_int,
            radius_int,
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
    n3, n2, n1 = ft_array.shape

    candidates: list[tuple[float, int, int, int]] = []
    candidate_mask = (ep_array > np.float32(planarity_threshold)) & (
        ft_array > np.float32(threshold)
    )
    for i3, i2, i1 in np.argwhere(candidate_mask):
        candidates.append(
            (
                float(ft_array[i3, i2, i1]),
                operator.index(i3),
                operator.index(i2),
                operator.index(i1),
            ),
        )

    candidates.sort(
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2], candidate[3]),
    )

    mark = np.zeros((n3, n2, n1), dtype=np.bool_)
    seeds: list[_SkinCell] = []
    for _, i3, i2, i1 in candidates:
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        b3 = max(i3 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        e3 = min(i3 + distance, n3 - 1)
        if mark[b3 : e3 + 1, b2 : e2 + 1, b1 : e1 + 1].any():
            continue

        seeds.append(
            _SkinCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            ),
        )
        mark[i3, i2, i1] = True

    return seeds
