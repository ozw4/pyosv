"""Helpers for OSV-style dynamic-programming path kernels."""

from __future__ import annotations

import math
import operator

import numpy as np

__all__ = [
    "shift_range",
    "strain_to_bstrain",
    "update_shift_ranges",
    "validate_cost_2d",
]


def validate_cost_2d(cost: np.ndarray) -> np.ndarray:
    """Validate and normalize a 2D ``(ni, nl)`` cost array."""

    cost_array = np.asarray(cost)
    if cost_array.ndim != 2:
        raise ValueError("cost must have shape (ni, nl)")

    cost_float32 = cost_array.astype(np.float32, copy=False)
    if not np.isfinite(cost_float32).all():
        raise ValueError("cost must contain only finite values")

    return cost_float32


def strain_to_bstrain(strain_max: float) -> int:
    """Convert a maximum strain bound to the integer DP step spacing."""

    try:
        strain = float(strain_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("strain_max must satisfy 0 < strain_max <= 1") from exc

    if not math.isfinite(strain) or strain <= 0.0 or strain > 1.0:
        raise ValueError("strain_max must satisfy 0 < strain_max <= 1")

    return int(math.ceil(1.0 / strain))


def shift_range(ru: int) -> tuple[int, int, int]:
    """Return ``(lmin, lmax, nl)`` for the fault-normal shift radius ``ru``."""

    ru_int = _validate_nonnegative_int(ru, "ru")
    lmin = -ru_int
    lmax = ru_int
    nl = 1 + lmax - lmin
    return lmin, lmax, nl


def update_shift_ranges(ru: int, rv: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Java-reference ``_lmins`` and ``_lmaxs`` arrays for shift bounds."""

    ru_int = _validate_nonnegative_int(ru, "ru")
    rv_int = _validate_nonnegative_int(rv, "rv")
    lmin, lmax, _ = shift_range(ru_int)

    nv = 2 * rv_int + 1
    lmins = np.zeros(nv, dtype=np.int32)
    lmaxs = np.zeros(nv, dtype=np.int32)

    for iv in range(-rv_int, rv_int + 1):
        if abs(iv) > 2:
            index = iv + rv_int
            lmins[index] = max(-abs(iv), lmin)
            lmaxs[index] = min(abs(iv), lmax)

    return lmins, lmaxs


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")

    try:
        value_int = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

    return value_int
