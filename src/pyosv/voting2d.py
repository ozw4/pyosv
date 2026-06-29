"""2D optimal-path voting entry points."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np

from pyosv.dp import shift_range, strain_to_bstrain, update_shift_ranges

__all__ = ["OptimalPathVoter"]


class OptimalPathVoter:
    """Configuration and state holder for 2D optimal-path voting."""

    def __init__(self, ru: int, rv: int) -> None:
        self.ru = _validate_nonnegative_int(ru, "ru")
        self.rv = _validate_nonnegative_int(rv, "rv")
        self.lmin, self.lmax, self.nl = shift_range(self.ru)
        self.bstrain1 = 4
        self.attribute_smoothing = 1
        self.path_smoothing1 = 2.0
        self.lmins: np.ndarray
        self.lmaxs: np.ndarray
        self._update_shift_ranges()

    def set_strain_max(self, strain_max1: float) -> None:
        """Set the maximum fault-curve strain in the first dimension."""

        self.bstrain1 = strain_to_bstrain(strain_max1)
        self._update_shift_ranges()

    def set_attribute_smoothing(self, attribute_smoothing: int) -> None:
        """Set the number of nonlinear smoothings for fault attributes."""

        self.attribute_smoothing = _validate_nonnegative_int(
            attribute_smoothing,
            "attribute_smoothing",
        )

    def set_path_smoothing(self, path_smoothing1: float) -> None:
        """Set the smoothing extent used for extracted fault paths."""

        self.path_smoothing1 = _validate_nonnegative_float(
            path_smoothing1,
            "path_smoothing1",
        )

    def _update_shift_ranges(self) -> None:
        self.lmins, self.lmaxs = update_shift_ranges(
            self.ru,
            self.rv,
            bstrain=self.bstrain1,
        )


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_nonnegative_int(value: int, name: str) -> int:
    try:
        value_int = _validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

    return value_int


def _validate_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")

    value_float = float(value)

    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float
