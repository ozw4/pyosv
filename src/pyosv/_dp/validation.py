"""Shared validation helpers for dynamic-programming kernels."""

from __future__ import annotations

import math
import operator

import numpy as np


def validate_cost_2d(cost: np.ndarray) -> np.ndarray:
    """Validate and normalize a 2D ``(ni, nl)`` cost array."""

    cost_array = np.asarray(cost)
    if cost_array.ndim != 2:
        raise ValueError("cost must have shape (ni, nl)")

    cost_float32 = cost_array.astype(np.float32, copy=False)
    if not np.isfinite(cost_float32).all():
        raise ValueError("cost must contain only finite values")
    return cost_float32


def validate_cost_3d(cost: np.ndarray) -> np.ndarray:
    """Validate and normalize a 3D local surface cost volume."""

    cost_array = np.asarray(cost)
    if cost_array.ndim != 3:
        raise ValueError("cost must have shape (nw, nv, nu)")

    cost_float32 = cost_array.astype(np.float32, copy=False)
    if not np.isfinite(cost_float32).all():
        raise ValueError("cost must contain only finite values")
    return cost_float32


def validate_direction(direction: int) -> int:
    direction_int = validate_int(direction, "direction")
    if direction_int not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return direction_int


def validate_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def validate_positive_int(value: int, name: str) -> int:
    value_int = validate_int(value, name)
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def validate_nonnegative_float(value: float, name: str) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value_float


def validate_nonnegative_int(value: int, name: str) -> int:
    try:
        value_int = validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value_int
