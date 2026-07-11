"""Validation shared by internal skinner implementations."""

from __future__ import annotations

from collections.abc import Iterable
import math
import numbers
import operator

import numpy as np


def _validate_nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float


def _validate_unit_interval_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0")

    return value_float


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")

    return value


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


def _validate_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer or None")

    try:
        value_int = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer or None") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")

    return value_int


def _validate_connectivity(connectivity: str) -> str:
    if not isinstance(connectivity, str) or connectivity not in {"face", "edge", "corner"}:
        raise ValueError("connectivity must be 'face', 'edge', or 'corner'")

    return connectivity


def _validate_matching_finite_arrays3_many(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    validated = _validate_matching_arrays3(arrays, names)
    finite_arrays: list[np.ndarray] = []
    for array, name in zip(validated, names):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                finite_array = array.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric finite values") from exc

        if not np.isfinite(finite_array).all():
            raise ValueError(f"{name} must contain only finite values")

        finite_arrays.append(finite_array)

    return tuple(finite_arrays)


def _validate_matching_arrays3(
    arrays: Iterable[np.ndarray],
    names: Iterable[str],
) -> tuple[np.ndarray, ...]:
    arrays_tuple = tuple(arrays)
    names_tuple = tuple(names)
    if len(arrays_tuple) != len(names_tuple):
        raise ValueError("arrays and names must have the same length")
    if not arrays_tuple:
        raise ValueError("at least one array is required")

    validated = tuple(
        _validate_array3(array, name) for array, name in zip(arrays_tuple, names_tuple)
    )
    shape = validated[0].shape
    first_name = names_tuple[0]
    for array, name in zip(validated[1:], names_tuple[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated


def _validate_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")

    return array
