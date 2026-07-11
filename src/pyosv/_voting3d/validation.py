"""Validation helpers shared by 3D voting components."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np


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


def _validate_positive_int(value: int, name: str) -> int:
    try:
        value_int = _validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _validate_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value_float


def _validate_fraction_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return value_float


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
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    if len(arrays) != len(names):
        raise ValueError("arrays and names must have the same length")
    if not arrays:
        raise ValueError("at least one array is required")
    validated = tuple(_validate_array3(array, name) for array, name in zip(arrays, names))
    shape = validated[0].shape
    first_name = names[0]
    for array, name in zip(validated[1:], names[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")
    return validated


def _validate_finite_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = _validate_array3(array, name)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_finite_array2(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")
    return array


def _validate_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = np.asarray(vector, dtype=np.float32)
    if vector_array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return vector_array


def _validate_finite_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = _validate_vector3(vector, name)
    if not np.isfinite(vector_array).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector_array
