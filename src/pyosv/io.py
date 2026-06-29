"""Binary DAT I/O helpers."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Literal

from math import prod
from numbers import Integral

import numpy as np

Endian = Literal["big", "little", ">", "<"]


def _endian_prefix(endian: Endian) -> str:
    if endian in ("big", ">"):
        return ">"
    if endian in ("little", "<"):
        return "<"
    raise ValueError("endian must be one of 'big', 'little', '>', or '<'")


def _storage_dtype(dtype: np.dtype | type, endian: Endian) -> np.dtype:
    return np.dtype(dtype).newbyteorder(_endian_prefix(endian))


def _native_dtype(dtype: np.dtype | type) -> np.dtype:
    return np.dtype(dtype).newbyteorder("=")


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError("shape must be a non-empty tuple of positive integers")

    validated = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0:
            raise ValueError("shape must be a non-empty tuple of positive integers")
        validated.append(int(dim))
    return tuple(validated)


def read_dat(
    path: str | PathLike[str],
    shape: tuple[int, ...],
    *,
    endian: Endian = "big",
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Read raw binary scalar values and reshape them in C order."""
    file_path = Path(path)
    valid_shape = _validate_shape(shape)
    storage_dtype = _storage_dtype(dtype, endian)
    element_count = prod(valid_shape)
    expected_bytes = element_count * storage_dtype.itemsize
    actual_bytes = file_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{file_path}: expected {expected_bytes} bytes for shape {valid_shape}, "
            f"got {actual_bytes} bytes"
        )

    array = np.fromfile(file_path, dtype=storage_dtype, count=element_count)
    array = array.reshape(valid_shape, order="C")
    return np.ascontiguousarray(array.astype(_native_dtype(dtype), copy=False))


def write_dat(
    path: str | PathLike[str],
    array: np.ndarray,
    *,
    endian: Endian = "big",
    dtype: np.dtype | type = np.float32,
    create_parents: bool = True,
) -> Path:
    """Write an array as raw binary scalar values in C order."""
    file_path = Path(path)
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    output = np.ascontiguousarray(np.asarray(array, dtype=_storage_dtype(dtype, endian)))
    output.tofile(file_path)
    return file_path
