"""Metadata helpers for the public F3 3D reference dataset."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from os import PathLike
from pathlib import Path

import numpy as np

from pyosv.io import read_dat

F3D_ENV_VAR = "PYOSV_F3D_DATA_ROOT"
F3D_SHAPE = (420, 400, 100)
F3D_DTYPE = ">f4"
F3D_EXPECTED_BYTES = 67_200_000
F3D_FILENAMES = ("xs.dat", "ep.dat", "fl.dat", "fv.dat", "fvt.dat")


@dataclass(frozen=True)
class F3DFileSpec:
    """Metadata for one F3 3D reference data file."""

    filename: str
    meaning: str
    expected_bytes: int


_F3D_FILE_MEANINGS = {
    "xs.dat": "input seismic image",
    "ep.dat": "reference planarity",
    "fl.dat": "reference fault likelihood",
    "fv.dat": "reference fault votes",
    "fvt.dat": "reference thinned fault votes",
}


def _validate_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return result


def _validate_shape3(name: str, value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 dimensions")

    return tuple(
        _validate_int(f"{name}[{axis}]", size, minimum=1) for axis, size in enumerate(value)
    )


def _validate_center3(
    center: tuple[int, int, int], full_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    if len(center) != 3:
        raise ValueError("center must contain exactly 3 coordinates")

    validated = tuple(_validate_int(f"center[{axis}]", index) for axis, index in enumerate(center))
    for axis, (index, size) in enumerate(zip(validated, full_shape, strict=True)):
        if index < 0 or index >= size:
            raise ValueError(f"center[{axis}] must be inside full_shape[{axis}]")

    return validated


def parse_shape3(text: str) -> tuple[int, int, int]:
    """Parse a 3D shape string as ``(n3, n2, n1)``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    parts = [part for part in re.split(r"[\s, xX]+", text.strip()) if part]
    if len(parts) != 3:
        raise ValueError("shape must contain exactly 3 dimensions")

    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError("shape dimensions must be integers") from error

    return _validate_shape3("shape", shape)


def pick_reference_centers(
    fv: np.ndarray,
    *,
    count: int = 3,
    percentile: float = 99.9,
    min_separation: float = 48.0,
) -> list[tuple[int, int, int]]:
    """Pick deterministic high-value ``fv`` centers in ``(i3, i2, i1)`` order."""
    fv = np.asarray(fv)
    if fv.ndim != 3:
        raise ValueError("fv must be a 3D array with shape (n3, n2, n1)")
    _validate_shape3("fv.shape", fv.shape)
    count = _validate_int("count", count, minimum=0)

    if not isinstance(percentile, Real) or isinstance(percentile, bool):
        raise TypeError("percentile must be a real number")
    percentile = float(percentile)
    if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be finite and between 0 and 100")

    if not isinstance(min_separation, Real) or isinstance(min_separation, bool):
        raise TypeError("min_separation must be a real number")
    min_separation = float(min_separation)
    if not np.isfinite(min_separation) or min_separation < 0.0:
        raise ValueError("min_separation must be finite and >= 0")

    if count == 0:
        return []
    if not np.isfinite(fv).all():
        raise ValueError("fv must contain only finite values")

    threshold = float(np.percentile(fv, percentile))
    candidates = np.argwhere(fv >= threshold)
    values = fv[tuple(candidates.T)]
    order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], -values))

    selected: list[tuple[int, int, int]] = []
    min_separation_sq = min_separation * min_separation
    for candidate in candidates[order]:
        center = tuple(int(index) for index in candidate)
        if all(
            sum((candidate[axis] - existing[axis]) ** 2 for axis in range(3)) >= min_separation_sq
            for existing in selected
        ):
            selected.append(center)
            if len(selected) == count:
                break

    return selected


def crop_slices(
    center: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int] = F3D_SHAPE,
) -> tuple[slice, slice, slice]:
    """Return bounded crop slices ordered as ``(slice_i3, slice_i2, slice_i1)``."""
    full_shape = _validate_shape3("full_shape", full_shape)
    crop_shape = _validate_shape3("crop_shape", crop_shape)
    center = _validate_center3(center, full_shape)

    starts = []
    for axis, (center_index, crop_size, full_size) in enumerate(
        zip(center, crop_shape, full_shape, strict=True)
    ):
        if crop_size > full_size:
            raise ValueError(f"crop_shape[{axis}] must be <= full_shape[{axis}]")

        start = center_index - crop_size // 2
        start = min(max(start, 0), full_size - crop_size)
        starts.append(start)

    return tuple(slice(start, start + size) for start, size in zip(starts, crop_shape, strict=True))


def interior_mask(shape: tuple[int, int, int], margin: int = 16) -> np.ndarray:
    """Return a boolean mask that excludes ``margin`` samples from each boundary."""
    slices = interior_slices(shape, margin=margin)
    mask = np.zeros(shape, dtype=bool)
    mask[slices] = True
    return mask


def interior_slices(shape: tuple[int, int, int], margin: int = 16) -> tuple[slice, slice, slice]:
    """Return crop-local interior slices excluding ``margin`` samples at each boundary."""
    shape = _validate_shape3("shape", shape)
    margin = _validate_int("margin", margin, minimum=0)
    if any(2 * margin >= size for size in shape):
        raise ValueError("margin is too large for shape")

    if margin == 0:
        return tuple(slice(0, size) for size in shape)
    return tuple(slice(margin, size - margin) for size in shape)


def resolve_f3d_data_root(data_root: str | PathLike[str] | None = None) -> Path:
    """Resolve the F3 3D reference data root without requiring it to exist."""
    if data_root is not None:
        return Path(data_root)

    env_root = os.environ.get(F3D_ENV_VAR)
    if env_root is not None:
        return Path(env_root)

    raise ValueError(f"F3 3D reference data root is required; pass data_root or set {F3D_ENV_VAR}")


def f3d_file_specs() -> dict[str, F3DFileSpec]:
    """Return metadata for known F3 3D reference files keyed by filename."""
    return {
        filename: F3DFileSpec(
            filename=filename,
            meaning=_F3D_FILE_MEANINGS[filename],
            expected_bytes=F3D_EXPECTED_BYTES,
        )
        for filename in F3D_FILENAMES
    }


def f3d_file_paths(data_root: str | PathLike[str] | None = None) -> dict[str, Path]:
    """Return absolute or relative paths for known F3 3D reference files."""
    root = resolve_f3d_data_root(data_root)
    return {filename: root / filename for filename in F3D_FILENAMES}


def read_f3d_file(name: str, data_root: str | PathLike[str] | None = None) -> np.ndarray:
    """Read one known F3 3D reference file as a native float32 array."""
    paths = f3d_file_paths(data_root)
    try:
        path = paths[name]
    except KeyError as error:
        expected = ", ".join(F3D_FILENAMES)
        raise ValueError(
            f"unknown F3 3D reference file: {name}; expected one of {expected}"
        ) from error

    return read_dat(path, F3D_SHAPE, endian="big", dtype=np.float32)
