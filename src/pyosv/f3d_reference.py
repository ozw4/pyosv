"""Metadata helpers for the public F3 3D reference dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
