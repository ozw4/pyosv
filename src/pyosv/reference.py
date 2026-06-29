"""Reference OSV dataset metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod
from os import PathLike
from pathlib import Path
from typing import Literal

Endian = Literal["big", "little"]


@dataclass(frozen=True)
class ReferenceDataset:
    """Metadata for a known reference dataset."""

    name: str
    relative_dir: Path
    shape: tuple[int, ...]
    n1: int
    n2: int | None = None
    n3: int | None = None
    endian: Endian = "big"
    files: tuple[str, ...] = ()

    @property
    def ndim(self) -> int:
        """Number of dimensions in the stored array shape."""
        return len(self.shape)

    @property
    def sample_count(self) -> int:
        """Total number of scalar samples in the dataset shape."""
        return prod(self.shape)


REFERENCE_DATASETS_2D = {
    "f3d2d": ReferenceDataset(
        name="f3d2d",
        relative_dir=Path("data/2d/f3d2d"),
        shape=(440, 222),
        n1=222,
        n2=440,
        endian="big",
        files=("f3d75s.dat", "el.dat", "ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
    ),
    "campos": ReferenceDataset(
        name="campos",
        relative_dir=Path("data/2d/campos"),
        shape=(550, 300),
        n1=300,
        n2=550,
        endian="big",
        files=("camposL.dat", "gx373.dat", "el.dat", "ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
    ),
}

REFERENCE_DATASETS = {**REFERENCE_DATASETS_2D}


def reference_root(root: str | PathLike[str] | None = None) -> Path:
    """Resolve the reference_osv root path without requiring it to exist."""
    if root is not None:
        return Path(root)

    env_root = os.environ.get("PYOSV_REFERENCE_OSV")
    if env_root is not None:
        return Path(env_root)

    return Path.cwd() / "reference_osv"


def resolve_reference_file(
    dataset: str | ReferenceDataset,
    file_name: str,
    *,
    root: str | PathLike[str] | None = None,
) -> Path:
    """Construct a path to a file under a known reference dataset."""
    if isinstance(dataset, ReferenceDataset):
        metadata = dataset
    else:
        try:
            metadata = REFERENCE_DATASETS[dataset]
        except KeyError as error:
            raise KeyError(f"unknown reference dataset: {dataset}") from error

    return reference_root(root) / metadata.relative_dir / file_name
