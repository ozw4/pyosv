"""Fault skin containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from pyosv.cells import FaultCell

__all__ = ["FaultSkin"]


@dataclass(slots=True)
class FaultSkin:
    """Minimal grouped container for 3D :class:`FaultCell` objects."""

    cells: list[FaultCell] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cells = list(self.cells)
        for cell in self.cells:
            self._validate_cell(cell)

    @classmethod
    def from_cells(cls, cells: Iterable[FaultCell]) -> FaultSkin:
        return cls(list(cells))

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[FaultCell]:
        return iter(self.cells)

    def append(self, cell: FaultCell) -> None:
        """Append one fault cell to this skin."""

        self._validate_cell(cell)
        self.cells.append(cell)

    def add(self, cell: FaultCell) -> None:
        """Add one fault cell to this skin."""

        self.append(cell)

    def indices(self) -> np.ndarray:
        """Return cell indices as an ``(n, 3)`` array in ``(i1, i2, i3)`` order."""

        if not self.cells:
            return np.empty((0, 3), dtype=np.int32)
        return np.asarray([cell.index for cell in self.cells], dtype=np.int32)

    def likelihoods(self) -> np.ndarray:
        """Return cell likelihood values as an ``(n,)`` ``float32`` array."""

        return np.asarray([cell.fl for cell in self.cells], dtype=np.float32)

    @staticmethod
    def _validate_cell(cell: FaultCell) -> None:
        if not isinstance(cell, FaultCell):
            msg = "FaultSkin cells must be FaultCell instances"
            raise TypeError(msg)
