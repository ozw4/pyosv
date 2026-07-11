"""Sparse grid for internal skin cells."""

import operator

from pyosv._skinner.models import _SkinCell
from pyosv._skinner.validation import _validate_nonnegative_int


class _SkinCellGrid:
    """Sparse grid keyed by rounded fault-cell indices."""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int, int], _SkinCell] = {}

    def set(self, cell: _SkinCell) -> None:
        self._validate_cell(cell)
        self._cells[cell.index] = cell

    def get(self, i1: int, i2: int, i3: int) -> _SkinCell | None:
        return self._cells.get(_index_key(i1, i2, i3))

    def set_cells_in_box(self, cell: _SkinCell, r1: int, r2: int, r3: int) -> None:
        self._validate_cell(cell)
        radius1 = _validate_nonnegative_int(r1, "r1")
        radius2 = _validate_nonnegative_int(r2, "r2")
        radius3 = _validate_nonnegative_int(r3, "r3")

        for i3 in range(cell.i3 - radius3, cell.i3 + radius3 + 1):
            for i2 in range(cell.i2 - radius2, cell.i2 + radius2 + 1):
                for i1 in range(cell.i1 - radius1, cell.i1 + radius1 + 1):
                    self._cells[(i1, i2, i3)] = cell

    def find_cells_in_box(
        self,
        i1: int,
        i2: int,
        i3: int,
        r1: int,
        r2: int,
        r3: int,
    ) -> list[_SkinCell]:
        center1, center2, center3 = _index_key(i1, i2, i3)
        radius1 = _validate_nonnegative_int(r1, "r1")
        radius2 = _validate_nonnegative_int(r2, "r2")
        radius3 = _validate_nonnegative_int(r3, "r3")

        found: list[_SkinCell] = []
        seen: set[int] = set()
        for j3 in range(center3 - radius3, center3 + radius3 + 1):
            for j2 in range(center2 - radius2, center2 + radius2 + 1):
                for j1 in range(center1 - radius1, center1 + radius1 + 1):
                    cell = self._cells.get((j1, j2, j3))
                    if cell is not None and id(cell) not in seen:
                        found.append(cell)
                        seen.add(id(cell))

        found.sort(key=lambda cell: cell.index)
        return found

    @staticmethod
    def _validate_cell(cell: _SkinCell) -> None:
        if not isinstance(cell, _SkinCell):
            msg = "_SkinCellGrid only stores _SkinCell instances"
            raise TypeError(msg)


def _index_key(i1: int, i2: int, i3: int) -> tuple[int, int, int]:
    try:
        return (operator.index(i1), operator.index(i2), operator.index(i3))
    except TypeError as exc:
        raise ValueError("indices must be integers") from exc
