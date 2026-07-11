"""Internal fault-cell models and linking helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyosv.cells import FaultCell, _java_round


@dataclass(slots=True, eq=False)
class _SkinCell:
    """Mutable internal fault cell for reference-like linking and growth."""

    x1: float
    x2: float
    x3: float
    fl: float
    fp: float
    ft: float
    i1: int = field(init=False)
    i2: int = field(init=False)
    i3: int = field(init=False)
    ca: _SkinCell | None = field(default=None, repr=False)
    cb: _SkinCell | None = field(default=None, repr=False)
    cl: _SkinCell | None = field(default=None, repr=False)
    cr: _SkinCell | None = field(default=None, repr=False)
    skin_id: int | None = None

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.x2 = float(self.x2)
        self.x3 = float(self.x3)
        self.fl = float(self.fl)
        self.fp = float(self.fp)
        self.ft = float(self.ft)
        self.i1 = _java_round(self.x1)
        self.i2 = _java_round(self.x2)
        self.i3 = _java_round(self.x3)

    @property
    def index(self) -> tuple[int, int, int]:
        return (self.i1, self.i2, self.i3)

    def to_fault_cell(self) -> FaultCell:
        return FaultCell(self.x1, self.x2, self.x3, self.fl, self.fp, self.ft)

    def fault_normal(self) -> np.ndarray:
        return self.to_fault_cell().fault_normal()

    def fault_dip_vector(self) -> np.ndarray:
        return self.to_fault_cell().fault_dip_vector()

    def fault_strike_vector(self) -> np.ndarray:
        return self.to_fault_cell().fault_strike_vector()


@dataclass(slots=True, frozen=True)
class _LocalTransformMap:
    """Local grow offsets where u=normal, v=dip, and w=strike."""

    us: np.ndarray
    vs: np.ndarray
    ws: np.ndarray


def _validate_skin_cell(cell: _SkinCell, name: str) -> None:
    if not isinstance(cell, _SkinCell):
        raise TypeError(f"{name} must be a _SkinCell")


def link_above_below(a: _SkinCell, b: _SkinCell) -> None:
    """Link two internal cells in the above/below direction."""

    _validate_skin_cell(a, "a")
    _validate_skin_cell(b, "b")
    a.cb = b
    b.ca = a


def link_left_right(left: _SkinCell, right: _SkinCell) -> None:
    """Link two internal cells in the left/right direction."""

    _validate_skin_cell(left, "left")
    _validate_skin_cell(right, "right")
    left.cr = right
    right.cl = left
