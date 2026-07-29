"""Fault cell containers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
)


def _java_round(value: float) -> int:
    return math.floor(float(value) + 0.5)


FAULT_CELL_GENERATION_GROWN = "grown"
FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED = "existing_cells_reskinned"
FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED = "dense_reskin_observed"
FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED = "dense_reskin_generated"
FAULT_CELL_GENERATION_CONNECTED_COMPONENT = "connected_component"
FaultCellGeneration = Literal[
    "grown",
    "existing_cells_reskinned",
    "dense_reskin_observed",
    "dense_reskin_generated",
    "connected_component",
]
_FAULT_CELL_GENERATIONS = {
    FAULT_CELL_GENERATION_GROWN,
    FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
}
_DENSE_RESKIN_GENERATIONS = {
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
}


@dataclass(slots=True, frozen=True)
class FaultCell2:
    """Minimal 2D fault seed cell."""

    i1: int
    i2: int
    fl: float
    fp: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "i1", int(self.i1))
        object.__setattr__(self, "i2", int(self.i2))
        object.__setattr__(self, "fl", float(self.fl))
        object.__setattr__(self, "fp", float(self.fp))

    @property
    def index(self) -> tuple[int, int]:
        return (self.i1, self.i2)

    def fault_normal(self) -> np.ndarray:
        p = np.deg2rad(self.fp)
        return np.array([np.sin(p), np.cos(p)], dtype=np.float32)

    def fault_strike_vector(self) -> np.ndarray:
        p = np.deg2rad(self.fp)
        return np.array([-np.cos(p), np.sin(p)], dtype=np.float32)


@dataclass(slots=True, frozen=True)
class FaultCell:
    """Minimal 3D fault seed cell."""

    x1: float
    x2: float
    x3: float
    fl: float
    fp: float
    ft: float
    generation: FaultCellGeneration = field(
        default=FAULT_CELL_GENERATION_GROWN,
        compare=False,
        kw_only=True,
    )
    reskin_support: float | None = field(default=None, compare=False, kw_only=True)
    ca: FaultCell | None = field(default=None, init=False, repr=False, compare=False)
    cb: FaultCell | None = field(default=None, init=False, repr=False, compare=False)
    cl: FaultCell | None = field(default=None, init=False, repr=False, compare=False)
    cr: FaultCell | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x1", float(self.x1))
        object.__setattr__(self, "x2", float(self.x2))
        object.__setattr__(self, "x3", float(self.x3))
        object.__setattr__(self, "fl", float(self.fl))
        object.__setattr__(self, "fp", float(self.fp))
        object.__setattr__(self, "ft", float(self.ft))
        if not isinstance(self.generation, str) or self.generation not in _FAULT_CELL_GENERATIONS:
            raise ValueError(
                "generation must be 'grown', 'existing_cells_reskinned', "
                "'dense_reskin_observed', 'dense_reskin_generated', or "
                "'connected_component'",
            )
        support = self.reskin_support
        if self.generation in _DENSE_RESKIN_GENERATIONS:
            if (
                isinstance(support, (bool, np.bool_))
                or not isinstance(support, (int, float, np.integer, np.floating))
                or not math.isfinite(float(support))
                or not 0.0 <= float(support) <= 1.0
            ):
                raise ValueError(
                    "reskin_support must be a finite number in [0, 1] for dense reskin cells",
                )
            object.__setattr__(self, "reskin_support", float(support))
        elif support is not None:
            raise ValueError(
                "reskin_support must be None unless generation is a dense reskin value"
            )

    @property
    def i1(self) -> int:
        return _java_round(self.x1)

    @property
    def i2(self) -> int:
        return _java_round(self.x2)

    @property
    def i3(self) -> int:
        return _java_round(self.x3)

    @property
    def index(self) -> tuple[int, int, int]:
        return (self.i1, self.i2, self.i3)

    def fault_normal(self) -> np.ndarray:
        return fault_normal_vector_from_strike_and_dip(self.fp, self.ft)

    def fault_dip_vector(self) -> np.ndarray:
        return fault_dip_vector_from_strike_and_dip(self.fp, self.ft)

    def fault_strike_vector(self) -> np.ndarray:
        return fault_strike_vector_from_strike_and_dip(self.fp, self.ft)
