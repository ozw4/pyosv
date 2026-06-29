"""Fault cell containers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
)


def _java_round(value: float) -> int:
    return math.floor(float(value) + 0.5)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "x1", float(self.x1))
        object.__setattr__(self, "x2", float(self.x2))
        object.__setattr__(self, "x3", float(self.x3))
        object.__setattr__(self, "fl", float(self.fl))
        object.__setattr__(self, "fp", float(self.fp))
        object.__setattr__(self, "ft", float(self.ft))

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
