"""Fault cell containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
