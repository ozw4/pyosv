"""Immutable configuration snapshots for 3D surface-voting policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SurfaceVoterConfig:
    """Configuration used by one surface-voting execution."""

    ru: int
    rv: int
    rw: int
    lmin: int
    bstrain1: int
    bstrain2: int
    attribute_smoothing: int
    surface_smoothing1: float
    surface_smoothing2: float
    surface_orientation_smoothing: float
    surface_orientation_backend: str
    surface_support_min_fraction: float
    surface_support_exponent: float
