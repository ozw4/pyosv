"""Typed arrays for single-patch fault interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaultSurfaceFitConfig:
    """Cubic control grid, evidence corridor, and derived geometry resolution.

    ``search_radius`` and ``search_step`` are dependent-axis voxel distances.
    Sparse-seed plane initialization uses a local sphere of at most
    ``min(search_radius, 16)`` voxels before the dependent-axis search.
    ``smoothness`` penalizes second differences of spline coefficients.
    ``support_threshold`` applies to the original DL probability, not OSV votes.
    """

    control_shape: tuple[int, int] = (6, 6)
    mesh_shape: tuple[int, int] = (33, 33)
    search_radius: float = 6.0
    search_step: float = 0.5
    smoothness: float = 0.05
    support_threshold: float = 0.2
    stick_count: int = 4
    stick_point_count: int = 5


@dataclass(frozen=True)
class FaultSurfaceModel:
    """Durable cubic graph; axes index point columns (sample, xline, iline).

    ``bounds`` contains independent-axis [minimum, maximum] rows.
    ``coefficients`` has one axis for each independent coordinate.
    Open-uniform cubic knots are implied by the coefficient shape.
    """

    dependent_axis: int
    independent_axes: tuple[int, int]
    bounds: np.ndarray
    coefficients: np.ndarray
    volume_shape: tuple[int, int, int]
    mesh_shape: tuple[int, int] = (33, 33)
    smoothness: float = 0.05
    stick_count: int = 4
    stick_point_count: int = 5


@dataclass(frozen=True)
class FaultSurfaceStick:
    """Derived section samples, not independent fitted observations.

    ``axis`` indexes point columns. All points lie on coordinate ``index``.
    """

    axis: int
    index: float
    points: np.ndarray


@dataclass(frozen=True)
class FaultSurfaceFitResult:
    """Continuous derived mesh and sparse native-section sticks.

    Arrays are independent, read-only snapshots. ``probability`` and
    ``supported`` are per-vertex DL evidence, absent for probability-free
    refits. A False support value does not remove a triangle.
    """

    model: FaultSurfaceModel
    vertices: np.ndarray
    triangles: np.ndarray
    sticks: tuple[FaultSurfaceStick, ...]
    probability: np.ndarray | None = None
    supported: np.ndarray | None = None
