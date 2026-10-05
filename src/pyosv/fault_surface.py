"""Continuous low-dimensional surfaces from fault probability and candidate points.

This native extension represents a single nonfolded fault patch as a cubic
B-spline graph. Volumes use ``(iline, xline, sample)`` and points use
``(sample, xline, iline)``; coordinates are floating-point voxel indices.
"""

from pyosv._fault_surface.fit import fit_fault_surface, refit_fault_surface
from pyosv._fault_surface.geometry import evaluate_fault_surface, sample_fault_surface
from pyosv._fault_surface.models import (
    FaultSurfaceFitConfig,
    FaultSurfaceFitResult,
    FaultSurfaceModel,
    FaultSurfaceStick,
)

__all__ = [
    "FaultSurfaceFitConfig",
    "FaultSurfaceFitResult",
    "FaultSurfaceModel",
    "FaultSurfaceStick",
    "fit_fault_surface",
    "sample_fault_surface",
    "evaluate_fault_surface",
    "refit_fault_surface",
]
