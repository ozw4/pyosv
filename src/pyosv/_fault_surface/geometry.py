"""Deterministic continuous mesh and native-axis section sampling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyosv._fault_surface.basis import basis_matrix
from pyosv._fault_surface.models import FaultSurfaceFitResult, FaultSurfaceModel, FaultSurfaceStick
from pyosv._fault_surface.validation import validate_model


def readonly(array: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def numerical_grid_shape(control_shape: tuple[int, int]) -> tuple[int, int]:
    return tuple(max(33, 4 * count + 1) for count in control_shape)


def grid_coordinates(bounds: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    axes = [np.linspace(low, high, count) for (low, high), count in zip(bounds, shape)]
    first, second = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((first.ravel(), second.ravel()))


def evaluate_graph(model: FaultSurfaceModel, uv: np.ndarray) -> np.ndarray:
    matrix = basis_matrix(uv, np.asarray(model.bounds), np.shape(model.coefficients))
    points = np.empty((len(uv), 3), dtype=np.float64)
    points[:, model.independent_axes] = uv
    points[:, model.dependent_axis] = matrix @ np.asarray(model.coefficients).ravel()
    return points


def evaluate_fault_surface(model: FaultSurfaceModel, points_uv: np.ndarray) -> np.ndarray:
    """Evaluate float64 XYZ coordinates at in-domain independent coordinates.

    Double precision retains the exact manual constraints represented by the
    model; render vertices returned by ``sample_fault_surface`` are float32.
    """
    validate_model(model)
    uv = np.asarray(points_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv)):
        raise ValueError("points_uv must be finite (N, 2) independent coordinates")
    if np.any(uv < model.bounds[:, 0]) or np.any(uv > model.bounds[:, 1]):
        raise ValueError("Evaluation coordinates lie outside the model domain")
    return readonly(evaluate_graph(model, uv))


def mesh_triangles(shape: tuple[int, int]) -> np.ndarray:
    rows, columns = shape
    first = (np.arange(rows - 1)[:, None] * columns + np.arange(columns - 1)).ravel()
    triangles = np.empty((2 * len(first), 3), dtype=np.int32)
    triangles[::2] = np.column_stack((first, first + columns, first + 1))
    triangles[1::2] = np.column_stack((first + columns, first + columns + 1, first + 1))
    return triangles


def native_sticks(model: FaultSurfaceModel) -> tuple[FaultSurfaceStick, ...]:
    low, high = np.ceil(model.bounds[1, 0]), np.floor(model.bounds[1, 1])
    if low > high:
        raise ValueError("Surface domain does not intersect a native integer section")
    indices = np.unique(np.rint(np.linspace(low, high, model.stick_count)).astype(int))
    first = np.linspace(*model.bounds[0], model.stick_point_count)
    sticks = []
    for index in indices:
        uv = np.column_stack((first, np.full_like(first, index)))
        sticks.append(
            FaultSurfaceStick(
                axis=model.independent_axes[1],
                index=int(index),
                points=readonly(evaluate_graph(model, uv), np.float64),
            )
        )
    return tuple(sticks)


def sample_fault_surface(model: FaultSurfaceModel) -> FaultSurfaceFitResult:
    """Create a complete regular mesh and sparse integer-section sticks.

    No triangle is removed on account of missing prediction support. Models
    whose sampled surface leaves the volume are rejected, never clipped.
    """
    validate_model(model)
    frozen = replace(
        model, bounds=readonly(model.bounds), coefficients=readonly(model.coefficients)
    )
    points = evaluate_graph(frozen, grid_coordinates(frozen.bounds, frozen.mesh_shape))
    dense_shape = tuple(max(65, count) for count in frozen.mesh_shape)
    dense = evaluate_graph(frozen, grid_coordinates(frozen.bounds, dense_shape))
    sticks = native_sticks(frozen)
    maximum = np.asarray(frozen.volume_shape[::-1]) - 1
    checks = (points, dense, *(stick.points for stick in sticks))
    for check in checks:
        if (
            not np.all(np.isfinite(check))
            or np.any(check < -1e-5)
            or np.any(check > maximum + 1e-5)
        ):
            raise ValueError("Fitted surface leaves the volume; reduce or revise the constraints")
    return FaultSurfaceFitResult(
        model=frozen,
        vertices=readonly(points, np.float32),
        triangles=readonly(mesh_triangles(frozen.mesh_shape)),
        sticks=sticks,
    )
