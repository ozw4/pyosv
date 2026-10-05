"""Fail-fast array and graph-domain validation."""

from __future__ import annotations

import numbers

import numpy as np

from pyosv._fault_surface.models import FaultSurfaceFitConfig, FaultSurfaceModel


def integer(value: int, name: str, low: int, high: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def shape_pair(value: tuple[int, int], name: str, low: int, high: int) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two dimensions")
    for entry in value:
        integer(entry, name, low, high)


def validate_config(config: FaultSurfaceFitConfig) -> None:
    shape_pair(config.control_shape, "control_shape", 4, 12)
    shape_pair(config.mesh_shape, "mesh_shape", 2, 257)
    integer(config.stick_count, "stick_count", 2, config.control_shape[1])
    integer(config.stick_point_count, "stick_point_count", 2, config.control_shape[0])
    for name in ("search_radius", "search_step", "smoothness", "support_threshold"):
        value = getattr(config, name)
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.search_step <= 0 or config.search_radius / config.search_step > 256:
        raise ValueError(
            "search_step must be positive and the corridor must use at most 513 samples"
        )
    if not 0 < config.support_threshold <= 1:
        raise ValueError("support_threshold must be in (0, 1]")


def validate_volume(probability: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    volume = np.asarray(probability)
    if volume.ndim != 3 or min(volume.shape) < 2 or not np.issubdtype(volume.dtype, np.floating):
        raise ValueError("probability must be a floating (iline, xline, sample) volume")
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != volume.shape or mask.dtype != np.bool_:
            raise ValueError("valid_mask must be Boolean with probability's shape")
    return volume


def validate_points(points: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("Points must be finite (N, 3) sample/xline/iline coordinates")
    maximum = np.asarray(volume_shape[::-1]) - 1
    if np.any(array < 0) or np.any(array > maximum):
        raise ValueError("Points must be inside the probability volume")
    return array


def validate_model(model: FaultSurfaceModel) -> None:
    integer(model.dependent_axis, "dependent_axis", 0, 2)
    axes = tuple(axis for axis in range(3) if axis != model.dependent_axis)
    if tuple(model.independent_axes) != axes:
        raise ValueError("independent_axes must be the remaining axes in ascending order")
    if len(model.volume_shape) != 3:
        raise ValueError("volume_shape must contain three dimensions")
    for size in model.volume_shape:
        integer(size, "volume_shape", 2, 2**31 - 1)
    bounds = np.asarray(model.bounds)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("bounds must be finite (2, 2) independent-axis min/max pairs")
    maxima = np.asarray(model.volume_shape[::-1])[list(axes)] - 1
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > maxima):
        raise ValueError("Model domain lies outside its volume")
    if np.any(bounds[:, 1] - bounds[:, 0] <= 1e-6):
        raise ValueError("Model domain must span two independent axes")
    coefficients = np.asarray(model.coefficients)
    if coefficients.ndim != 2 or not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be a finite two-dimensional array")
    config = FaultSurfaceFitConfig(
        control_shape=coefficients.shape,
        mesh_shape=model.mesh_shape,
        smoothness=model.smoothness,
        stick_count=model.stick_count,
        stick_point_count=model.stick_point_count,
    )
    validate_config(config)
