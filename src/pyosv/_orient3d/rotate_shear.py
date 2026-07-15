"""Rotate, shear, and axis-aligned smoothing kernels."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from pyosv._orient3d.geometry import _coordinate_grids3
from pyosv._orient3d.interpolation import _sample2_with_constant, _sample3_with_constant
from pyosv._orient3d.structured_linear import (
    _rotate3_axis1_structured,
    _shear2_structured,
    _unrotate3_axis1_structured,
    _unshear2_structured,
)


def _rotate3_axis1(
    volume: np.ndarray,
    phi_degrees: float,
    *,
    interpolation_order: int,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    """Rotate a volume around axis 1 so strike is aligned with rotated axis 3."""

    n3, n2, _ = volume.shape
    phi = math.radians(phi_degrees)
    sin_phi = np.float32(math.sin(phi))
    cos_phi = np.float32(math.cos(phi))
    nrot2, nrot3, origin2, origin3 = _rotated_axis1_grid(
        n2,
        n3,
        sin_phi=float(sin_phi),
        cos_phi=float(cos_phi),
    )
    rotated_shape = (nrot3, nrot2, volume.shape[2])
    center2 = np.float32(0.5 * (n2 - 1))
    center3 = np.float32(0.5 * (n3 - 1))
    if interpolation_backend == "structured_linear":
        return _rotate3_axis1_structured(
            volume,
            nrot3,
            nrot2,
            origin2,
            origin3,
            center2,
            center3,
            sin_phi,
            cos_phi,
            np.float32(1.0),
        )

    i1, j2, j3 = _coordinate_grids3(rotated_shape)
    d2 = j2 + origin2
    d3 = j3 + origin3
    source_x2 = center2 + d2 * cos_phi + d3 * sin_phi
    source_x3 = center3 - d2 * sin_phi + d3 * cos_phi
    rotated = _sample3_with_constant(
        volume,
        i1,
        source_x2.astype(np.float32, copy=False),
        source_x3.astype(np.float32, copy=False),
        order=interpolation_order,
        fill_value=1.0,
    )
    return np.asarray(rotated, dtype=np.float32)


def _unrotate3_axis1(
    rotated: np.ndarray,
    shape: tuple[int, int, int],
    phi_degrees: float,
    *,
    interpolation_order: int,
    fill_value: float = 0.0,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    """Unrotate an axis-1 rotated volume back to global coordinates."""

    n3, n2, _ = shape
    center2 = np.float32(0.5 * (n2 - 1))
    center3 = np.float32(0.5 * (n3 - 1))
    phi = math.radians(phi_degrees)
    sin_phi = np.float32(math.sin(phi))
    cos_phi = np.float32(math.cos(phi))
    _, _, origin2, origin3 = _rotated_axis1_grid(
        n2,
        n3,
        sin_phi=float(sin_phi),
        cos_phi=float(cos_phi),
    )
    if interpolation_backend == "structured_linear":
        return _unrotate3_axis1_structured(
            rotated,
            shape,
            origin2,
            origin3,
            center2,
            center3,
            sin_phi,
            cos_phi,
            np.float32(fill_value),
        )

    i1, i2, i3 = _coordinate_grids3(shape)
    d2 = i2 - center2
    d3 = i3 - center3
    source_x2 = d2 * cos_phi - d3 * sin_phi - origin2
    source_x3 = d2 * sin_phi + d3 * cos_phi - origin3
    unrotated = _sample3_with_constant(
        rotated,
        i1,
        source_x2.astype(np.float32, copy=False),
        source_x3.astype(np.float32, copy=False),
        order=interpolation_order,
        fill_value=fill_value,
    )
    return np.asarray(unrotated, dtype=np.float32)


def _rotated_axis1_grid(
    n2: int,
    n3: int,
    *,
    sin_phi: float,
    cos_phi: float,
) -> tuple[int, int, np.float32, np.float32]:
    center2 = 0.5 * (n2 - 1)
    center3 = 0.5 * (n3 - 1)
    corners = (
        (-center2, -center3),
        (-center2, (n3 - 1) - center3),
        ((n2 - 1) - center2, -center3),
        ((n2 - 1) - center2, (n3 - 1) - center3),
    )
    rotated2_values = []
    rotated3_values = []
    for d2, d3 in corners:
        rotated2_values.append(d2 * cos_phi - d3 * sin_phi)
        rotated3_values.append(d2 * sin_phi + d3 * cos_phi)

    min2 = min(rotated2_values)
    max2 = max(rotated2_values)
    min3 = min(rotated3_values)
    max3 = max(rotated3_values)
    radius2 = max(abs(min2), abs(max2))
    radius3 = max(abs(min3), abs(max3))
    nrot2 = _symmetric_sample_count_covering_radius(radius2)
    nrot3 = _symmetric_sample_count_covering_radius(radius3)
    return nrot2, nrot3, np.float32(-radius2), np.float32(-radius3)


def _symmetric_sample_count_covering_radius(radius: float) -> int:
    nearest_integer = round(radius)
    if math.isclose(radius, nearest_integer, abs_tol=1.0e-6):
        half_width = int(nearest_integer)
    else:
        half_width = math.ceil(radius)

    return 2 * half_width + 1


def _shear2(
    image: np.ndarray,
    shear: float,
    *,
    interpolation_order: int,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    """Shear a ``(n2, n1)`` slice in axis 2 by ``shear * axis1``."""

    n2, n1 = image.shape
    center1 = np.float32(0.5 * (n1 - 1))
    if interpolation_backend == "structured_linear":
        return _shear2_structured(
            image,
            np.float32(shear),
            center1,
            np.float32(1.0),
        )

    i2, i1 = np.indices((n2, n1), dtype=np.float32)
    source_x2 = i2 - np.float32(shear) * (i1 - center1)
    sheared = _sample2_with_constant(
        image,
        i1,
        source_x2.astype(np.float32, copy=False),
        order=interpolation_order,
        fill_value=1.0,
    )
    return np.asarray(sheared, dtype=np.float32)


def _unshear2(
    sheared: np.ndarray,
    shear: float,
    *,
    interpolation_order: int,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    """Restore a same-shape slice previously transformed by :func:`_shear2`."""

    if interpolation_backend == "structured_linear":
        center1 = np.float32(0.5 * (sheared.shape[1] - 1))
        return _unshear2_structured(
            sheared,
            np.float32(shear),
            center1,
            np.float32(1.0),
        )

    return _shear2(sheared, -shear, interpolation_order=interpolation_order)


def _shear_rotated_volume(
    rotated: np.ndarray,
    shear: float,
    *,
    interpolation_order: int,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    sheared = np.empty_like(rotated, dtype=np.float32)
    for i3 in range(rotated.shape[0]):
        sheared[i3] = _shear2(
            rotated[i3],
            shear,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
        )
    return sheared


def _unshear_rotated_volume(
    sheared: np.ndarray,
    shear: float,
    *,
    interpolation_order: int,
    interpolation_backend: str = "scipy",
) -> np.ndarray:
    unsheared = np.empty_like(sheared, dtype=np.float32)
    for i3 in range(sheared.shape[0]):
        unsheared[i3] = _unshear2(
            sheared[i3],
            shear,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
        )
    return unsheared


def _smooth_rotated_strike_axis(rotated: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return rotated.astype(np.float32, copy=True)

    smoothed = ndimage.gaussian_filter1d(
        rotated,
        sigma=sigma,
        axis=0,
        mode="nearest",
    )
    return smoothed.astype(np.float32, copy=False)


def _smooth_sheared_dip_axis(
    sheared: np.ndarray,
    *,
    sigma: float,
    theta_degrees: float,
) -> np.ndarray:
    if sigma <= 0.0:
        return sheared.astype(np.float32, copy=True)

    sin_theta = abs(math.sin(math.radians(theta_degrees)))
    axis_sigma = max(0.0, sigma * sin_theta)
    if axis_sigma <= 0.0:
        return sheared.astype(np.float32, copy=True)

    smoothed = ndimage.gaussian_filter1d(
        sheared,
        sigma=axis_sigma,
        axis=2,
        mode="nearest",
    )
    return smoothed.astype(np.float32, copy=False)


def _dip_shear_from_theta(theta_degrees: float) -> np.float32:
    theta = math.radians(theta_degrees)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    if abs(cos_theta) <= 1.0e-6:
        return np.float32(0.0)
    if abs(sin_theta) <= 1.0e-6:
        return np.float32(math.copysign(1.0e4, -cos_theta))

    shear = -cos_theta / sin_theta
    return np.float32(np.clip(shear, -1.0e4, 1.0e4))
