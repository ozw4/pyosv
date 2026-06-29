"""Coordinate sampling adapters for OSV-style array layouts."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["rotate2d", "sample2", "sample3", "warp2d"]


def sample2(image, x1, x2, *, order: int = 1, mode: str = "nearest"):
    """Sample a 2D image at Java-style ``(i1, i2)`` coordinates.

    ``image`` must use pyosv's 2D shape convention ``(n2, n1)``. SciPy expects
    coordinates in array-axis order, so this adapter passes ``(x2, x1)`` to
    ``map_coordinates``.
    """

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("image must have shape (n2, n1)")

    x1_array, x2_array = np.asarray(x1, dtype=np.float32), np.asarray(x2, dtype=np.float32)
    scalar_input = x1_array.ndim == 0 and x2_array.ndim == 0
    x1_broadcast, x2_broadcast = np.broadcast_arrays(x1_array, x2_array)
    coordinates = _coordinates_for_map((x2_broadcast, x1_broadcast), scalar_input)

    sampled = ndimage.map_coordinates(image_array, coordinates, order=order, mode=mode)
    return _restore_sampled_type(sampled, image_array.dtype, scalar_input)


def sample3(volume, x1, x2, x3, *, order: int = 1, mode: str = "nearest"):
    """Sample a 3D volume at Java-style ``(i1, i2, i3)`` coordinates.

    ``volume`` must use pyosv's 3D shape convention ``(n3, n2, n1)``. SciPy
    expects coordinates in array-axis order, so this adapter passes
    ``(x3, x2, x1)`` to ``map_coordinates``.
    """

    volume_array = np.asarray(volume)
    if volume_array.ndim != 3:
        raise ValueError("volume must have shape (n3, n2, n1)")

    x1_array = np.asarray(x1, dtype=np.float32)
    x2_array = np.asarray(x2, dtype=np.float32)
    x3_array = np.asarray(x3, dtype=np.float32)
    scalar_input = x1_array.ndim == 0 and x2_array.ndim == 0 and x3_array.ndim == 0
    x1_broadcast, x2_broadcast, x3_broadcast = np.broadcast_arrays(
        x1_array,
        x2_array,
        x3_array,
    )
    coordinates = _coordinates_for_map(
        (x3_broadcast, x2_broadcast, x1_broadcast),
        scalar_input,
    )

    sampled = ndimage.map_coordinates(volume_array, coordinates, order=order, mode=mode)
    return _restore_sampled_type(sampled, volume_array.dtype, scalar_input)


def warp2d(image, x1, x2, *, order: int = 1, mode: str = "nearest") -> np.ndarray:
    """Warp a 2D image using Java-style coordinate arrays.

    ``image`` must have shape ``(n2, n1)``. ``x1`` and ``x2`` are broadcast to
    the output shape and interpreted as coordinates along axis 1 and axis 0.
    """

    image_array = np.asarray(image)
    warped = np.asarray(sample2(image_array, x1, x2, order=order, mode=mode))
    return _restore_array_type(warped, image_array.dtype)


def rotate2d(
    image,
    angle_degrees: float,
    *,
    reshape: bool = False,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """Rotate a 2D ``(n2, n1)`` image by ``angle_degrees``."""

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("image must have shape (n2, n1)")

    rotated = ndimage.rotate(
        image_array,
        angle_degrees,
        reshape=reshape,
        order=order,
        mode=mode,
    )
    return _restore_array_type(rotated, image_array.dtype)


def _coordinates_for_map(axis_coordinates, scalar_input: bool) -> np.ndarray:
    coordinates = np.stack(axis_coordinates)
    if scalar_input:
        return coordinates.reshape(len(axis_coordinates), 1)
    return coordinates


def _restore_sampled_type(sampled: np.ndarray, input_dtype: np.dtype, scalar_input: bool):
    if scalar_input:
        return float(sampled[0])
    return _restore_array_type(sampled, input_dtype)


def _restore_array_type(sampled: np.ndarray, input_dtype: np.dtype) -> np.ndarray:
    if input_dtype == np.dtype("float32") and sampled.dtype != np.float32:
        return sampled.astype(np.float32, copy=False)
    return sampled
