"""Structured linear interpolation kernels for rotate/shear transforms."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE, njit


def _rotate3_axis1_structured_python(
    volume: np.ndarray,
    nrot3: int,
    nrot2: int,
    origin2: np.float32,
    origin3: np.float32,
    center2: np.float32,
    center3: np.float32,
    sin_phi: np.float32,
    cos_phi: np.float32,
    fill_value: np.float32,
) -> np.ndarray:
    """Rotate validated ``(n3, n2, n1)`` data with direct bilinear sampling."""

    n3, n2, n1 = volume.shape
    output = np.empty((nrot3, nrot2, n1), dtype=np.float32)
    one = np.float32(1.0)
    for j3 in range(nrot3):
        d3 = np.float32(j3) + origin3
        for j2 in range(nrot2):
            d2 = np.float32(j2) + origin2
            x2 = np.float32(center2 + d2 * cos_phi + d3 * sin_phi)
            x3 = np.float32(center3 - d2 * sin_phi + d3 * cos_phi)
            if x2 < 0.0 or x2 > n2 - 1 or x3 < 0.0 or x3 > n3 - 1:
                for i1 in range(n1):
                    output[j3, j2, i1] = fill_value
                continue

            i2 = int(math.floor(x2))
            i3 = int(math.floor(x3))
            k2 = min(i2 + 1, n2 - 1)
            k3 = min(i3 + 1, n3 - 1)
            w2 = np.float32(x2 - np.float32(i2))
            w3 = np.float32(x3 - np.float32(i3))
            a2 = np.float32(one - w2)
            a3 = np.float32(one - w3)
            for i1 in range(n1):
                lower = np.float32(a2 * volume[i3, i2, i1] + w2 * volume[i3, k2, i1])
                upper = np.float32(a2 * volume[k3, i2, i1] + w2 * volume[k3, k2, i1])
                output[j3, j2, i1] = np.float32(a3 * lower + w3 * upper)
    return output


def _unrotate3_axis1_structured_python(
    rotated: np.ndarray,
    shape: tuple[int, int, int],
    origin2: np.float32,
    origin3: np.float32,
    center2: np.float32,
    center3: np.float32,
    sin_phi: np.float32,
    cos_phi: np.float32,
    fill_value: np.float32,
) -> np.ndarray:
    """Unrotate validated data with direct bilinear sampling."""

    n3, n2, n1 = shape
    nrot3, nrot2, _ = rotated.shape
    output = np.empty(shape, dtype=np.float32)
    one = np.float32(1.0)
    for i3_out in range(n3):
        d3 = np.float32(i3_out) - center3
        for i2_out in range(n2):
            d2 = np.float32(i2_out) - center2
            x2 = np.float32(d2 * cos_phi - d3 * sin_phi - origin2)
            x3 = np.float32(d2 * sin_phi + d3 * cos_phi - origin3)
            if x2 < 0.0 or x2 > nrot2 - 1 or x3 < 0.0 or x3 > nrot3 - 1:
                for i1 in range(n1):
                    output[i3_out, i2_out, i1] = fill_value
                continue

            i2 = int(math.floor(x2))
            i3 = int(math.floor(x3))
            k2 = min(i2 + 1, nrot2 - 1)
            k3 = min(i3 + 1, nrot3 - 1)
            w2 = np.float32(x2 - np.float32(i2))
            w3 = np.float32(x3 - np.float32(i3))
            a2 = np.float32(one - w2)
            a3 = np.float32(one - w3)
            for i1 in range(n1):
                lower = np.float32(a2 * rotated[i3, i2, i1] + w2 * rotated[i3, k2, i1])
                upper = np.float32(a2 * rotated[k3, i2, i1] + w2 * rotated[k3, k2, i1])
                output[i3_out, i2_out, i1] = np.float32(a3 * lower + w3 * upper)
    return output


def _shear2_structured_python(
    image: np.ndarray,
    shear: np.float32,
    center1: np.float32,
    fill_value: np.float32,
) -> np.ndarray:
    """Shear validated ``(n2, n1)`` data with direct linear sampling."""

    n2, n1 = image.shape
    output = np.empty((n2, n1), dtype=np.float32)
    one = np.float32(1.0)
    for i2_out in range(n2):
        for i1 in range(n1):
            x2 = np.float32(i2_out) - shear * (np.float32(i1) - center1)
            x2 = np.float32(x2)
            if x2 < 0.0 or x2 > n2 - 1:
                output[i2_out, i1] = fill_value
                continue

            i2 = int(math.floor(x2))
            k2 = min(i2 + 1, n2 - 1)
            w2 = np.float32(x2 - np.float32(i2))
            output[i2_out, i1] = np.float32(
                np.float32(one - w2) * image[i2, i1] + w2 * image[k2, i1]
            )
    return output


def _unshear2_structured_python(
    sheared: np.ndarray,
    shear: np.float32,
    center1: np.float32,
    fill_value: np.float32,
) -> np.ndarray:
    """Unshear validated ``(n2, n1)`` data with direct linear sampling."""

    n2, n1 = sheared.shape
    output = np.empty((n2, n1), dtype=np.float32)
    one = np.float32(1.0)
    for i2_out in range(n2):
        for i1 in range(n1):
            x2 = np.float32(i2_out) + shear * (np.float32(i1) - center1)
            x2 = np.float32(x2)
            if x2 < 0.0 or x2 > n2 - 1:
                output[i2_out, i1] = fill_value
                continue

            i2 = int(math.floor(x2))
            k2 = min(i2 + 1, n2 - 1)
            w2 = np.float32(x2 - np.float32(i2))
            output[i2_out, i1] = np.float32(
                np.float32(one - w2) * sheared[i2, i1] + w2 * sheared[k2, i1]
            )
    return output


_rotate3_axis1_structured_numba = njit(cache=True)(_rotate3_axis1_structured_python)
_unrotate3_axis1_structured_numba = njit(cache=True)(_unrotate3_axis1_structured_python)
_shear2_structured_numba = njit(cache=True)(_shear2_structured_python)
_unshear2_structured_numba = njit(cache=True)(_unshear2_structured_python)


def _structured_kernel(python_kernel, numba_kernel):
    return numba_kernel if NUMBA_AVAILABLE else python_kernel


_rotate3_axis1_structured = _structured_kernel(
    _rotate3_axis1_structured_python,
    _rotate3_axis1_structured_numba,
)
_unrotate3_axis1_structured = _structured_kernel(
    _unrotate3_axis1_structured_python,
    _unrotate3_axis1_structured_numba,
)
_shear2_structured = _structured_kernel(
    _shear2_structured_python,
    _shear2_structured_numba,
)
_unshear2_structured = _structured_kernel(
    _unshear2_structured_python,
    _unshear2_structured_numba,
)
