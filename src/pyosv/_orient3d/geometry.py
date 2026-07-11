"""Geometry and derivative helpers for 3D orientation scanning."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _fault_normal_components_from_strike_and_dip(
    phi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.deg2rad(phi).astype(np.float32, copy=False)
    t = np.deg2rad(theta).astype(np.float32, copy=False)
    cp = np.cos(p)
    sp = np.sin(p)
    ct = np.cos(t)
    st = np.sin(t)
    w1 = -ct
    w2 = st * cp
    w3 = -st * sp
    return (
        w1.astype(np.float32, copy=False),
        w2.astype(np.float32, copy=False),
        w3.astype(np.float32, copy=False),
    )


def _gaussian_derivatives(
    image: np.ndarray,
    sigma: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    d1 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 0, 1), mode="nearest")
    d2 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 1, 0), mode="nearest")
    d3 = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 0, 0), mode="nearest")
    d11 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 0, 2), mode="nearest")
    d22 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2, 0), mode="nearest")
    d33 = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0, 0), mode="nearest")
    d12 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 1, 1), mode="nearest")
    d13 = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 0, 1), mode="nearest")
    d23 = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1, 0), mode="nearest")
    return (
        d1.astype(np.float32, copy=False),
        d2.astype(np.float32, copy=False),
        d3.astype(np.float32, copy=False),
        d11.astype(np.float32, copy=False),
        d22.astype(np.float32, copy=False),
        d33.astype(np.float32, copy=False),
        d12.astype(np.float32, copy=False),
        d13.astype(np.float32, copy=False),
        d23.astype(np.float32, copy=False),
    )


def _coordinate_grids3(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i3, i2, i1 = np.indices(shape, dtype=np.float32)
    return i1, i2, i3
