"""Constant-fill and oriented-volume sampling helpers."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from pyosv.interp import sample3


def _sample2_with_constant(
    image: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    order: int,
    fill_value: float,
) -> np.ndarray:
    coordinates = np.stack((x2, x1))
    sampled = ndimage.map_coordinates(
        image,
        coordinates,
        order=order,
        mode="constant",
        cval=float(fill_value),
    )
    return sampled.astype(np.float32, copy=False)


def _sample3_with_constant(
    volume: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    *,
    order: int,
    fill_value: float,
) -> np.ndarray:
    coordinates = np.stack((x3, x2, x1))
    sampled = ndimage.map_coordinates(
        volume,
        coordinates,
        order=order,
        mode="constant",
        cval=float(fill_value),
    )
    return sampled.astype(np.float32, copy=False)


def _sample_oriented_volume(
    volume: np.ndarray,
    *,
    direction: np.ndarray,
    offset: float,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
) -> np.ndarray:
    i1, i2, i3 = grids
    d1, d2, d3 = direction.astype(np.float32, copy=False)
    sampled = sample3(
        volume,
        i1 + np.float32(offset) * d1,
        i2 + np.float32(offset) * d2,
        i3 + np.float32(offset) * d3,
        order=interpolation_order,
        mode="nearest",
    )
    return np.asarray(sampled, dtype=np.float32)


def _smooth_oriented_response(
    image: np.ndarray,
    *,
    strike: np.ndarray,
    dip: np.ndarray,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
    smoothing_sigma: float,
) -> np.ndarray:
    if smoothing_sigma <= 0.0:
        return image.astype(np.float32, copy=True)

    smoothed = _directional_gaussian_smooth(
        image,
        direction=strike,
        sigma=smoothing_sigma,
        grids=grids,
        interpolation_order=interpolation_order,
    )
    smoothed = _directional_gaussian_smooth(
        smoothed,
        direction=dip,
        sigma=smoothing_sigma,
        grids=grids,
        interpolation_order=interpolation_order,
    )
    return smoothed.astype(np.float32, copy=False)


def _directional_gaussian_smooth(
    volume: np.ndarray,
    *,
    direction: np.ndarray,
    sigma: float,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
) -> np.ndarray:
    radius = math.ceil(3.0 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2).astype(np.float32)
    weights /= np.sum(weights, dtype=np.float32)

    smoothed = np.zeros_like(volume, dtype=np.float32)
    for offset, weight in zip(offsets, weights):
        smoothed += weight * _sample_oriented_volume(
            volume,
            direction=direction,
            offset=float(offset),
            grids=grids,
            interpolation_order=interpolation_order,
        )
    return smoothed
