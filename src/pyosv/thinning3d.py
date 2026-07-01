"""Reference-like 3D thinning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy import ndimage

__all__ = ["reference_like_3d_nms_mask", "reference_like_3d_thin_values"]


def reference_like_3d_nms_mask(
    values: np.ndarray,
    strike: np.ndarray,
    *,
    sigma: float = 1.0,
    strict: bool = True,
) -> np.ndarray:
    """Return a strike-binned local-maximum mask in the ``i2-i3`` plane.

    ``values`` and ``strike`` must be finite 3D arrays with matching
    ``(n3, n2, n1)`` shapes. ``strike`` is interpreted in degrees. Values are
    smoothed along ``i3`` and ``i2`` before comparison, but not along ``i1``.
    """

    values_array, strike_array = _validate_matching_finite_arrays3(
        (values, strike),
        ("values", "strike"),
    )
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    _validate_bool(strict, "strict")

    smoothed = _smooth_i2_i3(values_array, sigma_float)
    return _reference_like_3d_nms_mask_from_smoothed(smoothed, strike_array, strict=strict)


def reference_like_3d_thin_values(
    values: np.ndarray,
    strike: np.ndarray,
    *,
    sigma: float = 1.0,
    reinforce_vertical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-like thinned values and their retained-sample mask.

    The local-maximum mask is computed from values smoothed only along the
    ``i3`` and ``i2`` axes. Retained output samples receive those same smoothed
    values, matching the reference Java thinning value flow.
    """

    values_array, strike_array = _validate_matching_finite_arrays3(
        (values, strike),
        ("values", "strike"),
    )
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    _validate_bool(reinforce_vertical, "reinforce_vertical")

    smoothed = _smooth_i2_i3(values_array, sigma_float)
    keep = _reference_like_3d_nms_mask_from_smoothed(smoothed, strike_array, strict=True)
    thinned = np.zeros(values_array.shape, dtype=np.float32)
    thinned[keep] = smoothed[keep]

    if reinforce_vertical and keep.any():
        strike360 = np.mod(strike_array, np.float32(360.0))
        folded = np.where(
            strike360 > np.float32(180.0),
            np.float32(360.0) - strike360,
            strike360,
        )
        reinforced = keep & (folded > np.float32(60.0)) & (folded < np.float32(120.0))
        if reinforced.any():
            i3, i2, i1 = np.nonzero(reinforced)
            thinned[np.maximum(i3 - 1, 0), i2, i1] = smoothed[i3, i2, i1]

    return thinned, keep


def _reference_like_3d_nms_mask_from_smoothed(
    smoothed: np.ndarray,
    strike: np.ndarray,
    *,
    strict: bool,
) -> np.ndarray:
    strike180 = np.mod(strike, np.float32(180.0))
    keep = np.zeros(smoothed.shape, dtype=np.bool_)

    horizontal = (strike180 < np.float32(22.5)) | (strike180 >= np.float32(157.5))
    positive_diagonal = (strike180 >= np.float32(22.5)) & (strike180 < np.float32(67.5))
    vertical = (strike180 >= np.float32(67.5)) & (strike180 < np.float32(112.5))
    negative_diagonal = (strike180 >= np.float32(112.5)) & (strike180 < np.float32(157.5))

    _accumulate_direction_mask(keep, smoothed, horizontal, d3=0, d2=1, strict=strict)
    _accumulate_direction_mask(
        keep,
        smoothed,
        positive_diagonal,
        d3=1,
        d2=-1,
        strict=strict,
    )
    _accumulate_direction_mask(keep, smoothed, vertical, d3=1, d2=0, strict=strict)
    _accumulate_direction_mask(
        keep,
        smoothed,
        negative_diagonal,
        d3=1,
        d2=1,
        strict=strict,
    )
    return keep


def _accumulate_direction_mask(
    keep: np.ndarray,
    values: np.ndarray,
    direction: np.ndarray,
    *,
    d3: int,
    d2: int,
    strict: bool,
) -> None:
    start3 = abs(d3)
    stop3 = values.shape[0] - abs(d3)
    start2 = abs(d2)
    stop2 = values.shape[1] - abs(d2)
    if stop3 <= start3 or stop2 <= start2:
        return

    center_slice = (slice(start3, stop3), slice(start2, stop2), slice(None))
    plus_slice = (
        slice(start3 + d3, stop3 + d3),
        slice(start2 + d2, stop2 + d2),
        slice(None),
    )
    minus_slice = (
        slice(start3 - d3, stop3 - d3),
        slice(start2 - d2, stop2 - d2),
        slice(None),
    )

    center = values[center_slice]
    plus = values[plus_slice]
    minus = values[minus_slice]
    if strict:
        local_maximum = (center > plus) & (center > minus)
    else:
        local_maximum = (center >= plus) & (center >= minus)
    keep[center_slice] |= direction[center_slice] & local_maximum


def _smooth_i2_i3(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0 or values.size == 0:
        return values.astype(np.float32, copy=True)

    smoothed = ndimage.gaussian_filter(
        values,
        sigma=(sigma, sigma, 0.0),
        mode="nearest",
    )
    return smoothed.astype(np.float32, copy=False)


def _validate_matching_finite_arrays3(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    validated = tuple(_validate_finite_array3(array, name) for array, name in zip(arrays, names))
    shape = validated[0].shape
    first_name = names[0]
    for array, name in zip(validated[1:], names[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated


def _validate_finite_array3(array: np.ndarray, name: str) -> np.ndarray:
    array_nd = np.asarray(array)
    if array_nd.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (n3, n2, n1)")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array_float32 = array_nd.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc

    if not np.isfinite(array_float32).all():
        raise ValueError(f"{name} must contain only finite values")

    return array_float32


def _validate_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
