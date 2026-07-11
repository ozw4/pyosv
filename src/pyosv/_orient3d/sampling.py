"""Angle sampling and scanner input validation."""

from __future__ import annotations

import math
import numbers

import numpy as np


def _angle_sampling(
    angle_min: float,
    angle_max: float,
    *,
    sigma: float,
    min_name: str,
    max_name: str,
    sigma_name: str,
) -> np.ndarray:
    amin = _validate_angle(angle_min, min_name)
    amax = _validate_angle(angle_max, max_name)
    if amax < amin:
        raise ValueError(f"{max_name} must be greater than or equal to {min_name}")

    amin32 = np.float32(amin)
    amax32 = np.float32(amax)
    if not np.isfinite(amin32) or not np.isfinite(amax32):
        raise ValueError(f"{min_name} and {max_name} must be finite float32 values")

    if amin == amax:
        return np.array([amin32], dtype=np.float32)

    target_step = math.degrees(0.5 / sigma)
    if not math.isfinite(target_step) or target_step <= 0.0:
        raise ValueError(f"{sigma_name} produces an invalid angle sampling interval")

    count_float = 1.0 + (amax - amin) / target_step
    if not math.isfinite(count_float) or count_float > 1_000_000:
        raise ValueError(f"{sigma_name} produces too many angle samples")

    count = max(2, int(count_float))
    return np.linspace(amin, amax, count, dtype=np.float32)


def _reference_like_strike_sampling(phi_min: float, phi_max: float) -> np.ndarray:
    pmin = _validate_angle(phi_min, "phi_min")
    pmax = _validate_angle(phi_max, "phi_max")
    if pmax < pmin:
        raise ValueError("phi_max must be greater than or equal to phi_min")

    pmin32 = np.float32(pmin)
    pmax32 = np.float32(pmax)
    if not np.isfinite(pmin32) or not np.isfinite(pmax32):
        raise ValueError("phi_min and phi_max must be finite float32 values")
    if pmin == pmax:
        return np.array([pmin32], dtype=np.float32)

    java_grid = np.arange(18, dtype=np.float32) * np.float32(20.0)
    samples = java_grid[(java_grid >= pmin32) & (java_grid <= pmax32)]
    if samples.size == 0:
        return np.array([pmin32], dtype=np.float32)
    return samples.astype(np.float32, copy=False)


def _reference_like_dip_sampling(theta_min: float, theta_max: float) -> np.ndarray:
    tmin = _validate_angle(theta_min, "theta_min")
    tmax = _validate_angle(theta_max, "theta_max")
    if tmax < tmin:
        raise ValueError("theta_max must be greater than or equal to theta_min")

    tmin32 = np.float32(tmin)
    tmax32 = np.float32(tmax)
    if not np.isfinite(tmin32) or not np.isfinite(tmax32):
        raise ValueError("theta_min and theta_max must be finite float32 values")
    if tmin == tmax:
        return np.array([tmin32], dtype=np.float32)

    count = max(2, int(round((tmax - tmin) / 5.0)) + 1)
    return np.linspace(tmin, tmax, count, dtype=np.float32)


def _refined_reference_like_sampling(
    base_samples: np.ndarray,
    *,
    refinement_factor: int,
) -> np.ndarray:
    factor = _validate_refinement_factor(refinement_factor)
    base = np.asarray(base_samples, dtype=np.float32)
    if factor == 1 or base.size <= 1:
        return base.astype(np.float32, copy=True)

    refined = [base]
    fractions = np.arange(1, factor, dtype=np.float32) / np.float32(factor)
    lower = base[:-1]
    upper = base[1:]
    for fraction in fractions:
        refined.append(lower + fraction * (upper - lower))

    samples = np.concatenate(refined).astype(np.float32, copy=False)
    return np.unique(samples).astype(np.float32, copy=False)


def _validate_refinement_factor(refinement_factor: int) -> int:
    if isinstance(refinement_factor, bool) or not isinstance(refinement_factor, numbers.Integral):
        raise ValueError("refinement_factor must be an integer from 1 to 4")

    factor = int(refinement_factor)
    if factor < 1 or factor > 4:
        raise ValueError("refinement_factor must be an integer from 1 to 4")
    return factor


def _validate_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive number")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")

    return value_float


def _validate_angle(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")

    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be a finite number")

    return value_float


def _validate_interpolation_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, numbers.Integral):
        raise ValueError("interpolation_order must be an integer from 0 to 5")

    order_int = int(order)
    if order_int < 0 or order_int > 5:
        raise ValueError("interpolation_order must be an integer from 0 to 5")

    return order_int


def _validate_optional_nonnegative_float(
    value: float | None,
    name: str,
    *,
    default: float,
) -> float:
    if value is None:
        return float(default)

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number or None")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number or None")

    return value_float


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")

    return value


def _validate_reference_like_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise ValueError("backend must be 'rotate_shear' or 'directional'")

    if backend not in {"rotate_shear", "directional"}:
        raise ValueError("backend must be 'rotate_shear' or 'directional'")

    return backend


def _validate_finite_image3(image: np.ndarray, name: str) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (n3, n2, n1)")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            image_float32 = image_array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc

    if not np.isfinite(image_float32).all():
        raise ValueError(f"{name} must contain only finite values")

    return image_float32


def _validate_matching_finite_images3(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    validated = tuple(_validate_finite_image3(array, name) for array, name in zip(arrays, names))
    shape = validated[0].shape
    first_name = names[0]
    for array, name in zip(validated[1:], names[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated
