"""2D fault-orientation scanning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = ["FaultOrientScanner2"]


class FaultOrientScanner2:
    """Configuration holder for approximate 2D fault-orientation scanning."""

    def __init__(self, sigma1: float) -> None:
        self.sigma1 = _validate_positive_float(sigma1, "sigma1")

    def theta_sampling(self, theta_min: float, theta_max: float) -> np.ndarray:
        """Return dip-angle samples in degrees.

        The first and last samples match the requested endpoints after float32
        conversion. Endpoint comparisons should allow normal float32 roundoff.
        """

        amin = _validate_angle(theta_min, "theta_min")
        amax = _validate_angle(theta_max, "theta_max")
        if amax < amin:
            raise ValueError("theta_max must be greater than or equal to theta_min")

        amin32 = np.float32(amin)
        amax32 = np.float32(amax)
        if not np.isfinite(amin32) or not np.isfinite(amax32):
            raise ValueError(
                "theta_min and theta_max must be representable as finite float32 values"
            )

        if amin == amax:
            return np.array([amin32], dtype=np.float32)

        target_step = math.degrees(0.5 / self.sigma1)
        if not math.isfinite(target_step) or target_step <= 0.0:
            raise ValueError("sigma1 produces an invalid theta sampling interval")

        count = max(2, 1 + int((amax - amin) / target_step))
        return np.linspace(amin, amax, count, dtype=np.float32)

    def validate_image(self, image: np.ndarray, name: str = "image") -> np.ndarray:
        """Return a finite 2D image as a float32 array."""

        return _validate_finite_image2(image, name)


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


def _validate_finite_image2(image: np.ndarray, name: str) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            image_float32 = image_array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc

    if not np.isfinite(image_float32).all():
        raise ValueError(f"{name} must contain only finite values")

    return image_float32
