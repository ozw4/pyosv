"""3D fault-orientation scanning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = ["FaultOrientScanner3"]


class FaultOrientScanner3:
    """Configuration holder for approximate 3D fault-orientation scanning.

    Parameters
    ----------
    sigma1, sigma2:
        Positive smoothing/scanning controls in samples. ``sigma1`` controls
        dip sampling density and ``sigma2`` controls strike sampling density.
    """

    def __init__(self, sigma1: float, sigma2: float) -> None:
        self.sigma1 = _validate_positive_float(sigma1, "sigma1")
        self.sigma2 = _validate_positive_float(sigma2, "sigma2")

    def strike_sampling(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return strike-angle samples in degrees as a finite float32 array.

        The first and last samples match the requested endpoints after float32
        conversion. Endpoint comparisons should allow normal float32 roundoff.
        """

        return _angle_sampling(
            phi_min,
            phi_max,
            sigma=self.sigma2,
            min_name="phi_min",
            max_name="phi_max",
            sigma_name="sigma2",
        )

    def dip_sampling(self, theta_min: float, theta_max: float) -> np.ndarray:
        """Return dip-angle samples in degrees as a finite float32 array.

        The first and last samples match the requested endpoints after float32
        conversion. Endpoint comparisons should allow normal float32 roundoff.
        """

        return _angle_sampling(
            theta_min,
            theta_max,
            sigma=self.sigma1,
            min_name="theta_min",
            max_name="theta_max",
            sigma_name="sigma1",
        )

    def validate_image(self, image: np.ndarray, name: str = "image") -> np.ndarray:
        """Return a finite global 3D image volume as a float32 array.

        Global 3D volumes use OSV's Python shape convention ``(n3, n2, n1)``.
        """

        return _validate_finite_image3(image, name)

    def scan(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate scanner inputs for a future 3D orientation scan.

        Full 3D scan response computation is intentionally outside this
        bootstrap implementation.
        """

        self.strike_sampling(phi_min, phi_max)
        self.dip_sampling(theta_min, theta_max)
        self.validate_image(g, "g")
        raise NotImplementedError("3D scan response computation is not implemented yet")


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
