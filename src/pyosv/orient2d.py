"""2D fault-orientation scanning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy import ndimage

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

    def scan(
        self,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan a 2D image for approximate fault likelihood and dip.

        The returned arrays have shape ``(n2, n1)``. ``ft`` is normalized to
        ``[0, 1]`` and ``pt`` contains the selected dip angle in degrees.
        """

        image = self.validate_image(g, "g")
        theta_sampling = self.theta_sampling(theta_min, theta_max)
        if float(np.max(image) - np.min(image)) == 0.0:
            ft = np.zeros_like(image, dtype=np.float32)
            pt = np.full_like(image, theta_sampling[0], dtype=np.float32)
            return ft, pt

        return self._scan_theta(theta_sampling, image)

    def scan_dip(
        self,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias for :meth:`scan` using Python naming for the reference API."""

        return self.scan(theta_min, theta_max, g)

    def _scan_theta(
        self,
        theta_sampling: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        derivative_sigma = max(0.75, 0.5 * self.sigma1)
        derivatives = _gaussian_derivatives(image, derivative_sigma)
        d1, d2, d11, d22, d12 = derivatives

        best_score = np.zeros_like(image, dtype=np.float32)
        best_theta = np.full_like(image, theta_sampling[0], dtype=np.float32)

        for theta in theta_sampling:
            theta_radians = math.radians(float(theta))
            normal1 = -math.sin(theta_radians)
            normal2 = math.cos(theta_radians)

            edge = np.abs(normal1 * d1 + normal2 * d2)
            ridge = np.abs(
                normal1 * normal1 * d11 + 2.0 * normal1 * normal2 * d12 + normal2 * normal2 * d22
            )
            score = edge + derivative_sigma * ridge
            score = score.astype(np.float32, copy=False)

            better = score > best_score
            best_score[better] = score[better]
            best_theta[better] = theta

        return _normalize_likelihood(best_score), best_theta.astype(np.float32, copy=False)


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


def _gaussian_derivatives(
    image: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d1 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 1), mode="nearest")
    d2 = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 0), mode="nearest")
    d11 = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2), mode="nearest")
    d22 = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0), mode="nearest")
    d12 = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1), mode="nearest")
    return (
        d1.astype(np.float32, copy=False),
        d2.astype(np.float32, copy=False),
        d11.astype(np.float32, copy=False),
        d22.astype(np.float32, copy=False),
        d12.astype(np.float32, copy=False),
    )


def _normalize_likelihood(score: np.ndarray) -> np.ndarray:
    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    high = float(np.percentile(score_float32, 99.5))
    if not math.isfinite(high) or high <= 0.0:
        return np.zeros_like(score_float32, dtype=np.float32)

    normalized = np.clip(score_float32 / np.float32(high), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)
