"""3D fault-orientation scanning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy import ndimage

from pyosv.geometry import fault_normal_vector_from_strike_and_dip
from pyosv.interp import sample3

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
        """Scan a 3D image for approximate fault likelihood, strike, and dip.

        The returned arrays have shape ``(n3, n2, n1)``. ``ft`` is normalized
        to ``[0, 1]``, ``pt`` contains strike in degrees, and ``tt`` contains
        dip in degrees.
        """

        phi_sampling = self.strike_sampling(phi_min, phi_max)
        theta_sampling = self.dip_sampling(theta_min, theta_max)
        image = self.validate_image(g, "g")
        if float(np.max(image) - np.min(image)) == 0.0:
            ft = np.zeros_like(image, dtype=np.float32)
            pt = np.full_like(image, phi_sampling[0], dtype=np.float32)
            tt = np.full_like(image, theta_sampling[0], dtype=np.float32)
            return ft, pt, tt

        return self._scan_orientation_bank(phi_sampling, theta_sampling, image)

    def thin(
        self,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep likelihood maxima along the local fault normal field.

        ``ft``, ``pt``, and ``tt`` must be finite 3D arrays with matching
        ``(n3, n2, n1)`` shapes. ``pt`` and ``tt`` are interpreted as strike
        and dip angles in degrees. The returned arrays are float32; retained
        orientation values match the input, and non-retained samples use zero
        as the orientation sentinel.
        """

        ft_array, pt_array, tt_array = _validate_matching_finite_images3(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        n3, n2, n1 = ft_array.shape
        i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
        w1, w2, w3 = _fault_normal_components_from_strike_and_dip(pt_array, tt_array)

        fp = sample3(ft_array, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
        fm = sample3(ft_array, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
        keep = (ft_array > np.float32(0.0)) & (fp < ft_array) & (fm < ft_array)

        thinned_ft = np.zeros((n3, n2, n1), dtype=np.float32)
        thinned_pt = np.zeros((n3, n2, n1), dtype=np.float32)
        thinned_tt = np.zeros((n3, n2, n1), dtype=np.float32)
        thinned_ft[keep] = ft_array[keep]
        thinned_pt[keep] = pt_array[keep]
        thinned_tt[keep] = tt_array[keep]
        return thinned_ft, thinned_pt, thinned_tt

    def _scan_orientation_bank(
        self,
        phi_sampling: np.ndarray,
        theta_sampling: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        derivative_sigma = max(0.75, 0.5 * min(self.sigma1, self.sigma2))
        derivatives = _gaussian_derivatives(image, derivative_sigma)
        d1, d2, d3, d11, d22, d33, d12, d13, d23 = derivatives

        best_score = np.zeros_like(image, dtype=np.float32)
        best_phi = np.full_like(image, phi_sampling[0], dtype=np.float32)
        best_theta = np.full_like(image, theta_sampling[0], dtype=np.float32)

        for phi in phi_sampling:
            for theta in theta_sampling:
                w1, w2, w3 = fault_normal_vector_from_strike_and_dip(
                    float(phi),
                    float(theta),
                )
                edge = np.abs(w1 * d1 + w2 * d2 + w3 * d3)
                ridge = np.abs(
                    w1 * w1 * d11
                    + w2 * w2 * d22
                    + w3 * w3 * d33
                    + 2.0 * w1 * w2 * d12
                    + 2.0 * w1 * w3 * d13
                    + 2.0 * w2 * w3 * d23
                )
                score = (edge + derivative_sigma * ridge).astype(
                    np.float32,
                    copy=False,
                )

                better = score > best_score
                best_score[better] = score[better]
                best_phi[better] = phi
                best_theta[better] = theta

        return (
            _normalize_likelihood(best_score),
            best_phi.astype(np.float32, copy=False),
            best_theta.astype(np.float32, copy=False),
        )


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


def _normalize_likelihood(score: np.ndarray) -> np.ndarray:
    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    high = float(np.percentile(score_float32, 99.5))
    if not math.isfinite(high) or high <= 0.0:
        return np.zeros_like(score_float32, dtype=np.float32)

    normalized = np.clip(score_float32 / np.float32(high), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)
