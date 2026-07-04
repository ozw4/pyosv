"""2D fault-orientation scanning helpers."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy import ndimage

from pyosv.interp import sample2

__all__ = ["FaultOrientScanner2"]


class FaultOrientScanner2:
    """Configuration holder for reference-first 2D fault-orientation scanning."""

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
        """Scan a 2D image with the reference-like backend.

        The returned arrays have shape ``(n2, n1)``. ``ft`` is normalized to
        ``[0, 1]`` and ``pt`` contains the selected voter-compatible fault
        orientation angle in degrees.
        """

        return self.scan_reference_like(theta_min, theta_max, g)

    def scan_fast(
        self,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan with the legacy derivative-bank backend.

        This practical backend is faster than :meth:`scan_reference_like` but
        does not follow the Java reference scanner's rotate/smooth/unrotate
        control flow. Returned orientations use the same
        ``FaultCell2``/voting-compatible angle convention as :meth:`scan`.
        """

        image = self.validate_image(g, "g")
        theta_sampling = self.theta_sampling(theta_min, theta_max)
        if float(np.max(image) - np.min(image)) == 0.0:
            ft = np.zeros_like(image, dtype=np.float32)
            pt = np.full_like(
                image,
                _feature_angle_to_fault_cell_angle(theta_sampling[0]),
                dtype=np.float32,
            )
            return ft, pt

        return self._scan_theta_fast(theta_sampling, image)

    def scan_reference_like(
        self,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
        *,
        interpolation_order: int = 1,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan using an approximate Java reference-like theta sweep.

        This backend follows the reference scanner's rotate, separable smooth,
        unrotate, and ``1 - smoothed**4`` control flow. It is a SciPy-based
        approximation of Mines JTK interpolation and recursive exponential
        filtering, and is the default :meth:`scan` backend. Returned
        orientations use the existing ``FaultCell2``/voting-compatible angle
        convention.
        """

        theta_sampling = self.theta_sampling(theta_min, theta_max)
        image = self.validate_image(g, "g")
        order = _validate_interpolation_order(interpolation_order)
        normalize_output = _validate_bool(normalize, "normalize")
        if float(np.max(image) - np.min(image)) == 0.0:
            ft = np.zeros_like(image, dtype=np.float32)
            pt = np.full_like(
                image,
                _feature_angle_to_fault_cell_angle(theta_sampling[0]),
                dtype=np.float32,
            )
            return ft, pt

        return self._scan_theta_reference_like(
            theta_sampling,
            image,
            interpolation_order=order,
            normalize=normalize_output,
        )

    def scan_dip(
        self,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan the two reference dip-angle branches and keep the stronger sample.

        The two feature-angle scans cover ``90 - theta_max`` to
        ``90 - theta_min`` and ``90 + theta_min`` to ``90 + theta_max``. Both
        scans return voter-compatible orientations; this method selects the
        ``(ft, pt)`` pair with larger likelihood at each sample.
        """

        ft_left, pt_left = self.scan(90.0 - theta_max, 90.0 - theta_min, g)
        ft_right, pt_right = self.scan(90.0 + theta_min, 90.0 + theta_max, g)
        use_right = ft_right > ft_left
        ft = np.where(use_right, ft_right, ft_left).astype(np.float32, copy=False)
        pt = np.where(use_right, pt_right, pt_left).astype(np.float32, copy=False)
        return ft, pt

    def thin(self, ft: np.ndarray, pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep likelihood maxima across local dip and zero orientation elsewhere.

        ``ft`` and ``pt`` must be finite 2D arrays with matching ``(n2, n1)``
        shapes. ``pt`` is interpreted with the same normal-angle convention as
        :class:`pyosv.cells.FaultCell2`. The returned likelihood and
        orientation arrays are float32; retained orientation values match the
        input, and non-retained samples use zero as the orientation sentinel.
        """

        ft_array, pt_array = _validate_matching_finite_images2(ft, pt, "ft", "pt")
        n2, n1 = ft_array.shape
        i2, i1 = np.indices((n2, n1), dtype=np.float32)

        theta = np.deg2rad(pt_array).astype(np.float32, copy=False)
        d1 = np.sin(theta).astype(np.float32, copy=False)
        d2 = np.cos(theta).astype(np.float32, copy=False)

        fp = sample2(ft_array, i1 + d1, i2 + d2)
        fm = sample2(ft_array, i1 - d1, i2 - d2)
        keep = (ft_array > np.float32(0.0)) & (fp < ft_array) & (fm < ft_array)

        thinned_ft = np.zeros((n2, n1), dtype=np.float32)
        thinned_pt = np.zeros((n2, n1), dtype=np.float32)
        thinned_ft[keep] = ft_array[keep]
        thinned_pt[keep] = pt_array[keep]
        return thinned_ft, thinned_pt

    def _scan_theta_fast(
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
            fault_cell_angle = _feature_angle_to_fault_cell_angle(theta)

            edge = np.abs(normal1 * d1 + normal2 * d2)
            ridge = np.abs(
                normal1 * normal1 * d11 + 2.0 * normal1 * normal2 * d12 + normal2 * normal2 * d22
            )
            score = edge + derivative_sigma * ridge
            score = score.astype(np.float32, copy=False)

            better = score > best_score
            best_score[better] = score[better]
            best_theta[better] = fault_cell_angle

        return _normalize_likelihood(best_score), best_theta.astype(np.float32, copy=False)

    def _scan_theta_reference_like(
        self,
        theta_sampling: np.ndarray,
        image: np.ndarray,
        *,
        interpolation_order: int,
        normalize: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        best_score = np.zeros_like(image, dtype=np.float32)
        best_theta = np.full_like(
            image,
            _feature_angle_to_fault_cell_angle(theta_sampling[0]),
            dtype=np.float32,
        )
        angle_map = _rotation_angle_map_2d(image.shape)

        for theta in theta_sampling:
            rotation_angle = -float(theta)
            rotated = _rotate_reference_like_2d(
                image,
                rotation_angle,
                angle_map=angle_map,
                interpolation_order=interpolation_order,
            )
            smoothed = _smooth_reference_like_2d(rotated, sigma1=self.sigma1)
            unrotated = _unrotate_reference_like_2d(
                smoothed,
                image.shape,
                rotation_angle,
                angle_map=angle_map,
                interpolation_order=interpolation_order,
            )
            score = np.float32(1.0) - unrotated.astype(np.float32, copy=False) ** np.float32(
                4.0,
            )
            score = score.astype(np.float32, copy=False)
            better = score > best_score
            best_score[better] = score[better]
            best_theta[better] = _feature_angle_to_fault_cell_angle(theta)

        if normalize:
            ft = _normalize_reference_like_likelihood(best_score)
        else:
            ft = np.maximum(best_score, np.float32(0.0)).astype(np.float32, copy=False)
        return ft, best_theta.astype(np.float32, copy=False)


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


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")

    return value


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


def _validate_matching_finite_images2(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = _validate_finite_image2(first, first_name)
    second_array = _validate_finite_image2(second, second_name)
    if first_array.shape != second_array.shape:
        raise ValueError(f"{first_name} and {second_name} shapes must match")

    return first_array, second_array


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


def _rotation_angle_map_2d(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reference-style radius, sine, and cosine maps for 2D rotation."""

    n2, n1 = shape
    h1 = math.ceil(n1 / 2.0)
    h2 = math.ceil(n2 / 2.0)
    hr = round(math.sqrt(h1 * h1 + h2 * h2))
    k2, k1 = np.mgrid[-hr : hr + 1, -hr : hr + 1].astype(np.float32)
    radius = np.sqrt(k1 * k1 + k2 * k2, dtype=np.float32)
    sine = np.zeros_like(radius, dtype=np.float32)
    cosine = np.zeros_like(radius, dtype=np.float32)
    nonzero = radius != np.float32(0.0)
    sine[nonzero] = k2[nonzero] / radius[nonzero]
    cosine[nonzero] = k1[nonzero] / radius[nonzero]
    return radius, sine, cosine


def _rotate_reference_like_2d(
    image: np.ndarray,
    theta_degrees: float,
    *,
    angle_map: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
) -> np.ndarray:
    """Rotate to a reference-like expanded canvas using SciPy interpolation."""

    n2, n1 = image.shape
    h2 = math.floor(n2 / 2.0)
    h1 = math.floor(n1 / 2.0)
    theta = math.radians(theta_degrees)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    r2 = max(
        abs(round(h2 * cos_theta + h1 * sin_theta)),
        abs(round(h2 * cos_theta - h1 * sin_theta)),
    )
    r1 = max(
        abs(round(h1 * cos_theta - h2 * sin_theta)),
        abs(round(h1 * cos_theta + h2 * sin_theta)),
    )
    radius, sine, cosine = angle_map
    center = (radius.shape[0] - 1) // 2
    rows = slice(center - r2, center + r2 + 1)
    cols = slice(center - r1, center + r1 + 1)
    local_radius = radius[rows, cols]
    local_sine = sine[rows, cols]
    local_cosine = cosine[rows, cols]

    source_x1 = local_radius * (local_cosine * cos_theta + local_sine * sin_theta) + h1
    source_x2 = local_radius * (local_sine * cos_theta - local_cosine * sin_theta) + h2
    rotated = sample2(
        image,
        source_x1.astype(np.float32, copy=False),
        source_x2.astype(np.float32, copy=False),
        order=interpolation_order,
        mode="nearest",
    )
    return np.asarray(rotated, dtype=np.float32)


def _unrotate_reference_like_2d(
    rotated: np.ndarray,
    shape: tuple[int, int],
    theta_degrees: float,
    *,
    angle_map: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
) -> np.ndarray:
    """Unrotate a reference-like canvas back to the original ``(n2, n1)`` shape."""

    n2, n1 = shape
    h2 = math.floor(n2 / 2.0)
    h1 = math.floor(n1 / 2.0)
    r2 = (rotated.shape[0] - 1) // 2
    r1 = (rotated.shape[1] - 1) // 2
    theta = math.radians(theta_degrees)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    radius, sine, cosine = angle_map
    center = (radius.shape[0] - 1) // 2
    rows = slice(center - h2, center - h2 + n2)
    cols = slice(center - h1, center - h1 + n1)
    local_radius = radius[rows, cols]
    local_sine = sine[rows, cols]
    local_cosine = cosine[rows, cols]

    source_x1 = local_radius * (local_cosine * cos_theta - local_sine * sin_theta) + r1
    source_x2 = local_radius * (local_sine * cos_theta + local_cosine * sin_theta) + r2
    unrotated = sample2(
        rotated,
        source_x1.astype(np.float32, copy=False),
        source_x2.astype(np.float32, copy=False),
        order=interpolation_order,
        mode="nearest",
    )
    return np.asarray(unrotated, dtype=np.float32)


def _smooth_reference_like_2d(rotated: np.ndarray, *, sigma1: float) -> np.ndarray:
    """Approximate ``ref2.apply2`` then ``ref1.apply1`` with separable Gaussians."""

    smoothed2 = ndimage.gaussian_filter1d(
        rotated,
        sigma=1.0,
        axis=0,
        mode="nearest",
    )
    smoothed1 = ndimage.gaussian_filter1d(
        smoothed2,
        sigma=sigma1,
        axis=1,
        mode="nearest",
    )
    return smoothed1.astype(np.float32, copy=False)


def _feature_angle_to_fault_cell_angle(theta_degrees: float) -> np.float32:
    return np.float32((180.0 - float(theta_degrees)) % 180.0)


def _normalize_reference_like_likelihood(score: np.ndarray) -> np.ndarray:
    return np.clip(score.astype(np.float32, copy=False), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _normalize_likelihood(score: np.ndarray) -> np.ndarray:
    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    high = float(np.percentile(score_float32, 99.5))
    if not math.isfinite(high) or high <= 0.0:
        return np.zeros_like(score_float32, dtype=np.float32)

    normalized = np.clip(score_float32 / np.float32(high), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)
