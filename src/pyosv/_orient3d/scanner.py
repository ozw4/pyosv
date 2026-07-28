"""FaultOrientScanner3 implementation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from pyosv._orient3d.geometry import (
    _coordinate_grids3,
    _fault_normal_components_from_strike_and_dip,
    _gaussian_derivatives,
)
from pyosv._orient3d.normalization import (
    _normalize_likelihood,
    _normalize_reference_like_likelihood,
)
from pyosv._orient3d.rotate_shear import (
    _dip_shear_from_theta,
    _rotate3_axis1,
    _shear_rotated_volume,
    _smooth_rotated_strike_axis,
    _smooth_sheared_dip_axis,
    _unrotate3_axis1,
    _unshear_rotated_volume,
)
from pyosv._orient3d.sampling import (
    _angle_sampling,
    _reference_like_dip_sampling,
    _reference_like_strike_sampling,
    _refined_reference_like_sampling,
    _validate_bool,
    _validate_finite_image3,
    _validate_interpolation_backend,
    _validate_interpolation_order,
    _validate_matching_finite_images3,
    _validate_optional_nonnegative_float,
    _validate_positive_float,
    _validate_reference_like_backend,
)
from pyosv._orient3d.scoring import (
    _orientation_basis_from_strike_and_dip,
    _orientation_confidence_from_scores,
    _reference_like_orientation_score,
    _reference_like_planarity_to_likelihood,
    _update_best_second_orientation,
)
from pyosv.geometry import fault_normal_vector_from_strike_and_dip
from pyosv.interp import sample3
from pyosv.thinning3d import reference_like_3d_thin_values, remove_reference_edge_effects_3d

__all__ = ["FaultOrientScanner3"]


def _orientation_code_dtype(orientation_count: int) -> np.dtype:
    """Return the smallest supported unsigned dtype for orientation codes."""

    if orientation_count <= 0:
        raise ValueError("orientation_count must be positive")
    if orientation_count <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if orientation_count <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if orientation_count <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError("orientation_count exceeds uint32 code capacity")


def _new_orientation_codes(
    shape: tuple[int, ...],
    phi_sampling: np.ndarray,
    theta_sampling: np.ndarray,
) -> np.ndarray:
    orientation_count = len(phi_sampling) * len(theta_sampling)
    return np.zeros(shape, dtype=_orientation_code_dtype(orientation_count))


def _encode_orientation_code(phi_index: int, theta_index: int, theta_count: int) -> int:
    return phi_index * theta_count + theta_index


def _decode_orientation_codes(
    codes: np.ndarray,
    phi_sampling: np.ndarray,
    theta_sampling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode sweep-order codes into independent contiguous angle volumes."""

    theta_count = len(theta_sampling)
    phi_indices = codes // theta_count
    theta_indices = codes % theta_count
    phis = np.asarray(phi_sampling, dtype=np.float32)[phi_indices]
    thetas = np.asarray(theta_sampling, dtype=np.float32)[theta_indices]
    return (
        np.ascontiguousarray(phis, dtype=np.float32),
        np.ascontiguousarray(thetas, dtype=np.float32),
    )


def _update_best_orientation(
    score: np.ndarray,
    orientation_code: int,
    best_score: np.ndarray,
    second_score: np.ndarray | None,
    best_code: np.ndarray,
) -> None:
    if second_score is not None:
        code = best_code.dtype.type(orientation_code)
        _update_best_second_orientation(
            score,
            code,
            code,
            best_score,
            second_score,
            best_code,
            best_code,
        )
        return

    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    better = score_float32 > best_score
    best_score[better] = score_float32[better]
    best_code[better] = orientation_code


class FaultOrientScanner3:
    """Configuration holder for reference-first 3D fault-orientation scanning.

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

    def reference_like_strike_sampling(
        self,
        phi_min: float,
        phi_max: float,
    ) -> np.ndarray:
        """Return Java-inspired strike samples for ``scan_reference_like``.

        Reference-like mode uses the Java scanner's fixed 18-sample strike grid
        at 20 degree spacing from 0 degrees, clipped to the requested range.
        Narrow ranges with no fixed-grid sample return the lower endpoint so
        explicit valid ranges remain callable.
        """

        return _reference_like_strike_sampling(phi_min, phi_max)

    def reference_like_dip_sampling(
        self,
        theta_min: float,
        theta_max: float,
    ) -> np.ndarray:
        """Return Java-inspired dip samples for ``scan_reference_like``.

        Reference-like mode uses approximately 5 degree dip spacing while
        preserving the requested endpoints.
        """

        return _reference_like_dip_sampling(theta_min, theta_max)

    def refined_reference_like_strike_sampling(
        self,
        phi_min: float,
        phi_max: float,
        *,
        refinement_factor: int = 2,
    ) -> np.ndarray:
        """Return reference-like strike samples with optional interval refinement."""

        return _refined_reference_like_sampling(
            self.reference_like_strike_sampling(phi_min, phi_max),
            refinement_factor=refinement_factor,
        )

    def refined_reference_like_dip_sampling(
        self,
        theta_min: float,
        theta_max: float,
        *,
        refinement_factor: int = 2,
    ) -> np.ndarray:
        """Return reference-like dip samples with optional interval refinement."""

        return _refined_reference_like_sampling(
            self.reference_like_dip_sampling(theta_min, theta_max),
            refinement_factor=refinement_factor,
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
        *,
        backend: str = "rotate_shear",
        interpolation_order: int = 1,
        interpolation_backend: str = "scipy",
        smoothing_sigma: float | None = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scan a 3D image with the reference-like backend.

        The returned arrays have shape ``(n3, n2, n1)``. ``ft`` is normalized
        to ``[0, 1]``, ``pt`` contains strike in degrees, and ``tt`` contains
        dip in degrees.
        """

        return self.scan_reference_like(
            phi_min,
            phi_max,
            theta_min,
            theta_max,
            g,
            backend=backend,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
            smoothing_sigma=smoothing_sigma,
            normalize=normalize,
        )

    def scan_fast(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scan with the legacy derivative-bank backend.

        This practical backend is faster than :meth:`scan_reference_like` but
        does not follow the Java reference scanner's orientation sweep
        semantics. Returned strike and dip angles use the same convention as
        :meth:`scan`.
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

    def scan_reference_like(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
        *,
        backend: str = "rotate_shear",
        interpolation_order: int = 1,
        interpolation_backend: str = "scipy",
        smoothing_sigma: float | None = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scan using an approximate Java reference-like orientation sweep.

        ``backend="rotate_shear"`` is the default reference-first path. It uses
        SciPy interpolation to approximate the Java scanner's rotate, strike
        smooth, dip shear, dip smooth, unshear, likelihood, and unrotate stages.
        ``backend="directional"`` keeps the previous practical approximation
        that directly smooths along candidate fault-parallel directions.

        Neither backend is a bit-exact Mines JTK port. Both convert smoothed
        planarity values to likelihood with ``1 - smoothed**4`` and keep the
        best orientation.
        """

        ft, pt, tt = self._scan_reference_like_with_confidence(
            phi_min,
            phi_max,
            theta_min,
            theta_max,
            g,
            backend=backend,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
            smoothing_sigma=smoothing_sigma,
            normalize=normalize,
            include_confidence=False,
        )
        return ft, pt, tt

    def scan_with_confidence(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
        *,
        backend: str = "rotate_shear",
        interpolation_order: int = 1,
        interpolation_backend: str = "scipy",
        smoothing_sigma: float | None = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scan and return an orientation confidence diagnostic volume.

        The first three arrays have the same semantics as
        :meth:`scan_reference_like`. ``confidence`` is a normalized
        ``float32`` map in ``[0, 1]`` based on the response gap between the
        best and second-best sampled orientations.
        """

        return self._scan_reference_like_with_confidence(
            phi_min,
            phi_max,
            theta_min,
            theta_max,
            g,
            backend=backend,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
            smoothing_sigma=smoothing_sigma,
            normalize=normalize,
            include_confidence=True,
        )

    def scan_quality(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
        *,
        backend: str = "rotate_shear",
        refinement_factor: int = 2,
        interpolation_order: int = 1,
        interpolation_backend: str = "scipy",
        smoothing_sigma: float | None = None,
        normalize: bool = True,
        return_confidence: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Scan with opt-in refined reference-like sampling for quality studies."""

        include_confidence = _validate_bool(return_confidence, "return_confidence")
        phi_sampling = self.refined_reference_like_strike_sampling(
            phi_min,
            phi_max,
            refinement_factor=refinement_factor,
        )
        theta_sampling = self.refined_reference_like_dip_sampling(
            theta_min,
            theta_max,
            refinement_factor=refinement_factor,
        )
        return self._scan_reference_like_samples_with_confidence(
            phi_sampling,
            theta_sampling,
            g,
            backend=backend,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
            smoothing_sigma=smoothing_sigma,
            normalize=normalize,
            include_confidence=include_confidence,
        )

    def _scan_reference_like_with_confidence(
        self,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
        g: np.ndarray,
        *,
        backend: str,
        interpolation_order: int,
        interpolation_backend: str,
        smoothing_sigma: float | None,
        normalize: bool,
        include_confidence: bool,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        phi_sampling = self.reference_like_strike_sampling(phi_min, phi_max)
        theta_sampling = self.reference_like_dip_sampling(theta_min, theta_max)
        return self._scan_reference_like_samples_with_confidence(
            phi_sampling,
            theta_sampling,
            g,
            backend=backend,
            interpolation_order=interpolation_order,
            interpolation_backend=interpolation_backend,
            smoothing_sigma=smoothing_sigma,
            normalize=normalize,
            include_confidence=include_confidence,
        )

    def _scan_reference_like_samples_with_confidence(
        self,
        phi_sampling: np.ndarray,
        theta_sampling: np.ndarray,
        g: np.ndarray,
        *,
        backend: str,
        interpolation_order: int,
        interpolation_backend: str,
        smoothing_sigma: float | None,
        normalize: bool,
        include_confidence: bool,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        image = self.validate_image(g, "g")
        backend_name = _validate_reference_like_backend(backend)
        order = _validate_interpolation_order(interpolation_order)
        interpolation_backend_name = _validate_interpolation_backend(interpolation_backend)
        if interpolation_backend_name == "structured_linear" and order != 1:
            raise ValueError(
                "interpolation_backend='structured_linear' requires interpolation_order=1"
            )
        if interpolation_backend_name == "structured_linear" and backend_name != "rotate_shear":
            raise ValueError(
                "interpolation_backend='structured_linear' requires backend='rotate_shear'"
            )
        sigma = _validate_optional_nonnegative_float(
            smoothing_sigma,
            "smoothing_sigma",
            default=max(1.0, 0.5 * (self.sigma1 + self.sigma2)),
        )
        normalize_output = _validate_bool(normalize, "normalize")
        if float(np.max(image) - np.min(image)) == 0.0:
            ft = np.zeros_like(image, dtype=np.float32)
            pt = np.full_like(image, phi_sampling[0], dtype=np.float32)
            tt = np.full_like(image, theta_sampling[0], dtype=np.float32)
            if include_confidence:
                confidence = np.zeros_like(image, dtype=np.float32)
                return ft, pt, tt, confidence
            return ft, pt, tt

        if backend_name == "rotate_shear":
            return self._scan_rotate_shear_reference_like(
                phi_sampling,
                theta_sampling,
                image,
                interpolation_order=order,
                interpolation_backend=interpolation_backend_name,
                smoothing_sigma=sigma,
                normalize=normalize_output,
                include_confidence=include_confidence,
            )

        return self._scan_reference_like_orientation_sweep(
            phi_sampling,
            theta_sampling,
            image,
            interpolation_order=order,
            smoothing_sigma=sigma,
            normalize=normalize_output,
            include_confidence=include_confidence,
        )

    def thin(
        self,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
        *,
        mode: str = "reference",
        reference_sigma: float = 1.0,
        remove_edge_effects: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep likelihood maxima using the selected 3D thinning mode.

        ``ft``, ``pt``, and ``tt`` must be finite 3D arrays with matching
        ``(n3, n2, n1)`` shapes. ``pt`` and ``tt`` are interpreted as strike
        and dip angles in degrees. The returned arrays are float32; retained
        orientation values match the input, and non-retained samples use zero
        as the orientation sentinel.

        ``mode="reference"`` applies the default reference-like strike-binned
        local-maximum suppression in the ``i2-i3`` plane, using
        ``reference_sigma`` for smoothing inside the shared helper. It also
        applies scanner-style edge-effect removal by default; set
        ``remove_edge_effects=False`` only for diagnostics.
        ``mode="normal"`` preserves the legacy local fault-normal thinning as
        an explicit opt-in mode.
        """

        ft_array, pt_array, tt_array = _validate_matching_finite_images3(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        remove_edges = _validate_bool(remove_edge_effects, "remove_edge_effects")
        n3, n2, n1 = ft_array.shape
        if mode == "normal":
            i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
            w1, w2, w3 = _fault_normal_components_from_strike_and_dip(pt_array, tt_array)

            fp = sample3(ft_array, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
            fm = sample3(ft_array, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
            keep = (ft_array > np.float32(0.0)) & (fp < ft_array) & (fm < ft_array)
            thinned_ft = np.zeros((n3, n2, n1), dtype=np.float32)
            thinned_ft[keep] = ft_array[keep]
        elif mode == "reference":
            thinned_ft, keep = reference_like_3d_thin_values(
                ft_array,
                pt_array,
                sigma=reference_sigma,
                reinforce_vertical=False,
            )
            if remove_edges:
                thinned_ft, thinned_pt, thinned_tt, keep = remove_reference_edge_effects_3d(
                    thinned_ft,
                    pt_array,
                    tt_array,
                )
                return thinned_ft, thinned_pt, thinned_tt
        else:
            raise ValueError("mode must be 'normal' or 'reference'")

        thinned_pt = np.zeros((n3, n2, n1), dtype=np.float32)
        thinned_tt = np.zeros((n3, n2, n1), dtype=np.float32)
        thinned_pt[keep] = pt_array[keep]
        thinned_tt[keep] = tt_array[keep]
        return thinned_ft, thinned_pt, thinned_tt

    def _scan_orientation_bank(
        self,
        phi_sampling: np.ndarray,
        theta_sampling: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the legacy derivative-bank scan used by :meth:`scan_fast`."""

        derivative_sigma = max(0.75, 0.5 * min(self.sigma1, self.sigma2))
        derivatives = _gaussian_derivatives(image, derivative_sigma)
        d1, d2, d3, d11, d22, d33, d12, d13, d23 = derivatives

        best_score = np.zeros_like(image, dtype=np.float32)
        best_code = _new_orientation_codes(image.shape, phi_sampling, theta_sampling)

        for iphi, phi in enumerate(phi_sampling):
            for itheta, theta in enumerate(theta_sampling):
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

                orientation_code = _encode_orientation_code(
                    iphi,
                    itheta,
                    len(theta_sampling),
                )
                _update_best_orientation(
                    score,
                    orientation_code,
                    best_score,
                    None,
                    best_code,
                )

        best_phi, best_theta = _decode_orientation_codes(
            best_code,
            phi_sampling,
            theta_sampling,
        )

        return (
            _normalize_likelihood(best_score),
            best_phi,
            best_theta,
        )

    def _scan_rotate_shear_reference_like(
        self,
        phi_sampling: np.ndarray,
        theta_sampling: np.ndarray,
        image: np.ndarray,
        *,
        interpolation_order: int,
        interpolation_backend: str,
        smoothing_sigma: float,
        normalize: bool,
        include_confidence: bool,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        best_score = np.zeros_like(image, dtype=np.float32)
        second_score = np.zeros_like(image, dtype=np.float32) if include_confidence else None
        best_code = _new_orientation_codes(image.shape, phi_sampling, theta_sampling)

        for iphi, phi in enumerate(phi_sampling):
            rotated = _rotate3_axis1(
                image,
                float(phi),
                interpolation_order=interpolation_order,
                interpolation_backend=interpolation_backend,
            )
            strike_smoothed = _smooth_rotated_strike_axis(
                rotated,
                sigma=smoothing_sigma,
            )
            rotated_scores = self._scan_theta_shear_reference_like(
                theta_sampling,
                strike_smoothed,
                interpolation_order=interpolation_order,
                interpolation_backend=interpolation_backend,
                smoothing_sigma=smoothing_sigma,
            )
            for itheta, (theta, rotated_score) in enumerate(
                zip(theta_sampling, rotated_scores),
            ):
                score = _unrotate3_axis1(
                    rotated_score,
                    image.shape,
                    float(phi),
                    interpolation_order=interpolation_order,
                    interpolation_backend=interpolation_backend,
                )
                score = np.clip(score, np.float32(0.0), np.float32(1.0)).astype(
                    np.float32,
                    copy=False,
                )
                orientation_code = _encode_orientation_code(
                    iphi,
                    itheta,
                    len(theta_sampling),
                )
                _update_best_orientation(
                    score,
                    orientation_code,
                    best_score,
                    second_score,
                    best_code,
                )

        if normalize:
            ft = _normalize_reference_like_likelihood(best_score)
        else:
            ft = np.maximum(best_score, np.float32(0.0)).astype(np.float32, copy=False)

        best_phi, best_theta = _decode_orientation_codes(
            best_code,
            phi_sampling,
            theta_sampling,
        )
        theta_min = np.float32(theta_sampling[0])
        theta_max = np.float32(theta_sampling[-1])
        result = (
            ft,
            best_phi,
            np.clip(best_theta, theta_min, theta_max).astype(np.float32, copy=False),
        )
        if second_score is None:
            return result
        confidence = _orientation_confidence_from_scores(best_score, second_score)
        return (*result, confidence)

    def _scan_theta_shear_reference_like(
        self,
        theta_sampling: np.ndarray,
        rotated: np.ndarray,
        *,
        interpolation_order: int,
        interpolation_backend: str = "scipy",
        smoothing_sigma: float,
    ) -> Iterator[np.ndarray]:
        for theta in theta_sampling:
            shear = _dip_shear_from_theta(float(theta))
            sheared = _shear_rotated_volume(
                rotated,
                shear,
                interpolation_order=interpolation_order,
                interpolation_backend=interpolation_backend,
            )
            dip_smoothed = _smooth_sheared_dip_axis(
                sheared,
                sigma=smoothing_sigma,
                theta_degrees=float(theta),
            )
            unsheared = _unshear_rotated_volume(
                dip_smoothed,
                shear,
                interpolation_order=interpolation_order,
                interpolation_backend=interpolation_backend,
            )
            yield _reference_like_planarity_to_likelihood(unsheared)

    def _scan_reference_like_orientation_sweep(
        self,
        phi_sampling: np.ndarray,
        theta_sampling: np.ndarray,
        image: np.ndarray,
        *,
        interpolation_order: int,
        smoothing_sigma: float,
        normalize: bool,
        include_confidence: bool,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        best_score = np.zeros_like(image, dtype=np.float32)
        second_score = np.zeros_like(image, dtype=np.float32) if include_confidence else None
        best_code = _new_orientation_codes(image.shape, phi_sampling, theta_sampling)
        grids = _coordinate_grids3(image.shape)

        for iphi, phi in enumerate(phi_sampling):
            for itheta, theta in enumerate(theta_sampling):
                _, strike, dip = _orientation_basis_from_strike_and_dip(
                    float(phi),
                    float(theta),
                )
                score = _reference_like_orientation_score(
                    image,
                    strike=strike,
                    dip=dip,
                    grids=grids,
                    interpolation_order=interpolation_order,
                    smoothing_sigma=smoothing_sigma,
                )
                orientation_code = _encode_orientation_code(
                    iphi,
                    itheta,
                    len(theta_sampling),
                )
                _update_best_orientation(
                    score,
                    orientation_code,
                    best_score,
                    second_score,
                    best_code,
                )

        if normalize:
            ft = _normalize_reference_like_likelihood(best_score)
        else:
            ft = np.maximum(best_score, np.float32(0.0)).astype(np.float32, copy=False)
        best_phi, best_theta = _decode_orientation_codes(
            best_code,
            phi_sampling,
            theta_sampling,
        )
        result = (
            ft,
            best_phi,
            best_theta,
        )
        if second_score is None:
            return result
        confidence = _orientation_confidence_from_scores(best_score, second_score)
        return (*result, confidence)
