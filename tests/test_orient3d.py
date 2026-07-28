from pathlib import Path

import numpy as np
import pytest

import pyosv
import pyosv._orient3d.rotate_shear as rotate_shear_module
import pyosv._orient3d.scanner as scanner_module
import pyosv._orient3d.structured_linear as structured_linear_module
from pyosv._accel import NUMBA_AVAILABLE
from pyosv.geometry import fault_normal_vector_from_strike_and_dip
from pyosv.metrics import buffered_ridge_overlap, strike_dip_angle_error
from pyosv.orient3d import (
    FaultOrientScanner3,
    _dip_shear_from_theta,
    _rotate3_axis1,
    _shear2,
    _unrotate3_axis1,
    _unshear2,
)
from pyosv.thinning3d import reference_like_3d_thin_values, remove_reference_edge_effects_3d
from pyosv.voting3d import OptimalSurfaceVoter


def test_fault_orient_scanner3_import_does_not_change_package_root_api() -> None:
    assert isinstance(pyosv.__version__, str)
    assert not hasattr(pyosv, "FaultOrientScanner3")


def test_strike_sampling_returns_float32_monotonic_angles_with_endpoints() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    phis = scanner.strike_sampling(phi_min=0.0, phi_max=360.0)

    assert phis.dtype == np.float32
    assert phis.ndim == 1
    assert np.isfinite(phis).all()
    assert np.diff(phis).min() > 0.0
    np.testing.assert_allclose(phis[0], np.float32(0.0), atol=1e-6)
    np.testing.assert_allclose(phis[-1], np.float32(360.0), atol=1e-6)
    np.testing.assert_array_equal(
        phis,
        scanner.strike_sampling(phi_min=0.0, phi_max=360.0),
    )


def test_dip_sampling_returns_float32_monotonic_angles_with_endpoints() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    thetas = scanner.dip_sampling(theta_min=35.0, theta_max=85.0)

    assert thetas.dtype == np.float32
    assert thetas.ndim == 1
    assert np.isfinite(thetas).all()
    assert np.diff(thetas).min() > 0.0
    np.testing.assert_allclose(thetas[0], np.float32(35.0), atol=1e-6)
    np.testing.assert_allclose(thetas[-1], np.float32(85.0), atol=1e-6)
    np.testing.assert_array_equal(
        thetas,
        scanner.dip_sampling(theta_min=35.0, theta_max=85.0),
    )


def test_sampling_single_angle_returns_one_endpoint_sample() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    phis = scanner.strike_sampling(phi_min=12.5, phi_max=12.5)
    thetas = scanner.dip_sampling(theta_min=45.0, theta_max=45.0)

    np.testing.assert_array_equal(phis, np.array([12.5], dtype=np.float32))
    np.testing.assert_array_equal(thetas, np.array([45.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("orientation_count", "expected_dtype"),
    [
        (1, np.uint8),
        (256, np.uint8),
        (257, np.uint16),
        (65_536, np.uint16),
        (65_537, np.uint32),
    ],
)
def test_orientation_code_dtype_uses_smallest_supported_unsigned_type(
    orientation_count: int,
    expected_dtype: type[np.unsignedinteger],
) -> None:
    assert scanner_module._orientation_code_dtype(orientation_count) == np.dtype(expected_dtype)


def test_orientation_code_decode_round_trip_returns_independent_contiguous_float32() -> None:
    phis = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    thetas = np.array([40.0, 50.0], dtype=np.float32)
    codes = np.array(
        [
            scanner_module._encode_orientation_code(iphi, itheta, len(thetas))
            for iphi in range(len(phis))
            for itheta in range(len(thetas))
        ],
        dtype=np.uint8,
    ).reshape(2, 3)

    decoded_phi, decoded_theta = scanner_module._decode_orientation_codes(
        codes,
        phis,
        thetas,
    )

    np.testing.assert_array_equal(
        decoded_phi,
        np.array([[10.0, 10.0, 20.0], [20.0, 30.0, 30.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        decoded_theta,
        np.array([[40.0, 50.0, 40.0], [50.0, 40.0, 50.0]], dtype=np.float32),
    )
    for decoded in (decoded_phi, decoded_theta):
        assert decoded.dtype == np.float32
        assert decoded.flags.c_contiguous
        assert not np.shares_memory(decoded, phis)
        assert not np.shares_memory(decoded, thetas)


@pytest.mark.parametrize("include_confidence", [False, True])
def test_orientation_code_updates_keep_strict_ties_and_first_middle_last_best(
    include_confidence: bool,
) -> None:
    best_score = np.zeros(4, dtype=np.float32)
    second_score = np.zeros(4, dtype=np.float32) if include_confidence else None
    best_code = np.zeros(4, dtype=np.uint8)
    scores = (
        np.array([3.0, 1.0, 1.0, 2.0], dtype=np.float32),
        np.array([2.0, 4.0, 1.0, 2.0], dtype=np.float32),
        np.array([1.0, 3.0, 5.0, 2.0], dtype=np.float32),
    )

    for code, score in enumerate(scores):
        scanner_module._update_best_orientation(
            score,
            code,
            best_score,
            second_score,
            best_code,
        )

    np.testing.assert_array_equal(best_code, np.array([0, 1, 2, 0], dtype=np.uint8))
    np.testing.assert_array_equal(best_score, np.array([3.0, 4.0, 5.0, 2.0], dtype=np.float32))


def test_reference_like_sampling_uses_java_inspired_angle_grid() -> None:
    scanner = FaultOrientScanner3(sigma1=8.0, sigma2=8.0)

    phis = scanner.reference_like_strike_sampling(phi_min=0.0, phi_max=360.0)
    thetas = scanner.reference_like_dip_sampling(theta_min=65.0, theta_max=80.0)

    np.testing.assert_array_equal(phis, np.arange(0.0, 360.0, 20.0, dtype=np.float32))
    np.testing.assert_array_equal(thetas, np.array([65.0, 70.0, 75.0, 80.0], dtype=np.float32))
    assert not np.array_equal(phis, scanner.strike_sampling(phi_min=0.0, phi_max=360.0))


def test_reference_like_sampling_keeps_narrow_valid_ranges_callable() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    phis = scanner.reference_like_strike_sampling(phi_min=7.5, phi_max=8.0)
    thetas = scanner.reference_like_dip_sampling(theta_min=42.5, theta_max=42.5)

    np.testing.assert_array_equal(phis, np.array([7.5], dtype=np.float32))
    np.testing.assert_array_equal(thetas, np.array([42.5], dtype=np.float32))


def test_refined_reference_like_sampling_factor_one_matches_base_sampling() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    np.testing.assert_array_equal(
        scanner.refined_reference_like_strike_sampling(
            0.0,
            100.0,
            refinement_factor=1,
        ),
        scanner.reference_like_strike_sampling(0.0, 100.0),
    )
    np.testing.assert_array_equal(
        scanner.refined_reference_like_dip_sampling(
            45.0,
            60.0,
            refinement_factor=1,
        ),
        scanner.reference_like_dip_sampling(45.0, 60.0),
    )


def test_refined_reference_like_sampling_factor_two_includes_midpoints() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    base_phis = scanner.reference_like_strike_sampling(0.0, 60.0)
    refined_phis = scanner.refined_reference_like_strike_sampling(
        0.0,
        60.0,
        refinement_factor=2,
    )
    base_thetas = scanner.reference_like_dip_sampling(45.0, 60.0)
    refined_thetas = scanner.refined_reference_like_dip_sampling(
        45.0,
        60.0,
        refinement_factor=2,
    )

    assert refined_phis.dtype == np.float32
    assert refined_thetas.dtype == np.float32
    np.testing.assert_array_equal(
        refined_phis,
        np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        refined_thetas,
        np.array([45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0], dtype=np.float32),
    )
    assert set(base_phis.tolist()).issubset(set(refined_phis.tolist()))
    assert set(base_thetas.tolist()).issubset(set(refined_thetas.tolist()))
    assert np.diff(refined_phis).min() > 0.0
    assert np.diff(refined_thetas).min() > 0.0
    assert refined_phis[0] >= np.float32(0.0)
    assert refined_phis[-1] <= np.float32(60.0)
    assert refined_thetas[0] >= np.float32(45.0)
    assert refined_thetas[-1] <= np.float32(60.0)


def test_refined_reference_like_sampling_dispatches_to_base_sampling_overrides() -> None:
    class OverrideScanner(FaultOrientScanner3):
        def reference_like_strike_sampling(
            self,
            phi_min: float,
            phi_max: float,
        ) -> np.ndarray:
            del phi_min, phi_max
            return np.asarray([10.0, 30.0], dtype=np.float32)

        def reference_like_dip_sampling(
            self,
            theta_min: float,
            theta_max: float,
        ) -> np.ndarray:
            del theta_min, theta_max
            return np.asarray([5.0, 15.0], dtype=np.float32)

    scanner = OverrideScanner(sigma1=2.0, sigma2=2.0)

    strike = scanner.refined_reference_like_strike_sampling(
        10.0,
        30.0,
        refinement_factor=2,
    )
    dip = scanner.refined_reference_like_dip_sampling(
        5.0,
        15.0,
        refinement_factor=2,
    )

    np.testing.assert_array_equal(
        strike,
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dip,
        np.asarray([5.0, 10.0, 15.0], dtype=np.float32),
    )
    assert strike.dtype == np.float32
    assert dip.dtype == np.float32


@pytest.mark.parametrize("refinement_factor", [0, 5, 1.5, True])
def test_refined_reference_like_sampling_rejects_invalid_refinement_factor(
    refinement_factor: object,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match="refinement_factor"):
        scanner.refined_reference_like_strike_sampling(
            0.0,
            60.0,
            refinement_factor=refinement_factor,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("sigma1", "sigma2", "message"),
    [
        (0.0, 2.0, "sigma1"),
        (-1.0, 2.0, "sigma1"),
        (np.nan, 2.0, "sigma1"),
        (np.inf, 2.0, "sigma1"),
        (True, 2.0, "sigma1"),
        ("2.0", 2.0, "sigma1"),
        (2.0, 0.0, "sigma2"),
        (2.0, -1.0, "sigma2"),
        (2.0, np.nan, "sigma2"),
        (2.0, np.inf, "sigma2"),
        (2.0, True, "sigma2"),
        (2.0, "2.0", "sigma2"),
    ],
)
def test_constructor_rejects_invalid_sigmas(
    sigma1: object,
    sigma2: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("phi_min", "phi_max", "message"),
    [
        (np.nan, 360.0, "phi_min"),
        (0.0, np.inf, "phi_max"),
        (True, 360.0, "phi_min"),
        (0.0, "360.0", "phi_max"),
        (360.0, 0.0, "phi_max"),
    ],
)
def test_strike_sampling_rejects_invalid_angles(
    phi_min: object,
    phi_max: object,
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.strike_sampling(phi_min=phi_min, phi_max=phi_max)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("theta_min", "theta_max", "message"),
    [
        (np.nan, 85.0, "theta_min"),
        (35.0, np.inf, "theta_max"),
        (True, 85.0, "theta_min"),
        (35.0, "85.0", "theta_max"),
        (85.0, 35.0, "theta_max"),
    ],
)
def test_dip_sampling_rejects_invalid_angles(
    theta_min: object,
    theta_max: object,
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.dip_sampling(theta_min=theta_min, theta_max=theta_max)  # type: ignore[arg-type]


def test_validate_image_accepts_finite_3d_numeric_array_as_float32() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    validated = scanner.validate_image(image)

    assert validated.shape == (2, 3, 4)
    assert validated.dtype == np.float32
    np.testing.assert_allclose(validated, image)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "3D array"),
        (np.array([[[0.0, np.nan]]], dtype=np.float32), "finite"),
        (np.array([[[0.0, np.inf]]], dtype=np.float32), "finite"),
        (np.array([[["bad"]]], dtype=object), "numeric finite"),
    ],
)
def test_validate_image_rejects_invalid_inputs(image: object, message: str) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.validate_image(image)  # type: ignore[arg-type]


def test_scan_validates_sampling_inputs_before_image_response() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="phi_max"):
        scanner.scan(360.0, 0.0, 35.0, 85.0, image)


def test_scan_reference_like_method_exists() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    assert callable(scanner.scan_reference_like)


def test_scan_fast_method_exists() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    assert callable(scanner.scan_fast)


@pytest.mark.parametrize(
    ("phi_min", "phi_max", "theta_min", "theta_max", "message"),
    [
        (360.0, 0.0, 35.0, 85.0, "phi_max"),
        (0.0, 360.0, 85.0, 35.0, "theta_max"),
        (np.nan, 360.0, 35.0, 85.0, "phi_min"),
        (0.0, 360.0, np.inf, 85.0, "theta_min"),
    ],
)
def test_scan_reference_like_validates_angle_ranges(
    phi_min: object,
    phi_max: object,
    theta_min: object,
    theta_max: object,
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        scanner.scan_reference_like(
            phi_min,  # type: ignore[arg-type]
            phi_max,  # type: ignore[arg-type]
            theta_min,  # type: ignore[arg-type]
            theta_max,  # type: ignore[arg-type]
            image,
        )


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "3D array"),
        (np.array([[[0.0, np.nan]]], dtype=np.float32), "finite"),
        (np.array([[[0.0, np.inf]]], dtype=np.float32), "finite"),
    ],
)
def test_scan_reference_like_validates_image(
    image: object,
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.scan_reference_like(0.0, 90.0, 35.0, 85.0, image)  # type: ignore[arg-type]


@pytest.mark.parametrize("interpolation_order", [-1, 6, 1.5, True])
def test_scan_reference_like_validates_interpolation_order(
    interpolation_order: object,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="interpolation_order"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            interpolation_order=interpolation_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("interpolation_backend", ["missing", 1, None])
def test_scan_reference_like_validates_interpolation_backend(
    interpolation_backend: object,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="interpolation_backend"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            interpolation_backend=interpolation_backend,  # type: ignore[arg-type]
        )


def test_structured_linear_backend_requires_order_one_and_rotate_shear() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="requires interpolation_order=1"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            interpolation_order=0,
            interpolation_backend="structured_linear",
        )
    with pytest.raises(ValueError, match="requires backend='rotate_shear'"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            backend="directional",
            interpolation_backend="structured_linear",
        )


@pytest.mark.parametrize("smoothing_sigma", [-1.0, np.nan, np.inf, True, "1.0"])
def test_scan_reference_like_validates_smoothing_sigma(smoothing_sigma: object) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="smoothing_sigma"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            smoothing_sigma=smoothing_sigma,  # type: ignore[arg-type]
        )


def test_scan_reference_like_validates_normalize() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="normalize"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            normalize=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("backend", ["missing", 1])
def test_scan_reference_like_validates_backend(backend: object) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="backend"):
        scanner.scan_reference_like(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            backend=backend,  # type: ignore[arg-type]
        )


def test_scan_reference_like_backend_selector_keeps_directional_approximation() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(60.0, 60.0, shape=(9, 10, 11), width=1.0)

    rotate_shear = scanner.scan_reference_like(
        0.0,
        80.0,
        40.0,
        80.0,
        image,
        backend="rotate_shear",
        smoothing_sigma=0.75,
    )
    directional = scanner.scan_reference_like(
        0.0,
        80.0,
        40.0,
        80.0,
        image,
        backend="directional",
        smoothing_sigma=0.75,
    )

    for outputs in (rotate_shear, directional):
        for array in outputs:
            assert array.shape == image.shape
            assert array.dtype == np.float32
            assert np.isfinite(array).all()
        assert float(outputs[0].min()) >= 0.0
        assert float(outputs[0].max()) <= 1.0


def test_scan_with_confidence_returns_float32_finite_unit_confidence() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(9, 10, 11), width=1.0)

    ft, pt, tt, confidence = scanner.scan_with_confidence(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        smoothing_sigma=0.75,
    )

    for array in (ft, pt, tt, confidence):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert float(ft.min()) >= 0.0
    assert float(ft.max()) <= 1.0
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0


def test_scan_with_confidence_first_three_arrays_match_reference_like() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(9, 10, 11), width=1.0)

    reference_like = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        smoothing_sigma=0.75,
    )
    with_confidence = scanner.scan_with_confidence(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        smoothing_sigma=0.75,
    )

    for reference_array, confidence_array in zip(reference_like, with_confidence[:3]):
        np.testing.assert_array_equal(reference_array, confidence_array)


@pytest.mark.parametrize("backend", ["rotate_shear", "directional"])
def test_scans_without_confidence_skip_confidence_work(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image = np.linspace(0.0, 1.0, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    def fail_confidence(*args: object, **kwargs: object) -> None:
        raise AssertionError("confidence calculation must be skipped")

    def fail_second_best(*args: object, **kwargs: object) -> None:
        raise AssertionError("second-best update must be skipped")

    monkeypatch.setattr(
        scanner_module,
        "_orientation_confidence_from_scores",
        fail_confidence,
    )
    monkeypatch.setattr(
        scanner_module,
        "_update_best_second_orientation",
        fail_second_best,
    )

    reference_like = scanner.scan_reference_like(
        0.0,
        20.0,
        50.0,
        55.0,
        image,
        backend=backend,
        smoothing_sigma=0.5,
    )
    quality = scanner.scan_quality(
        0.0,
        20.0,
        50.0,
        55.0,
        image,
        backend=backend,
        refinement_factor=1,
        smoothing_sigma=0.5,
        return_confidence=False,
    )

    assert len(reference_like) == 3
    assert len(quality) == 3


def test_theta_shear_scores_are_generated_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    theta_sampling = np.array([45.0, 50.0, 55.0], dtype=np.float32)
    rotated = np.linspace(0.0, 1.0, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    original_likelihood = scanner_module._reference_like_planarity_to_likelihood
    calls = 0

    def count_likelihood(smoothed: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_likelihood(smoothed)

    monkeypatch.setattr(
        scanner_module,
        "_reference_like_planarity_to_likelihood",
        count_likelihood,
    )

    scores = scanner._scan_theta_shear_reference_like(
        theta_sampling,
        rotated,
        interpolation_order=1,
        smoothing_sigma=0.5,
    )

    assert iter(scores) is scores
    assert calls == 0
    first = next(scores)
    first_copy = first.copy()
    assert calls == 1
    second = next(scores)
    assert calls == 2
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, first_copy)
    assert len(list(scores)) == 1
    assert calls == len(theta_sampling)


def test_reference_like_strict_ties_keep_first_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image = np.linspace(0.0, 1.0, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    def tied_score(
        image: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        return np.ones_like(image, dtype=np.float32)

    monkeypatch.setattr(scanner_module, "_reference_like_orientation_score", tied_score)

    ft, pt, tt = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        backend="directional",
    )

    np.testing.assert_array_equal(ft, np.ones_like(image, dtype=np.float32))
    np.testing.assert_array_equal(pt, np.zeros_like(image, dtype=np.float32))
    np.testing.assert_array_equal(tt, np.full_like(image, 50.0, dtype=np.float32))


def test_scan_with_confidence_constant_input_returns_zero_confidence() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.full((5, 6, 7), 3.0, dtype=np.float64)

    ft, pt, tt, confidence = scanner.scan_with_confidence(10.0, 40.0, 30.0, 60.0, image)

    for array in (ft, pt, tt, confidence):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    np.testing.assert_array_equal(ft, np.zeros(image.shape, dtype=np.float32))
    np.testing.assert_array_equal(confidence, np.zeros(image.shape, dtype=np.float32))


def test_scan_with_confidence_directional_backend_returns_confidence() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(8, 9, 10), width=1.0)

    ft, pt, tt, confidence = scanner.scan_with_confidence(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        backend="directional",
        smoothing_sigma=0.75,
    )

    for array in (ft, pt, tt, confidence):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0


def test_scan_quality_factor_one_matches_reference_like() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(8, 9, 10), width=1.0)

    reference_like = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        smoothing_sigma=0.75,
    )
    quality = scanner.scan_quality(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        refinement_factor=1,
        smoothing_sigma=0.75,
    )

    for reference_array, quality_array in zip(reference_like, quality):
        np.testing.assert_array_equal(reference_array, quality_array)


def test_scan_quality_uses_sampling_derived_from_base_overrides() -> None:
    observed: dict[str, np.ndarray] = {}

    class OverrideScanner(FaultOrientScanner3):
        def reference_like_strike_sampling(
            self,
            phi_min: float,
            phi_max: float,
        ) -> np.ndarray:
            del phi_min, phi_max
            return np.asarray([10.0, 30.0], dtype=np.float32)

        def reference_like_dip_sampling(
            self,
            theta_min: float,
            theta_max: float,
        ) -> np.ndarray:
            del theta_min, theta_max
            return np.asarray([5.0, 15.0], dtype=np.float32)

        def _scan_reference_like_samples_with_confidence(
            self,
            phi_sampling: np.ndarray,
            theta_sampling: np.ndarray,
            g: np.ndarray,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            del self, kwargs
            observed["strike"] = phi_sampling
            observed["dip"] = theta_sampling
            zeros = np.zeros_like(g, dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy()

    scanner = OverrideScanner(sigma1=2.0, sigma2=2.0)
    image = np.zeros((1, 2, 3), dtype=np.float32)

    result = scanner.scan_quality(
        10.0,
        30.0,
        5.0,
        15.0,
        image,
        refinement_factor=2,
    )

    np.testing.assert_array_equal(
        observed["strike"],
        np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observed["dip"],
        np.asarray([5.0, 10.0, 15.0], dtype=np.float32),
    )
    assert observed["strike"].dtype == np.float32
    assert observed["dip"].dtype == np.float32
    assert len(result) == 3


def test_scan_quality_can_return_confidence() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(8, 9, 10), width=1.0)

    without_confidence = scanner.scan_quality(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        refinement_factor=2,
        smoothing_sigma=0.75,
        return_confidence=False,
    )
    with_confidence = scanner.scan_quality(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        refinement_factor=2,
        smoothing_sigma=0.75,
        return_confidence=True,
    )
    ft, pt, tt, confidence = with_confidence

    assert len(without_confidence) == 3
    assert len(with_confidence) == 4
    for plain_array, confidence_array in zip(without_confidence, with_confidence[:3]):
        np.testing.assert_array_equal(plain_array, confidence_array)
    for array in (ft, pt, tt, confidence):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert float(ft.min()) >= 0.0
    assert float(ft.max()) <= 1.0
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "missing"}, "backend"),
        ({"backend": 1}, "backend"),
        ({"interpolation_order": -1}, "interpolation_order"),
        ({"interpolation_order": 6}, "interpolation_order"),
        ({"interpolation_order": True}, "interpolation_order"),
        ({"smoothing_sigma": -1.0}, "smoothing_sigma"),
        ({"smoothing_sigma": np.nan}, "smoothing_sigma"),
        ({"smoothing_sigma": True}, "smoothing_sigma"),
    ],
)
def test_scan_with_confidence_validates_reference_like_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        scanner.scan_with_confidence(
            0.0,
            90.0,
            35.0,
            85.0,
            image,
            **kwargs,
        )


def test_rotate3_axis1_unrotate3_axis1_return_finite_float32_shapes() -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    rotated = _rotate3_axis1(volume, 35.0, interpolation_order=1)
    unrotated = _unrotate3_axis1(rotated, volume.shape, 35.0, interpolation_order=1)

    assert rotated.ndim == 3
    assert rotated.shape[0] >= volume.shape[0]
    assert rotated.shape[1] >= volume.shape[1]
    assert rotated.shape[2] == volume.shape[2]
    assert rotated.dtype == np.float32
    assert unrotated.shape == volume.shape
    assert unrotated.dtype == np.float32
    assert np.isfinite(rotated).all()
    assert np.isfinite(unrotated).all()


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 5, 4)])
@pytest.mark.parametrize("angle", [0.0, 27.5, -35.0])
def test_structured_rotation_primitives_match_scipy(
    shape: tuple[int, int, int],
    angle: float,
) -> None:
    volume = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)

    scipy_rotated = _rotate3_axis1(volume, angle, interpolation_order=1)
    structured_rotated = _rotate3_axis1(
        volume,
        angle,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )
    scipy_unrotated = _unrotate3_axis1(
        scipy_rotated,
        shape,
        angle,
        interpolation_order=1,
        fill_value=0.375,
    )
    structured_unrotated = _unrotate3_axis1(
        scipy_rotated,
        shape,
        angle,
        interpolation_order=1,
        fill_value=0.375,
        interpolation_backend="structured_linear",
    )

    np.testing.assert_allclose(structured_rotated, scipy_rotated, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        structured_unrotated,
        scipy_unrotated,
        atol=1.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("shape", [(2, 2), (5, 6)])
@pytest.mark.parametrize("shear", [0.0, 0.5, -0.375])
def test_structured_shear_primitives_match_scipy_boundary_coordinates(
    shape: tuple[int, int],
    shear: float,
) -> None:
    image = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)

    scipy_sheared = _shear2(image, shear, interpolation_order=1)
    structured_sheared = _shear2(
        image,
        shear,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )
    scipy_unsheared = _unshear2(image, shear, interpolation_order=1)
    structured_unsheared = _unshear2(
        image,
        shear,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )

    np.testing.assert_allclose(structured_sheared, scipy_sheared, atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        structured_unsheared,
        scipy_unsheared,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_structured_linear_sampling_matches_known_boundary_values() -> None:
    image = np.repeat(
        np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32),
        5,
        axis=1,
    )

    sheared = _shear2(
        image,
        0.5,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )

    np.testing.assert_array_equal(
        sheared[0],
        np.array([20.0, 15.0, 10.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sheared[2],
        np.array([40.0, 35.0, 30.0, 25.0, 20.0], dtype=np.float32),
    )


def test_structured_unrotation_matches_scipy_with_nonzero_cval() -> None:
    rotated = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    shape = (4, 5, 3)

    scipy = _unrotate3_axis1(
        rotated,
        shape,
        27.5,
        interpolation_order=1,
        fill_value=0.375,
    )
    structured = _unrotate3_axis1(
        rotated,
        shape,
        27.5,
        interpolation_order=1,
        fill_value=0.375,
        interpolation_backend="structured_linear",
    )

    assert np.any(scipy == np.float32(0.375))
    np.testing.assert_allclose(structured, scipy, atol=1.0e-6, rtol=0.0)


def test_structured_transforms_do_not_build_coordinate_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    def fail_grid(*args: object, **kwargs: object) -> None:
        raise AssertionError("structured transforms must not build coordinate grids")

    monkeypatch.setattr(rotate_shear_module, "_coordinate_grids3", fail_grid)
    monkeypatch.setattr(rotate_shear_module.np, "indices", fail_grid)
    rotated = _rotate3_axis1(
        volume,
        27.5,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )
    _unrotate3_axis1(
        rotated,
        volume.shape,
        27.5,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )
    _shear2(
        volume[0],
        0.5,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )
    _unshear2(
        volume[0],
        0.5,
        interpolation_order=1,
        interpolation_backend="structured_linear",
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_structured_python_and_numba_primitives_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = np.linspace(0.0, 1.0, 3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)

    def transformed_with(kernels: tuple[object, object, object, object]):
        monkeypatch.setattr(rotate_shear_module, "_rotate3_axis1_structured", kernels[0])
        monkeypatch.setattr(rotate_shear_module, "_unrotate3_axis1_structured", kernels[1])
        monkeypatch.setattr(rotate_shear_module, "_shear2_structured", kernels[2])
        monkeypatch.setattr(rotate_shear_module, "_unshear2_structured", kernels[3])
        rotated = _rotate3_axis1(
            volume,
            -27.5,
            interpolation_order=1,
            interpolation_backend="structured_linear",
        )
        return (
            rotated,
            _unrotate3_axis1(
                rotated,
                volume.shape,
                -27.5,
                interpolation_order=1,
                interpolation_backend="structured_linear",
            ),
            _shear2(
                volume[0],
                0.375,
                interpolation_order=1,
                interpolation_backend="structured_linear",
            ),
            _unshear2(
                volume[0],
                0.375,
                interpolation_order=1,
                interpolation_backend="structured_linear",
            ),
        )

    python_outputs = transformed_with(
        (
            structured_linear_module._rotate3_axis1_structured_python,
            structured_linear_module._unrotate3_axis1_structured_python,
            structured_linear_module._shear2_structured_python,
            structured_linear_module._unshear2_structured_python,
        )
    )
    numba_outputs = transformed_with(
        (
            structured_linear_module._rotate3_axis1_structured_numba,
            structured_linear_module._unrotate3_axis1_structured_numba,
            structured_linear_module._shear2_structured_numba,
            structured_linear_module._unshear2_structured_numba,
        )
    )

    for python_output, numba_output in zip(python_outputs, numba_outputs):
        np.testing.assert_array_equal(python_output, numba_output)


@pytest.mark.parametrize("shape", [(4, 6, 5), (5, 5, 6)])
def test_rotate3_axis1_unrotate3_axis1_zero_degrees_are_identity(
    shape: tuple[int, int, int],
) -> None:
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    rotated = _rotate3_axis1(volume, 0.0, interpolation_order=1)
    unrotated = _unrotate3_axis1(rotated, volume.shape, 0.0, interpolation_order=1)

    assert rotated.shape[0] >= volume.shape[0]
    assert rotated.shape[1] >= volume.shape[1]
    assert rotated.shape[2] == volume.shape[2]
    np.testing.assert_array_equal(
        rotated[: volume.shape[0], : volume.shape[1], :],
        volume,
    )
    np.testing.assert_array_equal(unrotated, volume)


def test_rotate3_axis1_unrotate3_axis1_round_trip_is_close_for_smooth_volume() -> None:
    shape = (9, 10, 11)
    i3, i2, i1 = np.indices(shape, dtype=np.float32)
    c3 = np.float32(0.5 * (shape[0] - 1))
    c2 = np.float32(0.5 * (shape[1] - 1))
    c1 = np.float32(0.5 * (shape[2] - 1))
    radius2 = (i1 - c1) ** 2 + (i2 - c2) ** 2 + (i3 - c3) ** 2
    volume = (1.0 - 0.5 * np.exp(-radius2 / np.float32(18.0))).astype(np.float32)

    rotated = _rotate3_axis1(volume, 35.0, interpolation_order=1)
    unrotated = _unrotate3_axis1(rotated, volume.shape, 35.0, interpolation_order=1)

    np.testing.assert_allclose(
        unrotated[2:-2, 2:-2, 2:-2],
        volume[2:-2, 2:-2, 2:-2],
        atol=0.06,
        rtol=0.0,
    )


def test_shear2_unshear2_are_deterministic_and_near_vertical_shear_is_stable() -> None:
    image = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    shear = _dip_shear_from_theta(90.0)

    first = _shear2(image, float(shear), interpolation_order=1)
    second = _shear2(image, float(shear), interpolation_order=1)
    restored = _unshear2(first, float(shear), interpolation_order=1)

    assert shear == np.float32(0.0)
    assert first.shape == image.shape
    assert restored.shape == image.shape
    assert first.dtype == np.float32
    assert restored.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.isfinite(restored).all()
    np.testing.assert_array_equal(first, second)


def test_scan_reference_like_constant_input_returns_zero_likelihood_and_finite_angles() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.full((5, 6, 7), 3.0, dtype=np.float64)

    ft, pt, tt = scanner.scan_reference_like(10.0, 40.0, 30.0, 60.0, image)

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert tt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert tt.dtype == np.float32
    np.testing.assert_array_equal(ft, np.zeros(image.shape, dtype=np.float32))
    np.testing.assert_array_equal(
        pt,
        np.full(
            image.shape,
            scanner.reference_like_strike_sampling(10.0, 40.0)[0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        tt,
        np.full(
            image.shape,
            scanner.reference_like_dip_sampling(30.0, 60.0)[0],
            dtype=np.float32,
        ),
    )
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()


def test_scan_reference_like_returns_float32_normalized_outputs() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(60.0, 60.0, shape=(15, 16, 17), width=1.0)

    ft, pt, tt = scanner.scan_reference_like(
        0.0,
        90.0,
        30.0,
        90.0,
        image,
        smoothing_sigma=1.0,
    )

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert tt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert tt.dtype == np.float32
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()
    assert float(ft.min()) >= 0.0
    assert float(ft.max()) <= 1.0


def test_scan_reference_like_is_deterministic() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(60.0, 60.0, shape=(13, 14, 15), width=1.0)

    first = scanner.scan_reference_like(
        0.0,
        90.0,
        30.0,
        90.0,
        image,
        smoothing_sigma=0.75,
    )
    second = scanner.scan_reference_like(
        0.0,
        90.0,
        30.0,
        90.0,
        image,
        smoothing_sigma=0.75,
    )

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_scan_reference_like_scores_low_planarity_plane_above_background() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, distance = _low_planarity_fault(0.0, 90.0, shape=(11, 12, 13), width=0.75)

    ft, pt, tt = scanner.scan_reference_like(
        0.0,
        0.0,
        90.0,
        90.0,
        image,
        smoothing_sigma=0.0,
        normalize=False,
    )

    near_plane = np.abs(distance) <= 0.5
    far_from_plane = np.abs(distance) >= 3.0
    assert float(np.mean(ft[near_plane])) > 0.75
    assert float(np.mean(ft[far_from_plane])) < 0.05
    np.testing.assert_array_equal(pt, np.zeros_like(pt, dtype=np.float32))
    np.testing.assert_array_equal(tt, np.full_like(tt, 90.0, dtype=np.float32))


def test_scan_reference_like_localizes_synthetic_planar_fault_orientation() -> None:
    true_phi = 60.0
    true_theta = 60.0
    image, distance = _low_planarity_fault(
        true_phi,
        true_theta,
        shape=(21, 22, 23),
        width=1.0,
    )
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)

    ft, pt, tt = scanner.scan_reference_like(
        0.0,
        90.0,
        30.0,
        90.0,
        image,
        smoothing_sigma=1.5,
    )

    near_plane = np.abs(distance) <= 1.0
    far_from_plane = np.abs(distance) >= 5.0
    assert float(np.mean(ft[near_plane])) > 2.0 * float(np.mean(ft[far_from_plane]))

    ridge_target = np.exp(-0.5 * (distance / np.float32(1.0)) ** 2).astype(np.float32)
    overlap = buffered_ridge_overlap(
        ridge_target,
        ft,
        percentile=98.0,
        radius=2.5,
    )
    assert overlap["reference_count"] > 0
    assert overlap["candidate_count"] > 0
    assert overlap["buffered_precision"] >= 0.70

    high_likelihood = ft >= np.percentile(ft, 98.0)
    angle_error = strike_dip_angle_error(
        pt[high_likelihood],
        tt[high_likelihood],
        expected_strike=true_phi,
        expected_dip=true_theta,
        strike_period=180.0,
    )
    assert float(np.median(angle_error["strike"])) <= 31.0
    assert float(np.median(angle_error["dip"])) <= 31.0


def test_scan_reference_like_crossing_planes_selects_deterministic_maximum() -> None:
    shape = (17, 18, 19)
    plane_a, distance_a = _planar_gaussian_fault(0.0, 90.0, shape=shape, width=0.8)
    plane_b, distance_b = _planar_gaussian_fault(60.0, 60.0, shape=shape, width=0.8)
    image = (np.float32(1.0) - np.maximum(plane_a, plane_b * np.float32(0.9))).astype(
        np.float32,
        copy=False,
    )
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)

    first = scanner.scan_reference_like(
        0.0,
        100.0,
        50.0,
        90.0,
        image,
        smoothing_sigma=0.75,
    )
    second = scanner.scan_reference_like(
        0.0,
        100.0,
        50.0,
        90.0,
        image,
        smoothing_sigma=0.75,
    )
    ft, pt, tt = first

    for first_array, second_array in zip(first, second):
        assert first_array.shape == image.shape
        assert first_array.dtype == np.float32
        assert np.isfinite(first_array).all()
        np.testing.assert_array_equal(first_array, second_array)

    max_mask = ft == np.max(ft)
    assert np.count_nonzero(max_mask) > 0
    assert np.all(np.abs(distance_a[max_mask]) <= 1.0)
    assert np.all(np.abs(distance_b[max_mask]) <= 1.0)
    assert np.unique(pt[max_mask]).size == 1
    assert np.unique(tt[max_mask]).size == 1
    np.testing.assert_array_equal(
        np.unique(pt[max_mask]),
        np.intersect1d(
            np.unique(pt[max_mask]),
            scanner.reference_like_strike_sampling(0.0, 100.0),
        ),
    )
    np.testing.assert_array_equal(
        np.unique(tt[max_mask]),
        np.intersect1d(
            np.unique(tt[max_mask]),
            scanner.reference_like_dip_sampling(50.0, 90.0),
        ),
    )


def test_scan_reference_like_does_not_call_default_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("scan() must not be called")

    monkeypatch.setattr(FaultOrientScanner3, "scan", fail_scan)
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(60.0, 60.0, shape=(9, 10, 11), width=1.0)

    ft, pt, tt = scanner.scan_reference_like(
        60.0,
        60.0,
        60.0,
        60.0,
        image,
        smoothing_sigma=0.5,
    )

    for array in (ft, pt, tt):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_f3_validation_examples_use_current_scan_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_paths = [
        repo_root / "examples" / "run_3d_f3d_crop_validation.py",
        repo_root / "examples" / "run_3d_f3d_full.py",
    ]

    for path in example_paths:
        source = path.read_text()
        assert "scan_reference_like" not in source
        assert ".scan(" in source


def test_scan_matches_reference_like_default_backend() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(60.0, 60.0, shape=(9, 10, 11), width=1.0)

    default = scanner.scan(0.0, 90.0, 30.0, 90.0, image)
    explicit = scanner.scan(
        0.0,
        90.0,
        30.0,
        90.0,
        image,
        backend="rotate_shear",
        interpolation_order=1,
        interpolation_backend="scipy",
        smoothing_sigma=None,
        normalize=True,
    )

    for default_array, explicit_array in zip(default, explicit):
        np.testing.assert_array_equal(default_array, explicit_array)


def test_scan_forwards_all_reference_like_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)
    expected = tuple(np.zeros_like(image) for _ in range(3))
    observed: dict[str, object] = {}

    def scan_reference_like(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(scanner, "scan_reference_like", scan_reference_like)

    actual = scanner.scan(
        10.0,
        20.0,
        30.0,
        40.0,
        image,
        backend="directional",
        interpolation_order=3,
        interpolation_backend="scipy",
        smoothing_sigma=1.25,
        normalize=False,
    )

    assert actual is expected
    assert observed == {
        "args": (10.0, 20.0, 30.0, 40.0, image),
        "kwargs": {
            "backend": "directional",
            "interpolation_order": 3,
            "interpolation_backend": "scipy",
            "smoothing_sigma": 1.25,
            "normalize": False,
        },
    }


def test_default_scipy_interpolation_backend_is_exactly_unchanged() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(6, 7, 8), width=1.0)

    implicit = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        smoothing_sigma=0.75,
    )
    explicit = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        interpolation_backend="scipy",
        smoothing_sigma=0.75,
    )

    for implicit_array, explicit_array in zip(implicit, explicit):
        np.testing.assert_array_equal(implicit_array, explicit_array)


def test_structured_backend_end_to_end_metrics_and_public_entry_propagation() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _low_planarity_fault(40.0, 55.0, shape=(6, 7, 8), width=1.0)
    options = {
        "interpolation_backend": "structured_linear",
        "smoothing_sigma": 0.75,
    }

    scipy = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        interpolation_backend="scipy",
        smoothing_sigma=0.75,
    )
    structured = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        **options,
    )
    scan = scanner.scan(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        interpolation_backend="structured_linear",
    )
    confidence = scanner.scan_with_confidence(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        **options,
    )
    quality = scanner.scan_quality(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        refinement_factor=1,
        **options,
    )

    likelihood_max_abs_diff = float(np.max(np.abs(scipy[0] - structured[0])))
    strike_periodic_diff = np.minimum(
        np.abs(scipy[1] - structured[1]),
        np.float32(180.0) - np.abs(scipy[1] - structured[1]),
    )
    dip_diff = np.abs(scipy[2] - structured[2])
    orientation_bin_change_rate = float(np.mean((strike_periodic_diff > 0.0) | (dip_diff > 0.0)))

    assert likelihood_max_abs_diff <= 1.0e-6
    assert float(np.max(strike_periodic_diff)) == 0.0
    assert float(np.max(dip_diff)) == 0.0
    assert orientation_bin_change_rate == 0.0
    for structured_array, confidence_array, quality_array in zip(
        structured,
        confidence[:3],
        quality,
    ):
        np.testing.assert_array_equal(structured_array, confidence_array)
        np.testing.assert_array_equal(structured_array, quality_array)
    explicit_scan = scanner.scan_reference_like(
        0.0,
        40.0,
        50.0,
        60.0,
        image,
        interpolation_backend="structured_linear",
    )
    for scan_array, explicit_array in zip(scan, explicit_scan):
        np.testing.assert_array_equal(scan_array, explicit_array)


def test_scan_constant_input_returns_zero_likelihood_and_finite_angles() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.full((5, 6, 7), 3.0, dtype=np.float64)

    ft, pt, tt = scanner.scan(10.0, 40.0, 30.0, 60.0, image)

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert tt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert tt.dtype == np.float32
    np.testing.assert_array_equal(ft, np.zeros(image.shape, dtype=np.float32))
    np.testing.assert_array_equal(
        pt,
        np.full(
            image.shape,
            scanner.reference_like_strike_sampling(10.0, 40.0)[0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        tt,
        np.full(
            image.shape,
            scanner.reference_like_dip_sampling(30.0, 60.0)[0],
            dtype=np.float32,
        ),
    )
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()


def test_scan_uses_reference_like_sampling_for_constant_input() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.full((3, 4, 5), 1.0, dtype=np.float32)

    _, pt, tt = scanner.scan(10.0, 40.0, 30.0, 60.0, image)

    assert np.unique(pt).tolist() == [20.0]
    assert np.unique(tt).tolist() == [30.0]


def test_scan_localizes_synthetic_planar_fault_and_recovers_orientation() -> None:
    true_phi = 60.0
    true_theta = 50.0
    image, distance = _low_planarity_fault(true_phi, true_theta)
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)

    ft, pt, tt = scanner.scan(0.0, 90.0, 20.0, 80.0, image)

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert tt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert tt.dtype == np.float32
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()
    assert float(ft.min()) >= 0.0
    assert float(ft.max()) <= 1.0

    near_plane = np.abs(distance) <= 1.0
    far_from_plane = np.abs(distance) >= 8.0
    assert float(np.mean(ft[near_plane])) > 2.0 * float(np.mean(ft[far_from_plane]))

    high_likelihood = ft >= np.percentile(ft, 98.0)
    angle_error = strike_dip_angle_error(
        pt[high_likelihood],
        tt[high_likelihood],
        expected_strike=true_phi,
        expected_dip=true_theta,
        strike_period=180.0,
    )
    assert float(np.median(angle_error["strike"])) <= 31.0
    assert float(np.median(angle_error["dip"])) <= 31.0


def test_scan_and_scan_fast_are_finite_on_same_synthetic_planar_volume() -> None:
    image, _ = _low_planarity_fault(45.0, 50.0, shape=(17, 18, 19), width=1.0)
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    scan_outputs = scanner.scan(0.0, 90.0, 20.0, 80.0, image)
    fast_outputs = scanner.scan_fast(0.0, 90.0, 20.0, 80.0, image)

    for array in (*scan_outputs, *fast_outputs):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_scan_fast_returns_finite_normalized_outputs() -> None:
    true_phi = 45.0
    true_theta = 50.0
    image, _ = _planar_gaussian_fault(true_phi, true_theta, shape=(17, 18, 19))
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    ft, pt, tt = scanner.scan_fast(0.0, 90.0, 20.0, 80.0, image)

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert tt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert tt.dtype == np.float32
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()
    assert float(ft.min()) >= 0.0
    assert float(ft.max()) <= 1.0


def test_scan_fast_constant_input_uses_derivative_bank_sampling() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.full((5, 6, 7), 3.0, dtype=np.float64)

    ft, pt, tt = scanner.scan_fast(10.0, 40.0, 30.0, 60.0, image)

    np.testing.assert_array_equal(ft, np.zeros(image.shape, dtype=np.float32))
    np.testing.assert_array_equal(
        pt,
        np.full(image.shape, scanner.strike_sampling(10.0, 40.0)[0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tt,
        np.full(image.shape, scanner.dip_sampling(30.0, 60.0)[0], dtype=np.float32),
    )


def test_thin_normal_mode_keeps_planar_likelihood_maxima_along_fault_normal() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((7, 9, 7), dtype=np.float32)
    ft[1:6, 3, 1:6] = 0.6
    ft[1:6, 4, 1:6] = 1.0
    ft[1:6, 5, 1:6] = 0.6
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft_before = ft.copy()
    pt_before = pt.copy()
    tt_before = tt.copy()

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(ft, pt, tt, mode="normal")

    for array in (thinned_ft, thinned_pt, thinned_tt):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    np.testing.assert_array_equal(ft, ft_before)
    np.testing.assert_array_equal(pt, pt_before)
    np.testing.assert_array_equal(tt, tt_before)
    assert np.count_nonzero(thinned_ft) == 25
    np.testing.assert_array_equal(thinned_ft[:, 4, :], ft[:, 4, :])
    np.testing.assert_array_equal(thinned_ft[:, :4, :], np.zeros_like(thinned_ft[:, :4, :]))
    np.testing.assert_array_equal(thinned_ft[:, 5:, :], np.zeros_like(thinned_ft[:, 5:, :]))
    np.testing.assert_array_equal(thinned_pt[thinned_ft > 0.0], pt[thinned_ft > 0.0])
    np.testing.assert_array_equal(thinned_tt[thinned_ft > 0.0], tt[thinned_ft > 0.0])
    np.testing.assert_array_equal(thinned_pt[thinned_ft == 0.0], 0.0)
    np.testing.assert_array_equal(thinned_tt[thinned_ft == 0.0], 0.0)


def test_thin_default_matches_reference_mode() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((7, 7, 2), dtype=np.float32)
    ft[3, 2:5, :] = np.array([[0.4, 0.2], [1.0, 0.8], [0.3, 0.1]], dtype=np.float32)
    pt = np.full_like(ft, 10.0)
    tt = np.full_like(ft, 55.0)

    default_ft, default_pt, default_tt = scanner.thin(ft, pt, tt)
    reference_ft, reference_pt, reference_tt = scanner.thin(ft, pt, tt, mode="reference")

    for array in (default_ft, default_pt, default_tt):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert float(default_ft.min()) >= 0.0
    assert float(default_ft.max()) <= 1.0
    np.testing.assert_array_equal(default_ft, reference_ft)
    np.testing.assert_array_equal(default_pt, reference_pt)
    np.testing.assert_array_equal(default_tt, reference_tt)


def test_thin_default_reference_mode_applies_edge_effect_removal() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((12, 12, 1), dtype=np.float32)
    ft[0, 6, 0] = 1.0
    ft[1, 6, 0] = 3.0
    ft[2, 6, 0] = 1.0
    pt = np.full_like(ft, 90.0)
    tt = np.full_like(ft, 90.0)

    default_ft, default_pt, default_tt = scanner.thin(
        ft,
        pt,
        tt,
        reference_sigma=0.0,
    )
    reference_ft, reference_pt, reference_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=0.0,
        remove_edge_effects=True,
    )

    assert default_ft[1, 6, 0] == np.float32(0.0)
    assert default_pt[1, 6, 0] == np.float32(0.0)
    assert default_tt[1, 6, 0] == np.float32(0.0)
    np.testing.assert_array_equal(default_ft, reference_ft)
    np.testing.assert_array_equal(default_pt, reference_pt)
    np.testing.assert_array_equal(default_tt, reference_tt)


def test_thin_reference_mode_returns_float32_arrays_and_preserves_values() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((5, 5, 1), dtype=np.float32)
    ft[2, 1, 0] = 1.0
    ft[2, 2, 0] = 3.0
    ft[2, 3, 0] = 2.0
    pt = np.full_like(ft, 10.0)
    tt = np.full_like(ft, 55.0)
    ft_before = ft.copy()
    pt_before = pt.copy()
    tt_before = tt.copy()

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=0.0,
    )

    for array in (thinned_ft, thinned_pt, thinned_tt):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    np.testing.assert_array_equal(ft, ft_before)
    np.testing.assert_array_equal(pt, pt_before)
    np.testing.assert_array_equal(tt, tt_before)
    assert thinned_ft[2, 2, 0] == np.float32(3.0)
    assert thinned_pt[2, 2, 0] == np.float32(10.0)
    assert thinned_tt[2, 2, 0] == np.float32(55.0)
    assert np.count_nonzero(thinned_ft) == 1
    np.testing.assert_array_equal(thinned_pt[thinned_ft == 0.0], 0.0)
    np.testing.assert_array_equal(thinned_tt[thinned_ft == 0.0], 0.0)


def test_thin_reference_mode_can_keep_boundary_artifact_for_diagnostics() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((12, 12, 1), dtype=np.float32)
    ft[0, 6, 0] = 1.0
    ft[1, 6, 0] = 3.0
    ft[2, 6, 0] = 1.0
    pt = np.full_like(ft, 90.0)
    tt = np.full_like(ft, 90.0)

    cleaned_ft, cleaned_pt, cleaned_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=0.0,
        remove_edge_effects=True,
    )
    diagnostic_ft, diagnostic_pt, diagnostic_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=0.0,
        remove_edge_effects=False,
    )

    assert cleaned_ft[1, 6, 0] == np.float32(0.0)
    assert cleaned_pt[1, 6, 0] == np.float32(0.0)
    assert cleaned_tt[1, 6, 0] == np.float32(0.0)
    assert diagnostic_ft[1, 6, 0] == np.float32(3.0)
    assert diagnostic_pt[1, 6, 0] == np.float32(90.0)
    assert diagnostic_tt[1, 6, 0] == np.float32(90.0)


def test_thin_reference_mode_preserves_interior_sample_with_edge_effect_removal() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((12, 12, 1), dtype=np.float32)
    ft[5, 6, 0] = 1.0
    ft[6, 6, 0] = 3.0
    ft[7, 6, 0] = 1.0
    pt = np.full_like(ft, 90.0)
    tt = np.full_like(ft, 90.0)

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=0.0,
        remove_edge_effects=True,
    )

    assert thinned_ft[6, 6, 0] == np.float32(3.0)
    assert thinned_pt[6, 6, 0] == np.float32(90.0)
    assert thinned_tt[6, 6, 0] == np.float32(90.0)


def test_thin_reference_mode_matches_smoothed_value_helper_mask() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((7, 7, 1), dtype=np.float32)
    ft[3, 3, 0] = 10.0
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 55.0)
    expected_ft, keep = reference_like_3d_thin_values(
        ft,
        pt,
        sigma=1.0,
        reinforce_vertical=False,
    )
    expected_ft, expected_pt, expected_tt, keep = remove_reference_edge_effects_3d(
        expected_ft,
        pt,
        tt,
    )

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=1.0,
    )

    np.testing.assert_allclose(thinned_ft, expected_ft)
    np.testing.assert_array_equal(thinned_pt, expected_pt)
    np.testing.assert_array_equal(thinned_tt, expected_tt)
    np.testing.assert_array_equal(thinned_pt[~keep], 0.0)
    np.testing.assert_array_equal(thinned_tt[~keep], 0.0)


def test_thin_normal_mode_ignores_remove_edge_effects_flag() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((12, 12, 1), dtype=np.float32)
    ft[0, 6, 0] = 1.0
    ft[1, 6, 0] = 3.0
    ft[2, 6, 0] = 1.0
    pt = np.full_like(ft, 90.0)
    tt = np.full_like(ft, 90.0)

    enabled = scanner.thin(
        ft,
        pt,
        tt,
        mode="normal",
        remove_edge_effects=True,
    )
    disabled = scanner.thin(
        ft,
        pt,
        tt,
        mode="normal",
        remove_edge_effects=False,
    )

    for enabled_array, disabled_array in zip(enabled, disabled):
        np.testing.assert_array_equal(enabled_array, disabled_array)


def test_thin_rejects_invalid_mode() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((3, 3, 1), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="mode"):
        scanner.thin(ft, pt, tt, mode="bad")


@pytest.mark.parametrize("mode", ["normal", "reference"])
@pytest.mark.parametrize(
    ("ft", "pt", "tt", "message"),
    [
        (
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3, 1), dtype=np.float32),
            np.zeros((3, 3, 1), dtype=np.float32),
            "3D array",
        ),
        (
            np.array([[[0.0, np.nan]]], dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            "ft",
        ),
        (
            np.zeros((3, 3, 1), dtype=np.float32),
            np.zeros((3, 4, 1), dtype=np.float32),
            np.zeros((3, 3, 1), dtype=np.float32),
            "shapes must match",
        ),
    ],
)
def test_thin_validates_inputs_for_modes(
    mode: str,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    message: str,
) -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.thin(ft, pt, tt, mode=mode)


def test_thin_default_uses_reference_45_degree_diagonal() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    ft = np.zeros((4, 4, 1), dtype=np.float32)
    ft[1, 1, 0] = 1.0
    ft[0, 0, 0] = 3.0
    pt = np.full_like(ft, 45.0)
    tt = np.full_like(ft, 90.0)

    reference_ft, reference_pt, reference_tt = scanner.thin(
        ft,
        pt,
        tt,
        reference_sigma=0.0,
    )

    assert reference_ft[1, 1, 0] == np.float32(1.0)
    assert reference_pt[1, 1, 0] == np.float32(45.0)
    assert reference_tt[1, 1, 0] == np.float32(90.0)


def test_scan_output_feeds_voting_and_thinning_on_small_planar_volume() -> None:
    true_phi = 0.0
    true_theta = 90.0
    image, distance = _low_planarity_fault(
        true_phi,
        true_theta,
        shape=(15, 15, 15),
        width=1.0,
    )
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)

    ft, pt, tt = scanner.scan(true_phi, true_phi, true_theta, true_theta, image)
    fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    fvt = voter.thin(fv, vp, vt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    for array in (ft, pt, tt, fv, vp, vt, fvt):
        assert array.shape == image.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert fv.min() >= -1e-6
    assert fv.max() <= 1.0 + 1e-6
    assert fv.max() > 0.0
    assert fvt.max() > 0.0

    near_plane = np.abs(distance) <= 1.0
    far_from_plane = np.abs(distance) >= 5.0
    assert float(fv[near_plane].mean()) > float(fv[far_from_plane].mean())
    max_samples = fvt == fvt.max()
    assert float(np.mean(np.abs(distance[max_samples]))) <= 1.0
    assert not np.any(max_samples & far_from_plane)
    for first_array, second_array in zip((fv, vp, vt), second):
        np.testing.assert_array_equal(first_array, second_array)


def _planar_gaussian_fault(
    phi: float,
    theta: float,
    *,
    shape: tuple[int, int, int] = (40, 42, 44),
    width: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    n3, n2, n1 = shape
    i3, i2, i1 = np.indices(shape, dtype=np.float32)
    center1 = np.float32(0.5 * (n1 - 1))
    center2 = np.float32(0.5 * (n2 - 1))
    center3 = np.float32(0.5 * (n3 - 1))
    w1, w2, w3 = fault_normal_vector_from_strike_and_dip(phi, theta)
    distance = w1 * (i1 - center1) + w2 * (i2 - center2) + w3 * (i3 - center3)
    image = np.exp(-0.5 * (distance / np.float32(width)) ** 2)
    return image.astype(np.float32, copy=False), distance.astype(np.float32, copy=False)


def _low_planarity_fault(
    phi: float,
    theta: float,
    *,
    shape: tuple[int, int, int] = (40, 42, 44),
    width: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    high_likelihood, distance = _planar_gaussian_fault(phi, theta, shape=shape, width=width)
    planarity = np.float32(1.0) - high_likelihood
    return planarity.astype(np.float32, copy=False), distance
