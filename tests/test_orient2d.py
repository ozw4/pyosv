import numpy as np
import pytest

import pyosv
from pyosv.orient2d import FaultOrientScanner2


def test_fault_orient_scanner_import_does_not_change_package_root_api() -> None:
    assert isinstance(pyosv.__version__, str)
    assert not hasattr(pyosv, "FaultOrientScanner2")


def test_theta_sampling_returns_float32_monotonic_angles_with_endpoints() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)

    thetas = scanner.theta_sampling(theta_min=-65.0, theta_max=65.0)

    assert thetas.dtype == np.float32
    assert thetas.ndim == 1
    assert np.isfinite(thetas).all()
    assert np.diff(thetas).min() > 0.0
    np.testing.assert_allclose(thetas[0], np.float32(-65.0), atol=1e-6)
    np.testing.assert_allclose(thetas[-1], np.float32(65.0), atol=1e-6)
    np.testing.assert_array_equal(
        thetas,
        scanner.theta_sampling(theta_min=-65.0, theta_max=65.0),
    )


def test_theta_sampling_single_angle_returns_one_endpoint_sample() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)

    thetas = scanner.theta_sampling(theta_min=12.5, theta_max=12.5)

    np.testing.assert_array_equal(thetas, np.array([12.5], dtype=np.float32))


@pytest.mark.parametrize("sigma1", [0.0, -1.0, np.nan, np.inf, True, "2.0"])
def test_constructor_rejects_invalid_sigma1(sigma1: object) -> None:
    with pytest.raises(ValueError, match="sigma1"):
        FaultOrientScanner2(sigma1=sigma1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("theta_min", "theta_max", "message"),
    [
        (np.nan, 65.0, "theta_min"),
        (-65.0, np.inf, "theta_max"),
        (True, 65.0, "theta_min"),
        (-65.0, "65.0", "theta_max"),
        (65.0, -65.0, "theta_max"),
    ],
)
def test_theta_sampling_rejects_invalid_angles(
    theta_min: object,
    theta_max: object,
    message: str,
) -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.theta_sampling(theta_min=theta_min, theta_max=theta_max)  # type: ignore[arg-type]


def test_validate_image_accepts_finite_2d_numeric_array_as_float32() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    validated = scanner.validate_image(image)

    assert validated.shape == (2, 2)
    assert validated.dtype == np.float32
    np.testing.assert_allclose(validated, image)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((1, 2, 3), dtype=np.float32), "2D array"),
        (np.array([[0.0, np.nan]], dtype=np.float32), "finite"),
        (np.array([[0.0, np.inf]], dtype=np.float32), "finite"),
        (np.array([["bad"]], dtype=object), "numeric finite"),
    ],
)
def test_validate_image_rejects_invalid_inputs(image: object, message: str) -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)

    with pytest.raises(ValueError, match=message):
        scanner.validate_image(image)  # type: ignore[arg-type]


def test_scan_returns_float32_arrays_matching_input_shape() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image = np.zeros((12, 10), dtype=np.float64)

    ft, pt = scanner.scan(-45.0, 45.0, image)

    assert ft.shape == image.shape
    assert pt.shape == image.shape
    assert ft.dtype == np.float32
    assert pt.dtype == np.float32
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()


def test_scan_constant_input_has_zero_finite_likelihood() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image = np.full((24, 20), 3.0, dtype=np.float32)

    ft, pt = scanner.scan(-60.0, 60.0, image)

    np.testing.assert_array_equal(ft, np.zeros_like(image, dtype=np.float32))
    assert np.isfinite(pt).all()


def test_scan_detects_dipping_synthetic_lineament() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    theta = 30.0
    image, distance = _dipping_gaussian_lineament(theta)

    ft, pt = scanner.scan(-60.0, 60.0, image)

    near_line = np.abs(distance) <= 1.5
    far_from_line = np.abs(distance) >= 12.0
    assert float(np.mean(ft[near_line])) > float(np.mean(ft[far_from_line])) + 0.25

    high_likelihood = near_line & (ft >= np.percentile(ft, 90.0))
    assert np.count_nonzero(high_likelihood) > 0
    assert float(np.median(np.abs(pt[high_likelihood] - theta))) <= 15.0


def test_scan_dip_matches_scan() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image, _ = _dipping_gaussian_lineament(-20.0)

    scan_ft, scan_pt = scanner.scan(-45.0, 45.0, image)
    dip_ft, dip_pt = scanner.scan_dip(-45.0, 45.0, image)

    np.testing.assert_array_equal(dip_ft, scan_ft)
    np.testing.assert_array_equal(dip_pt, scan_pt)


def _dipping_gaussian_lineament(theta_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    n2, n1 = 96, 128
    x2, x1 = np.mgrid[:n2, :n1].astype(np.float32)
    x1 -= np.float32((n1 - 1) / 2.0)
    x2 -= np.float32((n2 - 1) / 2.0)

    theta_radians = np.deg2rad(theta_degrees)
    distance = x2 * np.cos(theta_radians) - x1 * np.sin(theta_radians)
    image = np.exp(-0.5 * (distance / np.float32(1.2)) ** 2)
    return image.astype(np.float32), distance.astype(np.float32)
