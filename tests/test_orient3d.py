import numpy as np
import pytest

import pyosv
from pyosv.orient3d import FaultOrientScanner3


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


def test_scan_skeleton_validates_inputs_before_not_implemented() -> None:
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    image = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="3D scan response"):
        scanner.scan(0.0, 360.0, 35.0, 85.0, image)

    with pytest.raises(ValueError, match="phi_max"):
        scanner.scan(360.0, 0.0, 35.0, 85.0, image)
