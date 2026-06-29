import numpy as np
import pytest

import pyosv
from pyosv.orient2d import FaultOrientScanner2
from pyosv.voting2d import OptimalPathVoter


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


@pytest.mark.parametrize(
    ("theta", "theta_min", "theta_max"),
    [
        (0.0, -45.0, 45.0),
        (30.0, -60.0, 60.0),
        (90.0, -90.0, 90.0),
    ],
)
def test_scan_detects_synthetic_lineament_orientation(
    theta: float,
    theta_min: float,
    theta_max: float,
) -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image, distance = _dipping_gaussian_lineament(theta)

    ft, pt = scanner.scan(theta_min, theta_max, image)

    near_line = np.abs(distance) <= 1.5
    far_from_line = np.abs(distance) >= 12.0
    assert float(np.mean(ft[near_line])) > float(np.mean(ft[far_from_line])) + 0.25

    high_likelihood = near_line & (ft >= np.percentile(ft, 90.0))
    assert np.count_nonzero(high_likelihood) > 0
    expected_pt = _voter_angle_from_feature_angle(theta)
    angle_error = _orientation_error_degrees(pt[high_likelihood], expected_pt)
    assert float(np.median(angle_error)) <= 15.0


def test_scan_outputs_voting_angle_for_down_right_diagonal_lineament() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image, distance = _dipping_gaussian_lineament(45.0, n2=48, n1=48)

    ft, pt = scanner.scan(30.0, 60.0, image)

    near_line = np.abs(distance) <= 1.5
    high_likelihood = near_line & (ft >= np.percentile(ft, 90.0))
    assert np.count_nonzero(high_likelihood) > 0
    angle_error = _orientation_error_degrees(pt[high_likelihood], 135.0)
    assert float(np.median(angle_error)) <= 10.0

    voter = OptimalPathVoter(ru=2, rv=5)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    fv, _, _ = voter.apply_voting(d=3, fm=0.45, ft=ft, pt=pt)

    far_from_line = np.abs(distance) >= 10.0
    assert float(np.mean(fv[near_line])) > float(np.mean(fv[far_from_line])) + 0.35


def test_scan_output_feeds_apply_voting_on_synthetic_lineament() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image, distance = _dipping_gaussian_lineament(25.0, n2=48, n1=64)
    ft, pt = scanner.scan(-75.0, 75.0, image)
    near_line = np.abs(distance) <= 1.5
    far_from_line = np.abs(distance) >= 10.0
    voter = OptimalPathVoter(ru=2, rv=5)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)

    fv, w1, w2 = voter.apply_voting(d=3, fm=0.45, ft=ft, pt=pt)

    assert fv.shape == image.shape
    assert w1.shape == image.shape
    assert w2.shape == image.shape
    assert fv.dtype == np.float32
    assert w1.dtype == np.float32
    assert w2.dtype == np.float32
    assert np.isfinite(fv).all()
    assert np.isfinite(w1).all()
    assert np.isfinite(w2).all()
    assert fv.max() > 0.0
    assert float(np.mean(fv[near_line])) > float(np.mean(fv[far_from_line])) + 0.35


def test_scan_dip_matches_scan() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    image, _ = _dipping_gaussian_lineament(-20.0)

    scan_ft, scan_pt = scanner.scan(-45.0, 45.0, image)
    dip_ft, dip_pt = scanner.scan_dip(-45.0, 45.0, image)

    np.testing.assert_array_equal(dip_ft, scan_ft)
    np.testing.assert_array_equal(dip_pt, scan_pt)


def test_thin_returns_float32_arrays_without_modifying_inputs() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.zeros((3, 5), dtype=np.float64)
    ft[1] = np.array([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float64)
    pt = np.full_like(ft, 90.0)
    ft_before = ft.copy()
    pt_before = pt.copy()

    thinned_ft, thinned_pt = scanner.thin(ft, pt)

    assert thinned_ft.shape == ft.shape
    assert thinned_pt.shape == pt.shape
    assert thinned_ft.dtype == np.float32
    assert thinned_pt.dtype == np.float32
    assert np.isfinite(thinned_ft).all()
    assert np.isfinite(thinned_pt).all()
    np.testing.assert_array_equal(ft, ft_before)
    np.testing.assert_array_equal(pt, pt_before)


def test_thin_narrows_broad_vertical_likelihood_ridge() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.zeros((5, 7), dtype=np.float32)
    ft[:, 2] = 1.0
    ft[:, 3] = 3.0
    ft[:, 4] = 1.0
    pt = np.full_like(ft, 90.0)

    thinned_ft, thinned_pt = scanner.thin(ft, pt)

    expected_ft = np.zeros_like(ft)
    expected_ft[:, 3] = 3.0
    expected_pt = np.zeros_like(pt)
    expected_pt[:, 3] = 90.0
    assert np.count_nonzero(thinned_ft) < np.count_nonzero(ft)
    np.testing.assert_array_equal(thinned_ft, expected_ft)
    np.testing.assert_array_equal(thinned_pt, expected_pt)


def test_thin_narrows_broad_diagonal_likelihood_ridge_with_voting_angle() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    n = 9
    i2, i1 = np.indices((n, n), dtype=np.intp)
    distance = i2 - i1
    ft = np.zeros((n, n), dtype=np.float32)
    ft[np.abs(distance) == 1] = 1.0
    ft[distance == 0] = 3.0
    pt = np.full_like(ft, 135.0)

    thinned_ft, thinned_pt = scanner.thin(ft, pt)

    expected_ft = np.zeros_like(ft)
    np.fill_diagonal(expected_ft, 3.0)
    expected_pt = np.zeros_like(pt)
    np.fill_diagonal(expected_pt, 135.0)
    assert np.count_nonzero(thinned_ft) < np.count_nonzero(ft)
    np.testing.assert_array_equal(thinned_ft, expected_ft)
    np.testing.assert_array_equal(thinned_pt, expected_pt)


def test_thin_preserves_orientation_only_where_likelihood_survives() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.zeros((5, 5), dtype=np.float32)
    ft[2, 1:4] = np.array([1.0, 4.0, 1.0], dtype=np.float32)
    pt = np.full_like(ft, 90.0)
    pt[2, 2] = 35.0

    thinned_ft, thinned_pt = scanner.thin(ft, pt)

    assert thinned_ft[2, 2] == np.float32(4.0)
    assert thinned_pt[2, 2] == np.float32(35.0)
    np.testing.assert_array_equal(thinned_pt[thinned_ft == 0.0], 0.0)


def test_thin_returns_zero_for_flat_likelihood_map() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.ones((5, 6), dtype=np.float32)
    pt = np.full_like(ft, 25.0)

    thinned_ft, thinned_pt = scanner.thin(ft, pt)

    np.testing.assert_array_equal(thinned_ft, np.zeros_like(ft))
    np.testing.assert_array_equal(thinned_pt, np.zeros_like(pt))


def test_thin_rejects_mismatched_shapes() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.zeros((2, 3), dtype=np.float32)
    pt = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shapes must match"):
        scanner.thin(ft, pt)


@pytest.mark.parametrize(
    ("ft_value", "pt_value", "message"),
    [(np.nan, 0.0, "ft"), (0.0, np.inf, "pt")],
)
def test_thin_rejects_nonfinite_inputs(
    ft_value: float,
    pt_value: float,
    message: str,
) -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)
    ft = np.zeros((3, 3), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[1, 1] = ft_value
    pt[1, 1] = pt_value

    with pytest.raises(ValueError, match=message):
        scanner.thin(ft, pt)


def test_thin_rejects_non_2d_arrays() -> None:
    scanner = FaultOrientScanner2(sigma1=2.0)

    with pytest.raises(ValueError, match="ft must be a 2D array"):
        scanner.thin(
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        )


def _orientation_error_degrees(
    actual: np.ndarray,
    expected_degrees: float,
) -> np.ndarray:
    return np.abs((actual - np.float32(expected_degrees) + 90.0) % 180.0 - 90.0)


def _voter_angle_from_feature_angle(theta_degrees: float) -> np.float32:
    return np.float32((180.0 - theta_degrees) % 180.0)


def _dipping_gaussian_lineament(
    theta_degrees: float,
    *,
    n2: int = 96,
    n1: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    x2, x1 = np.mgrid[:n2, :n1].astype(np.float32)
    x1 -= np.float32((n1 - 1) / 2.0)
    x2 -= np.float32((n2 - 1) / 2.0)

    theta_radians = np.deg2rad(theta_degrees)
    distance = x2 * np.cos(theta_radians) - x1 * np.sin(theta_radians)
    image = np.exp(-0.5 * (distance / np.float32(1.2)) ** 2)
    return image.astype(np.float32), distance.astype(np.float32)
