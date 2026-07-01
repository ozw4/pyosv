from pathlib import Path

import numpy as np
import pytest

import pyosv
from pyosv.geometry import fault_normal_vector_from_strike_and_dip
from pyosv.orient3d import FaultOrientScanner3
from pyosv.thinning3d import reference_like_3d_thin_values
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
    np.testing.assert_array_equal(pt, np.full(image.shape, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(tt, np.full(image.shape, 30.0, dtype=np.float32))
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()


def test_scan_reference_like_returns_float32_normalized_outputs() -> None:
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _planar_gaussian_fault(60.0, 60.0, shape=(15, 16, 17), width=1.0)

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
    image, _ = _planar_gaussian_fault(60.0, 60.0, shape=(13, 14, 15), width=1.0)

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


def test_scan_reference_like_localizes_synthetic_planar_fault_orientation() -> None:
    true_phi = 60.0
    true_theta = 60.0
    image, distance = _planar_gaussian_fault(
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

    high_likelihood = ft >= np.percentile(ft, 98.0)
    phi_error = _periodic_angle_error(pt[high_likelihood], true_phi, period=180.0)
    theta_error = np.abs(tt[high_likelihood] - np.float32(true_theta))
    assert float(np.median(phi_error)) <= 31.0
    assert float(np.median(theta_error)) <= 31.0


def test_scan_reference_like_does_not_call_default_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("scan() must not be called")

    monkeypatch.setattr(FaultOrientScanner3, "scan", fail_scan)
    scanner = FaultOrientScanner3(sigma1=1.0, sigma2=1.0)
    image, _ = _planar_gaussian_fault(60.0, 60.0, shape=(9, 10, 11), width=1.0)

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
    np.testing.assert_array_equal(pt, np.full(image.shape, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(tt, np.full(image.shape, 30.0, dtype=np.float32))
    assert np.isfinite(ft).all()
    assert np.isfinite(pt).all()
    assert np.isfinite(tt).all()


def test_scan_localizes_synthetic_planar_fault_and_recovers_orientation() -> None:
    true_phi = 45.0
    true_theta = 50.0
    image, distance = _planar_gaussian_fault(true_phi, true_theta)
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)

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
    phi_error = _periodic_angle_error(pt[high_likelihood], true_phi, period=180.0)
    theta_error = np.abs(tt[high_likelihood] - np.float32(true_theta))
    assert float(np.median(phi_error)) <= 20.0
    assert float(np.median(theta_error)) <= 20.0


def test_thin_keeps_planar_likelihood_maxima_along_fault_normal() -> None:
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

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(ft, pt, tt)
    normal_ft, normal_pt, normal_tt = scanner.thin(ft, pt, tt, mode="normal")

    for array in (thinned_ft, thinned_pt, thinned_tt):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    np.testing.assert_array_equal(normal_ft, thinned_ft)
    np.testing.assert_array_equal(normal_pt, thinned_pt)
    np.testing.assert_array_equal(normal_tt, thinned_tt)
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

    thinned_ft, thinned_pt, thinned_tt = scanner.thin(
        ft,
        pt,
        tt,
        mode="reference",
        reference_sigma=1.0,
    )

    np.testing.assert_allclose(thinned_ft, expected_ft)
    np.testing.assert_array_equal(thinned_pt[keep], pt[keep])
    np.testing.assert_array_equal(thinned_tt[keep], tt[keep])
    np.testing.assert_array_equal(thinned_pt[~keep], 0.0)
    np.testing.assert_array_equal(thinned_tt[~keep], 0.0)


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


def test_thin_reference_mode_uses_reference_45_degree_diagonal() -> None:
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
        mode="reference",
        reference_sigma=0.0,
    )

    assert reference_ft[1, 1, 0] == np.float32(1.0)
    assert reference_pt[1, 1, 0] == np.float32(45.0)
    assert reference_tt[1, 1, 0] == np.float32(90.0)


def test_scan_output_feeds_voting_and_thinning_on_small_planar_volume() -> None:
    true_phi = 0.0
    true_theta = 90.0
    image, distance = _planar_gaussian_fault(
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


def _periodic_angle_error(
    actual: np.ndarray,
    expected: float,
    *,
    period: float,
) -> np.ndarray:
    half_period = np.float32(0.5 * period)
    return np.abs((actual - np.float32(expected) + half_period) % period - half_period)
