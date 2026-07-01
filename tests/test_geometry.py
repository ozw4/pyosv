import numpy as np
import pytest

from pyosv.geometry import (
    cross_product,
    fault_dip_from_dip_vector,
    fault_dip_from_normal_vector,
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_from_dip_vector,
    fault_strike_from_normal_vector,
    fault_strike_from_strike_vector,
    fault_strike_vector_from_strike_and_dip,
    range180,
    range360,
    strike_and_dip_from_normal,
)


def test_range360_scalar_boundaries() -> None:
    assert range360(0.0) == 0.0
    assert range360(360.0) == 0.0
    assert range360(-1.0) == 359.0
    assert range360(720.0) == 0.0


def test_range180_scalar_boundaries() -> None:
    assert range180(180.0) == 180.0
    assert range180(-180.0) == -180.0
    assert range180(540.0) == 180.0
    assert range180(-540.0) == -180.0
    assert range180(181.0) == -179.0
    assert range180(-181.0) == 179.0


def test_scalar_inputs_return_python_float() -> None:
    assert isinstance(range360(np.float32(-1.0)), float)
    assert isinstance(range180(np.float32(181.0)), float)


def test_range360_array_input() -> None:
    angles = np.array([-720.0, -1.0, 0.0, 360.0, 721.0], dtype=np.float32)

    wrapped = range360(angles)

    assert isinstance(wrapped, np.ndarray)
    np.testing.assert_allclose(wrapped, np.array([0.0, 359.0, 0.0, 0.0, 1.0]))


def test_range180_array_input() -> None:
    angles = np.array([-540.0, -181.0, -180.0, 180.0, 181.0, 540.0], dtype=np.float32)

    wrapped = range180(angles)

    assert isinstance(wrapped, np.ndarray)
    np.testing.assert_allclose(wrapped, np.array([-180.0, 179.0, -180.0, 180.0, -179.0, 180.0]))


def test_large_finite_angles_remain_finite() -> None:
    angles = np.array([-1.0e6, 1.0e6, -123456.0, 123456.0], dtype=np.float32)

    wrapped360 = range360(angles)
    wrapped180 = range180(angles)

    assert np.isfinite(wrapped360).all()
    assert np.isfinite(wrapped180).all()
    assert ((0.0 <= wrapped360) & (wrapped360 < 360.0)).all()
    assert ((-180.0 <= wrapped180) & (wrapped180 <= 180.0)).all()


def test_fault_vectors_match_reference_formulas() -> None:
    u = fault_dip_vector_from_strike_and_dip(0.0, 30.0)
    v = fault_strike_vector_from_strike_and_dip(90.0, 60.0)
    w = fault_normal_vector_from_strike_and_dip(180.0, 90.0)

    assert u.dtype == np.float32
    assert v.dtype == np.float32
    assert w.dtype == np.float32
    assert u.shape == (3,)
    assert v.shape == (3,)
    assert w.shape == (3,)
    np.testing.assert_allclose(u, np.array([0.5, 0.8660254, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(v, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(w, np.array([0.0, -1.0, 0.0], dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("phi", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("theta", [30.0, 60.0, 65.0, 80.0, 90.0])
def test_fault_vectors_are_unit_orthogonal_and_cross_related(phi: float, theta: float) -> None:
    # Vector components are ordered by image axes as (i1, i2, i3).
    u = fault_dip_vector_from_strike_and_dip(phi, theta)
    v = fault_strike_vector_from_strike_and_dip(phi, theta)
    w = fault_normal_vector_from_strike_and_dip(phi, theta)

    assert u.dtype == np.float32
    assert v.dtype == np.float32
    assert w.dtype == np.float32
    assert np.isfinite(u).all()
    assert np.isfinite(v).all()
    assert np.isfinite(w).all()

    np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-6)

    np.testing.assert_allclose(np.dot(u, v), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(u, w), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(v, w), 0.0, atol=1e-6)

    np.testing.assert_allclose(cross_product(v, w), u, atol=1e-6)
    np.testing.assert_allclose(cross_product(w, u), v, atol=1e-6)
    np.testing.assert_allclose(cross_product(u, v), w, atol=1e-6)


@pytest.mark.parametrize(
    ("phi", "expected_dip", "expected_strike", "expected_normal"),
    [
        (0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        (90.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        (180.0, [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
        (270.0, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ],
)
def test_fault_vectors_cardinal_vertical_dip_sign_convention(
    phi: float,
    expected_dip: list[float],
    expected_strike: list[float],
    expected_normal: list[float],
) -> None:
    # Vector components are ordered by image axes as (i1, i2, i3).
    theta = 90.0

    dip = fault_dip_vector_from_strike_and_dip(phi, theta)
    strike = fault_strike_vector_from_strike_and_dip(phi, theta)
    normal = fault_normal_vector_from_strike_and_dip(phi, theta)

    np.testing.assert_allclose(dip, np.array(expected_dip, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(strike, np.array(expected_strike, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(normal, np.array(expected_normal, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("phi", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("theta", [30.0, 60.0, 65.0, 80.0, 90.0])
def test_fault_angle_recovery_from_vectors(phi: float, theta: float) -> None:
    u = fault_dip_vector_from_strike_and_dip(phi, theta)
    v = fault_strike_vector_from_strike_and_dip(phi, theta)
    w = fault_normal_vector_from_strike_and_dip(phi, theta)

    np.testing.assert_allclose(fault_strike_from_dip_vector(u), phi, atol=1e-5)
    np.testing.assert_allclose(fault_dip_from_dip_vector(u), theta, atol=1e-5)
    np.testing.assert_allclose(fault_strike_from_strike_vector(v), phi, atol=1e-5)

    phi_from_w = fault_strike_from_normal_vector(w)
    theta_from_w = fault_dip_from_normal_vector(w)
    recovered_w = fault_normal_vector_from_strike_and_dip(phi_from_w, theta_from_w)

    np.testing.assert_allclose(recovered_w, w, atol=1e-5)


@pytest.mark.parametrize(
    ("phi", "theta"),
    [
        (15.0, 65.0),
        (135.0, 80.0),
        (225.0, 65.0),
        (315.0, 80.0),
    ],
)
def test_normal_vector_round_trip_allows_reference_sign_flip(phi: float, theta: float) -> None:
    # Vector components are ordered by image axes as (i1, i2, i3).
    normal = fault_normal_vector_from_strike_and_dip(phi, theta)
    reversed_normal = -normal

    recovered_phi = fault_strike_from_normal_vector(normal)
    recovered_theta = fault_dip_from_normal_vector(normal)
    recovered_normal = fault_normal_vector_from_strike_and_dip(recovered_phi, recovered_theta)

    flipped_phi = fault_strike_from_normal_vector(reversed_normal)
    flipped_theta = fault_dip_from_normal_vector(reversed_normal)
    recovered_flipped_normal = fault_normal_vector_from_strike_and_dip(flipped_phi, flipped_theta)

    np.testing.assert_allclose(recovered_normal, normal, atol=1e-5)
    np.testing.assert_allclose(np.abs(np.dot(recovered_flipped_normal, normal)), 1.0, atol=1e-5)


def test_invalid_dip_and_normal_vectors_raise_value_error() -> None:
    with pytest.raises(ValueError, match="dip vector is not vertical"):
        fault_strike_from_dip_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="normal vector is not vertical"):
        fault_strike_from_normal_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_strike_and_dip_from_normal_returns_float32_angle_volumes() -> None:
    u1 = np.array(
        [[[-0.5, -0.5], [0.5, 0.5]]],
        dtype=np.float32,
    )
    u2 = np.array(
        [[[0.8660254, 0.0], [-0.8660254, 0.0]]],
        dtype=np.float32,
    )
    u3 = np.array(
        [[[0.0, -0.8660254], [0.0, 0.8660254]]],
        dtype=np.float32,
    )

    fp, ft = strike_and_dip_from_normal(u1, u2, u3)

    assert fp.shape == u1.shape
    assert ft.shape == u1.shape
    assert fp.dtype == np.float32
    assert ft.dtype == np.float32
    np.testing.assert_allclose(
        fp,
        np.array([[[0.0, 90.0], [0.0, 90.0]]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(ft, np.full_like(u1, 60.0), atol=1e-5)


def test_strike_and_dip_from_normal_matches_scalar_geometry_helpers() -> None:
    # Volumes are stored as (n3, n2, n1), with vector components (i1, i2, i3).
    normal = fault_normal_vector_from_strike_and_dip(270.0, 30.0)
    u1 = np.full((2, 1, 3), normal[0], dtype=np.float32)
    u2 = np.full_like(u1, normal[1])
    u3 = np.full_like(u1, normal[2])

    fp, ft = strike_and_dip_from_normal(u1, u2, u3)

    np.testing.assert_allclose(fp, np.full_like(u1, 270.0), atol=1e-5)
    np.testing.assert_allclose(ft, np.full_like(u1, 30.0), atol=1e-5)


def test_strike_and_dip_from_normal_round_trips_small_volume_with_sign_flip() -> None:
    # Volumes are stored as (n3, n2, n1), with vector components (i1, i2, i3).
    shape = (2, 2, 3)
    phi = np.array(
        [
            [[0.0, 90.0, 180.0], [270.0, 15.0, 135.0]],
            [[225.0, 315.0, 45.0], [120.0, 240.0, 300.0]],
        ],
        dtype=np.float32,
    )
    theta = np.array(
        [
            [[65.0, 80.0, 90.0], [65.0, 80.0, 90.0]],
            [[65.0, 80.0, 90.0], [65.0, 80.0, 90.0]],
        ],
        dtype=np.float32,
    )
    u1 = np.empty(shape, dtype=np.float32)
    u2 = np.empty(shape, dtype=np.float32)
    u3 = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        normal = fault_normal_vector_from_strike_and_dip(float(phi[index]), float(theta[index]))
        if index[0] == 1:
            normal = -normal
        u1[index], u2[index], u3[index] = normal

    fp, ft = strike_and_dip_from_normal(u1, u2, u3)

    assert fp.shape == shape
    assert ft.shape == shape
    assert fp.dtype == np.float32
    assert ft.dtype == np.float32
    for index in np.ndindex(shape):
        recovered_normal = fault_normal_vector_from_strike_and_dip(
            float(fp[index]), float(ft[index])
        )
        input_normal = np.array([u1[index], u2[index], u3[index]], dtype=np.float32)
        np.testing.assert_allclose(np.abs(np.dot(recovered_normal, input_normal)), 1.0, atol=1e-5)


def test_strike_and_dip_from_normal_rejects_shape_mismatch() -> None:
    u1 = np.zeros((2, 3, 4), dtype=np.float32)
    u2 = np.zeros((2, 3, 5), dtype=np.float32)
    u3 = np.zeros_like(u1)

    with pytest.raises(ValueError, match="shapes must match"):
        strike_and_dip_from_normal(u1, u2, u3)


def test_strike_and_dip_from_normal_rejects_non_3d_arrays() -> None:
    u1 = np.zeros((2, 3), dtype=np.float32)
    u2 = np.zeros_like(u1)
    u3 = np.zeros_like(u1)

    with pytest.raises(ValueError, match="u1 must be a 3D array"):
        strike_and_dip_from_normal(u1, u2, u3)
