import numpy as np
import pytest

from pyosv.interp import sample2, sample3


def test_sample2_constant_image_returns_constant() -> None:
    image = np.full((4, 5), 7.25, dtype=np.float32)
    x1 = np.array([0.0, 1.5, 4.0, -2.0], dtype=np.float32)
    x2 = np.array([0.0, 2.5, 3.0, 1.0], dtype=np.float32)

    sampled = sample2(image, x1, x2)

    assert sampled.dtype == np.float32
    np.testing.assert_allclose(sampled, np.full(x1.shape, 7.25, dtype=np.float32))


def test_sample2_integer_coordinates_match_ramp_values() -> None:
    n2, n1 = 4, 5
    i2, i1 = np.indices((n2, n1), dtype=np.float32)
    image = 10.0 * i2 + i1
    x1 = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    x2 = np.array([0.0, 1.0, 3.0], dtype=np.float32)

    sampled = sample2(image, x1, x2)

    np.testing.assert_allclose(sampled, np.array([0.0, 12.0, 34.0], dtype=np.float32))


def test_sample2_midpoint_coordinates_use_linear_interpolation() -> None:
    n2, n1 = 4, 5
    i2, i1 = np.indices((n2, n1), dtype=np.float32)
    image = 10.0 * i2 + i1

    sampled = sample2(image, 1.5, 2.25)

    assert isinstance(sampled, float)
    assert sampled == pytest.approx(24.0)


def test_sample3_constant_volume_returns_constant() -> None:
    volume = np.full((3, 4, 5), -2.5, dtype=np.float32)
    x1 = np.array([[0.0, 1.5], [4.0, 3.0]], dtype=np.float32)
    x2 = np.array([[0.0, 2.5], [1.0, 3.0]], dtype=np.float32)
    x3 = np.array([[0.0, 1.5], [2.0, -1.0]], dtype=np.float32)

    sampled = sample3(volume, x1, x2, x3)

    assert sampled.dtype == np.float32
    np.testing.assert_allclose(sampled, np.full(x1.shape, -2.5, dtype=np.float32))


def test_sample3_integer_coordinates_match_ramp_values() -> None:
    n3, n2, n1 = 3, 4, 5
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    volume = 100.0 * i3 + 10.0 * i2 + i1
    x1 = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    x2 = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    x3 = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    sampled = sample3(volume, x1, x2, x3)

    np.testing.assert_allclose(sampled, np.array([0.0, 112.0, 234.0], dtype=np.float32))


def test_scalar_and_array_inputs_have_expected_return_types() -> None:
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    scalar = sample2(image, 1.0, 2.0)
    array = sample2(image, np.array([1.0, 2.0], dtype=np.float32), 2.0)

    assert isinstance(scalar, float)
    assert isinstance(array, np.ndarray)
    assert array.shape == (2,)
    assert array.dtype == np.float32


def test_shape_conventions_are_axis_ordered() -> None:
    image = np.zeros((2, 3), dtype=np.float32)
    image[1, 2] = 9.0
    volume = np.zeros((2, 3, 4), dtype=np.float32)
    volume[1, 2, 3] = 11.0

    assert sample2(image, 2.0, 1.0) == pytest.approx(9.0)
    assert sample3(volume, 3.0, 2.0, 1.0) == pytest.approx(11.0)


def test_sample_functions_reject_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match=r"shape \(n2, n1\)"):
        sample2(np.zeros((2, 3, 4), dtype=np.float32), 0.0, 0.0)

    with pytest.raises(ValueError, match=r"shape \(n3, n2, n1\)"):
        sample3(np.zeros((2, 3), dtype=np.float32), 0.0, 0.0, 0.0)
