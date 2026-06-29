import numpy as np
import pytest

from pyosv.filters import smooth1d, smooth2d, smooth3d
from pyosv.interp import rotate2d, sample2, sample3, warp2d


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


def test_warp2d_identity_coordinates_reproduce_input() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    x2, x1 = np.indices(image.shape, dtype=np.float32)

    warped = warp2d(image, x1, x2)

    assert warped.shape == image.shape
    assert warped.dtype == np.float32
    np.testing.assert_allclose(warped, image)


def test_warp2d_constant_image_returns_constant() -> None:
    image = np.full((4, 5), 3.5, dtype=np.float32)
    x1 = np.array([[0.0, 1.5], [4.0, -2.0]], dtype=np.float32)
    x2 = np.array([[0.0, 2.5], [3.0, 1.0]], dtype=np.float32)

    warped = warp2d(image, x1, x2)

    assert warped.shape == x1.shape
    assert warped.dtype == np.float32
    np.testing.assert_allclose(warped, np.full(x1.shape, 3.5, dtype=np.float32))


def test_rotate2d_reshape_false_preserves_input_shape() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    rotated = rotate2d(image, 30.0, reshape=False)

    assert rotated.shape == image.shape
    assert rotated.dtype == np.float32


def test_rotate2d_constant_image_returns_constant() -> None:
    image = np.full((5, 6), -4.25, dtype=np.float32)

    rotated = rotate2d(image, 35.0)

    assert rotated.shape == image.shape
    assert rotated.dtype == np.float32
    np.testing.assert_allclose(rotated, image)


def test_rotate2d_zero_degrees_is_near_identity() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    rotated = rotate2d(image, 0.0)

    assert rotated.shape == image.shape
    assert rotated.dtype == np.float32
    np.testing.assert_allclose(rotated, image, atol=1.0e-6)


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


def test_smooth_constant_arrays_remain_constant() -> None:
    line = np.full(9, 4.5, dtype=np.float32)
    image = np.full((5, 6), -2.25, dtype=np.float32)
    volume = np.full((4, 5, 6), 1.75, dtype=np.float32)

    np.testing.assert_allclose(smooth1d(line, 1.5), line)
    np.testing.assert_allclose(smooth2d(image, 1.5), image)
    np.testing.assert_allclose(smooth3d(volume, 1.5), volume)


def test_smooth_impulse_spreads_to_neighbors() -> None:
    line = np.zeros(9, dtype=np.float32)
    line[4] = 1.0

    smoothed = smooth1d(line, 1.0)

    assert smoothed[4] < line[4]
    assert smoothed[3] > 0.0
    assert smoothed[5] > 0.0
    assert np.isfinite(smoothed).all()


def test_smooth_sigma_zero_returns_equal_copy() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)

    smoothed = smooth2d(image, 0.0)

    assert smoothed is not image
    np.testing.assert_array_equal(smoothed, image)


def test_smooth_does_not_modify_input() -> None:
    image = np.zeros((7, 7), dtype=np.float32)
    image[3, 3] = 1.0
    original = image.copy()

    smooth2d(image, 1.0)

    np.testing.assert_array_equal(image, original)


def test_smooth_preserves_shapes_for_1d_2d_and_3d() -> None:
    line = np.arange(8, dtype=np.float32)
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

    assert smooth1d(line, 0.75).shape == line.shape
    assert smooth2d(image, (0.75, 1.25)).shape == image.shape
    assert smooth3d(volume, (0.5, 0.75, 1.25)).shape == volume.shape


def test_smooth_float32_input_returns_float32() -> None:
    line = np.arange(8, dtype=np.float32)
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

    assert smooth1d(line, 1.0).dtype == np.float32
    assert smooth2d(image, 1.0).dtype == np.float32
    assert smooth3d(volume, 1.0).dtype == np.float32
