import numpy as np
import pytest

from pyosv.thinning3d import reference_like_3d_nms_mask


def test_reference_like_3d_nms_mask_validates_matching_3d_shapes() -> None:
    values = np.zeros((3, 4, 2), dtype=np.float32)
    strike = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="3D array"):
        reference_like_3d_nms_mask(values, strike)

    with pytest.raises(ValueError, match="shapes must match"):
        reference_like_3d_nms_mask(values, np.zeros((3, 5, 2), dtype=np.float32))


@pytest.mark.parametrize(
    ("values", "strike", "message"),
    [
        (
            np.array([[[0.0, np.nan]]], dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            "values",
        ),
        (
            np.zeros((1, 1, 2), dtype=np.float32),
            np.array([[[0.0, np.inf]]], dtype=np.float32),
            "strike",
        ),
        (
            np.array([[["bad"]]], dtype=object),
            np.zeros((1, 1, 1), dtype=np.float32),
            "numeric finite",
        ),
    ],
)
def test_reference_like_3d_nms_mask_rejects_non_finite_inputs(
    values: np.ndarray,
    strike: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reference_like_3d_nms_mask(values, strike)


def test_reference_like_3d_nms_mask_preserves_inputs_and_returns_bool_shape() -> None:
    values = np.zeros((5, 5, 2), dtype=np.float32)
    strike = np.full_like(values, 30.0)
    values[2, 2, 1] = 1.0
    values_before = values.copy()
    strike_before = strike.copy()

    mask = reference_like_3d_nms_mask(values, strike)

    assert mask.dtype == np.bool_
    assert mask.shape == values.shape
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(strike, strike_before)


def test_reference_like_3d_nms_mask_horizontal_bin_keeps_i2_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.zeros_like(values)
    values[2, 2, 0] = 3.0
    values[2, 1, 0] = 1.0
    values[2, 3, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[2, 1, 0]
    assert not mask[2, 3, 0]


def test_reference_like_3d_nms_mask_positive_diagonal_bin_keeps_diagonal_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 45.0)
    values[2, 2, 0] = 3.0
    values[1, 1, 0] = 1.0
    values[3, 3, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 1, 0]
    assert not mask[3, 3, 0]


def test_reference_like_3d_nms_mask_vertical_bin_keeps_i3_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    values[2, 2, 0] = 3.0
    values[1, 2, 0] = 1.0
    values[3, 2, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 2, 0]
    assert not mask[3, 2, 0]


def test_reference_like_3d_nms_mask_negative_diagonal_bin_keeps_diagonal_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 135.0)
    values[2, 2, 0] = 3.0
    values[1, 3, 0] = 1.0
    values[3, 1, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 3, 0]
    assert not mask[3, 1, 0]


@pytest.mark.parametrize("strike_value", [0.0, 45.0, 90.0, 135.0])
def test_reference_like_3d_nms_mask_does_not_retain_boundary_samples(
    strike_value: float,
) -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    values[0, 0, 0] = 10.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert not mask[0, 0, 0]


def test_reference_like_3d_nms_mask_constant_volume_has_no_strict_maxima() -> None:
    values = np.ones((5, 5, 2), dtype=np.float32)
    strike = np.full_like(values, 45.0)

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert not mask.any()


def test_reference_like_3d_nms_mask_non_strict_allows_flat_interior() -> None:
    values = np.ones((3, 3, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0, strict=False)

    assert mask[1, 1, 0]
    assert mask.sum() == 3
