import numpy as np
import pytest

from pyosv.dp import shift_range, strain_to_bstrain, update_shift_ranges, validate_cost_2d


def test_strain_to_bstrain_matches_reference_spacing() -> None:
    assert strain_to_bstrain(0.25) == 4
    assert strain_to_bstrain(1.0) == 1


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_strain_to_bstrain_rejects_invalid_strain(strain_max: float) -> None:
    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        strain_to_bstrain(strain_max)


def test_shift_range_returns_lag_bounds_and_count() -> None:
    assert shift_range(ru=3) == (-3, 3, 7)


def test_update_shift_ranges_shapes_match_strike_shift_count() -> None:
    lmins, lmaxs = update_shift_ranges(ru=3, rv=4)

    assert lmins.shape == (9,)
    assert lmaxs.shape == (9,)


def test_update_shift_ranges_matches_reference_semantics() -> None:
    lmins, lmaxs = update_shift_ranges(ru=3, rv=4)

    np.testing.assert_array_equal(lmins, np.array([-3, -3, 0, 0, 0, 0, 0, -3, -3]))
    np.testing.assert_array_equal(lmaxs, np.array([3, 3, 0, 0, 0, 0, 0, 3, 3]))


def test_validate_cost_2d_accepts_finite_array_as_float32() -> None:
    cost = np.arange(6, dtype=np.float64).reshape(2, 3)

    validated = validate_cost_2d(cost)

    assert validated.shape == (2, 3)
    assert validated.dtype == np.float32
    np.testing.assert_allclose(validated, cost.astype(np.float32))


def test_validate_cost_2d_preserves_float32_array_without_copy() -> None:
    cost = np.arange(6, dtype=np.float32).reshape(2, 3)

    validated = validate_cost_2d(cost)

    assert validated is cost


@pytest.mark.parametrize(
    "cost",
    [
        np.zeros(3, dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, np.inf]], dtype=np.float32),
    ],
)
def test_validate_cost_2d_rejects_invalid_inputs(cost: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_cost_2d(cost)
