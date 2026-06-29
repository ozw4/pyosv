import numpy as np
import pytest

from pyosv.dp import (
    accumulate_forward_2d,
    backtrack_reverse_2d,
    find_path_2d,
    shift_range,
    smooth_path_1d,
    strain_to_bstrain,
    update_shift_ranges,
    validate_cost_2d,
)


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


def test_update_shift_ranges_uses_fixed_reference_zero_radius() -> None:
    lmins, lmaxs = update_shift_ranges(ru=5, rv=3)

    np.testing.assert_array_equal(lmins, np.array([-3, 0, 0, 0, 0, 0, -3]))
    np.testing.assert_array_equal(lmaxs, np.array([3, 0, 0, 0, 0, 0, 3]))


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


def test_accumulate_and_backtrack_follow_straight_valley() -> None:
    cost = _valley_cost(np.zeros(12, dtype=np.float32), lmin=-3, nl=7)

    accumulated = accumulate_forward_2d(cost, bstrain=1)
    path = backtrack_reverse_2d(accumulated, cost, lmin=-3, bstrain=1)

    assert accumulated.shape == cost.shape
    assert path.shape == (12,)
    np.testing.assert_allclose(path, 0.0, atol=0.01)
    assert np.isfinite(path).all()


def test_find_path_2d_horizontal_valley_returns_constant_lag() -> None:
    expected = np.full(16, 2.0, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-4, nl=9)

    path = find_path_2d(cost, lmin=-4, bstrain=1, attribute_smoothing=0)

    assert path.shape == expected.shape
    assert path.dtype == np.float32
    np.testing.assert_allclose(path, expected, atol=0.01)


def test_find_path_2d_lower_boundary_valley_can_start_at_lag_zero() -> None:
    expected = np.full(12, -3.0, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-3, nl=7)

    path = find_path_2d(cost, lmin=-3, bstrain=1, attribute_smoothing=0)

    np.testing.assert_allclose(path, expected, atol=0.01)


def test_find_path_2d_linear_sloping_valley_within_strain_limit() -> None:
    expected = np.linspace(-2.0, 2.0, 21, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-3, nl=7)

    path = find_path_2d(cost, lmin=-3, bstrain=2, attribute_smoothing=0)

    assert np.max(np.abs(np.diff(path))) <= 0.5
    assert np.mean(np.abs(path - expected)) <= 0.2
    assert np.max(np.abs(path - expected)) <= 0.5


def test_find_path_2d_linear_sloping_valley_beyond_strain_limit_is_constrained() -> None:
    expected = np.array([-2, -2, -2, -2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
    cost = _valley_cost(expected, lmin=-2, nl=5)

    path = find_path_2d(cost, lmin=-2, bstrain=4, attribute_smoothing=0)

    assert np.max(np.abs(np.diff(path))) <= 0.25
    assert np.mean(np.abs(path - expected)) > 1.0
    assert np.any(np.abs(path - expected) > 1.0)


def test_find_path_2d_noisy_valley_with_smoothing_stays_near_valley() -> None:
    expected = np.full(48, 1.0, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-4, nl=9)
    rng = np.random.default_rng(20240629)
    noisy_cost = cost + rng.normal(0.0, 0.75, size=cost.shape).astype(np.float32)

    path = find_path_2d(
        noisy_cost,
        lmin=-4,
        bstrain=2,
        attribute_smoothing=1,
        path_smoothing=1.0,
    )

    assert path.shape == expected.shape
    assert np.isfinite(path).all()
    assert np.mean(np.abs(path - expected)) <= 0.2
    assert np.max(np.abs(path - expected)) <= 0.5


def test_find_path_2d_all_equal_cost_tie_breaks_to_center_lag() -> None:
    cost = np.zeros((11, 7), dtype=np.float32)

    path = find_path_2d(cost, lmin=-3, bstrain=3, attribute_smoothing=0)

    np.testing.assert_array_equal(path, np.zeros(11, dtype=np.float32))


def test_backtrack_flat_cost_prefers_center_lag() -> None:
    cost = np.zeros((8, 5), dtype=np.float32)

    accumulated = accumulate_forward_2d(cost, bstrain=2)
    path = backtrack_reverse_2d(accumulated, cost, lmin=-2, bstrain=2)

    np.testing.assert_allclose(path, 0.0)


def test_backtrack_boundary_same_step_does_not_skip_intermediate_decisions() -> None:
    accumulated = np.full((6, 3), 50.0, dtype=np.float32)
    accumulated[5] = [0.0, 10.0, 20.0]
    accumulated[4, 0] = 0.0
    accumulated[3, 0] = 10.0
    accumulated[2, 0] = 10.0
    accumulated[2, 1] = 10.0
    accumulated[1, 0] = 10.0
    accumulated[1, 1] = 0.0
    accumulated[0, 0] = 10.0
    accumulated[0, 1] = 0.0
    accumulated[0, 2] = 10.0
    cost = np.zeros_like(accumulated)

    path = backtrack_reverse_2d(accumulated, cost, lmin=0, bstrain=3)

    np.testing.assert_allclose(
        path,
        np.array([1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0], dtype=np.float32),
    )


def test_backtrack_respects_bstrain_slope_limit() -> None:
    target = np.array([-2, -2, -2, -2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
    cost = _valley_cost(target, lmin=-2, nl=5)

    path = find_path_2d(cost, lmin=-2, bstrain=4, attribute_smoothing=0)

    assert np.max(np.abs(np.diff(path))) <= 0.25


def test_find_path_2d_restores_straight_valley() -> None:
    expected = np.full(15, 2.0, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-4, nl=9)

    path = find_path_2d(cost, lmin=-4, bstrain=1, attribute_smoothing=1)

    assert path.shape == (15,)
    np.testing.assert_allclose(path, expected, atol=0.01)
    assert np.isfinite(path).all()


def test_find_path_2d_does_not_modify_input_cost() -> None:
    cost = _valley_cost(np.zeros(10, dtype=np.float32), lmin=-2, nl=5)
    original = cost.copy()

    find_path_2d(cost, lmin=-2, bstrain=1, attribute_smoothing=2)

    np.testing.assert_array_equal(cost, original)


def test_find_path_2d_allows_zero_attribute_smoothing() -> None:
    expected = np.full(9, -1.0, dtype=np.float32)
    cost = _valley_cost(expected, lmin=-3, nl=7)

    path = find_path_2d(cost, lmin=-3, bstrain=1, attribute_smoothing=0)

    np.testing.assert_allclose(path, expected, atol=0.01)


def test_find_path_2d_path_smoothing_zero_preserves_unsmoothed_path() -> None:
    expected = np.array([-1, -1, -1, 0, 1, 1, 1], dtype=np.float32)
    cost = _valley_cost(expected, lmin=-2, nl=5)

    direct = find_path_2d(cost, lmin=-2, bstrain=1, attribute_smoothing=0)
    unsmoothed = find_path_2d(
        cost,
        lmin=-2,
        bstrain=1,
        attribute_smoothing=0,
        path_smoothing=0.0,
    )

    np.testing.assert_array_equal(unsmoothed, direct)


def test_smooth_path_1d_reduces_abrupt_changes() -> None:
    path = np.array([0, 0, 0, 4, 4, 4], dtype=np.float32)

    smoothed = smooth_path_1d(path, 1.0, bstrain=1)

    assert smoothed.shape == path.shape
    assert np.max(np.abs(np.diff(smoothed))) < np.max(np.abs(np.diff(path)))
    assert np.isfinite(smoothed).all()


def test_find_path_2d_path_smoothing_reduces_abrupt_changes() -> None:
    expected = np.array([-2, -2, -2, 2, 2, 2], dtype=np.float32)
    cost = _valley_cost(expected, lmin=-2, nl=5)

    unsmoothed = find_path_2d(
        cost,
        lmin=-2,
        bstrain=1,
        attribute_smoothing=0,
        path_smoothing=0.0,
    )
    smoothed = find_path_2d(
        cost,
        lmin=-2,
        bstrain=1,
        attribute_smoothing=0,
        path_smoothing=1.0,
    )

    assert np.max(np.abs(np.diff(smoothed))) < np.max(np.abs(np.diff(unsmoothed)))


def _valley_cost(path: np.ndarray, *, lmin: int, nl: int) -> np.ndarray:
    lags = lmin + np.arange(nl, dtype=np.float32)
    return (lags[None, :] - path[:, None]) ** 2
