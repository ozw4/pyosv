import numpy as np
import pytest

from pyosv.dp import (
    accumulate_2d,
    accumulate_forward_2d,
    backtrack_reverse_2d,
    find_path_2d,
    find_surface_3d,
    shift_range,
    smooth_fault_attributes_2d,
    smooth_fault_attributes_3d,
    smooth_path_1d,
    smooth_surface_2d,
    strain_to_bstrain,
    update_shift_ranges,
    update_shift_ranges_3d,
    validate_cost_2d,
    validate_cost_3d,
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


def test_validate_cost_3d_accepts_finite_array_as_float32() -> None:
    cost = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    validated = validate_cost_3d(cost)

    assert validated.shape == (2, 3, 4)
    assert validated.dtype == np.float32
    np.testing.assert_allclose(validated, cost.astype(np.float32))


def test_validate_cost_3d_preserves_float32_array_without_copy() -> None:
    cost = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    validated = validate_cost_3d(cost)

    assert validated is cost


@pytest.mark.parametrize(
    "cost",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
        np.array([[[0.0, np.nan]]], dtype=np.float32),
        np.array([[[0.0, np.inf]]], dtype=np.float32),
    ],
)
def test_validate_cost_3d_rejects_invalid_inputs(cost: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_cost_3d(cost)


def test_update_shift_ranges_3d_shapes_match_surface_shift_counts() -> None:
    lmins, lmaxs = update_shift_ranges_3d(ru=3, rv=4, rw=5)

    assert lmins.shape == (11, 9)
    assert lmaxs.shape == (11, 9)
    assert lmins.dtype == np.int32
    assert lmaxs.dtype == np.int32


def test_update_shift_ranges_3d_zeroes_offsets_within_reference_radius() -> None:
    lmins, lmaxs = update_shift_ranges_3d(ru=4, rv=3, rw=3)

    for iw in range(-3, 4):
        for iv in range(-3, 4):
            if np.sqrt(iw * iw + iv * iv) <= 2.0:
                assert lmins[iw + 3, iv + 3] == 0
                assert lmaxs[iw + 3, iv + 3] == 0


def test_update_shift_ranges_3d_uses_java_rounding_and_clips_to_ru() -> None:
    lmins, lmaxs = update_shift_ranges_3d(ru=2, rv=3, rw=3)

    assert lmins[3 + 2, 3 + 1] == -2
    assert lmaxs[3 + 2, 3 + 1] == 2
    assert lmins[3 + 3, 3 + 3] == -2
    assert lmaxs[3 + 3, 3 + 3] == 2


def test_accumulate_and_backtrack_follow_straight_valley() -> None:
    cost = _valley_cost(np.zeros(12, dtype=np.float32), lmin=-3, nl=7)

    accumulated = accumulate_forward_2d(cost, bstrain=1)
    path = backtrack_reverse_2d(accumulated, cost, lmin=-3, bstrain=1)

    assert accumulated.shape == cost.shape
    assert path.shape == (12,)
    np.testing.assert_allclose(path, 0.0, atol=0.01)
    assert np.isfinite(path).all()


@pytest.mark.parametrize(("direction", "start_index"), [(1, 0), (-1, -1)])
def test_accumulate_2d_preserves_start_row_with_negative_cost(
    direction: int,
    start_index: int,
) -> None:
    cost = np.zeros((2, 3), dtype=np.float32)
    cost[start_index, 0] = -5.0

    accumulated = accumulate_2d(cost, bstrain=1, direction=direction)

    np.testing.assert_array_equal(accumulated[start_index], cost[start_index])


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


def test_smooth_surface_2d_zero_sigmas_preserve_surface_values() -> None:
    surface = np.arange(12, dtype=np.float32).reshape(3, 4)

    smoothed = smooth_surface_2d(surface, sigma1=0.0, sigma2=0.0)

    assert smoothed.shape == surface.shape
    assert smoothed.dtype == np.float32
    assert smoothed is not surface
    np.testing.assert_array_equal(smoothed, surface)


def test_smooth_surface_2d_reduces_abrupt_changes() -> None:
    surface = np.zeros((7, 9), dtype=np.float32)
    surface[:, 4:] = 4.0

    smoothed = smooth_surface_2d(surface, sigma1=1.0, sigma2=0.0)

    assert smoothed.shape == surface.shape
    assert smoothed.dtype == np.float32
    assert np.max(np.abs(np.diff(smoothed, axis=1))) < np.max(
        np.abs(np.diff(surface, axis=1)),
    )
    assert np.isfinite(smoothed).all()


@pytest.mark.parametrize(
    ("surface", "kwargs"),
    [
        (np.zeros(3, dtype=np.float32), {}),
        (np.array([[0.0, np.nan]], dtype=np.float32), {}),
        (np.zeros((2, 3), dtype=np.float32), {"sigma1": -1.0}),
        (np.zeros((2, 3), dtype=np.float32), {"sigma2": np.inf}),
    ],
)
def test_smooth_surface_2d_rejects_invalid_inputs(
    surface: np.ndarray,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        smooth_surface_2d(surface, **kwargs)


@pytest.mark.parametrize("attribute_smoothing", [0, 1])
def test_find_surface_3d_flat_valley_returns_constant_lag(
    attribute_smoothing: int,
) -> None:
    expected = np.full((4, 16), 2.0, dtype=np.float32)
    cost = _surface_cost(expected, lmin=-4, nu=9)

    surface = find_surface_3d(
        cost,
        lmin=-4,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=attribute_smoothing,
    )

    assert surface.shape == expected.shape
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    np.testing.assert_allclose(surface, expected, atol=0.01)


def test_find_surface_3d_linear_v_valley_returns_bounded_surface() -> None:
    nw, nv, nu = 5, 21, 9
    expected_path = np.linspace(-2.0, 2.0, nv, dtype=np.float32)
    expected = np.broadcast_to(expected_path, (nw, nv)).copy()
    cost = _surface_cost(expected, lmin=-4, nu=nu)

    surface = find_surface_3d(
        cost,
        lmin=-4,
        bstrain1=2,
        bstrain2=1,
        attribute_smoothing=0,
    )

    assert np.max(np.abs(np.diff(surface, axis=1))) <= 0.5
    assert np.mean(np.abs(surface - expected)) <= 0.2
    assert np.max(np.abs(surface - expected)) <= 0.5


@pytest.mark.parametrize("attribute_smoothing", [0, 1])
def test_find_surface_3d_linear_w_valley_returns_bounded_surface(
    attribute_smoothing: int,
) -> None:
    nw, nv, nu = 21, 5, 9
    expected_row = np.linspace(-2.0, 2.0, nw, dtype=np.float32)[:, None]
    expected = np.broadcast_to(expected_row, (nw, nv)).copy()
    cost = _surface_cost(expected, lmin=-4, nu=nu)

    surface = find_surface_3d(
        cost,
        lmin=-4,
        bstrain1=1,
        bstrain2=2,
        attribute_smoothing=attribute_smoothing,
    )

    assert surface.shape == expected.shape
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    assert np.mean(np.abs(surface - expected)) <= 0.25
    assert np.max(np.abs(surface - expected)) <= 0.5


def test_find_surface_3d_surface_smoothing_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.zeros((3, 7), dtype=np.float32)
    cost = _surface_cost(expected, lmin=-2, nu=5)

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("surface smoothing should not run")

    monkeypatch.setattr("pyosv.dp.smooth_surface_2d", fail_if_called)

    surface = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )

    np.testing.assert_allclose(surface, expected, atol=0.01)


def test_find_surface_3d_surface_smoothing_reduces_abrupt_changes() -> None:
    expected = np.zeros((5, 8), dtype=np.float32)
    expected[:, 4:] = 2.0
    cost = _surface_cost(expected, lmin=-2, nu=5)

    unsmoothed = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
    )
    smoothed = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=1.0,
    )

    assert np.max(np.abs(np.diff(smoothed, axis=1))) < np.max(
        np.abs(np.diff(unsmoothed, axis=1)),
    )


def test_find_surface_3d_surface_smoothing2_reduces_abrupt_changes() -> None:
    expected = np.zeros((8, 5), dtype=np.float32)
    expected[4:] = 2.0
    cost = _surface_cost(expected, lmin=-2, nu=5)

    unsmoothed = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
    )
    smoothed = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing2=1.0,
    )

    assert np.max(np.abs(np.diff(smoothed, axis=0))) < np.max(
        np.abs(np.diff(unsmoothed, axis=0)),
    )


def test_find_surface_3d_is_deterministic() -> None:
    expected = np.zeros((4, 9), dtype=np.float32)
    expected[:, 5:] = 1.0
    cost = _surface_cost(expected, lmin=-2, nu=5)

    surface1 = find_surface_3d(cost, lmin=-2, bstrain1=2, bstrain2=2)
    surface2 = find_surface_3d(cost, lmin=-2, bstrain1=2, bstrain2=2)

    np.testing.assert_array_equal(surface1, surface2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bstrain1": 0}, "bstrain1"),
        ({"bstrain2": 0}, "bstrain2"),
        ({"attribute_smoothing": -1}, "attribute_smoothing"),
        ({"surface_smoothing1": -1.0}, "surface_smoothing1"),
        ({"surface_smoothing2": np.nan}, "surface_smoothing2"),
    ],
)
def test_find_surface_3d_rejects_invalid_parameters(
    kwargs: dict[str, object],
    match: str,
) -> None:
    cost = np.zeros((2, 3, 4), dtype=np.float32)
    params = {
        "lmin": -1,
        "bstrain1": 1,
        "bstrain2": 1,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        find_surface_3d(cost, **params)


def test_smooth_fault_attributes_3d_preserves_constant_volume() -> None:
    cost = np.full((4, 5, 3), 2.5, dtype=np.float32)

    smoothed = smooth_fault_attributes_3d(cost, bstrain1=2, bstrain2=3)

    assert smoothed.shape == cost.shape
    assert smoothed.dtype == np.float32
    assert np.isfinite(smoothed).all()
    np.testing.assert_allclose(smoothed, smoothed[0, 0, 0])


def test_smooth_fault_attributes_3d_matches_staged_2d_smoothing() -> None:
    rng = np.random.default_rng(20240629)
    cost = rng.normal(size=(3, 4, 5)).astype(np.float32)

    expected_v = np.empty_like(cost)
    for iw in range(cost.shape[0]):
        expected_v[iw] = smooth_fault_attributes_2d(cost[iw], bstrain=2)

    expected = np.empty_like(cost)
    for iv in range(cost.shape[1]):
        expected[:, iv, :] = smooth_fault_attributes_2d(expected_v[:, iv, :], bstrain=1)

    smoothed = smooth_fault_attributes_3d(cost, bstrain1=2, bstrain2=1)

    assert smoothed.dtype == np.float32
    np.testing.assert_allclose(smoothed, expected)


def test_smooth_fault_attributes_3d_keeps_synthetic_surface_within_lag_bounds() -> None:
    nw, nv, nu = 5, 7, 9
    lmin = -4
    lags = lmin + np.arange(nu, dtype=np.float32)
    w = np.arange(nw, dtype=np.float32)[:, None]
    v = np.arange(nv, dtype=np.float32)[None, :]
    surface = np.clip(0.5 * (w - 2.0) + 0.25 * (v - 3.0), -3.0, 3.0)
    cost = (lags[None, None, :] - surface[:, :, None]) ** 2

    smoothed = smooth_fault_attributes_3d(cost, bstrain1=2, bstrain2=2)
    picked_lags = lmin + np.argmin(smoothed, axis=2)

    assert smoothed.shape == cost.shape
    assert smoothed.dtype == np.float32
    assert np.isfinite(smoothed).all()
    assert picked_lags.min() >= lmin
    assert picked_lags.max() <= lmin + nu - 1


@pytest.mark.parametrize(
    "cost",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
        np.array([[[0.0, np.nan]]], dtype=np.float32),
        np.array([[[0.0, np.inf]]], dtype=np.float32),
    ],
)
def test_smooth_fault_attributes_3d_rejects_invalid_cost(cost: np.ndarray) -> None:
    with pytest.raises(ValueError):
        smooth_fault_attributes_3d(cost, bstrain1=1, bstrain2=1)


@pytest.mark.parametrize(("bstrain1", "bstrain2"), [(0, 1), (1, 0), (1.5, 1)])
def test_smooth_fault_attributes_3d_rejects_invalid_bstrain(
    bstrain1: int,
    bstrain2: int,
) -> None:
    cost = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        smooth_fault_attributes_3d(cost, bstrain1=bstrain1, bstrain2=bstrain2)


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


def _surface_cost(surface: np.ndarray, *, lmin: int, nu: int) -> np.ndarray:
    lags = lmin + np.arange(nu, dtype=np.float32)
    return (lags[None, None, :] - surface[:, :, None]) ** 2
