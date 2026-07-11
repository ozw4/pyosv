import numpy as np
import pytest

from pyosv import dp
from pyosv.dp import (
    _find_surface_3d_masked,
    _project_surface_to_valid_mask_python,
    _surface_respects_masked_strain,
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


def test_legacy_private_validation_helpers_remain_available_from_facade() -> None:
    expected = {
        "_validate_direction",
        "_validate_int",
        "_validate_nonnegative_float",
        "_validate_nonnegative_int",
        "_validate_positive_int",
    }

    assert not expected.difference(vars(dp))


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


def test_find_surface_3d_reference_audit_flat_zero_lag_surface() -> None:
    # Audits OptimalSurfaceVoter.findSurface for a deterministic zero-lag valley.
    expected = np.zeros((5, 7), dtype=np.float32)
    cost = _surface_cost(expected, lmin=-3, nu=7)

    surface = find_surface_3d(
        cost,
        lmin=-3,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )

    assert surface.shape == expected.shape
    assert surface.dtype == np.float32
    np.testing.assert_array_equal(surface, expected)


def test_find_surface_3d_reference_audit_sloped_surface_is_strain_limited() -> None:
    # Audits OptimalSurfaceVoter.findSurface inverse-strain limiting on a local surface.
    preferred = np.zeros((3, 9), dtype=np.float32)
    preferred[:, 4:] = 3.0
    cost = _surface_cost(preferred, lmin=-3, nu=7)

    surface = find_surface_3d(
        cost,
        lmin=-3,
        bstrain1=2,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )

    expected = np.array(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0]] * 3,
        dtype=np.float32,
    )
    np.testing.assert_allclose(surface, expected, rtol=0.0, atol=1e-6)
    assert np.max(np.abs(np.diff(surface, axis=1))) <= 0.5


def test_find_surface_3d_reference_audit_attribute_smoothing_on_off() -> None:
    # Audits the optional smoothFaultAttributes pass used by findSurface.
    cost = np.array(
        [
            [
                [0.8506242, 0.63696164, 0.5111365, 0.26978666, 0.30782938],
                [0.04097348, 0.07524014, 0.01652759, 0.17526728, 0.8132702],
                [0.64941573, 0.91275555, 0.50362694, 0.60663575, 0.9707428],
                [0.72949654, 0.63227075, 0.54362494, 0.5599174, 0.9350724],
                [0.27734703, 0.81585354, 0.6708765, 0.00273848, 0.39414912],
            ],
            [
                [0.85740423, 0.5543149, 0.03358555, 0.7648899, 0.72965544],
                [0.8465752, 0.1756556, 0.08928674, 0.8631789, 0.02210194],
                [0.54146117, 0.08039963, 0.29971188, 0.48106134, 0.42268717],
                [0.40323848, 0.02831966, 0.00535262, 0.12428325, 0.00828427],
                [0.6706244, 0.5256177, 0.6471895, 0.25729978, 0.61538506],
            ],
            [
                [0.7640549, 0.38367754, 0.4609216, 0.9972099, 0.80498916],
                [0.9808353, 0.37952334, 0.6855419, 0.9501003, 0.65045923],
                [0.84031135, 0.6884467, 0.704001, 0.38892138, 0.8751561],
                [0.13509649, 0.57890344, 0.7214883, 0.84548056, 0.52535427],
                [0.37541664, 0.31024182, 0.42295915, 0.4858353, 0.7188217],
            ],
        ],
        dtype=np.float32,
    )

    disabled = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )
    enabled = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=1,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )

    expected_disabled = np.array(
        [[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0, 1.0], [-1.0, -1.0, -1.0, -2.0, -1.0]],
        dtype=np.float32,
    )
    expected_enabled = np.array(
        [[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, -1.0, 0.0], [-1.0, -1.0, -1.0, -2.0, -1.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(disabled, expected_disabled)
    np.testing.assert_array_equal(enabled, expected_enabled)
    assert not np.array_equal(disabled, enabled)


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


def test_find_surface_3d_uses_facade_find_path_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = np.zeros((2, 3, 5), dtype=np.float32)
    calls = 0

    def fake_find_path(cost_row: np.ndarray, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert cost_row.shape == (3, 5)
        assert kwargs["lmin"] == -2
        return np.full(3, 1.25, dtype=np.float32)

    monkeypatch.setattr(dp, "find_path_2d", fake_find_path)

    surface = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
    )

    assert calls == 2
    np.testing.assert_array_equal(surface, np.full((2, 3), 1.25, dtype=np.float32))


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


def test_find_surface_3d_masked_excludes_invalid_very_low_cost_lags() -> None:
    cost = np.full((3, 7, 5), 8.0, dtype=np.float32)
    cost[:, :, 0] = -1_000.0
    cost[:, :, 2] = 0.0
    valid_mask = np.ones_like(cost, dtype=np.bool_)
    valid_mask[:, :, 0] = False

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=1,
    )

    assert surface is not None
    assert surface.dtype == np.float32
    np.testing.assert_array_equal(surface, np.zeros((3, 7), dtype=np.float32))
    assert projection_count == 0


def test_find_surface_3d_masked_uses_facade_python_kernel_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = np.zeros((2, 3, 3), dtype=np.float32)
    valid_mask = np.ones_like(cost, dtype=np.bool_)
    accumulate_calls = 0
    backtrack_calls = 0

    def fake_accumulate(
        cost_row: np.ndarray,
        mask_row: np.ndarray,
        bstrain: int,
        direction: int,
    ) -> np.ndarray:
        nonlocal accumulate_calls
        accumulate_calls += 1
        assert bstrain == 1
        assert direction in (-1, 1)
        assert mask_row.all()
        return cost_row.copy()

    def fake_backtrack(
        accumulated: np.ndarray,
        cost_row: np.ndarray,
        mask_row: np.ndarray,
        lmin: int,
        bstrain: int,
        direction: int,
    ) -> tuple[np.ndarray, bool]:
        nonlocal backtrack_calls
        backtrack_calls += 1
        assert accumulated.shape == cost_row.shape == mask_row.shape
        assert (lmin, bstrain, direction) == (-1, 1, -1)
        return np.zeros(cost_row.shape[0], dtype=np.float32), True

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(dp, "_accumulate_2d_masked_python", fake_accumulate)
    monkeypatch.setattr(dp, "_backtrack_2d_masked_python", fake_backtrack)

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-1,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=1,
    )

    assert accumulate_calls == 12
    assert backtrack_calls == 2
    np.testing.assert_array_equal(surface, np.zeros((2, 3), dtype=np.float32))
    assert projection_count == 0


def test_find_surface_3d_masked_selects_only_valid_ridge_lags() -> None:
    ridge_indices = np.array([1, 2, 3, 2, 1], dtype=np.int32)
    cost = np.full((2, ridge_indices.size, 5), 100.0, dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for iw in range(cost.shape[0]):
        for iv, lag_index in enumerate(ridge_indices):
            cost[iw, iv, lag_index] = 0.0
            valid_mask[iw, iv, lag_index] = True

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=1,
    )

    assert surface is not None
    expected = np.broadcast_to(ridge_indices.astype(np.float32) - 2.0, surface.shape)
    np.testing.assert_array_equal(surface, expected)
    selected_indices = np.floor(surface.astype(np.float64) + 2.0 + 0.5).astype(np.intp)
    selected_valid = np.take_along_axis(valid_mask, selected_indices[:, :, None], axis=2)
    assert selected_valid.all()
    assert projection_count == 0


def test_find_surface_3d_masked_detects_strain_infeasible_path() -> None:
    cost = np.zeros((1, 3, 3), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    valid_mask[0, 0, 0] = True
    valid_mask[0, 1, 2] = True
    valid_mask[0, 2, 2] = True

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-1,
        bstrain1=1,
        bstrain2=2,
        attribute_smoothing=0,
    )

    assert surface is None
    assert projection_count == 0


def test_find_surface_3d_masked_detects_w_direction_strain_infeasibility() -> None:
    cost = np.zeros((2, 1, 3), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    valid_mask[0, 0, 0] = True
    valid_mask[1, 0, 2] = True

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-1,
        bstrain1=2,
        bstrain2=1,
        attribute_smoothing=0,
    )

    assert surface is None
    assert projection_count == 0


def test_find_surface_3d_masked_recovers_jointly_feasible_bidirectional_surface() -> None:
    cost = np.array(
        [
            [
                [-0.2588143, 1.0633872, -1.2434108, -1.1908289],
                [-0.4183583, 0.0651413, -0.337766, 1.9926934],
            ],
            [
                [-0.6038749, 0.6353444, -0.5863985, 0.3207008],
                [-0.4428899, -0.3930264, 1.6236607, -0.3370287],
            ],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [[False, False, True, False], [False, True, False, True]],
            [[True, False, False, True], [False, True, True, True]],
        ],
        dtype=np.bool_,
    )

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=0,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
    )

    assert surface is not None
    assert projection_count == 0
    selected = np.floor(surface.astype(np.float64) + 0.5).astype(np.intp)
    assert np.take_along_axis(valid_mask, selected[:, :, None], axis=2).all()
    assert np.max(np.abs(np.diff(surface, axis=0))) <= 1.0
    assert np.max(np.abs(np.diff(surface, axis=1))) <= 1.0


def test_find_surface_3d_masked_uses_java_rounding_cells_for_strain_feasibility() -> None:
    cost = np.zeros((2, 1, 2), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    valid_mask[0, 0, 0] = True
    valid_mask[1, 0, 1] = True

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=0,
        bstrain1=1,
        bstrain2=2,
        attribute_smoothing=0,
    )

    assert surface is not None
    assert projection_count == 0
    selected = np.floor(surface.astype(np.float64) + 0.5).astype(np.intp)
    assert np.take_along_axis(valid_mask, selected[:, :, None], axis=2).all()
    assert abs(float(surface[1, 0] - surface[0, 0])) <= 0.5 + 1.0e-6


def test_find_surface_3d_masked_uses_feasible_alternative_transition() -> None:
    cost = np.full((1, 5, 5), 100.0, dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    valid_mask[0, :, 2] = True
    cost[0, :, 2] = 10.0
    valid_mask[0, 3:, 3] = True
    cost[0, 4, 3] = -1_000.0

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=4,
        bstrain2=1,
        attribute_smoothing=0,
    )

    assert surface is not None
    np.testing.assert_array_equal(surface, np.zeros((1, 5), dtype=np.float32))
    assert projection_count == 0


def test_find_surface_3d_masked_projects_smoothed_lags_to_valid_states() -> None:
    ridge_indices = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int32)
    cost = np.full((1, ridge_indices.size, 3), 10.0, dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for iv, lag_index in enumerate(ridge_indices):
        cost[0, iv, lag_index] = 0.0
        valid_mask[0, iv, lag_index] = True

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=0,
        bstrain1=1,
        bstrain2=1,
        attribute_smoothing=0,
        surface_smoothing1=1.0,
    )

    assert surface is not None
    assert projection_count == 1
    selected_indices = np.floor(surface.astype(np.float64) + 0.5).astype(np.intp)
    selected_valid = np.take_along_axis(valid_mask, selected_indices[:, :, None], axis=2)
    assert selected_valid.all()
    np.testing.assert_array_equal(surface[0, 3], np.float32(0.5))


def test_find_surface_3d_masked_all_valid_strain_feasible_matches_unmasked_surface() -> None:
    rng = np.random.default_rng(40)
    cost = rng.random((3, 7, 5), dtype=np.float32)
    valid_mask = np.ones_like(cost, dtype=np.bool_)

    expected = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
    )
    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
    )

    assert surface is not None
    np.testing.assert_array_equal(surface, expected)
    assert projection_count == 0


def test_find_surface_3d_masked_all_valid_recovers_unmasked_strain_violation() -> None:
    rng = np.random.default_rng(4901)
    cost = rng.random((3, 7, 5), dtype=np.float32)
    valid_mask = np.ones_like(cost, dtype=np.bool_)
    expected = find_surface_3d(
        cost,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
    )

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
    )

    assert not _surface_respects_masked_strain(
        expected,
        valid_mask,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
    )
    assert surface is not None
    assert _surface_respects_masked_strain(
        surface,
        valid_mask,
        lmin=-2,
        bstrain1=2,
        bstrain2=2,
    )
    assert not np.array_equal(surface, expected)
    assert projection_count == 0


def test_find_surface_3d_masked_preserves_strain_after_smoothing_projection() -> None:
    cost = np.zeros((1, 4, 5), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for iv, (start, stop) in enumerate(((4, 4), (0, 4), (2, 3), (2, 3))):
        valid_mask[0, iv, start : stop + 1] = True

    results = [
        _find_surface_3d_masked(
            cost,
            valid_mask,
            lmin=-2,
            bstrain1=4,
            bstrain2=4,
            attribute_smoothing=0,
            surface_smoothing1=2.0,
            surface_smoothing2=2.0,
        )
        for _ in range(2)
    ]

    surface, projection_count = results[0]
    assert surface is not None
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    assert _surface_respects_masked_strain(
        surface,
        valid_mask,
        lmin=-2,
        bstrain1=4,
        bstrain2=4,
    )
    assert np.max(np.abs(np.diff(surface, axis=1))) <= np.float32(0.25 + 1.0e-6)
    selected_indices = np.floor(surface.astype(np.float64) + 2.0 + 0.5).astype(np.intp)
    assert np.take_along_axis(valid_mask, selected_indices[:, :, None], axis=2).all()
    assert projection_count == 1
    repeated_surface, repeated_projection_count = results[1]
    assert repeated_surface is not None
    np.testing.assert_array_equal(repeated_surface, surface)
    assert repeated_projection_count == projection_count


def test_project_masked_surface_preserves_java_rounding_cell_boundaries() -> None:
    valid_mask = np.zeros((1, 1, 5), dtype=np.bool_)
    valid_mask[0, 0, 2] = True
    lower_half = np.array([[1.5]], dtype=np.float32)

    projected_lower, lower_count, lower_ok = _project_surface_to_valid_mask_python(
        lower_half,
        valid_mask,
        0,
    )

    assert lower_ok
    assert lower_count == 0
    np.testing.assert_array_equal(projected_lower, lower_half)

    exact_upper = np.array([[2.5]], dtype=np.float32)
    projected_upper, upper_count, upper_ok = _project_surface_to_valid_mask_python(
        exact_upper,
        valid_mask,
        0,
    )
    upper_predecessor = np.nextafter(np.float32(2.5), np.float32(-np.inf))

    assert upper_ok
    assert upper_count == 1
    np.testing.assert_array_equal(projected_upper, np.array([[upper_predecessor]], np.float32))

    just_below_upper = np.array([[upper_predecessor]], dtype=np.float32)
    projected_predecessor, predecessor_count, predecessor_ok = (
        _project_surface_to_valid_mask_python(just_below_upper, valid_mask, 0)
    )

    assert predecessor_ok
    assert predecessor_count == 0
    np.testing.assert_array_equal(projected_predecessor, just_below_upper)


def test_project_masked_surface_clamps_lag_range_ends_and_breaks_ties_by_low_lag() -> None:
    endpoint_mask = np.zeros((1, 2, 5), dtype=np.bool_)
    endpoint_mask[0, 0, 0] = True
    endpoint_mask[0, 1, 4] = True
    outside_endpoints = np.array([[-0.25, 4.25]], dtype=np.float32)

    projected, projection_count, projection_ok = _project_surface_to_valid_mask_python(
        outside_endpoints,
        endpoint_mask,
        0,
    )

    assert projection_ok
    assert projection_count == 2
    np.testing.assert_array_equal(projected, np.array([[0.0, 4.0]], dtype=np.float32))

    tie_mask = np.zeros((1, 1, 3), dtype=np.bool_)
    tie_mask[0, 0, (0, 2)] = True
    tie_value = np.array([[1.0]], dtype=np.float32)
    tie_results = [_project_surface_to_valid_mask_python(tie_value, tie_mask, 0) for _ in range(2)]
    expected_tie = np.nextafter(np.float32(0.5), np.float32(-np.inf))

    for tie_surface, tie_count, tie_ok in tie_results:
        assert tie_ok
        assert tie_count == 1
        np.testing.assert_array_equal(
            tie_surface,
            np.array([[expected_tie]], dtype=np.float32),
        )


def test_find_surface_3d_masked_post_smoothing_invariant_sweep() -> None:
    rng = np.random.default_rng(20260710)
    configurations = (
        ((1, 3, 5), 1, 1, 0.0, 0.0),
        ((2, 3, 5), 2, 4, 0.75, 0.0),
        ((3, 4, 7), 4, 2, 1.25, 1.0),
    )
    for shape, bstrain1, bstrain2, smoothing1, smoothing2 in configurations:
        for _ in range(4):
            cost = rng.normal(size=shape).astype(np.float32)
            valid_mask = np.zeros(shape, dtype=np.bool_)
            for iw in range(shape[0]):
                for iv in range(shape[1]):
                    start = int(rng.integers(0, shape[2]))
                    stop = int(rng.integers(start, shape[2]))
                    valid_mask[iw, iv, start : stop + 1] = True

            kwargs = {
                "lmin": -(shape[2] // 2),
                "bstrain1": bstrain1,
                "bstrain2": bstrain2,
                "attribute_smoothing": 0,
                "surface_smoothing1": smoothing1,
                "surface_smoothing2": smoothing2,
            }
            first_surface, first_count = _find_surface_3d_masked(cost, valid_mask, **kwargs)
            second_surface, second_count = _find_surface_3d_masked(cost, valid_mask, **kwargs)

            assert (first_surface is None) is (second_surface is None)
            assert first_count == second_count
            if first_surface is None:
                assert first_count == 0
                continue
            assert second_surface is not None
            assert first_surface.dtype == np.float32
            assert np.isfinite(first_surface).all()
            assert _surface_respects_masked_strain(
                first_surface,
                valid_mask,
                lmin=kwargs["lmin"],
                bstrain1=bstrain1,
                bstrain2=bstrain2,
            )
            selected = np.floor(first_surface.astype(np.float64) - kwargs["lmin"] + 0.5).astype(
                np.intp
            )
            assert np.take_along_axis(valid_mask, selected[:, :, None], axis=2).all()
            np.testing.assert_array_equal(second_surface, first_surface)


def test_find_surface_3d_masked_returns_none_when_mask_and_strain_are_infeasible() -> None:
    cost = np.zeros((1, 2, 5), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    valid_mask[0, 0, 0] = True
    valid_mask[0, 1, 4] = True
    locally_valid = np.array([[-2.0, 2.0]], dtype=np.float32)
    projected, local_count, projection_ok = _project_surface_to_valid_mask_python(
        locally_valid,
        valid_mask,
        -2,
    )

    assert projection_ok
    assert local_count == 0
    assert not _surface_respects_masked_strain(
        projected,
        valid_mask,
        lmin=-2,
        bstrain1=4,
        bstrain2=4,
    )

    surface, projection_count = _find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=-2,
        bstrain1=4,
        bstrain2=4,
        attribute_smoothing=0,
        surface_smoothing1=2.0,
        surface_smoothing2=2.0,
    )

    assert surface is None
    assert projection_count == 0


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
