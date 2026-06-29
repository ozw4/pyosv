import numpy as np
import pytest

from pyosv.cells import FaultCell2
from pyosv.voting2d import OptimalPathVoter, _normalize_and_power_2d


def test_constructor_initializes_range_and_default_configuration() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    assert voter.ru == 3
    assert voter.rv == 4
    assert voter.lmin == -3
    assert voter.lmax == 3
    assert voter.nl == 7
    assert voter.bstrain1 == 4
    assert voter.attribute_smoothing == 1
    assert voter.path_smoothing1 == 2.0
    np.testing.assert_array_equal(
        voter.lmins,
        np.array([-3, -3, 0, 0, 0, 0, 0, -3, -3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        voter.lmaxs,
        np.array([3, 3, 0, 0, 0, 0, 0, 3, 3], dtype=np.int32),
    )


def test_shift_range_arrays_match_strike_radius_shape() -> None:
    voter = OptimalPathVoter(ru=5, rv=6)

    assert voter.lmins.shape == (2 * voter.rv + 1,)
    assert voter.lmaxs.shape == (2 * voter.rv + 1,)


def test_set_strain_max_updates_only_bstrain_spacing() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    lmins_before = voter.lmins.copy()
    lmaxs_before = voter.lmaxs.copy()

    voter.set_strain_max(1.0)

    assert voter.bstrain1 == 1
    np.testing.assert_array_equal(voter.lmins, lmins_before)
    np.testing.assert_array_equal(voter.lmaxs, lmaxs_before)


def test_set_strain_max_keeps_default_bstrain_spacing() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_strain_max(0.25)

    assert voter.bstrain1 == 4
    np.testing.assert_array_equal(
        voter.lmins,
        np.array([-3, -3, 0, 0, 0, 0, 0, -3, -3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        voter.lmaxs,
        np.array([3, 3, 0, 0, 0, 0, 0, 3, 3], dtype=np.int32),
    )


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_values(strain_max: float) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(strain_max)


@pytest.mark.parametrize("attribute_smoothing", [0, 1, np.int32(2)])
def test_set_attribute_smoothing_accepts_nonnegative_integers(
    attribute_smoothing: int,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_attribute_smoothing(attribute_smoothing)

    assert voter.attribute_smoothing == int(attribute_smoothing)


@pytest.mark.parametrize("attribute_smoothing", [-1, 1.5, True, "1"])
def test_set_attribute_smoothing_rejects_invalid_values(
    attribute_smoothing: object,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="attribute_smoothing"):
        voter.set_attribute_smoothing(attribute_smoothing)  # type: ignore[arg-type]


@pytest.mark.parametrize("path_smoothing1", [0.0, 1.25, np.float32(2.0)])
def test_set_path_smoothing_accepts_nonnegative_finite_numbers(
    path_smoothing1: float,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_path_smoothing(path_smoothing1)

    assert voter.path_smoothing1 == float(path_smoothing1)


@pytest.mark.parametrize("path_smoothing1", [-0.1, np.nan, np.inf, True, "1"])
def test_set_path_smoothing_rejects_invalid_values(path_smoothing1: object) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="path_smoothing1"):
        voter.set_path_smoothing(path_smoothing1)  # type: ignore[arg-type]


def test_pick_seeds_returns_no_seeds_when_no_sample_exceeds_threshold() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    pt = np.full_like(ft, 45.0)

    seeds = voter.pick_seeds(d=1, fm=0.4, ft=ft, pt=pt)

    assert seeds == []


def test_pick_seeds_returns_isolated_peak_with_cell_fields() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.zeros((3, 4), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[1, 2] = 0.75
    pt[1, 2] = 30.0

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt)

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.i1 == 2
    assert seed.i2 == 1
    assert seed.fl == pytest.approx(0.75)
    assert seed.fp == pytest.approx(30.0)


def test_pick_seeds_suppresses_nearby_lower_peak() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.zeros((5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[2, 2] = 0.9
    ft[3, 3] = 0.8
    pt[2, 2] = 20.0
    pt[3, 3] = 40.0

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt)

    assert [(seed.i1, seed.i2) for seed in seeds] == [(2, 2)]
    assert seeds[0].fl == pytest.approx(0.9)
    assert seeds[0].fp == pytest.approx(20.0)


def test_pick_seeds_with_zero_distance_returns_all_candidates() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.array(
        [
            [0.0, 0.7, 0.0],
            [0.6, 0.5, 0.8],
        ],
        dtype=np.float32,
    )
    pt = np.arange(ft.size, dtype=np.float32).reshape(ft.shape)

    seeds = voter.pick_seeds(d=0, fm=0.5, ft=ft, pt=pt)

    assert [(seed.i1, seed.i2) for seed in seeds] == [(2, 1), (1, 0), (0, 1)]
    assert [seed.fl for seed in seeds] == pytest.approx([0.8, 0.7, 0.6])


def test_pick_seeds_rejects_mismatched_shapes() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.zeros((2, 3), dtype=np.float32)
    pt = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shapes must match"):
        voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt)


def test_pick_seeds_does_not_modify_inputs() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    pt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    ft_before = ft.copy()
    pt_before = pt.copy()

    voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt)

    np.testing.assert_array_equal(ft, ft_before)
    np.testing.assert_array_equal(pt, pt_before)


def test_get_seeds_returns_seed_at_requested_sample() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    pt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    seeds = voter.get_seeds(c1=1, c2=0, ft=ft, pt=pt)

    assert len(seeds) == 1
    assert seeds[0].i1 == 1
    assert seeds[0].i2 == 0
    assert seeds[0].fl == pytest.approx(0.2)
    assert seeds[0].fp == pytest.approx(20.0)


@pytest.mark.parametrize(("c1", "c2"), [(-1, 0), (0, -1), (2, 0), (0, 2)])
def test_get_seeds_rejects_coordinates_outside_image(c1: int, c2: int) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    ft = np.zeros((2, 2), dtype=np.float32)
    pt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="image bounds"):
        voter.get_seeds(c1=c1, c2=c2, ft=ft, pt=pt)


def test_update_vector_map_radius_two_positive_columns_follow_vector() -> None:
    voter = OptimalPathVoter(ru=1, rv=1)

    vector_map = voter.update_vector_map(radius=2, vector=np.array([1.0, 0.0]))

    assert vector_map.shape == (2, 5)
    assert vector_map.dtype == np.float32
    np.testing.assert_array_equal(
        vector_map,
        np.array(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-0.0, -0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_update_vector_map_center_column_is_zero_displacement() -> None:
    voter = OptimalPathVoter(ru=1, rv=1)

    vector_map = voter.update_vector_map(radius=3, vector=np.array([2.0, -0.5]))

    np.testing.assert_array_equal(vector_map[:, 3], np.zeros(2, dtype=np.float32))


def test_samples_in_uv_box_returns_constant_cost_from_constant_fx() -> None:
    voter = OptimalPathVoter(ru=2, rv=3)
    fx = np.full((5, 6), 0.25, dtype=np.float32)

    costs = voter.samples_in_uv_box(
        c1=3,
        c2=2,
        normal=np.array([1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (2 * voter.rv + 1, 2 * voter.ru + 1)
    assert costs.dtype == np.float32
    expected = np.ones((7, 5), dtype=np.float32)
    for kv in range(expected.shape[0]):
        ku_min = voter.lmins[kv] + voter.ru
        ku_max = voter.lmaxs[kv] + voter.ru
        expected[kv, ku_min : ku_max + 1] = 0.75
    np.testing.assert_array_equal(costs, expected)


def test_samples_in_uv_box_respects_reference_shift_ranges() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    fx = np.full((20, 20), 0.25, dtype=np.float32)

    costs = voter.samples_in_uv_box(
        c1=10,
        c2=10,
        normal=np.array([1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (9, 7)
    assert costs.dtype == np.float32

    for kv in range(2 * voter.rv + 1):
        ku_min = voter.lmins[kv] + voter.ru
        ku_max = voter.lmaxs[kv] + voter.ru

        if ku_min > 0:
            np.testing.assert_array_equal(costs[kv, :ku_min], 1.0)
        if ku_max + 1 < costs.shape[1]:
            np.testing.assert_array_equal(costs[kv, ku_max + 1 :], 1.0)
        np.testing.assert_array_equal(costs[kv, ku_min : ku_max + 1], 0.75)


def test_samples_in_uv_box_central_strike_rows_sample_only_zero_normal_shift() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    fx = np.full((20, 20), 0.25, dtype=np.float32)

    costs = voter.samples_in_uv_box(
        c1=10,
        c2=10,
        normal=np.array([1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    for iv in range(-2, 3):
        kv = iv + voter.rv
        expected = np.ones(2 * voter.ru + 1, dtype=np.float32)
        expected[voter.ru] = 0.75
        np.testing.assert_array_equal(costs[kv], expected)


def test_samples_in_uv_box_v_then_u_shape_and_center_uses_seed_coordinate() -> None:
    voter = OptimalPathVoter(ru=2, rv=1)
    i2, i1 = np.indices((5, 6), dtype=np.float32)
    fx = 0.1 * i2 + 0.01 * i1

    costs = voter.samples_in_uv_box(
        c1=3,
        c2=2,
        normal=np.array([1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (3, 5)
    assert costs[voter.rv, voter.ru] == pytest.approx(1.0 - fx[2, 3])


def test_samples_in_uv_box_rounds_and_clamps_near_image_boundary() -> None:
    voter = OptimalPathVoter(ru=2, rv=2)
    i2, i1 = np.indices((3, 4), dtype=np.float32)
    fx = 0.1 * i2 + 0.01 * i1

    costs = voter.samples_in_uv_box(
        c1=0,
        c2=0,
        normal=np.array([1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (2 * voter.rv + 1, 2 * voter.ru + 1)
    assert np.isfinite(costs).all()
    assert costs[0, 0] == pytest.approx(1.0 - fx[0, 0])


def test_path_voting_adds_votes_on_high_likelihood_path() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((11, 11), dtype=np.float32)
    ft[5, 3:8] = 0.8
    fc = np.zeros_like(ft)
    fe = np.zeros_like(ft)
    w1 = np.zeros_like(ft)
    w2 = np.zeros_like(ft)

    voter._path_voting(FaultCell2(5, 5, 0.8, 0.0), ft, fc, fe, w1, w2)

    assert fe[5, 3:8].sum() == pytest.approx(4.0)
    assert np.count_nonzero(fe) == 5
    np.testing.assert_allclose(fe[5, 3:8], 0.8)
    np.testing.assert_allclose(fc[5, 3:8], 0.8)
    np.testing.assert_allclose(w1[5, 3:8], 0.0, atol=1e-7)
    np.testing.assert_allclose(w2[5, 3:8], -1.0, atol=1e-7)


def test_path_voting_keeps_stronger_vector_when_later_vote_is_weaker() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft_strong = np.zeros((11, 11), dtype=np.float32)
    ft_strong[5, 3:8] = 0.8
    ft_weak = np.zeros((11, 11), dtype=np.float32)
    ft_weak[3:8, 5] = 0.2
    fc = np.zeros_like(ft_strong)
    fe = np.zeros_like(ft_strong)
    w1 = np.zeros_like(ft_strong)
    w2 = np.zeros_like(ft_strong)

    voter._path_voting(FaultCell2(5, 5, 0.8, 0.0), ft_strong, fc, fe, w1, w2)
    voter._path_voting(FaultCell2(5, 5, 0.2, 90.0), ft_weak, fc, fe, w1, w2)

    assert fe[5, 5] == pytest.approx(1.0)
    assert fc[5, 5] == pytest.approx(0.8)
    assert w1[5, 5] == pytest.approx(0.0, abs=1e-7)
    assert w2[5, 5] == pytest.approx(-1.0, abs=1e-7)


def test_path_voting_skips_seed_with_no_valid_interior_path_samples() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    ft = np.ones((3, 3), dtype=np.float32)
    fc = np.zeros_like(ft)
    fe = np.zeros_like(ft)
    w1 = np.zeros_like(ft)
    w2 = np.zeros_like(ft)

    voter._path_voting(FaultCell2(0, 0, 1.0, 0.0), ft, fc, fe, w1, w2)

    for array in (fc, fe, w1, w2):
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, np.zeros_like(array))


def test_path_voting_is_deterministic_for_same_seed_and_inputs() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((11, 11), dtype=np.float32)
    ft[5, 3:8] = 0.8
    first = tuple(np.zeros_like(ft) for _ in range(4))
    second = tuple(np.zeros_like(ft) for _ in range(4))

    voter._path_voting(FaultCell2(5, 5, 0.8, 0.0), ft, *first)
    voter._path_voting(FaultCell2(5, 5, 0.8, 0.0), ft, *second)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_apply_voting_returns_zero_arrays_when_no_seeds_are_selected() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    ft = np.zeros((7, 8), dtype=np.float32)
    pt = np.zeros_like(ft)

    fv, w1, w2 = voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt)

    for array in (fv, w1, w2):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, np.zeros_like(ft))


def test_apply_voting_rejects_mismatched_shapes() -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    ft = np.zeros((2, 3), dtype=np.float32)
    pt = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shapes must match"):
        voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt)


@pytest.mark.parametrize(
    ("ft_value", "pt_value", "message"),
    [(np.nan, 0.0, "ft"), (0.0, np.inf, "pt")],
)
def test_apply_voting_rejects_nonfinite_inputs(
    ft_value: float,
    pt_value: float,
    message: str,
) -> None:
    voter = OptimalPathVoter(ru=1, rv=2)
    ft = np.zeros((3, 3), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[1, 1] = ft_value
    pt[1, 1] = pt_value

    with pytest.raises(ValueError, match=message):
        voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt)


def test_apply_voting_highlights_simple_fault_like_line() -> None:
    voter = OptimalPathVoter(ru=1, rv=3)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((15, 15), dtype=np.float64)
    pt = np.zeros_like(ft)
    ft[7, 3:12] = 0.9

    fv, w1, w2 = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    assert fv.shape == ft.shape
    assert w1.shape == ft.shape
    assert w2.shape == ft.shape
    assert fv.dtype == np.float32
    assert w1.dtype == np.float32
    assert w2.dtype == np.float32
    assert np.isfinite(fv).all()
    assert np.isfinite(w1).all()
    assert np.isfinite(w2).all()
    assert fv.min() >= -1e-6
    assert fv.max() <= 1.0 + 1e-6
    assert fv[7, 4:11].mean() > fv[[3, 11], 4:11].mean()


def test_apply_voting_highlights_vertical_constant_angle_line() -> None:
    voter = OptimalPathVoter(ru=1, rv=4)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((17, 17), dtype=np.float32)
    pt = np.full_like(ft, 90.0)
    ft[3:14, 8] = 0.9
    line_mask = np.zeros_like(ft, dtype=np.bool_)
    line_mask[4:13, 8] = True
    background_mask = np.zeros_like(ft, dtype=np.bool_)
    background_mask[4:13, 4] = True
    background_mask[4:13, 12] = True

    fv, w1, w2 = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    assert fv.shape == ft.shape
    assert w1.shape == ft.shape
    assert w2.shape == ft.shape
    assert np.isfinite(fv).all()
    assert np.isfinite(w1).all()
    assert np.isfinite(w2).all()
    assert fv.max() > 0.0
    assert fv[line_mask].mean() > fv[background_mask].mean() + 0.4


def test_apply_voting_highlights_diagonal_constant_angle_line() -> None:
    voter = OptimalPathVoter(ru=2, rv=5)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((21, 21), dtype=np.float32)
    pt = np.full_like(ft, 135.0)
    line_mask = np.zeros_like(ft, dtype=np.bool_)
    for i2 in range(5, 16):
        ft[i2, i2] = 0.9
        line_mask[i2, i2] = True
    background_mask = np.zeros_like(ft, dtype=np.bool_)
    background_mask[5:16, 4] = True
    background_mask[4, 5:16] = True

    fv, w1, w2 = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    assert fv.shape == ft.shape
    assert w1.shape == ft.shape
    assert w2.shape == ft.shape
    assert np.isfinite(fv).all()
    assert np.isfinite(w1).all()
    assert np.isfinite(w2).all()
    assert fv.max() > 0.0
    assert fv[line_mask].mean() > fv[background_mask].mean() + 0.4


def test_apply_voting_is_deterministic_for_same_inputs() -> None:
    voter = OptimalPathVoter(ru=1, rv=3)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((15, 15), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[7, 3:12] = 0.9

    first = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_seed_to_image_keeps_seed_values_above_threshold() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    cells = [FaultCell2(1, 0, 0.7, 20.0), FaultCell2(0, 1, 0.8, 30.0)]
    ep = np.array([[0.1, 0.9], [0.4, 0.0]], dtype=np.float32)

    image = voter.seed_to_image(fmin=0.5, shape=ep.shape, cells=cells, ep=ep)

    expected = np.array([[0.0, 0.9], [0.0, 0.0]], dtype=np.float32)
    assert image.dtype == np.float32
    np.testing.assert_array_equal(image, expected)


def test_seed_to_points_returns_two_by_seed_count_array() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    cells = [FaultCell2(3, 4, 0.7, 20.0), FaultCell2(5, 6, 0.8, 30.0)]

    points = voter.seed_to_points(cells)

    assert points.shape == (2, 2)
    assert points.dtype == np.float32
    np.testing.assert_array_equal(
        points,
        np.array([[3.0, 5.0], [4.0, 6.0]], dtype=np.float32),
    )


def test_normalize_and_power_2d_zero_array_returns_zero_scores() -> None:
    image = np.zeros((3, 4), dtype=np.float32)

    scores = _normalize_and_power_2d(image)

    assert scores.shape == image.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, image)


def test_normalize_and_power_2d_constant_nonzero_array_returns_zero_scores() -> None:
    image = np.full((2, 3), 7.5, dtype=np.float32)

    scores = _normalize_and_power_2d(image)

    assert scores.shape == image.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, np.zeros_like(image))


def test_normalize_and_power_2d_simple_ramp_uses_min_max_and_power() -> None:
    image = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)

    scores = _normalize_and_power_2d(image, sigma=0.0, power=4)

    expected = np.array([[0.0, 0.9375, 1.0]], dtype=np.float32)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-7)


def test_normalize_and_power_2d_clear_maximum_stays_finite_and_bounded() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = 10.0

    scores = _normalize_and_power_2d(image, sigma=1.0)

    assert scores.shape == image.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert scores.min() >= -1e-6
    assert scores.max() <= 1.0 + 1e-6
    assert scores[2, 2] == pytest.approx(1.0)
