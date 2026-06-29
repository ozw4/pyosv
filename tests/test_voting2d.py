import numpy as np
import pytest

from pyosv.cells import FaultCell2
from pyosv.voting2d import OptimalPathVoter


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
