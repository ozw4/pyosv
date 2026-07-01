import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
)
from pyosv.orient3d import FaultOrientScanner3
from pyosv.thinning3d import reference_like_3d_thin_values
from pyosv.voting3d import (
    OptimalSurfaceVoter,
    _normalize_and_power_3d,
    _smooth_fault_likelihood_3d,
    _surface_strike_and_dip,
)


def test_constructor_initializes_range_and_default_configuration() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    assert voter.ru == 3
    assert voter.rv == 2
    assert voter.rw == 2
    assert voter.lmin == -3
    assert voter.lmax == 3
    assert voter.nl == 7
    assert voter.bstrain1 == 4
    assert voter.bstrain2 == 4
    assert voter.attribute_smoothing == 1
    assert voter.surface_smoothing1 == 2.0
    assert voter.surface_smoothing2 == 2.0
    np.testing.assert_array_equal(
        voter.lmins,
        np.array(
            [
                [-3, -2, 0, -2, -3],
                [-2, 0, 0, 0, -2],
                [0, 0, 0, 0, 0],
                [-2, 0, 0, 0, -2],
                [-3, -2, 0, -2, -3],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(voter.lmaxs, -voter.lmins)


def test_shift_range_arrays_match_surface_radius_shape() -> None:
    voter = OptimalSurfaceVoter(ru=5, rv=6, rw=7)

    assert voter.lmins.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)
    assert voter.lmaxs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)


@pytest.mark.parametrize(
    ("ru", "rv", "rw"),
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1.5, 0, 0),
        (0, True, 0),
        (0, 0, "1"),
    ],
)
def test_constructor_rejects_invalid_radii(ru: object, rv: object, rw: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        OptimalSurfaceVoter(ru, rv, rw)  # type: ignore[arg-type]


def test_set_strain_max_updates_only_bstrain_spacing() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    lmins_before = voter.lmins.copy()
    lmaxs_before = voter.lmaxs.copy()

    voter.set_strain_max(1.0, 0.5)

    assert voter.bstrain1 == 1
    assert voter.bstrain2 == 2
    np.testing.assert_array_equal(voter.lmins, lmins_before)
    np.testing.assert_array_equal(voter.lmaxs, lmaxs_before)


def test_set_strain_max_keeps_default_bstrain_spacing() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_strain_max(0.25, 0.25)

    assert voter.bstrain1 == 4
    assert voter.bstrain2 == 4


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_first_strain(strain_max: float) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(strain_max, 0.25)


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_second_strain(strain_max: float) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(0.25, strain_max)


@pytest.mark.parametrize("attribute_smoothing", [0, 1, np.int32(2)])
def test_set_attribute_smoothing_accepts_nonnegative_integers(
    attribute_smoothing: int,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_attribute_smoothing(attribute_smoothing)

    assert voter.attribute_smoothing == int(attribute_smoothing)


@pytest.mark.parametrize("attribute_smoothing", [-1, 1.5, True, "1"])
def test_set_attribute_smoothing_rejects_invalid_values(
    attribute_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="attribute_smoothing"):
        voter.set_attribute_smoothing(attribute_smoothing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("surface_smoothing1", "surface_smoothing2"),
    [
        (0.0, 0.0),
        (1.25, 0.5),
        (np.float32(2.0), np.float32(3.0)),
    ],
)
def test_set_surface_smoothing_accepts_nonnegative_finite_numbers(
    surface_smoothing1: float,
    surface_smoothing2: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)

    assert voter.surface_smoothing1 == float(surface_smoothing1)
    assert voter.surface_smoothing2 == float(surface_smoothing2)


@pytest.mark.parametrize("surface_smoothing", [-0.1, np.nan, np.inf, True, "1"])
def test_set_surface_smoothing_rejects_invalid_first_value(
    surface_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_smoothing1"):
        voter.set_surface_smoothing(surface_smoothing, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("surface_smoothing", [-0.1, np.nan, np.inf, True, "1"])
def test_set_surface_smoothing_rejects_invalid_second_value(
    surface_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_smoothing2"):
        voter.set_surface_smoothing(0.0, surface_smoothing)  # type: ignore[arg-type]


def test_pick_seeds_returns_no_seeds_when_no_sample_exceeds_threshold() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    pt = np.full_like(ft, 45.0)
    tt = np.full_like(ft, 30.0)

    seeds = voter.pick_seeds(d=1, fm=0.4, ft=ft, pt=pt, tt=tt)

    assert seeds == []


def test_pick_seeds_returns_fault_cell_with_volume_indices_and_angles() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((2, 3, 4), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)
    ft[1, 2, 3] = 0.75
    pt[1, 2, 3] = 35.0
    tt[1, 2, 3] = 55.0

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert len(seeds) == 1
    seed = seeds[0]
    assert isinstance(seed, FaultCell)
    assert seed.i1 == 3
    assert seed.i2 == 2
    assert seed.i3 == 1
    assert seed.fl == pytest.approx(0.75)
    assert seed.fp == pytest.approx(35.0)
    assert seed.ft == pytest.approx(55.0)


def test_pick_seeds_suppresses_lower_candidate_inside_radius_box() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)
    ft[2, 2, 2] = 0.9
    ft[3, 3, 3] = 0.8
    pt[2, 2, 2] = 20.0
    tt[2, 2, 2] = 45.0
    pt[3, 3, 3] = 40.0
    tt[3, 3, 3] = 60.0

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert [(seed.i1, seed.i2, seed.i3) for seed in seeds] == [(2, 2, 2)]
    assert seeds[0].fl == pytest.approx(0.9)
    assert seeds[0].fp == pytest.approx(20.0)
    assert seeds[0].ft == pytest.approx(45.0)


def test_pick_seeds_preserves_candidates_outside_radius_box() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)
    ft[2, 2, 2] = 0.9
    ft[4, 2, 2] = 0.8

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert [(seed.i1, seed.i2, seed.i3) for seed in seeds] == [(2, 2, 2), (2, 2, 4)]
    assert [seed.fl for seed in seeds] == pytest.approx([0.9, 0.8])


def test_pick_seeds_with_zero_distance_returns_all_candidates_in_descending_likelihood() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((2, 2, 3), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)
    ft[0, 0, 1] = 0.7
    ft[0, 1, 0] = 0.6
    ft[1, 1, 2] = 0.8

    seeds = voter.pick_seeds(d=0, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert [(seed.i1, seed.i2, seed.i3) for seed in seeds] == [
        (2, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
    ]
    assert [seed.fl for seed in seeds] == pytest.approx([0.8, 0.7, 0.6])


def test_pick_seeds_rejects_mismatched_shapes() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((2, 3, 4), dtype=np.float32)
    pt = np.zeros((2, 4, 3), dtype=np.float32)
    tt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="shapes must match"):
        voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)


def test_pick_seeds_rejects_non_3d_inputs() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((3, 4), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="ft must be a 3D array"):
        voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)


@pytest.mark.parametrize(
    "bad_name",
    ["ft", "pt", "tt"],
)
def test_pick_seeds_rejects_nonfinite_inputs(bad_name: str) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    arrays = {
        "ft": np.zeros((2, 2, 2), dtype=np.float32),
        "pt": np.zeros((2, 2, 2), dtype=np.float32),
        "tt": np.zeros((2, 2, 2), dtype=np.float32),
    }
    arrays[bad_name][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=f"{bad_name} must contain only finite values"):
        voter.pick_seeds(
            d=1,
            fm=0.5,
            ft=arrays["ft"],
            pt=arrays["pt"],
            tt=arrays["tt"],
        )


def test_pick_seeds_does_not_modify_inputs() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.array([[[0.9, 0.8], [0.7, 0.6]]], dtype=np.float32)
    pt = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
    tt = np.array([[[50.0, 60.0], [70.0, 80.0]]], dtype=np.float32)
    ft_before = ft.copy()
    pt_before = pt.copy()
    tt_before = tt.copy()

    voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    np.testing.assert_array_equal(ft, ft_before)
    np.testing.assert_array_equal(pt, pt_before)
    np.testing.assert_array_equal(tt, tt_before)


def test_get_seeds_returns_seed_at_requested_sample() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    pt = ft + 10.0
    tt = ft + 20.0

    seeds = voter.get_seeds(c1=1, c2=0, c3=1, ft=ft, pt=pt, tt=tt)

    assert len(seeds) == 1
    assert seeds[0].i1 == 1
    assert seeds[0].i2 == 0
    assert seeds[0].i3 == 1
    assert seeds[0].fl == pytest.approx(5.0)
    assert seeds[0].fp == pytest.approx(15.0)
    assert seeds[0].ft == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("c1", "c2", "c3"),
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
)
def test_get_seeds_rejects_coordinates_outside_image(c1: int, c2: int, c3: int) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    ft = np.zeros((2, 2, 2), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="image bounds"):
        voter.get_seeds(c1=c1, c2=c2, c3=c3, ft=ft, pt=pt, tt=tt)


def test_update_vector_map_radius_two_offsets_are_symmetric() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=1, rw=1)

    vector_map = voter.update_vector_map(
        radius=2,
        vector=np.array([1.0, -0.5, 2.0]),
    )

    assert vector_map.shape == (3, 5)
    assert vector_map.dtype == np.float32
    np.testing.assert_array_equal(vector_map[:, 2], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(vector_map[:, 0], -vector_map[:, 4])
    np.testing.assert_array_equal(vector_map[:, 1], -vector_map[:, 3])
    np.testing.assert_array_equal(
        vector_map,
        np.array(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [1.0, 0.5, -0.0, -0.5, -1.0],
                [-4.0, -2.0, 0.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )


def test_samples_in_uvw_box_returns_constant_cost_from_constant_fx() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=2, rw=2)
    fx = np.full((7, 8, 9), 0.25, dtype=np.float32)

    costs = voter.samples_in_uvw_box(
        c1=4,
        c2=3,
        c3=2,
        normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dip=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1, 2 * voter.ru + 1)
    assert costs.dtype == np.float32
    expected = np.ones((5, 5, 5), dtype=np.float32)
    for kw in range(expected.shape[0]):
        for kv in range(expected.shape[1]):
            ku_min = voter.lmins[kw, kv] + voter.ru
            ku_max = voter.lmaxs[kw, kv] + voter.ru
            expected[kw, kv, ku_min : ku_max + 1] = 0.75
    np.testing.assert_array_equal(costs, expected)


def test_samples_in_uvw_box_respects_surface_shift_ranges() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=2, rw=2)
    fx = np.full((7, 8, 9), 0.25, dtype=np.float32)

    costs = voter.samples_in_uvw_box(
        c1=4,
        c2=3,
        c3=2,
        normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dip=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    for kw in range(2 * voter.rw + 1):
        for kv in range(2 * voter.rv + 1):
            ku_min = voter.lmins[kw, kv] + voter.ru
            ku_max = voter.lmaxs[kw, kv] + voter.ru

            if ku_min > 0:
                np.testing.assert_array_equal(costs[kw, kv, :ku_min], 1.0)
            if ku_max + 1 < costs.shape[2]:
                np.testing.assert_array_equal(costs[kw, kv, ku_max + 1 :], 1.0)
            np.testing.assert_array_equal(costs[kw, kv, ku_min : ku_max + 1], 0.75)


def test_samples_in_uvw_box_uses_n3_n2_n1_volume_indexing() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=2, rw=2)
    i3, i2, i1 = np.indices((7, 8, 9), dtype=np.float32)
    fx = 100.0 * i3 + 10.0 * i2 + i1

    costs = voter.samples_in_uvw_box(
        c1=4,
        c2=3,
        c3=2,
        normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dip=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fx=fx,
    )

    assert costs[voter.rw, voter.rv, voter.ru] == pytest.approx(1.0 - fx[2, 3, 4])
    assert costs[0, 0, 4] == pytest.approx(1.0 - fx[0, 1, 6])
    assert costs[4, 4, 0] == pytest.approx(1.0 - fx[4, 5, 2])


def test_samples_in_uvw_box_rounds_and_clamps_near_volume_boundary() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=2, rw=2)
    i3, i2, i1 = np.indices((3, 4, 5), dtype=np.float32)
    fx = 100.0 * i3 + 10.0 * i2 + i1

    costs = voter.samples_in_uvw_box(
        c1=0,
        c2=0,
        c3=0,
        normal=np.array([0.6, 0.0, 0.0], dtype=np.float32),
        dip=np.array([0.0, 0.6, 0.0], dtype=np.float32),
        strike=np.array([0.0, 0.0, 0.6], dtype=np.float32),
        fx=fx,
    )

    assert costs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1, 2 * voter.ru + 1)
    assert np.isfinite(costs).all()
    assert costs[0, 0, 0] == pytest.approx(1.0 - fx[0, 0, 0])
    assert costs[4, 4, 4] == pytest.approx(1.0 - fx[1, 1, 1])


def test_surface_voting_adds_votes_on_high_likelihood_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 3:8] = 0.8
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)

    voter._surface_voting(FaultCell(5, 5, 5, 0.8, 0.0, 90.0), ft, fe, vp, vt, vm)

    expected_mask = np.zeros_like(ft, dtype=np.bool_)
    expected_mask[3:8, 4:7, 3:8] = True
    assert fe[expected_mask].sum() == pytest.approx(60.0)
    assert np.count_nonzero(fe) == 75
    np.testing.assert_allclose(fe[expected_mask], 0.8)
    np.testing.assert_allclose(vm[expected_mask], 0.8)
    np.testing.assert_allclose(vp[expected_mask], 0.0, atol=1e-7)
    np.testing.assert_allclose(vt[expected_mask], 90.0, atol=1e-5)
    np.testing.assert_array_equal(fe[~expected_mask], np.zeros_like(fe[~expected_mask]))
    np.testing.assert_array_equal(vm[~expected_mask], np.zeros_like(vm[~expected_mask]))
    np.testing.assert_array_equal(vp[~expected_mask], np.full_like(vp[~expected_mask], -1.0))
    np.testing.assert_array_equal(vt[~expected_mask], np.full_like(vt[~expected_mask], -1.0))
    for array in (fe, vp, vt, vm):
        assert array.shape == ft.shape
        assert array.dtype == np.float32


def test_surface_voting_keeps_stronger_orientation_when_later_vote_is_weaker() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft_strong = np.zeros((11, 11, 11), dtype=np.float32)
    ft_strong[3:8, 5, 3:8] = 0.8
    ft_weak = np.zeros_like(ft_strong)
    ft_weak[5, 3:8, 3:8] = 0.2
    fe = np.zeros_like(ft_strong)
    vp = np.full_like(ft_strong, -1.0)
    vt = np.full_like(ft_strong, -1.0)
    vm = np.zeros_like(ft_strong)

    voter._surface_voting(FaultCell(5, 5, 5, 0.8, 0.0, 90.0), ft_strong, fe, vp, vt, vm)
    voter._surface_voting(FaultCell(5, 5, 5, 0.2, 90.0, 90.0), ft_weak, fe, vp, vt, vm)

    assert fe[5, 5, 5] == pytest.approx(1.0)
    assert vm[5, 5, 5] == pytest.approx(0.8)
    assert vp[5, 5, 5] == pytest.approx(0.0, abs=1e-7)
    assert vt[5, 5, 5] == pytest.approx(90.0, abs=1e-5)


def test_surface_voting_accepts_boundary_surface_samples() -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.ones((3, 3, 3), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)

    voter._surface_voting(FaultCell(0, 0, 0, 1.0, 0.0, 90.0), ft, fe, vp, vt, vm)

    expected_mask = np.zeros_like(ft, dtype=np.bool_)
    expected_mask[0:2, 0:2, 0:2] = True
    np.testing.assert_allclose(fe[expected_mask], 1.0)
    np.testing.assert_allclose(vm[expected_mask], 1.0)
    np.testing.assert_allclose(vp[expected_mask], 0.0, atol=1e-7)
    np.testing.assert_allclose(vt[expected_mask], 90.0, atol=1e-5)
    np.testing.assert_array_equal(fe[~expected_mask], np.zeros_like(fe[~expected_mask]))
    np.testing.assert_array_equal(vm[~expected_mask], np.zeros_like(vm[~expected_mask]))
    np.testing.assert_array_equal(vp[~expected_mask], np.full_like(vp[~expected_mask], -1.0))
    np.testing.assert_array_equal(vt[~expected_mask], np.full_like(vt[~expected_mask], -1.0))


def test_surface_voting_is_deterministic_for_same_seed_and_inputs() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 3:8] = 0.8
    first = (
        np.zeros_like(ft),
        np.full_like(ft, -1.0),
        np.full_like(ft, -1.0),
        np.zeros_like(ft),
    )
    second = tuple(array.copy() for array in first)

    voter._surface_voting(FaultCell(5, 5, 5, 0.8, 0.0, 90.0), ft, *first)
    voter._surface_voting(FaultCell(5, 5, 5, 0.8, 0.0, 90.0), ft, *second)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_apply_voting_returns_zero_arrays_when_no_seeds_are_selected() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    ft = np.zeros((7, 8, 9), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    fv, vp, vt = voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    for array in (fv, vp, vt):
        assert array.shape == ft.shape
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, np.zeros_like(ft))


def test_apply_voting_accepts_empty_n3_volume() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    ft = np.zeros((0, 8, 9), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    fv, vp, vt = voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    for array in (fv, vp, vt):
        assert array.shape == (0, 8, 9)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, np.zeros_like(ft))


def test_apply_voting_rejects_mismatched_shapes() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    ft = np.zeros((2, 3, 4), dtype=np.float32)
    pt = np.zeros((2, 4, 3), dtype=np.float32)
    tt = np.zeros_like(ft)

    with pytest.raises(ValueError, match="shapes must match"):
        voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)


@pytest.mark.parametrize(
    ("ft_value", "pt_value", "tt_value", "message"),
    [
        (np.nan, 0.0, 0.0, "ft"),
        (0.0, np.inf, 0.0, "pt"),
        (0.0, 0.0, np.nan, "tt"),
    ],
)
def test_apply_voting_rejects_nonfinite_inputs(
    ft_value: float,
    pt_value: float,
    tt_value: float,
    message: str,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    ft = np.zeros((3, 3, 3), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)
    ft[1, 1, 1] = ft_value
    pt[1, 1, 1] = pt_value
    tt[1, 1, 1] = tt_value

    with pytest.raises(ValueError, match=message):
        voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)


def test_thin_returns_finite_float32_volume_without_modifying_inputs() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[:, 1, :] = 0.5
    fv[:, 2, :] = 1.0
    fv[:, 3, :] = 0.5
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 45.0)
    fv_before = fv.copy()
    vp_before = vp.copy()
    vt_before = vt.copy()

    fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    np.testing.assert_array_equal(fv, fv_before)
    np.testing.assert_array_equal(vp, vp_before)
    np.testing.assert_array_equal(vt, vt_before)


def test_thin_normal_mode_matches_default_behavior() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[:, 1, :] = 0.5
    fv[:, 2, :] = 1.0
    fv[:, 3, :] = 0.5
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 45.0)

    default = voter.thin(fv, vp, vt)
    normal = voter.thin(fv, vp, vt, mode="normal")

    np.testing.assert_array_equal(normal, default)


def test_thin_reference_mode_returns_float32_shape_and_original_values() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 2), dtype=np.float32)
    fv[2, 1, :] = 0.25
    fv[2, 2, :] = [0.75, 1.0]
    fv[2, 3, :] = 0.5
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    fv_before = fv.copy()
    vp_before = vp.copy()
    vt_before = vt.copy()

    fvt = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    retained = fvt != 0.0
    assert retained.any()
    np.testing.assert_array_equal(fvt[retained], fv[retained])
    np.testing.assert_array_equal(fv, fv_before)
    np.testing.assert_array_equal(vp, vp_before)
    np.testing.assert_array_equal(vt, vt_before)


def test_thin_reference_mode_reinforces_vertical_strike_neighbor() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    fv = np.zeros((7, 7, 1), dtype=np.float32)
    fv[3, 3, 0] = 10.0
    vp = np.full_like(fv, 90.0)
    vt = np.full_like(fv, 45.0)
    expected, keep = reference_like_3d_thin_values(
        fv,
        vp,
        sigma=1.0,
        reinforce_vertical=True,
    )

    fvt = voter.thin(fv, vp, vt, mode="reference", reference_sigma=1.0)
    scanner_ft, _, _ = scanner.thin(
        fv,
        vp,
        vt,
        mode="reference",
        reference_sigma=1.0,
    )

    assert keep[3, 3, 0]
    np.testing.assert_allclose(fvt, expected)
    assert fvt[2, 3, 0] == pytest.approx(float(fvt[3, 3, 0]))
    assert scanner_ft[2, 3, 0] == np.float32(0.0)


def test_thin_rejects_invalid_mode() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="mode"):
        voter.thin(fv, vp, vt, mode="nearest")


def test_thin_reference_mode_uses_strike_bin_nms_in_i2_i3_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    normal = voter.thin(fv, vp, vt)
    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)

    assert np.count_nonzero(reference) > 0
    assert np.count_nonzero(normal) > 0
    assert np.count_nonzero(reference[:, 2, :]) == np.count_nonzero(reference)
    assert np.count_nonzero(normal[:, :, 2]) == np.count_nonzero(normal)
    assert np.count_nonzero(reference != normal) > 0


def test_thin_narrows_planar_ridge_along_fault_normal() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((7, 9, 7), dtype=np.float32)
    fv[1:6, 3, 1:6] = 0.6
    fv[1:6, 4, 1:6] = 1.0
    fv[1:6, 5, 1:6] = 0.6
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.count_nonzero(fvt) < np.count_nonzero(fv)
    assert np.count_nonzero(fvt[:, 4, :]) > 0
    assert np.count_nonzero(fvt[:, :4, :]) == 0
    assert np.count_nonzero(fvt[:, 5:, :]) == 0


def test_thin_suppresses_broad_planar_ridge_to_center_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((7, 9, 7), dtype=np.float32)
    fv[1:6, 2, 1:6] = 0.7
    fv[1:6, 3:6, 1:6] = 1.0
    fv[1:6, 6, 1:6] = 0.7
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    assert np.count_nonzero(fvt) < np.count_nonzero(fv)
    assert np.count_nonzero(fvt[:, 4, :]) == 25
    assert np.count_nonzero(fvt[:, :4, :]) == 0
    assert np.count_nonzero(fvt[:, 5:, :]) == 0
    np.testing.assert_array_equal(fvt[:, 4, :], fv[:, 4, :])


def test_thin_returns_zero_for_flat_volume() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.full((4, 5, 6), 0.75, dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    with np.errstate(all="raise"):
        fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    np.testing.assert_array_equal(fvt, np.zeros_like(fv))


def test_thin_zero_orientation_angles_return_finite_scores() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[:, :, 2] = 1.0
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with np.errstate(all="raise"):
        fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()


def test_thin_rejects_mismatched_shapes() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((2, 3, 4), dtype=np.float32)
    vp = np.zeros((2, 4, 3), dtype=np.float32)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="shapes must match"):
        voter.thin(fv, vp, vt)


@pytest.mark.parametrize(
    ("fv_value", "vp_value", "vt_value", "message"),
    [
        (np.nan, 0.0, 0.0, "fv"),
        (0.0, np.inf, 0.0, "vp"),
        (0.0, 0.0, np.nan, "vt"),
    ],
)
def test_thin_rejects_nonfinite_inputs(
    fv_value: float,
    vp_value: float,
    vt_value: float,
    message: str,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    fv[1, 1, 1] = fv_value
    vp[1, 1, 1] = vp_value
    vt[1, 1, 1] = vt_value

    with pytest.raises(ValueError, match=message):
        voter.thin(fv, vp, vt)


@pytest.mark.parametrize(
    "fv",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
    ],
)
def test_thin_rejects_non_3d_arrays(fv: np.ndarray) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)

    with pytest.raises(ValueError, match="fv must be a 3D array"):
        voter.thin(
            fv,
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((2, 3, 4), dtype=np.float32),
        )


def test_apply_voting_highlights_simple_fault_like_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.9
    plane_mask = np.zeros_like(ft, dtype=np.bool_)
    plane_mask[3:8, 5, 3:8] = True
    background_mask = np.zeros_like(ft, dtype=np.bool_)
    background_mask[3:8, 2, 3:8] = True
    background_mask[3:8, 8, 3:8] = True

    fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert fv.shape == ft.shape
    assert vp.shape == ft.shape
    assert vt.shape == ft.shape
    assert fv.dtype == np.float32
    assert vp.dtype == np.float32
    assert vt.dtype == np.float32
    assert np.isfinite(fv).all()
    assert np.isfinite(vp).all()
    assert np.isfinite(vt).all()
    assert fv.min() >= -1e-6
    assert fv.max() <= 1.0 + 1e-6
    assert fv.max() > 0.0
    assert fv[plane_mask].mean() > fv[background_mask].mean()
    assert np.count_nonzero(fv[plane_mask]) > 0


def test_apply_voting_then_thin_returns_sparse_plane_maxima() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.9
    plane_mask = np.zeros_like(ft, dtype=np.bool_)
    plane_mask[3:8, 5, 3:8] = True
    near_plane_mask = np.zeros_like(ft, dtype=np.bool_)
    near_plane_mask[:, 4:7, :] = True

    fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    fvt = voter.thin(fv, vp, vt)

    assert fvt.shape == ft.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    assert np.count_nonzero(fvt) < np.count_nonzero(fv)
    assert np.count_nonzero(fvt[plane_mask]) > 0
    assert fvt[plane_mask].mean() > fvt[~near_plane_mask].mean()
    assert np.argwhere(fvt == fvt.max())[:, 1].min() >= 4
    assert np.argwhere(fvt == fvt.max())[:, 1].max() <= 6


def test_apply_voting_is_deterministic_for_same_inputs() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.9

    first = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_apply_voting_localizes_broad_gently_dipping_ridge() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    n3, n2, n1 = 13, 14, 13
    ft = np.zeros((n3, n2, n1), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 75.0)
    near_surface = np.zeros_like(ft, dtype=np.bool_)
    far_from_surface = np.zeros_like(ft, dtype=np.bool_)
    center1 = 6.0
    center2 = 6.0
    slope = np.float32(np.tan(np.deg2rad(15.0)))

    for i3 in range(3, 10):
        for i1 in range(2, 11):
            surface_i2 = center2 + slope * (i1 - center1)
            for i2 in range(2, 12):
                distance = abs(i2 - surface_i2)
                if distance <= 1.25:
                    ft[i3, i2, i1] = np.exp(-0.5 * (distance / 0.75) ** 2)
                    near_surface[i3, i2, i1] = True
                if distance >= 4.0:
                    far_from_surface[i3, i2, i1] = True

    fv, vp, vt = voter.apply_voting(d=4, fm=0.55, ft=ft, pt=pt, tt=tt)
    fv_second, vp_second, vt_second = voter.apply_voting(d=4, fm=0.55, ft=ft, pt=pt, tt=tt)

    assert fv.shape == (n3, n2, n1)
    assert vp.shape == (n3, n2, n1)
    assert vt.shape == (n3, n2, n1)
    for array in (fv, vp, vt):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert fv.min() >= -1e-6
    assert fv.max() <= 1.0 + 1e-6
    assert np.count_nonzero(fv[near_surface]) > 0
    assert fv[near_surface].mean() > fv[far_from_surface].mean()
    np.testing.assert_array_equal(fv, fv_second)
    np.testing.assert_array_equal(vp, vp_second)
    np.testing.assert_array_equal(vt, vt_second)


def test_apply_voting_handles_small_volume_with_clipped_local_boxes() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=3, rw=3)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[1:4, 2, 1:4] = 0.9
    plane_mask = np.zeros_like(ft, dtype=np.bool_)
    plane_mask[1:4, 2, 1:4] = True

    fv, vp, vt = voter.apply_voting(d=2, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert fv.shape == (5, 5, 5)
    assert vp.shape == (5, 5, 5)
    assert vt.shape == (5, 5, 5)
    for array in (fv, vp, vt):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert fv.min() >= -1e-6
    assert fv.max() <= 1.0 + 1e-6
    assert fv.max() > 0.0
    assert fv[plane_mask].mean() > fv[~plane_mask].mean()
    max_mask = fv == fv.max()
    assert np.any(max_mask & plane_mask)


def test_normalize_and_power_3d_zero_dynamic_range_returns_finite_zeros() -> None:
    volume = np.full((2, 3, 4), 7.5, dtype=np.float32)

    with np.errstate(all="raise"):
        scores = _normalize_and_power_3d(volume)

    assert scores.shape == volume.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, np.zeros_like(volume))


def test_normalize_and_power_3d_simple_ramp_uses_min_max_and_power() -> None:
    volume = np.array([[[2.0, 3.0, 4.0]]], dtype=np.float32)

    scores = _normalize_and_power_3d(volume, sigma=0.0, power=4)

    expected = np.array([[[0.0, 0.9375, 1.0]]], dtype=np.float32)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-7)


def test_smooth_fault_likelihood_3d_preserves_shape_and_bounds() -> None:
    volume = np.zeros((5, 6, 7), dtype=np.float32)
    volume[2, 3, 4] = 10.0

    smoothed = _smooth_fault_likelihood_3d(volume, sigma=1.0)

    assert smoothed.shape == volume.shape
    assert smoothed.dtype == np.float32
    assert np.isfinite(smoothed).all()
    assert smoothed.min() >= -1e-6
    assert smoothed.max() <= 1.0 + 1e-6
    assert smoothed[2, 3, 4] == pytest.approx(1.0)


def test_smooth_fault_likelihood_3d_zero_dynamic_range_returns_finite_zeros() -> None:
    volume = np.full((3, 4, 5), 2.0, dtype=np.float32)

    with np.errstate(all="raise"):
        smoothed = _smooth_fault_likelihood_3d(volume, sigma=1.0)

    assert smoothed.shape == volume.shape
    assert smoothed.dtype == np.float32
    assert np.isfinite(smoothed).all()
    np.testing.assert_array_equal(smoothed, np.zeros_like(volume))


def test_surface_strike_and_dip_flat_surface_recovers_seed_orientation() -> None:
    strike_angle = 30.0
    dip_angle = 60.0
    normal = fault_normal_vector_from_strike_and_dip(strike_angle, dip_angle)
    dip = fault_dip_vector_from_strike_and_dip(strike_angle, dip_angle)
    strike = fault_strike_vector_from_strike_and_dip(strike_angle, dip_angle)
    surface = np.zeros((5, 5), dtype=np.float32)

    actual_strike, actual_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=None,
    )

    assert np.isfinite(actual_strike)
    assert np.isfinite(actual_dip)
    assert actual_strike == pytest.approx(strike_angle, abs=1e-5)
    assert actual_dip == pytest.approx(dip_angle, abs=1e-5)


def test_surface_strike_and_dip_sloped_surface_returns_finite_angles() -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    w, v = np.indices((5, 5), dtype=np.float32)
    surface = 0.25 * (v - 2.0) - 0.5 * (w - 2.0)

    actual_strike, actual_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=0.0,
    )

    assert np.isfinite(actual_strike)
    assert np.isfinite(actual_dip)
    assert 0.0 <= actual_strike < 360.0
    assert 0.0 <= actual_dip <= 180.0
