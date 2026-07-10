import numpy as np
import pytest

import pyosv.voting3d as voting3d
from pyosv.cells import FaultCell
from pyosv.dp import update_shift_ranges_3d
from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
    strike_and_dip_from_local_surface_derivatives,
)
from pyosv.orient3d import FaultOrientScanner3
from pyosv.thinning3d import reference_like_3d_thin_values, remove_reference_edge_effects_3d
from pyosv.voting3d import (
    OptimalSurfaceVoter,
    _accumulate_surface_votes,
    _normalize_and_power_3d,
    _smooth_fault_likelihood_3d,
    _surface_strike_and_dip,
    _surface_vote_average,
)


def _java_style_round(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _axis_aligned_ramp_volume(shape: tuple[int, int, int]) -> np.ndarray:
    i3, i2, i1 = np.indices(shape, dtype=np.float32)
    return (0.2 + 0.01 * i1 + 0.02 * i2 + 0.04 * i3).astype(np.float32)


def _angle_distance_degrees(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _expected_uvw_costs(
    voter: OptimalSurfaceVoter,
    c1: int,
    c2: int,
    c3: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fx.shape
    costs = np.ones((2 * voter.rw + 1, 2 * voter.rv + 1, 2 * voter.ru + 1), dtype=np.float32)
    for kw in range(costs.shape[0]):
        iw = kw - voter.rw
        for kv in range(costs.shape[1]):
            iv = kv - voter.rv
            ku_min = voter.lmins[kw, kv] + voter.ru
            ku_max = voter.lmaxs[kw, kv] + voter.ru
            for ku in range(ku_min, ku_max + 1):
                iu = ku - voter.ru
                x1 = np.float32(c1 + iw * strike[0] + iv * dip[0] + iu * normal[0])
                x2 = np.float32(c2 + iw * strike[1] + iv * dip[1] + iu * normal[1])
                x3 = np.float32(c3 + iw * strike[2] + iv * dip[2] + iu * normal[2])
                j1 = min(max(_java_style_round(float(x1)), 0), n1 - 1)
                j2 = min(max(_java_style_round(float(x2)), 0), n2 - 1)
                j3 = min(max(_java_style_round(float(x3)), 0), n3 - 1)
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]
    return costs


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
    assert voter.surface_orientation_smoothing == 2.0
    assert voter.final_normalization_smoothing == 0.0
    assert voter.surface_support_min_fraction == 0.0
    assert voter.surface_support_exponent == 0.0
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


def test_constructor_initializes_surface_orientation_smoothing_from_max_radius() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=5)

    assert voter.surface_orientation_smoothing == 5.0


def test_constructor_allows_zero_surface_orientation_smoothing_default() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=0, rw=0)

    assert voter.surface_orientation_smoothing == 0.0


def test_shift_range_arrays_match_surface_radius_shape() -> None:
    voter = OptimalSurfaceVoter(ru=5, rv=6, rw=7)

    assert voter.lmins.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)
    assert voter.lmaxs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)


def test_update_shift_ranges_3d_local_uvw_masks_use_rounded_radial_distance() -> None:
    ru, rv, rw = 4, 4, 4

    lmins, lmaxs = update_shift_ranges_3d(ru, rv, rw)

    assert lmins.shape == (2 * rw + 1, 2 * rv + 1)
    assert lmaxs.shape == (2 * rw + 1, 2 * rv + 1)
    np.testing.assert_array_equal(lmaxs, -lmins)
    for iw in range(-rw, rw + 1):
        for iv in range(-rv, rv + 1):
            kw = iw + rw
            kv = iv + rv
            radial_distance = float(np.sqrt(iw * iw + iv * iv))
            if radial_distance <= 2.0:
                assert lmins[kw, kv] == 0
                assert lmaxs[kw, kv] == 0
            else:
                expected_shift = min(_java_style_round(radial_distance), ru)
                assert lmins[kw, kv] == -expected_shift
                assert lmaxs[kw, kv] == expected_shift

    assert lmins[rw, rv + 2] == 0
    assert lmaxs[rw, rv + 2] == 0
    assert lmins[rw, rv + 3] == -3
    assert lmaxs[rw, rv + 3] == 3
    assert lmins[rw + 2, rv + 2] == -3
    assert lmaxs[rw + 2, rv + 2] == 3
    assert lmins[rw + 2, rv + 3] == -4
    assert lmaxs[rw + 2, rv + 3] == 4


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


@pytest.mark.parametrize("surface_orientation_smoothing", [0.0, np.float32(2.5)])
def test_set_surface_orientation_smoothing_accepts_nonnegative_finite_numbers(
    surface_orientation_smoothing: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_surface_orientation_smoothing(surface_orientation_smoothing)

    assert voter.surface_orientation_smoothing == float(surface_orientation_smoothing)
    assert isinstance(voter.surface_orientation_smoothing, float)


@pytest.mark.parametrize(
    "surface_orientation_smoothing",
    [-0.1, np.nan, np.inf, True, "1.0"],
)
def test_set_surface_orientation_smoothing_rejects_invalid_values(
    surface_orientation_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_orientation_smoothing"):
        voter.set_surface_orientation_smoothing(
            surface_orientation_smoothing,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("final_normalization_smoothing", [0.0, 1.0, np.float32(2.0)])
def test_set_final_normalization_smoothing_accepts_nonnegative_finite_numbers(
    final_normalization_smoothing: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_final_normalization_smoothing(final_normalization_smoothing)

    assert voter.final_normalization_smoothing == float(final_normalization_smoothing)
    assert isinstance(voter.final_normalization_smoothing, float)


@pytest.mark.parametrize(
    "final_normalization_smoothing",
    [-0.1, np.nan, np.inf, True, "1.0"],
)
def test_set_final_normalization_smoothing_rejects_invalid_values(
    final_normalization_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="final_normalization_smoothing"):
        voter.set_final_normalization_smoothing(
            final_normalization_smoothing,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("min_fraction", "exponent"),
    [
        (0.0, 0.0),
        (0.5, 1.0),
        (1.0, 2.5),
        (np.float32(0.25), np.float32(1.5)),
    ],
)
def test_set_surface_support_policy_accepts_valid_values(
    min_fraction: float,
    exponent: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_surface_support_policy(min_fraction=min_fraction, exponent=exponent)

    assert voter.surface_support_min_fraction == float(min_fraction)
    assert voter.surface_support_exponent == float(exponent)


@pytest.mark.parametrize("min_fraction", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_set_surface_support_policy_rejects_invalid_min_fraction(
    min_fraction: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_support_min_fraction"):
        voter.set_surface_support_policy(
            min_fraction=min_fraction,  # type: ignore[arg-type]
            exponent=0.0,
        )


@pytest.mark.parametrize("exponent", [-0.1, np.nan, np.inf, True, "1.0"])
def test_set_surface_support_policy_rejects_invalid_exponent(exponent: object) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_support_exponent"):
        voter.set_surface_support_policy(
            min_fraction=0.0,
            exponent=exponent,  # type: ignore[arg-type]
        )


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


def test_samples_in_uvw_box_local_axis_convention_maps_w_v_u_to_strike_dip_normal() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=3, rw=3)
    fx = _axis_aligned_ramp_volume((11, 12, 13))
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    costs = voter.samples_in_uvw_box(
        c1=6,
        c2=5,
        c3=4,
        normal=normal,
        dip=dip,
        strike=strike,
        fx=fx,
    )

    expected = _expected_uvw_costs(voter, 6, 5, 4, normal, dip, strike, fx)
    assert costs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1, 2 * voter.ru + 1)
    np.testing.assert_array_equal(costs, expected)
    assert costs[voter.rw + 3, voter.rv, voter.ru + 1] == pytest.approx(
        1.0 - fx[7, 5, 7],
    )
    assert costs[voter.rw, voter.rv + 3, voter.ru - 2] == pytest.approx(
        1.0 - fx[4, 8, 4],
    )
    assert costs[voter.rw + 2, voter.rv + 2, voter.ru + 3] == pytest.approx(
        1.0 - fx[6, 7, 9],
    )
    assert costs[voter.rw, voter.rv, voter.ru + 1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("c1", "c2", "c3"),
    [
        (0, 2, 2),
        (4, 2, 2),
        (2, 0, 2),
        (2, 4, 2),
        (2, 2, 0),
        (2, 2, 4),
        (0, 0, 0),
        (4, 4, 4),
    ],
)
def test_samples_in_uvw_box_local_boundary_clamps_faces_and_preserves_invalid_mask(
    c1: int,
    c2: int,
    c3: int,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=3, rw=3)
    fx = _axis_aligned_ramp_volume((5, 5, 5))
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    costs = voter.samples_in_uvw_box(c1, c2, c3, normal, dip, strike, fx)

    expected = _expected_uvw_costs(voter, c1, c2, c3, normal, dip, strike, fx)
    np.testing.assert_array_equal(costs, expected)
    valid_mask = np.zeros(costs.shape, dtype=np.bool_)
    for kw in range(costs.shape[0]):
        for kv in range(costs.shape[1]):
            ku_min = voter.lmins[kw, kv] + voter.ru
            ku_max = voter.lmaxs[kw, kv] + voter.ru
            valid_mask[kw, kv, ku_min : ku_max + 1] = True
    np.testing.assert_array_equal(costs[~valid_mask], np.float32(1.0))
    assert (costs[valid_mask] < np.float32(1.0)).all()


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


@pytest.mark.parametrize(
    ("surface_orientation_smoothing", "expected_sigma"),
    [(None, 3.0), (0.0, 0.0)],
)
def test_surface_voting_passes_configured_orientation_smoothing(
    monkeypatch: pytest.MonkeyPatch,
    surface_orientation_smoothing: float | None,
    expected_sigma: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=3, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    if surface_orientation_smoothing is not None:
        voter.set_surface_orientation_smoothing(surface_orientation_smoothing)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 2:9] = 0.8
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)
    sigmas: list[float | None] = []

    def wrapped_surface_strike_and_dip(
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        surface: np.ndarray,
        *,
        sigma: float | None = None,
    ) -> tuple[float, float]:
        sigmas.append(sigma)
        return _surface_strike_and_dip(
            normal,
            dip,
            strike,
            surface,
            sigma=sigma,
        )

    monkeypatch.setattr(
        voting3d,
        "_surface_strike_and_dip",
        wrapped_surface_strike_and_dip,
    )

    voter._surface_voting(FaultCell(5, 5, 5, 0.8, 0.0, 90.0), ft, fe, vp, vt, vm)

    assert sigmas == [expected_sigma]
    assert np.isfinite(vp[fe > 0.0]).all()
    assert np.isfinite(vt[fe > 0.0]).all()


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


def test_surface_voting_excludes_boundary_surface_samples() -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.ones((3, 3, 3), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)

    voter._surface_voting(FaultCell(0, 0, 0, 1.0, 0.0, 90.0), ft, fe, vp, vt, vm)

    np.testing.assert_array_equal(fe, np.zeros_like(fe))
    np.testing.assert_array_equal(vm, np.zeros_like(vm))
    np.testing.assert_array_equal(vp, np.full_like(vp, -1.0))
    np.testing.assert_array_equal(vt, np.full_like(vt, -1.0))


def test_surface_vote_average_reference_audit_returns_zero_for_edge_only_surface() -> None:
    ft = np.ones((3, 3, 3), dtype=np.float32)

    fa, valid_count = _surface_vote_average(
        1,
        0,
        1,
        0,
        0,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        ft,
    )

    assert valid_count == 0
    assert fa == np.float32(0.0)


def test_surface_voting_no_ops_when_surface_has_no_valid_samples() -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=0, rw=0)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.ones((3, 3, 3), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)

    voter._surface_voting(FaultCell(1, 0, 1, 1.0, 0.0, 90.0), ft, fe, vp, vt, vm)

    np.testing.assert_array_equal(fe, np.zeros_like(fe))
    np.testing.assert_array_equal(vm, np.zeros_like(vm))
    np.testing.assert_array_equal(vp, np.full_like(vp, -1.0))
    np.testing.assert_array_equal(vt, np.full_like(vt, -1.0))


def test_surface_voting_skips_vote_below_support_fraction_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_surface_support_policy(min_fraction=0.5, exponent=0.0)
    ft = np.ones((5, 5, 5), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)
    surface = np.zeros((3, 3), dtype=np.float32)
    accumulate_calls: list[np.float32] = []

    monkeypatch.setattr(voting3d, "find_surface_3d", lambda *args, **kwargs: surface)
    monkeypatch.setattr(
        voting3d,
        "_surface_vote_average",
        lambda *args: (np.float32(0.8), 4),
    )

    def accumulate_stub(
        c1: int,
        c2: int,
        c3: int,
        rv: int,
        rw: int,
        fa: np.float32,
        vp_value: np.float32,
        vt_value: np.float32,
        align_i3: bool,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        surface: np.ndarray,
        fe: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        vm: np.ndarray,
    ) -> None:
        del c1, c2, c3, rv, rw, vp_value, vt_value, align_i3, normal, dip, strike
        del surface, fe, vp, vt, vm
        accumulate_calls.append(fa)

    monkeypatch.setattr(voting3d, "_accumulate_surface_votes", accumulate_stub)

    voter._surface_voting(FaultCell(2, 2, 2, 1.0, 0.0, 90.0), ft, fe, vp, vt, vm)

    assert accumulate_calls == []
    np.testing.assert_array_equal(fe, np.zeros_like(fe))
    np.testing.assert_array_equal(vm, np.zeros_like(vm))


def test_surface_voting_downweights_vote_by_support_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_surface_support_policy(min_fraction=0.0, exponent=1.0)
    ft = np.ones((5, 5, 5), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.full_like(ft, -1.0)
    vt = np.full_like(ft, -1.0)
    vm = np.zeros_like(ft)
    surface = np.zeros((3, 3), dtype=np.float32)
    accumulated_fa: list[np.float32] = []

    monkeypatch.setattr(voting3d, "find_surface_3d", lambda *args, **kwargs: surface)
    monkeypatch.setattr(
        voting3d,
        "_surface_vote_average",
        lambda *args: (np.float32(0.9), 4),
    )
    monkeypatch.setattr(
        voting3d,
        "_surface_strike_and_dip",
        lambda *args, **kwargs: (0.0, 90.0),
    )

    def accumulate_stub(
        c1: int,
        c2: int,
        c3: int,
        rv: int,
        rw: int,
        fa: np.float32,
        vp_value: np.float32,
        vt_value: np.float32,
        align_i3: bool,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        surface: np.ndarray,
        fe: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        vm: np.ndarray,
    ) -> None:
        del c1, c2, c3, rv, rw, vp_value, vt_value, align_i3, normal, dip, strike
        del surface, fe, vp, vt, vm
        accumulated_fa.append(fa)

    monkeypatch.setattr(voting3d, "_accumulate_surface_votes", accumulate_stub)

    voter._surface_voting(FaultCell(2, 2, 2, 1.0, 0.0, 90.0), ft, fe, vp, vt, vm)

    assert len(accumulated_fa) == 1
    assert accumulated_fa[0] == pytest.approx(np.float32(0.9 * 4.0 / 9.0))


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


def test_surface_vote_average_reference_audit_counts_small_plane_samples() -> None:
    # Audits OptimalSurfaceVoter.surfaceVoting average fault-attribute sampling.
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    expected_samples = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    ft[1:4, 1:4, 2] = expected_samples

    fa, valid_count = _surface_vote_average(
        2,
        2,
        2,
        1,
        1,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        ft,
    )

    assert valid_count == 9
    assert fa == pytest.approx(float(expected_samples.mean()))


def test_accumulate_surface_votes_reference_audit_writes_and_orientation_threshold() -> None:
    # Audits surfaceVoting accumulation and the reference "fa > vm" orientation rule.
    fe = np.zeros((5, 5, 5), dtype=np.float32)
    vp = np.full_like(fe, -1.0)
    vt = np.full_like(fe, -1.0)
    vm = np.zeros_like(fe)
    vm[2, 2, 2] = np.float32(0.6)
    vp[2, 2, 2] = np.float32(99.0)
    vt[2, 2, 2] = np.float32(88.0)

    _accumulate_surface_votes(
        2,
        2,
        2,
        1,
        1,
        np.float32(0.25),
        np.float32(12.0),
        np.float32(34.0),
        False,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        fe,
        vp,
        vt,
        vm,
    )

    expected_fe = np.zeros_like(fe)
    for i3 in range(1, 4):
        expected_fe[i3, 0, 2] = 0.25
        expected_fe[i3, 1, 2] = 0.50
        expected_fe[i3, 2, 2] = 0.75
        expected_fe[i3, 3, 2] = 0.50
        expected_fe[i3, 4, 2] = 0.25
    np.testing.assert_array_equal(fe, expected_fe)
    assert vp[2, 2, 2] == np.float32(99.0)
    assert vt[2, 2, 2] == np.float32(88.0)
    assert vm[2, 2, 2] == np.float32(0.6)
    updated = expected_fe > 0.0
    updated[2, 2, 2] = False
    np.testing.assert_array_equal(
        vp[updated],
        np.full(np.count_nonzero(updated), 12.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        vt[updated],
        np.full(np.count_nonzero(updated), 34.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        vm[updated],
        np.full(np.count_nonzero(updated), 0.25, dtype=np.float32),
    )


def test_accumulate_surface_votes_reference_audit_excludes_edge_source_samples() -> None:
    fe = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fe)
    vt = np.zeros_like(fe)
    vm = np.zeros_like(fe)

    _accumulate_surface_votes(
        1,
        1,
        0,
        0,
        0,
        np.float32(0.5),
        np.float32(90.0),
        np.float32(45.0),
        True,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        fe,
        vp,
        vt,
        vm,
    )

    np.testing.assert_array_equal(fe, np.zeros_like(fe))
    np.testing.assert_array_equal(vm, np.zeros_like(vm))


def test_accumulate_surface_votes_reference_audit_reinforcement_uses_interior_sources() -> None:
    # Audits the neighbor reinforcement path used by surfaceVoting near image faces.
    fe = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fe)
    vt = np.zeros_like(fe)
    vm = np.zeros_like(fe)

    _accumulate_surface_votes(
        1,
        1,
        1,
        0,
        0,
        np.float32(0.5),
        np.float32(90.0),
        np.float32(45.0),
        True,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        fe,
        vp,
        vt,
        vm,
    )

    expected = np.zeros_like(fe)
    expected[0, 1, 1] = np.float32(0.5)
    expected[1, 1, 1] = np.float32(0.5)
    expected[2, 1, 1] = np.float32(0.5)
    np.testing.assert_array_equal(fe, expected)
    np.testing.assert_array_equal(vm, expected)


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


def test_apply_voting_from_seeds_matches_apply_voting_for_same_seed_set() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    ft = np.zeros((7, 7, 7), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[2, 3, 2] = 0.9
    ft[5, 3, 5] = 0.8

    seeds = voter.pick_seeds(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)
    default = voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)
    explicit = voter.apply_voting_from_seeds(seeds, ft=ft, pt=pt, tt=tt)

    for default_array, explicit_array in zip(default, explicit):
        np.testing.assert_array_equal(default_array, explicit_array)


def test_apply_voting_surface_support_default_policy_is_no_op() -> None:
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 3:8] = 0.8
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)

    default_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    default_voter.set_attribute_smoothing(0)
    default_voter.set_surface_smoothing(0.0, 0.0)
    default_voter.set_surface_orientation_smoothing(0.0)
    configured_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    configured_voter.set_attribute_smoothing(0)
    configured_voter.set_surface_smoothing(0.0, 0.0)
    configured_voter.set_surface_orientation_smoothing(0.0)
    configured_voter.set_surface_support_policy(min_fraction=0.0, exponent=0.0)

    default = default_voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)
    configured = configured_voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    for default_array, configured_array in zip(default, configured):
        np.testing.assert_array_equal(default_array, configured_array)


@pytest.mark.parametrize(
    ("final_normalization_smoothing", "expected_sigma"),
    [(None, 0.0), (1.25, 1.25)],
)
def test_apply_voting_passes_configured_final_normalization_smoothing(
    monkeypatch: pytest.MonkeyPatch,
    final_normalization_smoothing: float | None,
    expected_sigma: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    if final_normalization_smoothing is not None:
        voter.set_final_normalization_smoothing(final_normalization_smoothing)

    sigmas: list[float] = []

    def normalize_stub(
        volume: np.ndarray,
        *,
        sigma: float = 0.0,
        power: float = 8.0,
    ) -> np.ndarray:
        del power
        sigmas.append(sigma)
        return np.zeros_like(volume, dtype=np.float32)

    monkeypatch.setattr(voting3d, "_normalize_and_power_3d", normalize_stub)

    ft = np.zeros((2, 3, 4), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.zeros_like(ft)

    fv, _, _ = voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)

    assert sigmas == [expected_sigma]
    assert fv.shape == ft.shape


def test_apply_voting_final_normalization_smoothing_does_not_change_orientation_arrays() -> None:
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 3:8] = 0.8
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)

    default_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    default_voter.set_attribute_smoothing(0)
    default_voter.set_surface_smoothing(0.0, 0.0)
    default_voter.set_surface_orientation_smoothing(0.0)
    smoothed_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    smoothed_voter.set_attribute_smoothing(0)
    smoothed_voter.set_surface_smoothing(0.0, 0.0)
    smoothed_voter.set_surface_orientation_smoothing(0.0)
    smoothed_voter.set_final_normalization_smoothing(1.0)

    default_fv, default_vp, default_vt = default_voter.apply_voting(
        d=1,
        fm=0.5,
        ft=ft,
        pt=pt,
        tt=tt,
    )
    smoothed_fv, smoothed_vp, smoothed_vt = smoothed_voter.apply_voting(
        d=1,
        fm=0.5,
        ft=ft,
        pt=pt,
        tt=tt,
    )

    assert not np.allclose(default_fv, smoothed_fv)
    np.testing.assert_array_equal(default_vp, smoothed_vp)
    np.testing.assert_array_equal(default_vt, smoothed_vt)


def test_apply_voting_final_normalization_smoothing_opt_in_keeps_fv_bounded() -> None:
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    ft[3:8, 5, 3:8] = 0.8
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)

    default_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    default_voter.set_attribute_smoothing(0)
    default_voter.set_surface_smoothing(0.0, 0.0)
    default_voter.set_surface_orientation_smoothing(0.0)
    smoothed_voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    smoothed_voter.set_attribute_smoothing(0)
    smoothed_voter.set_surface_smoothing(0.0, 0.0)
    smoothed_voter.set_surface_orientation_smoothing(0.0)
    smoothed_voter.set_final_normalization_smoothing(1.0)

    default_fv, _, _ = default_voter.apply_voting(d=1, fm=0.5, ft=ft, pt=pt, tt=tt)
    smoothed_fv, _, _ = smoothed_voter.apply_voting(
        d=1,
        fm=0.5,
        ft=ft,
        pt=pt,
        tt=tt,
    )

    for fv in (default_fv, smoothed_fv):
        assert fv.dtype == np.float32
        assert np.isfinite(fv).all()
        assert fv.min() >= np.float32(0.0)
        assert fv.max() <= np.float32(1.0)
    assert not np.allclose(default_fv, smoothed_fv)


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


def test_thin_default_matches_reference_mode() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[:, 1, :] = 0.5
    fv[:, 2, :] = 1.0
    fv[:, 3, :] = 0.5
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 45.0)

    default = voter.thin(fv, vp, vt)
    reference = voter.thin(fv, vp, vt, mode="reference")

    np.testing.assert_array_equal(reference, default)


def test_thin_normal_mode_remains_explicit_fault_normal_path() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    normal = voter.thin(fv, vp, vt, mode="normal")
    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)

    assert np.count_nonzero(normal) > 0
    assert np.count_nonzero(reference) > 0
    assert np.count_nonzero(normal[:, :, 2]) == np.count_nonzero(normal)
    assert np.count_nonzero(reference[:, 2, :]) == np.count_nonzero(reference)
    assert np.count_nonzero(reference != normal) > 0


def test_thin_reference_mode_matches_reference_helper_regression() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    expected, _ = reference_like_3d_thin_values(
        fv,
        vp,
        sigma=0.0,
        reinforce_vertical=True,
    )

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)

    np.testing.assert_array_equal(reference, expected)


def test_thin_normal_mode_matches_fault_normal_regression() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((7, 9, 7), dtype=np.float32)
    fv[1:6, 2, 1:6] = 0.7
    fv[1:6, 3:6, 1:6] = 1.0
    fv[1:6, 6, 1:6] = 0.7
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    expected = np.zeros_like(fv)
    expected[:, 4, :] = fv[:, 4, :]

    normal = voter.thin(fv, vp, vt, mode="normal")

    np.testing.assert_array_equal(normal, expected)


def test_thin_hybrid_constant_orientation_matches_reference_mode() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 45.0)

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)
    hybrid = voter.thin(fv, vp, vt, mode="hybrid", reference_sigma=0.0)

    np.testing.assert_array_equal(hybrid, reference)


def test_thin_hybrid_v2_stable_plane_matches_reference_mode() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((9, 9, 9), dtype=np.float32)
    fv[2:7, 4, 2:7] = 1.0
    fv[2:7, 3, 2:7] = 0.4
    fv[2:7, 5, 2:7] = 0.4
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 45.0)

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)
    hybrid_v2 = voter.thin(fv, vp, vt, mode="hybrid_v2", reference_sigma=0.0)

    assert np.count_nonzero(hybrid_v2) == np.count_nonzero(reference)
    np.testing.assert_array_equal(hybrid_v2 > 0.0, reference > 0.0)
    np.testing.assert_array_equal(hybrid_v2, reference)


def test_orientation_roughness_ignores_orientation_sentinel_outside_support() -> None:
    fv = np.zeros((4, 4, 4), dtype=np.float32)
    fv[1:3, 1:3, 1:3] = 1.0
    support = fv > 0.0
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    vp[support] = 30.0
    vt[support] = 45.0

    roughness = voting3d._orientation_roughness_3d(vp, vt, support=support)
    unmasked_roughness = voting3d._orientation_roughness_3d(vp, vt)

    np.testing.assert_array_equal(roughness[support], np.zeros(np.count_nonzero(support)))
    assert np.max(unmasked_roughness[support]) > np.float32(8.0)


def test_orientation_roughness_detects_orientation_change_inside_support() -> None:
    support = np.zeros((4, 4, 4), dtype=np.bool_)
    support[1:3, 1:3, 1:3] = True
    vp = np.zeros(support.shape, dtype=np.float32)
    vt = np.zeros(support.shape, dtype=np.float32)
    vp[support] = 30.0
    vt[support] = 45.0
    vt[1, 1, 2] = 80.0

    roughness = voting3d._orientation_roughness_3d(vp, vt, support=support)

    assert roughness[1, 1, 1] > np.float32(20.0)
    assert roughness[1, 1, 2] > np.float32(20.0)


def test_thin_hybrid_ignores_zero_vote_orientation_sentinel_for_constant_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    support = fv > 0.0
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    vp[support] = 30.0
    vt[support] = 45.0

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)
    hybrid = voter.thin(fv, vp, vt, mode="hybrid", reference_sigma=0.0)

    np.testing.assert_array_equal(hybrid, reference)


def test_thin_hybrid_uses_normal_mode_in_unstable_orientation_region() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.zeros_like(fv)
    vp[:, 2:, :] = 20.0
    vt = np.zeros_like(fv)

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)
    normal = voter.thin(fv, vp, vt, mode="normal")
    hybrid = voter.thin(
        fv,
        vp,
        vt,
        mode="hybrid",
        reference_sigma=0.0,
        hybrid_orientation_gradient_threshold=1.0,
    )

    normal_selected = (reference != normal) & (hybrid == normal)
    assert np.count_nonzero(normal_selected) > 0
    assert np.count_nonzero(hybrid != reference) > 0


def test_thin_hybrid_v2_uses_positive_normal_candidates_in_unstable_region() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((7, 7, 7), dtype=np.float32)
    fv[2:5, 3, 2:5] = 1.0
    fv[2:5, 2, 2:5] = 0.4
    fv[2:5, 4, 2:5] = 0.4
    vp = np.zeros_like(fv)
    vp[:, 3:, :] = 20.0
    vt = np.zeros_like(fv)

    reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=0.0)
    normal = voter.thin(fv, vp, vt, mode="normal")
    hybrid_v2 = voter.thin(
        fv,
        vp,
        vt,
        mode="hybrid_v2",
        reference_sigma=0.0,
        hybrid_orientation_gradient_threshold=1.0,
    )

    normal_selected = (normal > 0.0) & (reference != normal) & (hybrid_v2 == normal)
    assert np.count_nonzero(normal_selected) > 0
    assert np.count_nonzero(hybrid_v2 != reference) > 0


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

    fvt = voter.thin(fv, vp, vt)
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


def test_thin_default_reference_mode_does_not_apply_scanner_edge_cleanup() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((11, 11, 1), dtype=np.float32)
    fv[2, 5, 0] = 10.0
    vp = np.full_like(fv, 90.0)
    vt = np.full_like(fv, 90.0)
    expected, keep = reference_like_3d_thin_values(
        fv,
        vp,
        sigma=1.0,
        reinforce_vertical=True,
    )

    fvt = voter.thin(fv, vp, vt)
    edge_cleaned, _, _, edge_keep = remove_reference_edge_effects_3d(fvt, vp, vt)

    assert keep[2, 5, 0]
    assert fvt[2, 5, 0] > np.float32(0.0)
    np.testing.assert_array_equal(fvt, expected)
    assert not edge_keep[2, 5, 0]
    assert edge_cleaned[2, 5, 0] == np.float32(0.0)


def test_thin_rejects_invalid_mode() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="reference.*normal.*hybrid"):
        voter.thin(fv, vp, vt, mode="nearest")


@pytest.mark.parametrize("threshold", [-1.0, np.inf, np.nan])
def test_thin_hybrid_rejects_invalid_orientation_gradient_threshold(
    threshold: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="hybrid_orientation_gradient_threshold"):
        voter.thin(
            fv,
            vp,
            vt,
            mode="hybrid",
            hybrid_orientation_gradient_threshold=threshold,
        )


def test_thin_reference_mode_uses_strike_bin_nms_in_i2_i3_plane() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[1:4, 2, 1:4] = 1.0
    fv[1:4, 1, 1:4] = 0.4
    fv[1:4, 3, 1:4] = 0.4
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    normal = voter.thin(fv, vp, vt, mode="normal")
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

    fvt = voter.thin(fv, vp, vt, mode="normal")

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

    fvt = voter.thin(fv, vp, vt, mode="normal")

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    assert np.count_nonzero(fvt) < np.count_nonzero(fv)
    assert np.count_nonzero(fvt[:, 4, :]) == 25
    assert np.count_nonzero(fvt[:, :4, :]) == 0
    assert np.count_nonzero(fvt[:, 5:, :]) == 0
    np.testing.assert_array_equal(fvt[:, 4, :], fv[:, 4, :])


def test_thin_normal_plateau_keeps_tie_breaker_max_layer() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 7, 5), dtype=np.float32)
    fv[1:4, 2:5, 1:4] = 1.0
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    tie_breaker = np.zeros_like(fv)
    tie_breaker[:, 2, :] = 0.2
    tie_breaker[:, 3, :] = 0.9
    tie_breaker[:, 4, :] = 0.6

    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode="normal_plateau",
        plateau_tie_breaker=tie_breaker,
        plateau_tolerance=1.0,
    )

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    assert np.count_nonzero(fvt[:, 3, :]) == 9
    assert np.count_nonzero(fvt[:, :3, :]) == 0
    assert np.count_nonzero(fvt[:, 4:, :]) == 0
    np.testing.assert_array_equal(fvt[:, 3, :], fv[:, 3, :])


def test_thin_normal_plateau_tie_breaker_equal_run_keeps_center_layer() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((5, 7, 5), dtype=np.float32)
    fv[1:4, 2:5, 1:4] = 1.0
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode="normal_plateau",
        plateau_tolerance=1.0,
    )

    assert np.count_nonzero(fvt[:, 3, :]) == 9
    assert np.count_nonzero(fvt[:, :3, :]) == 0
    assert np.count_nonzero(fvt[:, 4:, :]) == 0


def test_thin_normal_plateau_returns_zero_for_all_zero_volume() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((4, 5, 6), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    fvt = voter.thin(fv, vp, vt, mode="normal_plateau")

    assert fvt.shape == fv.shape
    assert fvt.dtype == np.float32
    assert np.isfinite(fvt).all()
    np.testing.assert_array_equal(fvt, np.zeros_like(fv))


def test_thin_hybrid_v2_uses_edge_plateau_fallback_when_normal_is_empty() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((9, 9, 9), dtype=np.float32)
    fv[2:7, 0:2, 2:7] = 1.0
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    normal = voter.thin(fv, vp, vt, mode="normal")
    plateau = voter.thin(
        fv,
        vp,
        vt,
        mode="normal_plateau",
        plateau_tolerance=1.0,
    )
    hybrid_v2 = voter.thin(
        fv,
        vp,
        vt,
        mode="hybrid_v2",
        reference_sigma=0.0,
        plateau_tolerance=1.0,
    )

    assert np.count_nonzero(normal) == 0
    assert np.count_nonzero(plateau[:, 0, :]) > 0
    np.testing.assert_array_equal(hybrid_v2[:, 0, :], plateau[:, 0, :])


def test_thin_returns_zero_for_flat_volume() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.full((4, 5, 6), 0.75, dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)

    with np.errstate(all="raise"):
        fvt = voter.thin(fv, vp, vt, mode="normal")

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
        fvt = voter.thin(fv, vp, vt, mode="normal")

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


def test_thin_normal_plateau_rejects_tie_breaker_shape_mismatch() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    tie_breaker = np.zeros((3, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="plateau_tie_breaker shapes must match"):
        voter.thin(
            fv,
            vp,
            vt,
            mode="normal_plateau",
            plateau_tie_breaker=tie_breaker,
        )


def test_thin_normal_plateau_rejects_nonfinite_tie_breaker() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    fv = np.zeros((3, 3, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    tie_breaker = np.zeros_like(fv)
    tie_breaker[1, 1, 1] = np.nan

    with pytest.raises(ValueError, match="plateau_tie_breaker"):
        voter.thin(
            fv,
            vp,
            vt,
            mode="normal_plateau",
            plateau_tie_breaker=tie_breaker,
        )


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

    scores = _normalize_and_power_3d(volume, power=4)

    expected = np.array([[[0.0, 0.9375, 1.0]]], dtype=np.float32)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-7)


def test_normalize_and_power_3d_reference_audit_range_and_monotonicity() -> None:
    volume = np.array([[[-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]]], dtype=np.float32)

    scores = _normalize_and_power_3d(volume, power=2)

    assert scores.shape == volume.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert scores.min() >= np.float32(0.0)
    assert scores.max() <= np.float32(1.0)
    assert np.all(np.diff(scores.ravel()) > 0.0)


def test_normalize_and_power_3d_reference_audit_all_constant_input() -> None:
    # Audits OptimalSurfaceVoter.normalization zero dynamic-range behavior.
    volume = np.full((2, 2, 3), 4.25, dtype=np.float32)

    with np.errstate(all="raise"):
        scores = _normalize_and_power_3d(volume, sigma=0.0, power=8)

    assert scores.shape == volume.shape
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, np.zeros_like(volume))


def test_normalize_and_power_3d_default_matches_reference_formula() -> None:
    volume = np.array(
        [
            [[-2.0, -1.0], [0.0, 1.5]],
            [[3.0, 4.5], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )

    scores = _normalize_and_power_3d(volume)

    expected = volume.copy()
    expected -= expected.min()
    expected /= expected.max()
    expected = np.float32(1.0) - np.power(np.float32(1.0) - expected, 8)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, expected.astype(np.float32), rtol=0.0, atol=1e-7)


def test_normalize_and_power_3d_explicit_sigma_opts_into_smoothing() -> None:
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[2, 2, 2] = 1.0

    default_scores = _normalize_and_power_3d(volume)
    smoothed_scores = _normalize_and_power_3d(volume, sigma=1.0)

    assert smoothed_scores.shape == volume.shape
    assert smoothed_scores.dtype == np.float32
    assert np.isfinite(smoothed_scores).all()
    assert smoothed_scores.min() >= np.float32(0.0)
    assert smoothed_scores.max() <= np.float32(1.0)
    assert not np.allclose(smoothed_scores, default_scores)


def test_final_normalization_default_keeps_isolated_spike_sparser_than_smoothing() -> None:
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[3, 3, 3] = 1.0

    default_scores = _normalize_and_power_3d(volume)
    smoothed_scores = _normalize_and_power_3d(volume, sigma=1.0)

    assert default_scores[3, 3, 3] == pytest.approx(1.0)
    assert smoothed_scores[3, 3, 3] == pytest.approx(1.0)
    assert np.count_nonzero(default_scores > 0.0) < np.count_nonzero(
        smoothed_scores > 0.0,
    )
    assert np.count_nonzero(default_scores > 0.1) < np.count_nonzero(
        smoothed_scores > 0.1,
    )
    assert default_scores.sum() < smoothed_scores.sum()


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


def test_surface_strike_and_dip_linear_surface_matches_geometry_convention() -> None:
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    w, v = np.indices((5, 5), dtype=np.float32)
    du_dv = 0.25
    du_dw = -0.5
    surface = du_dv * (v - 2.0) + du_dw * (w - 2.0)

    expected_strike, expected_dip = strike_and_dip_from_local_surface_derivatives(
        normal,
        dip,
        strike,
        du_dv,
        du_dw,
    )
    strike_none, dip_none = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=None,
    )
    strike_zero, dip_zero = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=0.0,
    )

    assert np.isfinite(strike_none)
    assert np.isfinite(dip_none)
    assert strike_none == pytest.approx(expected_strike, abs=1e-5)
    assert dip_none == pytest.approx(expected_dip, abs=1e-5)
    assert strike_zero == pytest.approx(strike_none)
    assert dip_zero == pytest.approx(dip_none)
    assert 0.0 <= strike_none < 360.0
    assert 0.0 <= dip_none <= 180.0


def test_surface_strike_and_dip_smoothing_changes_spiky_center_derivatives() -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    w, v = np.indices((7, 7), dtype=np.float32)
    plane = 0.1 * (v - 3.0) + 0.2 * (w - 3.0)
    surface = plane.copy()
    surface[3, 2] -= 4.0
    surface[3, 4] += 4.0

    raw_strike, raw_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=None,
    )
    smooth_strike, smooth_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        surface,
        sigma=2.0,
    )
    plane_strike, plane_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        plane,
        sigma=None,
    )

    assert np.isfinite([raw_strike, raw_dip, smooth_strike, smooth_dip]).all()
    assert smooth_strike != pytest.approx(raw_strike)
    assert smooth_dip != pytest.approx(raw_dip)
    assert _angle_distance_degrees(smooth_strike, plane_strike) < _angle_distance_degrees(
        raw_strike,
        plane_strike,
    )
    assert abs(smooth_dip - plane_dip) < abs(raw_dip - plane_dip)


def test_surface_strike_and_dip_smoothing_reduces_stair_step_jitter() -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    w, v = np.indices((9, 9), dtype=np.float32)
    plane = 0.12 * (v - 4.0) - 0.18 * (w - 4.0)
    ideal_strike, ideal_dip = _surface_strike_and_dip(
        normal,
        dip,
        strike,
        plane,
        sigma=None,
    )
    raw_orientations: list[tuple[float, float]] = []
    smoothed_orientations: list[tuple[float, float]] = []

    for shift_w, shift_v in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (-1, 0),
        (0, -1),
        (2, -1),
        (-2, 1),
    ]:
        checker = (((w + shift_w) + (v + shift_v)) % 2.0) * 2.0 - 1.0
        stair = np.where(v + shift_v >= 4.0, 1.0, -1.0) + np.where(
            w + shift_w >= 4.0,
            -1.0,
            1.0,
        )
        surface = (plane + 0.5 * checker + 0.35 * stair).astype(np.float32)
        raw_orientations.append(
            _surface_strike_and_dip(normal, dip, strike, surface, sigma=0.0),
        )
        smoothed_orientations.append(
            _surface_strike_and_dip(normal, dip, strike, surface, sigma=2.0),
        )

    raw = np.asarray(raw_orientations, dtype=np.float32)
    smoothed = np.asarray(smoothed_orientations, dtype=np.float32)
    raw_strike_error = np.asarray(
        [_angle_distance_degrees(value, ideal_strike) for value in raw[:, 0]],
    )
    smoothed_strike_error = np.asarray(
        [_angle_distance_degrees(value, ideal_strike) for value in smoothed[:, 0]],
    )
    raw_dip_error = np.abs(raw[:, 1] - ideal_dip)
    smoothed_dip_error = np.abs(smoothed[:, 1] - ideal_dip)

    assert np.isfinite(raw).all()
    assert np.isfinite(smoothed).all()
    assert np.median(smoothed_strike_error) <= np.median(raw_strike_error)
    assert np.median(smoothed_dip_error) <= np.median(raw_dip_error)
    assert np.std(smoothed[:, 0]) < np.std(raw[:, 0])
    assert np.std(smoothed[:, 1]) < np.std(raw[:, 1])


def test_surface_strike_and_dip_does_not_mutate_surface() -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface = np.zeros((5, 5), dtype=np.float32)
    surface[2, 1] = -2.0
    surface[2, 3] = 3.0
    expected = surface.copy()

    _surface_strike_and_dip(normal, dip, strike, surface, sigma=1.0)

    np.testing.assert_array_equal(surface, expected)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_surface_strike_and_dip_rejects_too_small_surfaces(
    shape: tuple[int, int],
) -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="at least three samples"):
        _surface_strike_and_dip(normal, dip, strike, surface, sigma=None)


@pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, True])
def test_surface_strike_and_dip_rejects_invalid_sigma(sigma: object) -> None:
    normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="sigma"):
        _surface_strike_and_dip(
            normal,
            dip,
            strike,
            surface,
            sigma=sigma,  # type: ignore[arg-type]
        )


def _expected_masked_uvw_samples(
    voter: OptimalSurfaceVoter,
    c1: int,
    c2: int,
    c3: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n3, n2, n1 = fx.shape
    shape = (2 * voter.rw + 1, 2 * voter.rv + 1, 2 * voter.ru + 1)
    costs = np.ones(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    admissible_count = 0
    in_bounds_count = 0
    for kw in range(shape[0]):
        iw = kw - voter.rw
        for kv in range(shape[1]):
            iv = kv - voter.rv
            for ku in range(shape[2]):
                iu = ku - voter.ru
                if not voter.lmins[kw, kv] <= iu <= voter.lmaxs[kw, kv]:
                    continue
                admissible_count += 1
                x1 = np.float32(
                    float(c1)
                    + float(iw) * float(strike[0])
                    + float(iv) * float(dip[0])
                    + float(iu) * float(normal[0])
                )
                x2 = np.float32(
                    float(c2)
                    + float(iw) * float(strike[1])
                    + float(iv) * float(dip[1])
                    + float(iu) * float(normal[1])
                )
                x3 = np.float32(
                    float(c3)
                    + float(iw) * float(strike[2])
                    + float(iv) * float(dip[2])
                    + float(iu) * float(normal[2])
                )
                j1 = _java_style_round(float(x1))
                j2 = _java_style_round(float(x2))
                j3 = _java_style_round(float(x3))
                if not (0 <= j1 < n1 and 0 <= j2 < n2 and 0 <= j3 < n3):
                    continue
                mask[kw, kv, ku] = True
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]
                in_bounds_count += 1
    return costs, mask, admissible_count, in_bounds_count


def test_surface_voting_boundary_policy_defaults_to_reference() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=3)

    assert voter.surface_voting_boundary_policy == "reference"
    assert voter.surface_voting_diagnostics == ()
    assert voter.surface_voting_diagnostic_summary() == {
        "policy": "reference",
        "seed_count": 0,
        "boundary_affected_seed_count": 0,
        "voted_seed_count": 0,
        "skipped_seed_count": 0,
        "support_fraction_min": 1.0,
        "support_fraction_mean": 1.0,
        "surface_projection_count": 0,
        "selected_invalid_sample_count": 0,
        "face_center_vote_count": 0,
    }


def test_set_surface_voting_boundary_policy_accepts_only_supported_values() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=1, rw=1)

    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    assert voter.surface_voting_boundary_policy == "masked_in_bounds"
    voter.set_surface_voting_boundary_policy("reference")
    assert voter.surface_voting_boundary_policy == "reference"


@pytest.mark.parametrize("policy", ["", "Reference", "masked", None, 1, True])
def test_set_surface_voting_boundary_policy_rejects_invalid_values(policy: object) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=1, rw=1)

    with pytest.raises(ValueError, match="'reference'.*'masked_in_bounds'"):
        voter.set_surface_voting_boundary_policy(policy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("c1", "c2", "c3"),
    [
        (0, 3, 3),
        (6, 3, 3),
        (3, 0, 3),
        (3, 6, 3),
        (3, 3, 0),
        (3, 3, 6),
        (0, 0, 0),
        (6, 6, 6),
    ],
)
def test_masked_uvw_sampling_marks_out_of_bounds_oblique_face_and_corner_lags(
    c1: int,
    c2: int,
    c3: int,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=3, rw=3)
    fx = _axis_aligned_ramp_volume((7, 7, 7)) * np.float32(8.0)
    normal = fault_normal_vector_from_strike_and_dip(37.0, 61.0)
    dip = fault_dip_vector_from_strike_and_dip(37.0, 61.0)
    strike = fault_strike_vector_from_strike_and_dip(37.0, 61.0)

    samples = voter._samples_in_uvw_box_masked(
        c1,
        c2,
        c3,
        normal,
        dip,
        strike,
        fx,
    )
    expected = _expected_masked_uvw_samples(
        voter,
        c1,
        c2,
        c3,
        normal,
        dip,
        strike,
        fx,
    )
    expected_costs, expected_mask, admissible_count, in_bounds_count = expected

    assert samples.costs.dtype == np.float32
    assert samples.valid_lag_mask.dtype == np.bool_
    assert samples.w_offset == 0
    assert samples.v_offset == 0
    assert samples.full_tangential_shape == (7, 7)
    assert samples.admissible_lag_count == admissible_count
    assert samples.in_bounds_lag_count == in_bounds_count
    assert in_bounds_count < admissible_count
    assert np.any(~expected_mask)
    np.testing.assert_array_equal(samples.valid_lag_mask, expected_mask)
    np.testing.assert_array_equal(samples.costs, expected_costs)
    np.testing.assert_array_equal(
        samples.costs[~samples.valid_lag_mask],
        np.ones(np.count_nonzero(~samples.valid_lag_mask), dtype=np.float32),
    )
    assert np.any(samples.costs[samples.valid_lag_mask] < np.float32(0.0))


def test_masked_uvw_sampling_masks_lags_outside_lmin_lmax() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=1, rw=1)
    fx = np.full((9, 9, 9), 0.25, dtype=np.float32)
    samples = voter._samples_in_uvw_box_masked(
        4,
        4,
        4,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        fx,
    )

    for kw in range(samples.costs.shape[0]):
        for kv in range(samples.costs.shape[1]):
            for ku in range(samples.costs.shape[2]):
                iu = ku - voter.ru
                admissible = voter.lmins[kw, kv] <= iu <= voter.lmaxs[kw, kv]
                assert bool(samples.valid_lag_mask[kw, kv, ku]) is bool(admissible)
                if not admissible:
                    assert samples.costs[kw, kv, ku] == np.float32(1.0)


@pytest.mark.parametrize(
    ("normal_component", "expected_valid"),
    [
        (np.float32(0.5), True),
        (np.nextafter(np.float32(0.5), np.float32(0.0)), True),
        (np.nextafter(np.float32(0.5), np.float32(1.0)), False),
    ],
)
def test_masked_uvw_sampling_preserves_java_rounding_at_negative_half_boundary(
    normal_component: np.float32,
    expected_valid: bool,
) -> None:
    fx = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    lmins = np.array([[-1]], dtype=np.int32)
    lmaxs = np.array([[1]], dtype=np.int32)

    costs, mask, admissible_count, in_bounds_count = voting3d._samples_in_uvw_box_masked_python(
        0,
        0,
        0,
        1,
        0,
        0,
        np.array([normal_component, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        fx,
        lmins,
        lmaxs,
    )

    assert admissible_count == 3
    assert bool(mask[0, 0, 0]) is expected_valid
    assert in_bounds_count == (3 if expected_valid else 2)
    if expected_valid:
        assert costs[0, 0, 0] == np.float32(1.0) - fx[0, 0, 0]
    else:
        assert costs[0, 0, 0] == np.float32(1.0)


def test_select_supported_origin_rectangle_returns_full_box_when_supported() -> None:
    supported = np.ones((5, 7), dtype=np.bool_)

    rectangle = voting3d._select_supported_origin_rectangle(
        supported,
        origin_w=2,
        origin_v=3,
    )

    assert rectangle == voting3d._TangentialRectangle(0, 0, 5, 7)
    assert rectangle.shape == (5, 7)
    assert rectangle.size == 35


def test_select_supported_origin_rectangle_uses_asymmetry_before_lexicographic_order() -> None:
    supported = np.zeros((4, 4), dtype=np.bool_)
    supported[0:3, 2] = True
    supported[2, 1:4] = True

    rectangle = voting3d._select_supported_origin_rectangle(
        supported,
        origin_w=2,
        origin_v=2,
    )

    assert rectangle == voting3d._TangentialRectangle(2, 1, 3, 4)


def test_select_supported_origin_rectangle_uses_lexicographic_final_tie_break() -> None:
    supported = np.zeros((5, 5), dtype=np.bool_)
    supported[1:4, 2] = True
    supported[2, 1:4] = True

    rectangles = [
        voting3d._select_supported_origin_rectangle(
            supported,
            origin_w=2,
            origin_v=2,
        )
        for _ in range(3)
    ]

    assert rectangles == [voting3d._TangentialRectangle(1, 2, 4, 3)] * 3


def test_select_supported_origin_rectangle_returns_none_for_unsupported_origin() -> None:
    supported = np.ones((3, 3), dtype=np.bool_)
    supported[1, 1] = False

    assert (
        voting3d._select_supported_origin_rectangle(
            supported,
            origin_w=1,
            origin_v=1,
        )
        is None
    )


def test_crop_masked_uvw_box_preserves_full_box_offsets_and_counts() -> None:
    costs = np.arange(5 * 5 * 3, dtype=np.float32).reshape(5, 5, 3)
    mask = costs % np.float32(2.0) == np.float32(0.0)
    samples = voting3d._MaskedUVWBoxSamples(
        costs=costs,
        valid_lag_mask=mask,
        w_offset=0,
        v_offset=0,
        full_tangential_shape=(5, 5),
        admissible_lag_count=45,
        in_bounds_lag_count=30,
    )
    rectangle = voting3d._TangentialRectangle(1, 0, 4, 3)

    cropped = voting3d._crop_masked_uvw_box(samples, rectangle)

    assert cropped.w_offset == 1
    assert cropped.v_offset == 0
    assert cropped.full_tangential_shape == (5, 5)
    assert cropped.admissible_lag_count == 45
    assert cropped.in_bounds_lag_count == 30
    np.testing.assert_array_equal(cropped.costs, costs[1:4, 0:3, :])
    np.testing.assert_array_equal(cropped.valid_lag_mask, mask[1:4, 0:3, :])


def test_masked_sampler_boundary_crop_extracts_valid_smoothed_strain_surface() -> None:
    shape = (11, 11, 11)
    c1, c2, c3 = 0, 5, 5
    strike_angle = 37.0
    dip_angle = 61.0
    normal = fault_normal_vector_from_strike_and_dip(strike_angle, dip_angle)
    dip = fault_dip_vector_from_strike_and_dip(strike_angle, dip_angle)
    strike = fault_strike_vector_from_strike_and_dip(strike_angle, dip_angle)
    i3, i2, i1 = np.indices(shape, dtype=np.float32)
    x1 = i1 - np.float32(c1)
    x2 = i2 - np.float32(c2)
    x3 = i3 - np.float32(c3)
    local_u = x1 * normal[0] + x2 * normal[1] + x3 * normal[2]
    local_v = x1 * dip[0] + x2 * dip[1] + x3 * dip[2]
    local_w = x1 * strike[0] + x2 * strike[1] + x3 * strike[2]
    signed_distance = local_u - np.float32(0.15) * local_v + np.float32(0.10) * local_w
    ft = np.exp(np.float32(-0.5) * (signed_distance / np.float32(0.6)) ** np.float32(2.0)).astype(
        np.float32
    )

    voter = OptimalSurfaceVoter(ru=3, rv=4, rw=4)
    voter.set_strain_max(0.25, 0.25)
    voter.set_surface_smoothing(2.0, 2.0)
    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    full_samples = voter._samples_in_uvw_box_masked(
        c1,
        c2,
        c3,
        normal,
        dip,
        strike,
        ft,
    )
    supported_columns = np.any(full_samples.valid_lag_mask, axis=2)
    rectangle = voting3d._select_supported_origin_rectangle(
        supported_columns,
        origin_w=voter.rw,
        origin_v=voter.rv,
    )

    assert ft.dtype == np.float32
    assert np.isfinite(ft).all()
    assert full_samples.in_bounds_lag_count < full_samples.admissible_lag_count
    assert rectangle is not None
    samples = voting3d._crop_masked_uvw_box(full_samples, rectangle)
    surface, _ = voting3d._find_surface_3d_masked(
        samples.costs,
        samples.valid_lag_mask,
        lmin=voter.lmin,
        bstrain1=voter.bstrain1,
        bstrain2=voter.bstrain2,
        attribute_smoothing=voter.attribute_smoothing,
        surface_smoothing1=voter.surface_smoothing1,
        surface_smoothing2=voter.surface_smoothing2,
    )

    assert surface is not None
    assert surface.dtype == np.float32
    assert np.isfinite(surface).all()
    selected_lags = np.floor(surface.astype(np.float64) - voter.lmin + 0.5).astype(np.intp)
    selected_mask = np.take_along_axis(
        samples.valid_lag_mask,
        selected_lags[:, :, np.newaxis],
        axis=2,
    )
    assert selected_mask.all()
    strain1 = np.float32(1.0 / voter.bstrain1)
    strain2 = np.float32(1.0 / voter.bstrain2)
    tolerance = np.float32(1.0e-6)
    assert np.max(np.abs(np.diff(surface, axis=1))) <= strain1 + tolerance
    assert np.max(np.abs(np.diff(surface, axis=0))) <= strain2 + tolerance


@pytest.mark.parametrize(
    "seed_index",
    [
        (0, 3, 3),
        (6, 3, 3),
        (3, 0, 3),
        (3, 6, 3),
        (3, 3, 0),
        (3, 3, 6),
    ],
)
def test_masked_surface_voting_writes_center_votes_on_all_six_faces(
    seed_index: tuple[int, int, int],
) -> None:
    c1, c2, c3 = seed_index
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    voter.set_surface_orientation_smoothing(0.0)
    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    ft = np.ones((7, 7, 7), dtype=np.float32)
    fe = np.zeros_like(ft)
    vp = np.zeros_like(ft)
    vt = np.zeros_like(ft)
    vm = np.zeros_like(ft)

    diagnostic = voter._surface_voting(
        FaultCell(c1, c2, c3, 1.0, 0.0, 90.0),
        ft,
        fe,
        vp,
        vt,
        vm,
    )

    assert not diagnostic.skipped
    assert diagnostic.selected_invalid_sample_count == 0
    assert diagnostic.center_vote_write_count > 0
    assert diagnostic.face_center_vote_count > 0
    assert fe[c3, c2, c1] > np.float32(0.0)
    assert vm[c3, c2, c1] > np.float32(0.0)


@pytest.mark.parametrize(
    "seed_index",
    [(3, 0, 3), (3, 6, 3), (3, 3, 0), (3, 3, 6)],
)
def test_masked_face_center_votes_do_not_change_reference_i2_i3_exclusion(
    seed_index: tuple[int, int, int],
) -> None:
    c1, c2, c3 = seed_index
    ft = np.ones((7, 7, 7), dtype=np.float32)
    reference_arrays = tuple(np.zeros_like(ft) for _ in range(4))
    masked_arrays = tuple(np.zeros_like(ft) for _ in range(4))
    cell = FaultCell(c1, c2, c3, 1.0, 0.0, 90.0)

    reference = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    reference.set_attribute_smoothing(0)
    reference.set_surface_smoothing(0.0, 0.0)
    reference.set_surface_orientation_smoothing(0.0)
    reference._surface_voting(cell, ft, *reference_arrays)

    masked = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    masked.set_attribute_smoothing(0)
    masked.set_surface_smoothing(0.0, 0.0)
    masked.set_surface_orientation_smoothing(0.0)
    masked.set_surface_voting_boundary_policy("masked_in_bounds")
    masked._surface_voting(cell, ft, *masked_arrays)

    assert reference_arrays[0][c3, c2, c1] == np.float32(0.0)
    assert masked_arrays[0][c3, c2, c1] > np.float32(0.0)


@pytest.mark.parametrize(
    ("seed_index", "align_i3"),
    [
        ((2, 2, 0), True),
        ((2, 2, 4), True),
        ((2, 0, 2), False),
        ((2, 4, 2), False),
    ],
)
def test_masked_surface_vote_reinforcement_stays_inside_volume(
    seed_index: tuple[int, int, int],
    align_i3: bool,
) -> None:
    c1, c2, c3 = seed_index
    shape = (5, 5, 5)
    arrays = (
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    )
    surface = np.zeros((1, 1), dtype=np.float32)
    valid_mask = np.ones((1, 1, 1), dtype=np.bool_)

    center_count, face_count, invalid_count = voting3d._accumulate_surface_votes_masked_python(
        c1,
        c2,
        c3,
        0,
        0,
        0,
        0,
        0,
        np.float32(0.75),
        np.float32(12.0),
        np.float32(34.0),
        align_i3,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        surface,
        valid_mask,
        *arrays,
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[c3, c2, c1] = np.float32(0.75)
    if align_i3:
        for i3 in (c3 - 1, c3 + 1):
            if 0 <= i3 < shape[0]:
                expected[i3, c2, c1] = np.float32(0.75)
    else:
        for i2 in (c2 - 1, c2 + 1):
            if 0 <= i2 < shape[1]:
                expected[c3, i2, c1] = np.float32(0.75)

    assert (center_count, face_count, invalid_count) == (1, 1, 0)
    np.testing.assert_array_equal(arrays[0], expected)
    np.testing.assert_array_equal(arrays[3], expected)


def test_interior_reference_and_masked_voting_are_identical() -> None:
    i3, i2, _ = np.indices((21, 21, 21), dtype=np.float32)
    ft = np.exp(-0.5 * ((i2 - np.float32(10.0)) / np.float32(1.0)) ** 2).astype(
        np.float32,
    )
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    cell = FaultCell(10, 10, 10, 1.0, 0.0, 90.0)
    normal = cell.fault_normal()
    dip = cell.fault_dip_vector()
    strike = cell.fault_strike_vector()

    reference = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    masked = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    for voter in (reference, masked):
        voter.set_attribute_smoothing(0)
        voter.set_surface_smoothing(0.0, 0.0)
        voter.set_surface_orientation_smoothing(0.0)
    masked.set_surface_voting_boundary_policy("masked_in_bounds")

    reference_costs = reference.samples_in_uvw_box(
        10,
        10,
        10,
        normal,
        dip,
        strike,
        ft,
    )
    masked_samples = masked._samples_in_uvw_box_masked(
        10,
        10,
        10,
        normal,
        dip,
        strike,
        ft,
    )
    np.testing.assert_array_equal(masked_samples.costs, reference_costs)
    assert masked_samples.in_bounds_lag_count == masked_samples.admissible_lag_count

    reference_surface = voting3d.find_surface_3d(
        reference_costs,
        lmin=reference.lmin,
        bstrain1=reference.bstrain1,
        bstrain2=reference.bstrain2,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )
    masked_surface, projection_count = voting3d._find_surface_3d_masked(
        masked_samples.costs,
        masked_samples.valid_lag_mask,
        lmin=masked.lmin,
        bstrain1=masked.bstrain1,
        bstrain2=masked.bstrain2,
        attribute_smoothing=0,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
    )
    assert projection_count == 0
    assert masked_surface is not None
    np.testing.assert_array_equal(masked_surface, reference_surface)

    reference_raw = tuple(np.zeros_like(ft) for _ in range(4))
    masked_raw = tuple(np.zeros_like(ft) for _ in range(4))
    reference_diagnostic = reference._surface_voting(cell, ft, *reference_raw)
    masked_diagnostic = masked._surface_voting(cell, ft, *masked_raw)
    for reference_array, masked_array in zip(reference_raw, masked_raw):
        np.testing.assert_array_equal(masked_array, reference_array)
    assert reference_diagnostic.support_fraction == 1.0
    assert masked_diagnostic.support_fraction == 1.0
    assert reference_diagnostic.surface_center_lag == masked_diagnostic.surface_center_lag
    assert reference_diagnostic.orientation_source == "surface"
    assert masked_diagnostic.orientation_source == "surface"

    reference_result = reference.apply_voting_from_seeds([cell], ft, pt, tt)
    masked_result = masked.apply_voting_from_seeds([cell], ft, pt, tt)
    for reference_array, masked_array in zip(reference_result, masked_result):
        np.testing.assert_array_equal(masked_array, reference_array)


def test_masked_voting_diagnostics_are_seed_ordered_summarized_and_reset() -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    voter.set_surface_orientation_smoothing(0.0)
    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    i3, i2, i1 = np.indices((7, 7, 7), dtype=np.float32)
    ft = (0.2 + 0.01 * i1 + 0.02 * i2 + 0.03 * i3).astype(np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    seeds = [
        FaultCell(3, 3, 3, float(ft[3, 3, 3]), 0.0, 90.0),
        FaultCell(3, 3, 0, float(ft[0, 3, 3]), 0.0, 90.0),
    ]

    voter.apply_voting_from_seeds(seeds, ft, pt, tt)

    diagnostics = voter.surface_voting_diagnostics
    assert diagnostics is voter.last_surface_voting_diagnostics
    assert tuple(item.seed_index for item in diagnostics) == ((3, 3, 3), (3, 3, 0))
    interior, boundary = diagnostics
    assert interior.support_fraction == 1.0
    assert interior.selected_tangential_column_count == 9
    assert interior.orientation_source == "surface"
    assert boundary.full_tangential_column_count == 9
    assert boundary.selected_tangential_column_count == 6
    assert boundary.support_fraction == pytest.approx(6.0 / 9.0)
    assert boundary.selected_invalid_sample_count == 0
    assert boundary.face_center_vote_count == 3
    assert boundary.orientation_source == "seed_boundary_fallback"
    assert not boundary.skipped
    with pytest.raises(AttributeError):
        boundary.skipped = True  # type: ignore[misc]

    summary = voter.surface_voting_diagnostic_summary()
    assert summary == {
        "policy": "masked_in_bounds",
        "seed_count": 2,
        "boundary_affected_seed_count": 1,
        "voted_seed_count": 2,
        "skipped_seed_count": 0,
        "support_fraction_min": pytest.approx(2.0 / 3.0),
        "support_fraction_mean": pytest.approx(5.0 / 6.0),
        "surface_projection_count": 0,
        "selected_invalid_sample_count": 0,
        "face_center_vote_count": 3,
    }

    voter.apply_voting_from_seeds([], ft, pt, tt)

    assert voter.surface_voting_diagnostics == ()
    reset_summary = voter.surface_voting_diagnostic_summary()
    assert reset_summary["seed_count"] == 0
    assert reset_summary["support_fraction_min"] == 1.0
    assert reset_summary["support_fraction_mean"] == 1.0

    voter.set_surface_voting_boundary_policy("reference")
    assert voter.surface_voting_diagnostic_summary()["policy"] == "masked_in_bounds"


def test_masked_support_threshold_uses_full_tangential_patch_area() -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    voter.set_surface_support_policy(min_fraction=0.7, exponent=0.0)
    ft = np.ones((7, 7, 7), dtype=np.float32)
    arrays = tuple(np.zeros_like(ft) for _ in range(4))

    diagnostic = voter._surface_voting(
        FaultCell(3, 3, 0, 1.0, 0.0, 90.0),
        ft,
        *arrays,
    )

    assert diagnostic.full_tangential_column_count == 9
    assert diagnostic.selected_tangential_column_count == 6
    assert diagnostic.support_fraction == pytest.approx(2.0 / 3.0)
    assert diagnostic.skipped
    assert diagnostic.skip_reason == "support_below_min_fraction"
    np.testing.assert_array_equal(arrays[0], np.zeros_like(ft))


def test_masked_single_seed_vote_matches_padded_vote_after_crop() -> None:
    original_shape = (7, 7, 7)
    padding = 2
    padded_shape = tuple(size + 2 * padding for size in original_shape)

    def run(
        shape: tuple[int, int, int],
        cell: FaultCell,
    ) -> tuple[tuple[np.ndarray, ...], object]:
        voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
        voter.set_attribute_smoothing(0)
        voter.set_surface_smoothing(0.0, 0.0)
        voter.set_surface_orientation_smoothing(0.0)
        voter.set_surface_voting_boundary_policy("masked_in_bounds")
        ft = np.ones(shape, dtype=np.float32)
        arrays = tuple(np.zeros(shape, dtype=np.float32) for _ in range(4))
        diagnostic = voter._surface_voting(cell, ft, *arrays)
        return arrays, diagnostic

    direct, direct_diagnostic = run(
        original_shape,
        FaultCell(3, 3, 0, 1.0, 0.0, 90.0),
    )
    padded, padded_diagnostic = run(
        padded_shape,
        FaultCell(3 + padding, 3 + padding, padding, 1.0, 0.0, 90.0),
    )
    crop = tuple(slice(padding, padding + size) for size in original_shape)

    assert direct_diagnostic.orientation_source == "seed_boundary_fallback"
    assert padded_diagnostic.orientation_source == "surface"
    for direct_array, padded_array in zip(direct, padded):
        np.testing.assert_allclose(
            direct_array,
            padded_array[crop],
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_masked_axis_permutation_preserves_votes_and_support_diagnostics() -> None:
    ft = np.ones((7, 7, 7), dtype=np.float32)
    outputs: list[tuple[np.ndarray, ...]] = []
    diagnostics = []
    for cell in (
        FaultCell(3, 3, 0, 1.0, 0.0, 90.0),
        FaultCell(0, 3, 3, 1.0, 0.0, 90.0),
    ):
        voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
        voter.set_attribute_smoothing(0)
        voter.set_surface_smoothing(0.0, 0.0)
        voter.set_surface_orientation_smoothing(0.0)
        voter.set_surface_voting_boundary_policy("masked_in_bounds")
        arrays = tuple(np.zeros_like(ft) for _ in range(4))
        diagnostics.append(voter._surface_voting(cell, ft, *arrays))
        outputs.append(arrays)

    first, second = diagnostics
    assert first.full_tangential_column_count == second.full_tangential_column_count
    assert first.selected_tangential_column_count == second.selected_tangential_column_count
    assert first.support_fraction == second.support_fraction
    assert first.center_vote_write_count == second.center_vote_write_count
    assert first.face_center_vote_count == second.face_center_vote_count
    assert first.orientation_source == second.orientation_source
    for first_array, second_array in zip(outputs[0], outputs[1]):
        np.testing.assert_array_equal(first_array.transpose(2, 1, 0), second_array)


def test_masked_score_and_vote_mapping_reuse_float32_sampling_rounding() -> None:
    normal = np.array([-0.23766495, -0.9159979, 0.32320765], dtype=np.float32)
    dip = np.array([0.9713472, -0.22412235, 0.07908102], dtype=np.float32)
    strike = np.array([0.0, -0.33274162, -0.943018], dtype=np.float32)
    surface = np.ones((1, 1), dtype=np.float32)
    valid_mask = np.ones((1, 1, 1), dtype=np.bool_)
    ft = np.zeros((10, 2, 10), dtype=np.float32)
    ft[8, 0, 8] = np.float32(0.25)
    ft[9, 0, 8] = np.float32(0.75)
    args = (
        0,
        0,
        0,
        8,
        8,
        0,
        16,
        1,
        normal,
        dip,
        strike,
        surface,
        valid_mask,
    )

    fa, valid_count, invalid_count = voting3d._surface_vote_average_masked_python(
        *args,
        ft,
    )

    assert (valid_count, invalid_count) == (1, 0)
    assert fa == np.float32(0.75)

    arrays = tuple(np.zeros_like(ft) for _ in range(4))
    counts = voting3d._accumulate_surface_votes_masked_python(
        *args[:8],
        fa,
        np.float32(12.0),
        np.float32(34.0),
        False,
        *args[8:],
        *arrays,
    )

    assert counts == (1, 1, 0)
    assert arrays[0][9, 0, 8] == np.float32(0.75)
    assert arrays[0][8, 0, 8] == np.float32(0.0)


@pytest.mark.parametrize(
    ("lower", "upper", "strike_angle", "dip_angle"),
    [
        ((0, 3, 3), (6, 3, 3), 0.0, 0.0),
        ((3, 0, 3), (3, 6, 3), 0.0, 90.0),
        ((3, 3, 0), (3, 3, 6), 90.0, 90.0),
    ],
)
def test_masked_parallel_plane_translation_is_symmetric_between_opposite_faces(
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    strike_angle: float,
    dip_angle: float,
) -> None:
    ft = np.ones((7, 7, 7), dtype=np.float32)
    diagnostics = []
    nonzero_counts = []
    for index in (lower, upper):
        voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
        voter.set_attribute_smoothing(0)
        voter.set_surface_smoothing(0.0, 0.0)
        voter.set_surface_orientation_smoothing(0.0)
        voter.set_surface_voting_boundary_policy("masked_in_bounds")
        arrays = tuple(np.zeros_like(ft) for _ in range(4))
        diagnostics.append(
            voter._surface_voting(
                FaultCell(*index, 1.0, strike_angle, dip_angle),
                ft,
                *arrays,
            )
        )
        nonzero_counts.append(int(np.count_nonzero(arrays[0])))

    first, second = diagnostics
    assert first.support_fraction == second.support_fraction == 1.0
    assert first.center_vote_write_count == second.center_vote_write_count == 9
    assert first.face_center_vote_count == second.face_center_vote_count == 9
    assert nonzero_counts[0] == nonzero_counts[1]
