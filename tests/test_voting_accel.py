import numpy as np
import pytest

from pyosv import voting2d, voting3d
from pyosv.cells import FaultCell, FaultCell2
from pyosv.voting2d import OptimalPathVoter
from pyosv.voting3d import OptimalSurfaceVoter


pytestmark = pytest.mark.skipif(
    not voting2d.NUMBA_AVAILABLE,
    reason="Numba acceleration is optional",
)


def test_numba_2d_local_sampling_matches_python_fallback() -> None:
    voter = OptimalPathVoter(ru=2, rv=3)
    i2, i1 = np.indices((8, 9), dtype=np.float32)
    fx = (0.03 * i1 + 0.07 * i2).astype(np.float32)
    normal = np.array([0.8, 0.35], dtype=np.float32)
    strike = np.array([-0.35, 0.8], dtype=np.float32)

    fallback = voting2d._samples_in_uv_box_python(
        4,
        3,
        voter.ru,
        voter.rv,
        normal,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )
    accelerated = voting2d._samples_in_uv_box_numba(
        4,
        3,
        voter.ru,
        voter.rv,
        normal,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )

    assert accelerated.dtype == np.float32
    np.testing.assert_array_equal(accelerated, fallback)


def test_numba_2d_vote_accumulation_matches_python_fallback() -> None:
    ft = np.zeros((11, 11), dtype=np.float32)
    ft[5, 3:8] = np.array([0.5, 0.7, 0.9, 0.7, 0.5], dtype=np.float32)
    path = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    normal = np.array([1.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 1.0], dtype=np.float32)
    fallback = tuple(np.zeros_like(ft) for _ in range(4))
    accelerated = tuple(np.zeros_like(ft) for _ in range(4))

    voting2d._accumulate_path_votes_python(5, 5, 2, normal, strike, path, ft, *fallback)
    voting2d._accumulate_path_votes_numba(5, 5, 2, normal, strike, path, ft, *accelerated)

    for accelerated_array, fallback_array in zip(accelerated, fallback):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(accelerated_array, fallback_array, rtol=1e-6, atol=1e-6)


def test_numba_2d_public_voting_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalPathVoter(ru=1, rv=3)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((15, 15), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[7, 3:12] = 0.9

    monkeypatch.setattr(voting2d, "NUMBA_AVAILABLE", False)
    fallback = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    monkeypatch.setattr(voting2d, "NUMBA_AVAILABLE", True)
    accelerated = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    for accelerated_array, fallback_array in zip(accelerated, fallback):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(accelerated_array, fallback_array, rtol=1e-6, atol=1e-6)


def test_numba_3d_local_sampling_matches_python_fallback() -> None:
    voter = OptimalSurfaceVoter(ru=2, rv=2, rw=2)
    i3, i2, i1 = np.indices((7, 8, 9), dtype=np.float32)
    fx = (0.02 * i1 + 0.05 * i2 + 0.09 * i3).astype(np.float32)
    normal = np.array([0.85, 0.25, 0.1], dtype=np.float32)
    dip = np.array([-0.25, 0.85, 0.15], dtype=np.float32)
    strike = np.array([0.0, -0.15, 0.9], dtype=np.float32)

    fallback = voting3d._samples_in_uvw_box_python(
        4,
        3,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )
    accelerated = voting3d._samples_in_uvw_box_numba(
        4,
        3,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )

    assert accelerated.dtype == np.float32
    np.testing.assert_array_equal(accelerated, fallback)


def test_numba_3d_local_sampling_matches_fallback_at_float32_half_boundary() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=3, rw=0)
    i3, i2, i1 = np.indices((5, 5, 5), dtype=np.float32)
    fx = (0.01 * i1 + 0.1 * i2 + 0.2 * i3).astype(np.float32)
    normal = np.array(
        [np.nextafter(np.float32(0.5), np.float32(0.0)), 0.0, 0.0],
        dtype=np.float32,
    )
    dip = np.zeros(3, dtype=np.float32)
    strike = np.zeros(3, dtype=np.float32)

    fallback = voting3d._samples_in_uvw_box_python(
        2,
        2,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )
    accelerated = voting3d._samples_in_uvw_box_numba(
        2,
        2,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )

    expected = np.float32(1.0) - fx[2, 2, 3]
    assert fallback[0, 6, 2] == expected
    np.testing.assert_array_equal(accelerated, fallback)


def test_numba_3d_local_uvw_sampling_matches_python_fallback_at_exact_half_samples() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=3, rw=3)
    i3, i2, i1 = np.indices((5, 5, 5), dtype=np.float32)
    fx = (0.01 * i1 + 0.1 * i2 + 0.2 * i3).astype(np.float32)
    normal = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    dip = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    fallback = voting3d._samples_in_uvw_box_python(
        2,
        2,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )
    accelerated = voting3d._samples_in_uvw_box_numba(
        2,
        2,
        2,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )

    assert fallback[6, 3, 2] == np.float32(1.0) - fx[4, 2, 3]
    assert fallback[3, 6, 0] == np.float32(1.0) - fx[2, 4, 2]
    assert fallback[6, 6, 2] == np.float32(1.0) - fx[4, 4, 3]
    np.testing.assert_array_equal(accelerated, fallback)


def test_numba_3d_vote_accumulation_matches_python_fallback() -> None:
    ft = np.zeros((9, 9, 9), dtype=np.float32)
    ft[2:7, 4, 2:7] = 0.8
    surface = np.zeros((5, 5), dtype=np.float32)
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dip = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fallback = (
        np.zeros_like(ft),
        np.full_like(ft, -1.0),
        np.full_like(ft, -1.0),
        np.zeros_like(ft),
    )
    accelerated = tuple(array.copy() for array in fallback)

    fallback_fa, fallback_count = voting3d._surface_vote_average_python(
        4,
        4,
        4,
        2,
        2,
        normal,
        dip,
        strike,
        surface,
        ft,
    )
    accelerated_fa, accelerated_count = voting3d._surface_vote_average_numba(
        4,
        4,
        4,
        2,
        2,
        normal,
        dip,
        strike,
        surface,
        ft,
    )
    assert accelerated_count == fallback_count
    assert accelerated_fa == pytest.approx(fallback_fa)

    voting3d._accumulate_surface_votes_python(
        4,
        4,
        4,
        2,
        2,
        fallback_fa,
        np.float32(0.0),
        np.float32(90.0),
        True,
        normal,
        dip,
        strike,
        surface,
        *fallback,
    )
    voting3d._accumulate_surface_votes_numba(
        4,
        4,
        4,
        2,
        2,
        accelerated_fa,
        np.float32(0.0),
        np.float32(90.0),
        True,
        normal,
        dip,
        strike,
        surface,
        *accelerated,
    )

    for accelerated_array, fallback_array in zip(accelerated, fallback):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(accelerated_array, fallback_array, rtol=1e-6, atol=1e-6)


def test_numba_3d_public_voting_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.8

    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", False)
    fallback = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", True)
    accelerated = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    for accelerated_array, fallback_array in zip(accelerated, fallback):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(accelerated_array, fallback_array, rtol=1e-6, atol=1e-6)


def test_numba_seed_order_determinism_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalPathVoter(ru=1, rv=3)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    ft = np.zeros((15, 15), dtype=np.float32)
    pt = np.zeros_like(ft)
    ft[7, 3:12] = 0.9
    seeds = [FaultCell2(5, 7, 0.9, 0.0), FaultCell2(9, 7, 0.9, 0.0)]

    monkeypatch.setattr(voter, "pick_seeds", lambda *_args, **_kwargs: list(seeds))
    monkeypatch.setattr(voting2d, "NUMBA_AVAILABLE", True)

    first = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_numba_3d_seed_order_determinism_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.8
    seeds = [
        FaultCell(4, 5, 4, 0.8, 0.0, 90.0),
        FaultCell(6, 5, 6, 0.8, 0.0, 90.0),
    ]

    monkeypatch.setattr(
        voter,
        "_pick_seeds_validated",
        lambda *_args, **_kwargs: list(seeds),
    )
    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", True)

    first = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


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
def test_numba_3d_masked_oblique_sampling_matches_python_fallback(
    c1: int,
    c2: int,
    c3: int,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=3, rw=3)
    i3, i2, i1 = np.indices((7, 7, 7), dtype=np.float32)
    fx = (1.5 + 0.03 * i1 + 0.07 * i2 + 0.11 * i3).astype(np.float32)
    normal = np.array([-0.47, 0.62, -0.33], dtype=np.float32)
    dip = np.array([-0.40, -0.71, -0.58], dtype=np.float32)
    strike = np.array([-0.78, -0.12, 0.61], dtype=np.float32)

    fallback = voting3d._samples_in_uvw_box_masked_python(
        c1,
        c2,
        c3,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )
    accelerated = voting3d._samples_in_uvw_box_masked_numba(
        c1,
        c2,
        c3,
        voter.ru,
        voter.rv,
        voter.rw,
        normal,
        dip,
        strike,
        fx,
        voter.lmins,
        voter.lmaxs,
    )

    fallback_costs, fallback_mask, fallback_admissible, fallback_in_bounds = fallback
    accelerated_costs, accelerated_mask, accelerated_admissible, accelerated_in_bounds = accelerated
    assert accelerated_costs.dtype == np.float32
    assert accelerated_mask.dtype == np.bool_
    assert accelerated_admissible == fallback_admissible
    assert accelerated_in_bounds == fallback_in_bounds
    np.testing.assert_array_equal(accelerated_costs, fallback_costs)
    np.testing.assert_array_equal(accelerated_mask, fallback_mask)


def test_numba_3d_masked_sampling_matches_fallback_at_float32_half_boundary() -> None:
    fx = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    lmins = np.array([[-1]], dtype=np.int32)
    lmaxs = np.array([[1]], dtype=np.int32)
    normal = np.array(
        [np.nextafter(np.float32(0.5), np.float32(1.0)), 0.0, 0.0],
        dtype=np.float32,
    )
    args = (
        0,
        0,
        0,
        1,
        0,
        0,
        normal,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        fx,
        lmins,
        lmaxs,
    )

    fallback = voting3d._samples_in_uvw_box_masked_python(*args)
    accelerated = voting3d._samples_in_uvw_box_masked_numba(*args)

    assert not fallback[1][0, 0, 0]
    assert fallback[2:] == (3, 2)
    np.testing.assert_array_equal(accelerated[0], fallback[0])
    np.testing.assert_array_equal(accelerated[1], fallback[1])
    assert accelerated[2:] == fallback[2:]


def test_numba_3d_masked_score_and_accumulation_match_python_fallback() -> None:
    i3, i2, i1 = np.indices((7, 7, 7), dtype=np.float32)
    ft = (0.2 + 0.01 * i1 + 0.03 * i2 + 0.05 * i3).astype(np.float32)
    surface = np.zeros((2, 3), dtype=np.float32)
    valid_lag_mask = np.ones((2, 3, 1), dtype=np.bool_)
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dip = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    score_args = (
        3,
        3,
        0,
        1,
        1,
        1,
        0,
        0,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
        ft,
    )

    fallback_score = voting3d._surface_vote_average_masked_python(*score_args)
    accelerated_score = voting3d._surface_vote_average_masked_numba(*score_args)

    assert accelerated_score[1:] == fallback_score[1:] == (6, 0)
    assert accelerated_score[0] == pytest.approx(fallback_score[0], abs=1.0e-7)

    fallback_arrays = (
        np.zeros_like(ft),
        np.full_like(ft, -1.0),
        np.full_like(ft, -1.0),
        np.zeros_like(ft),
    )
    accelerated_arrays = tuple(array.copy() for array in fallback_arrays)
    accumulation_args = (
        3,
        3,
        0,
        1,
        1,
        1,
        0,
        0,
        fallback_score[0],
        np.float32(12.0),
        np.float32(34.0),
        False,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
    )
    fallback_counts = voting3d._accumulate_surface_votes_masked_python(
        *accumulation_args,
        *fallback_arrays,
    )
    accelerated_counts = voting3d._accumulate_surface_votes_masked_numba(
        *accumulation_args,
        *accelerated_arrays,
    )

    assert accelerated_counts == fallback_counts == (6, 3, 0)
    for accelerated_array, fallback_array in zip(accelerated_arrays, fallback_arrays):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(
            accelerated_array,
            fallback_array,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_numba_3d_masked_mapping_matches_python_at_float32_half_boundary() -> None:
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
        ft,
    )

    fallback = voting3d._surface_vote_average_masked_python(*args)
    accelerated = voting3d._surface_vote_average_masked_numba(*args)

    assert fallback == (np.float32(0.75), 1, 0)
    assert accelerated == fallback


def test_numba_3d_public_masked_boundary_voting_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voter = OptimalSurfaceVoter(ru=0, rv=1, rw=1)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    voter.set_surface_orientation_smoothing(0.0)
    voter.set_surface_voting_boundary_policy("masked_in_bounds")
    i3, i2, i1 = np.indices((7, 7, 7), dtype=np.float32)
    ft = (0.2 + 0.01 * i1 + 0.03 * i2 + 0.05 * i3).astype(np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    seeds = [FaultCell(3, 3, 0, float(ft[0, 3, 3]), 0.0, 90.0)]

    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", False)
    fallback = voter.apply_voting_from_seeds(seeds, ft, pt, tt)
    fallback_diagnostics = voter.surface_voting_diagnostics

    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", True)
    accelerated = voter.apply_voting_from_seeds(seeds, ft, pt, tt)
    accelerated_diagnostics = voter.surface_voting_diagnostics

    assert accelerated_diagnostics == fallback_diagnostics
    assert accelerated_diagnostics[0].orientation_source == "seed_boundary_fallback"
    assert accelerated_diagnostics[0].selected_invalid_sample_count == 0
    for accelerated_array, fallback_array in zip(accelerated, fallback):
        assert accelerated_array.dtype == np.float32
        np.testing.assert_allclose(
            accelerated_array,
            fallback_array,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
