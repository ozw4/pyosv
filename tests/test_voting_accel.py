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

    expected = np.float32(1.0) - fx[2, 2, 2]
    assert fallback[0, 6, 2] == expected
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

    monkeypatch.setattr(voter, "pick_seeds", lambda *_args, **_kwargs: list(seeds))
    monkeypatch.setattr(voting3d, "NUMBA_AVAILABLE", True)

    first = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    second = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)
