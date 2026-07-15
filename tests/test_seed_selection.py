from __future__ import annotations

import numpy as np
import pytest

import pyosv._skinner.seeds as skinner_seeds_module
import pyosv.voting2d as voting2d_module
import pyosv.voting3d as voting3d_module
from pyosv._accel import NUMBA_AVAILABLE
from pyosv._seed_selection import (
    _greedy_suppress_2d_numba,
    _greedy_suppress_2d_python,
    _greedy_suppress_3d_numba,
    _greedy_suppress_3d_python,
    _select_skinner_seed_indices_3d,
    _select_voter_seed_indices_2d,
    _select_voter_seed_indices_3d,
    _sorted_skinner_flat_indices,
    _sorted_voter_flat_indices,
)
from pyosv._skinner.models import _SkinCell
from pyosv._skinner.seeds import _find_reference_seeds
from pyosv.cells import FaultCell, FaultCell2
from pyosv.voting2d import OptimalPathVoter
from pyosv.voting3d import OptimalSurfaceVoter


def _oracle(sorted_indices: np.ndarray, shape: tuple[int, ...], distance: int) -> np.ndarray:
    mark = np.zeros(shape, dtype=np.bool_)
    accepted: list[int] = []
    for flat_index in sorted_indices:
        index = np.unravel_index(flat_index, shape)
        slices = tuple(
            slice(
                max(int(coordinate) - distance, 0),
                min(int(coordinate) + distance + 1, size),
            )
            for coordinate, size in zip(index, shape)
        )
        if mark[slices].any():
            continue
        accepted.append(int(flat_index))
        mark[index] = True
    return np.asarray(accepted, dtype=np.int64)


def _prechange_voter_oracle(
    scores: np.ndarray,
    threshold: np.float32,
    distance: int,
) -> np.ndarray:
    """Reproduce the pre-change row-major, stable-sort-then-reverse selector."""

    candidates = [index for index in np.ndindex(scores.shape) if scores[index] > threshold]
    candidates.sort(key=lambda index: scores[index])
    sorted_indices = np.asarray(
        [np.ravel_multi_index(index, scores.shape) for index in reversed(candidates)],
        dtype=np.int64,
    )
    return _oracle(sorted_indices, scores.shape, distance)


def _prechange_skinner_oracle(
    planarity: np.ndarray,
    scores: np.ndarray,
    planarity_threshold: np.float32,
    threshold: np.float32,
    distance: int,
) -> np.ndarray:
    """Reproduce the pre-change ``(-score, i3, i2, i1)`` selector."""

    candidates = [
        index
        for index in np.ndindex(scores.shape)
        if planarity[index] > planarity_threshold and scores[index] > threshold
    ]
    candidates.sort(key=lambda index: (-scores[index], *index))
    sorted_indices = np.asarray(
        [np.ravel_multi_index(index, scores.shape) for index in candidates],
        dtype=np.int64,
    )
    return _oracle(sorted_indices, scores.shape, distance)


def _cell_values(cell: FaultCell2 | FaultCell | _SkinCell) -> tuple[float, ...]:
    if isinstance(cell, FaultCell2):
        return (cell.i1, cell.i2, cell.fl, cell.fp)
    return (cell.i1, cell.i2, cell.i3, cell.fl, cell.fp, cell.ft)


@pytest.mark.parametrize("distance", [0, 1, 100, int(np.iinfo(np.int64).max)])
def test_suppression_kernels_cover_empty_all_candidate_and_large_distance_cases(
    distance: int,
) -> None:
    cases = [
        (np.empty(0, dtype=np.int64), (2, 3)),
        (np.arange(6, dtype=np.int64), (2, 3)),
        (np.empty(0, dtype=np.int64), (2, 2, 3)),
        (np.arange(12, dtype=np.int64), (2, 2, 3)),
    ]
    for candidates, shape in cases:
        expected = _oracle(candidates, shape, distance)
        if len(shape) == 2:
            python = _greedy_suppress_2d_python(candidates, shape, distance)
            numba = _greedy_suppress_2d_numba(candidates, shape, distance)
        else:
            python = _greedy_suppress_3d_python(candidates, shape, distance)
            numba = _greedy_suppress_3d_numba(candidates, shape, distance)
        np.testing.assert_array_equal(python, expected)
        np.testing.assert_array_equal(numba, expected)


@pytest.mark.parametrize(
    "distance",
    [int(np.iinfo(np.int64).max), int(np.iinfo(np.int64).max) + 1],
)
def test_public_seed_apis_accept_arbitrarily_large_distances(
    distance: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ft2 = np.ones((1, 3), dtype=np.float32)
    ft3 = ft2.reshape(1, 1, 3)

    for use_numba in (False, True):
        monkeypatch.setattr(voting2d_module, "NUMBA_AVAILABLE", use_numba)
        seeds2 = OptimalPathVoter(0, 0).pick_seeds(distance, 0.5, ft2, ft2)
        assert [seed.index for seed in seeds2] == [(2, 0)]

        monkeypatch.setattr(voting3d_module, "NUMBA_AVAILABLE", use_numba)
        seeds3 = OptimalSurfaceVoter(0, 0, 0).pick_seeds(
            distance,
            0.5,
            ft3,
            ft3,
            ft3,
        )
        assert [seed.index for seed in seeds3] == [(2, 0, 0)]

        monkeypatch.setattr(skinner_seeds_module, "NUMBA_AVAILABLE", use_numba)
        skin_seeds = _find_reference_seeds(distance, 0.5, ft3, ft3, ft3, ft3)
        assert [seed.index for seed in skin_seeds] == [(0, 0, 0)]


def test_sorting_noncontiguous_scores_does_not_ravel_score_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = np.arange(24, dtype=np.float32).reshape(4, 3, 2).transpose(2, 1, 0)
    mask = scores > np.float32(10.0)
    original_ravel = np.ravel

    def tracked_ravel(array: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        if array is scores:
            raise AssertionError("the score volume must not be raveled")
        return original_ravel(array, *args, **kwargs)

    monkeypatch.setattr(np, "ravel", tracked_ravel)
    voter_indices = _sorted_voter_flat_indices(mask, scores)
    skinner_indices = _sorted_skinner_flat_indices(mask, scores)

    assert voter_indices.size == skinner_indices.size == np.count_nonzero(mask)


def test_voter_and_skinner_use_opposite_flat_index_ties() -> None:
    scores = np.ones((1, 1, 4), dtype=np.float32)
    planarity = np.ones_like(scores)
    voter = _select_voter_seed_indices_3d(
        scores,
        np.float32(0.5),
        0,
        use_numba=False,
    )
    skinner = _select_skinner_seed_indices_3d(
        planarity,
        scores,
        np.float32(0.5),
        np.float32(0.5),
        0,
        use_numba=False,
    )
    np.testing.assert_array_equal(voter, [3, 2, 1, 0])
    np.testing.assert_array_equal(skinner, [0, 1, 2, 3])


def test_all_equal_voter_scores_have_exact_2d_and_3d_seed_order() -> None:
    ft2 = np.ones((2, 3), dtype=np.float32)
    seeds2 = OptimalPathVoter(ru=0, rv=0).pick_seeds(0, 0.5, ft2, np.zeros_like(ft2))
    assert [seed.index for seed in seeds2] == [
        (2, 1),
        (1, 1),
        (0, 1),
        (2, 0),
        (1, 0),
        (0, 0),
    ]

    ft3 = np.ones((2, 2, 2), dtype=np.float32)
    seeds3 = OptimalSurfaceVoter(ru=0, rv=0, rw=0).pick_seeds(
        0,
        0.5,
        ft3,
        np.zeros_like(ft3),
        np.full_like(ft3, 90.0),
    )
    assert [seed.index for seed in seeds3] == [
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 0),
    ]


def test_skinner_threshold_and_planarity_boundaries_are_strict() -> None:
    ep = np.array([[[0.8, 0.9, 0.9]]], dtype=np.float32)
    ft = np.array([[[0.9, 0.5, 0.6]]], dtype=np.float32)
    seeds = _find_reference_seeds(
        0,
        0.5,
        ep,
        ft,
        np.zeros_like(ft),
        np.full_like(ft, 90.0),
        min_ep=0.8,
    )
    assert [seed.index for seed in seeds] == [(2, 0, 0)]


def test_voter_threshold_boundary_is_strict() -> None:
    threshold = np.float32(0.5)
    ft = np.array([[threshold, np.nextafter(threshold, np.float32(1.0))]], dtype=np.float32)
    seeds = OptimalPathVoter(0, 0).pick_seeds(0, float(threshold), ft, np.zeros_like(ft))
    assert [seed.index for seed in seeds] == [(1, 0)]


def test_cell_objects_are_created_only_for_accepted_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ft2 = np.ones((3, 3), dtype=np.float32)
    created2: list[object] = []
    fault_cell2 = voting2d_module.FaultCell2

    def make_cell2(*args: object) -> object:
        cell = fault_cell2(*args)
        created2.append(cell)
        return cell

    monkeypatch.setattr(voting2d_module, "FaultCell2", make_cell2)
    OptimalPathVoter(0, 0).pick_seeds(100, 0.5, ft2, ft2)
    assert len(created2) == 1

    ft3 = np.ones((3, 3, 3), dtype=np.float32)
    created3: list[object] = []
    fault_cell = voting3d_module.FaultCell

    def make_cell3(*args: object) -> object:
        cell = fault_cell(*args)
        created3.append(cell)
        return cell

    monkeypatch.setattr(voting3d_module, "FaultCell", make_cell3)
    OptimalSurfaceVoter(0, 0, 0).pick_seeds(100, 0.5, ft3, ft3, ft3)
    assert len(created3) == 1

    created_skin: list[object] = []
    skin_cell = skinner_seeds_module._SkinCell

    def make_skin_cell(*args: object) -> object:
        cell = skin_cell(*args)
        created_skin.append(cell)
        return cell

    monkeypatch.setattr(skinner_seeds_module, "_SkinCell", make_skin_cell)
    _find_reference_seeds(100, 0.5, ft3, ft3, ft3, ft3)
    assert len(created_skin) == 1


@pytest.mark.parametrize("ndim", [2, 3])
def test_random_selectors_exactly_match_oracle_for_python_and_numba(ndim: int) -> None:
    rng = np.random.default_rng(394)
    shapes = [(1, 7), (3, 4), (5, 2)] if ndim == 2 else [(1, 2, 5), (2, 3, 4), (4, 2, 3)]
    for shape in shapes:
        for density in (0.0, 0.25, 1.0):
            scores = rng.integers(0, 4, size=shape).astype(np.float32)
            mask = rng.random(shape) < density
            scores = np.where(mask, scores + np.float32(1.0), np.float32(0.0))
            candidates = np.flatnonzero(scores > np.float32(0.5))
            order = np.argsort(scores.ravel()[candidates], kind="stable")[::-1]
            sorted_indices = np.asarray(candidates[order], dtype=np.int64)
            for distance in (0, 1, 8):
                expected = _oracle(sorted_indices, shape, distance)
                selector = (
                    _select_voter_seed_indices_2d if ndim == 2 else _select_voter_seed_indices_3d
                )
                python = selector(scores, np.float32(0.5), distance, use_numba=False)
                numba = selector(scores, np.float32(0.5), distance, use_numba=True)
                np.testing.assert_array_equal(python, expected)
                np.testing.assert_array_equal(numba, expected)


@pytest.mark.parametrize("distance", [0, 1, 8])
def test_final_fault_cell2_values_and_order_match_prechange_oracle(distance: int) -> None:
    rng = np.random.default_rng(394)
    threshold = np.float32(1.5)
    for shape in ((1, 7), (3, 4), (5, 2)):
        ft = rng.integers(0, 4, size=shape).astype(np.float32)
        pt = rng.uniform(-90.0, 90.0, size=shape).astype(np.float32)
        expected_indices = _prechange_voter_oracle(ft, threshold, distance)
        expected = [
            FaultCell2(i1, i2, ft[i2, i1], pt[i2, i1])
            for flat_index in expected_indices
            for i2, i1 in [np.unravel_index(flat_index, shape)]
        ]

        actual = OptimalPathVoter(0, 0).pick_seeds(distance, float(threshold), ft, pt)

        assert [_cell_values(cell) for cell in actual] == [_cell_values(cell) for cell in expected]


@pytest.mark.parametrize("distance", [0, 1, 8])
def test_final_fault_cell_values_and_order_match_prechange_oracle(distance: int) -> None:
    rng = np.random.default_rng(394)
    threshold = np.float32(1.5)
    for shape in ((1, 2, 5), (2, 3, 4), (4, 2, 3)):
        ft = rng.integers(0, 4, size=shape).astype(np.float32)
        pt = rng.uniform(0.0, 360.0, size=shape).astype(np.float32)
        tt = rng.uniform(0.0, 90.0, size=shape).astype(np.float32)
        expected_indices = _prechange_voter_oracle(ft, threshold, distance)
        expected = [
            FaultCell(i1, i2, i3, ft[i3, i2, i1], pt[i3, i2, i1], tt[i3, i2, i1])
            for flat_index in expected_indices
            for i3, i2, i1 in [np.unravel_index(flat_index, shape)]
        ]

        actual = OptimalSurfaceVoter(0, 0, 0).pick_seeds(
            distance,
            float(threshold),
            ft,
            pt,
            tt,
        )

        assert [_cell_values(cell) for cell in actual] == [_cell_values(cell) for cell in expected]


@pytest.mark.parametrize("distance", [0, 1, 8])
def test_final_skin_cell_values_and_order_match_prechange_oracle(distance: int) -> None:
    rng = np.random.default_rng(394)
    planarity_threshold = np.float32(0.5)
    threshold = np.float32(1.5)
    for shape in ((1, 2, 5), (2, 3, 4), (4, 2, 3)):
        ep = (rng.integers(0, 3, size=shape) * np.float32(0.5)).astype(np.float32)
        ft = rng.integers(0, 4, size=shape).astype(np.float32)
        pt = rng.uniform(0.0, 360.0, size=shape).astype(np.float32)
        tt = rng.uniform(0.0, 90.0, size=shape).astype(np.float32)
        expected_indices = _prechange_skinner_oracle(
            ep,
            ft,
            planarity_threshold,
            threshold,
            distance,
        )
        expected = [
            _SkinCell(i1, i2, i3, ft[i3, i2, i1], pt[i3, i2, i1], tt[i3, i2, i1])
            for flat_index in expected_indices
            for i3, i2, i1 in [np.unravel_index(flat_index, shape)]
        ]

        actual = _find_reference_seeds(
            distance,
            float(threshold),
            ep,
            ft,
            pt,
            tt,
            min_ep=float(planarity_threshold),
        )

        assert [_cell_values(cell) for cell in actual] == [_cell_values(cell) for cell in expected]


def test_public_seed_apis_use_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    ft2 = np.array([[0.6, 0.7]], dtype=np.float32)
    monkeypatch.setattr("pyosv.voting2d.NUMBA_AVAILABLE", False)
    assert [seed.index for seed in OptimalPathVoter(0, 0).pick_seeds(0, 0.5, ft2, ft2)] == [
        (1, 0),
        (0, 0),
    ]

    ft3 = ft2.reshape(1, 1, 2)
    monkeypatch.setattr("pyosv.voting3d.NUMBA_AVAILABLE", False)
    assert [
        seed.index for seed in OptimalSurfaceVoter(0, 0, 0).pick_seeds(0, 0.5, ft3, ft3, ft3)
    ] == [(1, 0, 0), (0, 0, 0)]

    monkeypatch.setattr("pyosv._skinner.seeds.NUMBA_AVAILABLE", False)
    skin_seeds = _find_reference_seeds(0, 0.5, np.ones_like(ft3), ft3, ft3, ft3)
    assert [seed.index for seed in skin_seeds] == [(1, 0, 0), (0, 0, 0)]


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_numba_seed_kernels_return_int64_arrays() -> None:
    indices = np.array([5, 0], dtype=np.int64)
    assert _greedy_suppress_2d_numba(indices, (2, 3), 0).dtype == np.int64
    assert _greedy_suppress_3d_numba(indices, (1, 2, 3), 0).dtype == np.int64
