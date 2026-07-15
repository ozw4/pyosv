"""Private exact-greedy seed selection helpers."""

from __future__ import annotations

import numpy as np

from pyosv._accel import njit


def _sorted_voter_flat_indices(candidate_mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return score-descending candidates with descending flat-index ties."""

    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_scores = np.ravel(scores)[candidate_indices]
    order = np.argsort(candidate_scores, kind="stable")[::-1]
    return np.asarray(candidate_indices[order], dtype=np.int64)


def _sorted_skinner_flat_indices(candidate_mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return score-descending candidates with ascending flat-index ties."""

    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_scores = np.ravel(scores)[candidate_indices]
    order = np.argsort(-candidate_scores, kind="stable")
    return np.asarray(candidate_indices[order], dtype=np.int64)


def _greedy_suppress_2d_python(
    sorted_flat_indices: np.ndarray,
    shape: tuple[int, int],
    distance: int,
) -> np.ndarray:
    """Keep 2D candidates whose Chebyshev box contains no accepted center."""

    n2, n1 = shape
    mark = np.zeros(shape, dtype=np.bool_)
    accepted = np.empty(sorted_flat_indices.size, dtype=np.int64)
    accepted_count = 0
    for flat_index in sorted_flat_indices:
        i2, i1 = divmod(int(flat_index), n1)
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        if mark[b2 : e2 + 1, b1 : e1 + 1].any():
            continue
        accepted[accepted_count] = flat_index
        accepted_count += 1
        mark[i2, i1] = True
    return accepted[:accepted_count]


@njit(cache=True)
def _greedy_suppress_2d_numba(
    sorted_flat_indices: np.ndarray,
    shape: tuple[int, int],
    distance: int,
) -> np.ndarray:
    """Numba implementation of exact-greedy 2D Chebyshev suppression."""

    n2, n1 = shape
    mark = np.zeros(shape, dtype=np.bool_)
    accepted = np.empty(sorted_flat_indices.size, dtype=np.int64)
    accepted_count = 0
    for flat_index in sorted_flat_indices:
        i2 = flat_index // n1
        i1 = flat_index - i2 * n1
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        occupied = False
        for j2 in range(b2, e2 + 1):
            for j1 in range(b1, e1 + 1):
                if mark[j2, j1]:
                    occupied = True
                    break
            if occupied:
                break
        if occupied:
            continue
        accepted[accepted_count] = flat_index
        accepted_count += 1
        mark[i2, i1] = True
    return accepted[:accepted_count]


def _greedy_suppress_3d_python(
    sorted_flat_indices: np.ndarray,
    shape: tuple[int, int, int],
    distance: int,
) -> np.ndarray:
    """Keep 3D candidates whose Chebyshev box contains no accepted center."""

    n3, n2, n1 = shape
    plane_size = n2 * n1
    mark = np.zeros(shape, dtype=np.bool_)
    accepted = np.empty(sorted_flat_indices.size, dtype=np.int64)
    accepted_count = 0
    for flat_index in sorted_flat_indices:
        i3, remainder = divmod(int(flat_index), plane_size)
        i2, i1 = divmod(remainder, n1)
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        b3 = max(i3 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        e3 = min(i3 + distance, n3 - 1)
        if mark[b3 : e3 + 1, b2 : e2 + 1, b1 : e1 + 1].any():
            continue
        accepted[accepted_count] = flat_index
        accepted_count += 1
        mark[i3, i2, i1] = True
    return accepted[:accepted_count]


@njit(cache=True)
def _greedy_suppress_3d_numba(
    sorted_flat_indices: np.ndarray,
    shape: tuple[int, int, int],
    distance: int,
) -> np.ndarray:
    """Numba implementation of exact-greedy 3D Chebyshev suppression."""

    n3, n2, n1 = shape
    plane_size = n2 * n1
    mark = np.zeros(shape, dtype=np.bool_)
    accepted = np.empty(sorted_flat_indices.size, dtype=np.int64)
    accepted_count = 0
    for flat_index in sorted_flat_indices:
        i3 = flat_index // plane_size
        remainder = flat_index - i3 * plane_size
        i2 = remainder // n1
        i1 = remainder - i2 * n1
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        b3 = max(i3 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        e3 = min(i3 + distance, n3 - 1)
        occupied = False
        for j3 in range(b3, e3 + 1):
            for j2 in range(b2, e2 + 1):
                for j1 in range(b1, e1 + 1):
                    if mark[j3, j2, j1]:
                        occupied = True
                        break
                if occupied:
                    break
            if occupied:
                break
        if occupied:
            continue
        accepted[accepted_count] = flat_index
        accepted_count += 1
        mark[i3, i2, i1] = True
    return accepted[:accepted_count]


def _select_voter_seed_indices_2d(
    scores: np.ndarray,
    threshold: np.float32,
    distance: int,
    *,
    use_numba: bool,
) -> np.ndarray:
    """Select voter seeds while preserving reverse-stable tie ordering."""

    candidate_mask = scores > threshold
    sorted_indices = _sorted_voter_flat_indices(candidate_mask, scores)
    kernel = _greedy_suppress_2d_numba if use_numba else _greedy_suppress_2d_python
    return kernel(sorted_indices, scores.shape, distance)


def _select_voter_seed_indices_3d(
    scores: np.ndarray,
    threshold: np.float32,
    distance: int,
    *,
    use_numba: bool,
) -> np.ndarray:
    """Select voter seeds while preserving reverse-stable tie ordering."""

    candidate_mask = scores > threshold
    sorted_indices = _sorted_voter_flat_indices(candidate_mask, scores)
    kernel = _greedy_suppress_3d_numba if use_numba else _greedy_suppress_3d_python
    return kernel(sorted_indices, scores.shape, distance)


def _select_skinner_seed_indices_3d(
    planarity: np.ndarray,
    scores: np.ndarray,
    planarity_threshold: np.float32,
    threshold: np.float32,
    distance: int,
    *,
    use_numba: bool,
) -> np.ndarray:
    """Select skinner seeds while preserving ascending-flat-index ties."""

    candidate_mask = (planarity > planarity_threshold) & (scores > threshold)
    sorted_indices = _sorted_skinner_flat_indices(candidate_mask, scores)
    kernel = _greedy_suppress_3d_numba if use_numba else _greedy_suppress_3d_python
    return kernel(sorted_indices, scores.shape, distance)
