"""Local-u candidate path kernels for reference-like skin growth."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit


def _pick_candidate_local_u_path_python(
    candidate_slice: np.ndarray,
    max_jump: int,
    jump_penalty: float,
) -> np.ndarray:
    nrow, nu = candidate_slice.shape
    if nrow == 0:
        return np.empty(0, dtype=np.int32)
    if nu == 1:
        return np.zeros(nrow, dtype=np.int32)

    max_jump_int = min(max_jump, nu - 1)
    jump_penalty_float = float(jump_penalty)
    accumulated = np.empty((nrow, nu), dtype=np.float32)
    predecessor = np.full((nrow, nu), -1, dtype=np.int32)
    accumulated[0] = candidate_slice[0]

    for irow in range(1, nrow):
        for iu in range(nu):
            ib = max(0, iu - max_jump_int)
            ie = min(nu, iu + max_jump_int + 1)
            best_score = -math.inf
            best_previous = ib
            best_jump = nu
            for ju in range(ib, ie):
                jump = abs(iu - ju)
                score = float(accumulated[irow - 1, ju]) - jump_penalty_float * jump
                previous_distance = abs(2 * ju - (nu - 1))
                best_previous_distance = abs(2 * best_previous - (nu - 1))
                if score > best_score or (
                    score == best_score
                    and (
                        jump < best_jump
                        or (
                            jump == best_jump
                            and (
                                previous_distance < best_previous_distance
                                or (
                                    previous_distance == best_previous_distance
                                    and ju < best_previous
                                )
                            )
                        )
                    )
                ):
                    best_score = score
                    best_previous = ju
                    best_jump = jump

            accumulated[irow, iu] = float(candidate_slice[irow, iu]) + best_score
            predecessor[irow, iu] = best_previous

    path = np.empty(nrow, dtype=np.int32)
    best_u = 0
    best_score = float(accumulated[-1, 0])
    for iu in range(1, nu):
        score = float(accumulated[-1, iu])
        distance = abs(2 * iu - (nu - 1))
        best_distance = abs(2 * best_u - (nu - 1))
        if score > best_score or (
            score == best_score
            and (distance < best_distance or (distance == best_distance and iu < best_u))
        ):
            best_u = iu
            best_score = score

    iu = best_u
    for irow in range(nrow - 1, -1, -1):
        path[irow] = iu
        previous = predecessor[irow, iu]
        if previous >= 0:
            iu = int(previous)

    return path


@njit(cache=True)
def _pick_candidate_local_u_path_numba(
    candidate_slice: np.ndarray,
    max_jump: int,
    jump_penalty: float,
) -> np.ndarray:
    nrow, nu = candidate_slice.shape
    if nrow == 0:
        return np.empty(0, dtype=np.int32)
    if nu == 1:
        return np.zeros(nrow, dtype=np.int32)

    max_jump_int = min(max_jump, nu - 1)
    jump_penalty_float = float(jump_penalty)
    accumulated = np.empty((nrow, nu), dtype=np.float32)
    predecessor = np.full((nrow, nu), -1, dtype=np.int32)
    accumulated[0] = candidate_slice[0]

    for irow in range(1, nrow):
        for iu in range(nu):
            ib = max(0, iu - max_jump_int)
            ie = min(nu, iu + max_jump_int + 1)
            best_score = -math.inf
            best_previous = ib
            best_jump = nu
            for ju in range(ib, ie):
                jump = abs(iu - ju)
                score = float(accumulated[irow - 1, ju]) - jump_penalty_float * jump
                previous_distance = abs(2 * ju - (nu - 1))
                best_previous_distance = abs(2 * best_previous - (nu - 1))
                if score > best_score or (
                    score == best_score
                    and (
                        jump < best_jump
                        or (
                            jump == best_jump
                            and (
                                previous_distance < best_previous_distance
                                or (
                                    previous_distance == best_previous_distance
                                    and ju < best_previous
                                )
                            )
                        )
                    )
                ):
                    best_score = score
                    best_previous = ju
                    best_jump = jump

            accumulated[irow, iu] = float(candidate_slice[irow, iu]) + best_score
            predecessor[irow, iu] = best_previous

    path = np.empty(nrow, dtype=np.int32)
    best_u = 0
    best_score = float(accumulated[-1, 0])
    for iu in range(1, nu):
        score = float(accumulated[-1, iu])
        distance = abs(2 * iu - (nu - 1))
        best_distance = abs(2 * best_u - (nu - 1))
        if score > best_score or (
            score == best_score
            and (distance < best_distance or (distance == best_distance and iu < best_u))
        ):
            best_u = iu
            best_score = score

    iu = best_u
    for irow in range(nrow - 1, -1, -1):
        path[irow] = iu
        previous = predecessor[irow, iu]
        if previous >= 0:
            iu = int(previous)

    return path
