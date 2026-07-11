"""Orientation basis, scoring, and confidence helpers."""

from __future__ import annotations

import numpy as np

from pyosv._orient3d.interpolation import _smooth_oriented_response
from pyosv._orient3d.normalization import _normalize_unit_range
from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
)


def _reference_like_planarity_to_likelihood(smoothed: np.ndarray) -> np.ndarray:
    clipped = np.clip(smoothed, np.float32(0.0), np.float32(1.0))
    score = np.float32(1.0) - clipped ** np.float32(4.0)
    return np.clip(score, np.float32(0.0), np.float32(1.0)).astype(
        np.float32,
        copy=False,
    )


def _orientation_basis_from_strike_and_dip(
    phi: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = fault_normal_vector_from_strike_and_dip(phi, theta)
    strike = fault_strike_vector_from_strike_and_dip(phi, theta)
    dip = fault_dip_vector_from_strike_and_dip(phi, theta)
    return normal, strike, dip


def _reference_like_orientation_score(
    image: np.ndarray,
    *,
    strike: np.ndarray,
    dip: np.ndarray,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    interpolation_order: int,
    smoothing_sigma: float,
) -> np.ndarray:
    smoothed = _smooth_oriented_response(
        image,
        strike=strike,
        dip=dip,
        grids=grids,
        interpolation_order=interpolation_order,
        smoothing_sigma=smoothing_sigma,
    )
    smoothed = np.clip(smoothed, np.float32(0.0), np.float32(1.0))
    score = np.float32(1.0) - smoothed ** np.float32(4.0)
    return score.astype(np.float32, copy=False)


def _update_best_second_orientation(
    score: np.ndarray,
    phi: np.float32,
    theta: np.float32,
    best_score: np.ndarray,
    second_score: np.ndarray,
    best_phi: np.ndarray,
    best_theta: np.ndarray,
) -> None:
    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    better = score_float32 > best_score
    second_better = (~better) & (score_float32 > second_score)

    second_score[better] = best_score[better]
    second_score[second_better] = score_float32[second_better]
    best_score[better] = score_float32[better]
    best_phi[better] = phi
    best_theta[better] = theta


def _orientation_confidence_from_scores(
    best_score: np.ndarray,
    second_score: np.ndarray,
) -> np.ndarray:
    raw = np.maximum(
        best_score.astype(np.float32, copy=False) - second_score.astype(np.float32, copy=False),
        np.float32(0.0),
    )
    return _normalize_unit_range(raw)
