"""Probability sampling restricted to a selected candidate corridor."""

from __future__ import annotations

import numpy as np

from pyosv._fault_surface.models import FaultSurfaceFitConfig, FaultSurfaceModel


def sample_probability(
    probability: np.ndarray, points: np.ndarray, valid_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.asarray(probability.shape[::-1]) - 1
    valid = np.all((points >= 0) & (points <= maximum), axis=1)
    clipped = np.clip(points, 0, maximum)
    base = np.floor(clipped).astype(np.intp)
    fractions = clipped - base
    total = np.zeros(len(points), dtype=np.float64)
    invalid_values = np.zeros(len(points), dtype=bool)
    for corner in np.ndindex(2, 2, 2):
        indices = np.minimum(base + corner, maximum)
        weights = np.prod(np.where(corner, fractions, 1 - fractions), axis=1)
        active = weights > 0
        values = probability[tuple(indices[:, ::-1].T)]
        available = np.ones(len(points), dtype=bool)
        if valid_mask is not None:
            available = valid_mask[tuple(indices[:, ::-1].T)]
        invalid_values |= active & (~np.isfinite(values) | (values < 0) | (values > 1))
        valid &= ~active | available
        # Zero-weight NaN neighbors must not contaminate exact-grid samples.
        total += weights * np.where(active & available & np.isfinite(values), values, 0.0)
    if np.any(valid & invalid_values):
        raise ValueError("Sampled valid probability must be finite and within [0, 1]")
    return np.where(valid, np.clip(total, 0.0, 1.0), 0.0), valid


def reject_ambiguous_bands(
    values: np.ndarray,
    valid: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> None:
    if len(offsets) < 3:
        return
    order = np.argsort(offsets)
    ordered = values[:, order]
    coverage = valid[:, order]
    spacing = float(np.diff(np.sort(offsets))[0])
    for row, known in zip(ordered, coverage):
        supported = np.flatnonzero(known & (row >= threshold))
        for left, right in zip(supported[:-1], supported[1:]):
            gap = slice(left + 1, right)
            if (right - left - 1) * spacing >= 1.0 - 1e-9 and np.all(known[gap]):
                raise ValueError(
                    "Ambiguous prediction bands in the candidate corridor; narrow search_radius "
                    "or select a separate patch"
                )


def corridor_targets(
    probability: np.ndarray,
    valid_mask: np.ndarray | None,
    model: FaultSurfaceModel,
    guide: np.ndarray,
    config: FaultSurfaceFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.search_radius == 0:
        offsets = np.array([0.0])
    else:
        steps = int(np.ceil(config.search_radius / config.search_step))
        offsets = np.linspace(-config.search_radius, config.search_radius, 2 * steps + 1)
        offsets = offsets[np.lexsort((offsets, np.abs(offsets)))]
    samples = np.repeat(guide[:, None, :], len(offsets), axis=1)
    samples[:, :, model.dependent_axis] += offsets
    values, valid = sample_probability(probability, samples.reshape(-1, 3), valid_mask)
    values, valid = values.reshape(samples.shape[:2]), valid.reshape(samples.shape[:2])
    reject_ambiguous_bands(values, valid, offsets, config.support_threshold)
    scores = np.where(valid, values, -1.0)
    choice = np.argmax(scores, axis=1)
    best = scores[np.arange(len(guide)), choice]
    targets = samples[np.arange(len(guide)), choice, model.dependent_axis]
    supported = best >= config.support_threshold
    targets = np.where(supported, targets, guide[:, model.dependent_axis])
    weights = np.where(supported, best**2, 0.0)
    if not np.any(supported):
        raise ValueError("No valid prediction support was found in the candidate corridor")
    return targets, weights
