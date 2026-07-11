"""Likelihood normalization helpers."""

from __future__ import annotations

import math

import numpy as np


def _normalize_reference_like_likelihood(score: np.ndarray) -> np.ndarray:
    return np.clip(score.astype(np.float32, copy=False), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _normalize_unit_range(values: np.ndarray) -> np.ndarray:
    values_float32 = np.maximum(values.astype(np.float32, copy=False), np.float32(0.0))
    low = float(np.min(values_float32))
    high = float(np.max(values_float32))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros_like(values_float32, dtype=np.float32)

    normalized = (values_float32 - np.float32(low)) / np.float32(high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_likelihood(score: np.ndarray) -> np.ndarray:
    score_float32 = np.maximum(score.astype(np.float32, copy=False), np.float32(0.0))
    high = float(np.percentile(score_float32, 99.5))
    if not math.isfinite(high) or high <= 0.0:
        return np.zeros_like(score_float32, dtype=np.float32)

    normalized = np.clip(score_float32 / np.float32(high), 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)
