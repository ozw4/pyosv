"""Optional static-visualization helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def require_matplotlib() -> Any:
    """Return ``matplotlib.pyplot`` or explain how to install it."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for pyosv visualization helpers. "
            'Install it with `pip install "pyosv[viz]"`.'
        ) from exc

    if "matplotlib.pyplot" not in sys.modules and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def ensure_output_dir(path: str | Path) -> Path:
    """Create an output directory and return it as a ``Path``."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_percentile_threshold(volume: ArrayLike, percentile: float) -> float:
    """Return a finite percentile threshold for finite values in ``volume``."""
    _validate_percentile(percentile, "percentile")
    values = np.asarray(volume)
    if values.size == 0:
        return 0.0

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0

    threshold = float(np.percentile(finite, percentile))
    if not np.isfinite(threshold):
        return 0.0
    return threshold


def normalize_for_display(
    volume_or_slice: ArrayLike,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Normalize finite data into a float32 display array in the range ``[0, 1]``."""
    low_percentile, high_percentile = _validate_clip_percentiles(clip_percentiles)
    values = np.asarray(volume_or_slice, dtype=np.float32)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if values.size == 0:
        return normalized

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return normalized

    finite = values[finite_mask]
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return normalized

    clipped = np.clip(values, low, high)
    clipped = np.where(np.isfinite(clipped), clipped, low)
    normalized = (clipped - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def select_center_slices(shape: tuple[int, int, int]) -> dict[str, int]:
    """Return center slice indices for a 3D ``(n3, n2, n1)`` shape."""
    if len(shape) != 3:
        raise ValueError("shape must be a 3D (n3, n2, n1) tuple")
    n3, n2, n1 = shape
    if n3 <= 0 or n2 <= 0 or n1 <= 0:
        raise ValueError("shape dimensions must be positive")
    return {"i3": n3 // 2, "i2": n2 // 2, "i1": n1 // 2}


def _validate_percentile(percentile: float, name: str) -> None:
    if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"{name} must be finite and between 0 and 100")


def _validate_clip_percentiles(clip_percentiles: tuple[float, float]) -> tuple[float, float]:
    if len(clip_percentiles) != 2:
        raise ValueError("clip_percentiles must contain two values")
    low, high = clip_percentiles
    _validate_percentile(low, "low clip percentile")
    _validate_percentile(high, "high clip percentile")
    if high < low:
        raise ValueError("high clip percentile must be greater than or equal to low")
    return low, high
