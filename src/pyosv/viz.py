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


def slice_2d(volume: ArrayLike, axis: str | int, index: int) -> np.ndarray:
    """Return a 2D slice from a 3D ``(n3, n2, n1)`` volume."""
    values = np.asarray(volume)
    if values.ndim != 3:
        raise ValueError("volume must be a 3D (n3, n2, n1) array")

    axis_name, axis_number = _normalize_axis(axis)
    axis_size = values.shape[axis_number]
    if index < 0 or index >= axis_size:
        raise ValueError(f"{axis_name} index must be between 0 and {axis_size - 1}")

    if axis_number == 0:
        return values[index, :, :]
    if axis_number == 1:
        return values[:, index, :]
    return values[:, :, index]


def save_slice_panel(
    output_path: str | Path,
    panels: list[tuple[str, ArrayLike]],
    *,
    title: str | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    cmap: str = "gray",
) -> Path:
    """Save a row of normalized 2D slice panels as a PNG image."""
    if not panels:
        raise ValueError("panels must contain at least one panel")

    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.0 * len(panels), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    try:
        if title is not None:
            fig.suptitle(title)

        for ax, (panel_title, panel_values) in zip(axes[0], panels, strict=True):
            display = normalize_for_display(panel_values, clip_percentiles=clip_percentiles)
            if display.ndim != 2:
                raise ValueError("each panel must be a 2D array")
            ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper", aspect="auto")
            ax.set_title(panel_title)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)

    return output_file


def save_volume_comparison_slices(
    output_dir: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    name: str,
    slice_indices: dict[str, int] | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> dict[str, Path]:
    """Save reference/candidate/difference slice panels for each 3D axis."""
    reference_values = np.asarray(reference, dtype=np.float32)
    candidate_values = np.asarray(candidate, dtype=np.float32)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference_values.ndim != 3:
        raise ValueError("reference and candidate must be 3D (n3, n2, n1) arrays")

    indices = select_center_slices(reference_values.shape)
    if slice_indices is not None:
        for axis, index in slice_indices.items():
            axis_name, _ = _normalize_axis(axis)
            indices[axis_name] = index

    output_path = ensure_output_dir(output_dir)
    written: dict[str, Path] = {}
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        reference_slice = slice_2d(reference_values, axis, index)
        candidate_slice = slice_2d(candidate_values, axis, index)
        difference = np.abs(candidate_slice - reference_slice)
        panel_path = output_path / f"{name}_{axis}_{index}.png"
        written[axis] = save_slice_panel(
            panel_path,
            [
                ("reference", reference_slice),
                ("candidate", candidate_slice),
                ("absolute difference", difference),
            ],
            title=f"{name} {axis}={index}",
            clip_percentiles=clip_percentiles,
        )

    return written


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


def _normalize_axis(axis: str | int) -> tuple[str, int]:
    if axis == "i3" or axis == 0:
        return "i3", 0
    if axis == "i2" or axis == 1:
        return "i2", 1
    if axis == "i1" or axis == 2:
        return "i1", 2
    raise ValueError('axis must be one of "i3", "i2", "i1", 0, 1, or 2')


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
