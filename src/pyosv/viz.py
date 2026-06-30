"""Optional static-visualization helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyosv.metrics import _dilate_mask as _dilate_ridge_mask
from pyosv.metrics import top_percentile_mask

_RIDGE_REFERENCE_ONLY_RGB = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_RIDGE_CANDIDATE_ONLY_RGB = np.array([0.0, 0.25, 1.0], dtype=np.float32)
_RIDGE_EXACT_OVERLAP_RGB = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_RIDGE_BUFFERED_MATCH_RGB = np.array([0.0, 1.0, 1.0], dtype=np.float32)


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


def maximum_intensity_projection(volume: ArrayLike, axis: str | int) -> np.ndarray:
    """Return the maximum-intensity projection of a 3D ``(n3, n2, n1)`` volume."""
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("volume must be a 3D (n3, n2, n1) array")
    if any(size <= 0 for size in values.shape):
        raise ValueError("volume dimensions must be positive")

    _, axis_number = _normalize_axis(axis)
    return np.max(values, axis=axis_number).astype(np.float32, copy=False)


def save_mip_comparison(
    output_path: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    name: str,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save reference/candidate/difference MIP panels for all three axes."""
    reference_values, candidate_values = _validate_volume_pair(reference, candidate)
    _validate_clip_percentiles(clip_percentiles)

    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    plt = require_matplotlib()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9.0, 9.0),
        squeeze=False,
        constrained_layout=True,
    )
    try:
        fig.suptitle(f"{name} maximum-intensity projections")
        for row, axis in enumerate(("i3", "i2", "i1")):
            reference_mip = maximum_intensity_projection(reference_values, axis)
            candidate_mip = maximum_intensity_projection(candidate_values, axis)
            difference = np.abs(candidate_mip - reference_mip)
            panels = (
                ("reference", reference_mip),
                ("candidate", candidate_mip),
                ("absolute difference", difference),
            )
            for col, (panel_title, panel_values) in enumerate(panels):
                display = normalize_for_display(
                    panel_values,
                    clip_percentiles=clip_percentiles,
                )
                ax = axes[row, col]
                ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", aspect="auto")
                ax.set_title(f"{axis} {panel_title}")
                ax.set_xticks([])
                ax.set_yticks([])

        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)

    return output_file


def save_histogram_comparison(
    output_path: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    name: str,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
    log_count: bool = True,
) -> Path:
    """Save an overlaid histogram comparison for two volumes."""
    reference_values, candidate_values = _validate_volume_pair(reference, candidate)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if value_range is not None:
        _validate_value_range(value_range)

    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    reference_finite = _finite_values(reference_values)
    candidate_finite = _finite_values(candidate_values)

    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    try:
        ax.hist(
            reference_finite,
            bins=bins,
            range=value_range,
            histtype="step",
            linewidth=1.5,
            label="reference",
        )
        ax.hist(
            candidate_finite,
            bins=bins,
            range=value_range,
            histtype="step",
            linewidth=1.5,
            label="candidate",
        )
        ax.set_title(f"{name} value histogram")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        if log_count:
            ax.set_yscale("log")
        ax.legend()
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)

    return output_file


def save_volume_diagnostics(
    output_dir: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    name: str,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> dict[str, Path]:
    """Save deterministic MIP and histogram diagnostics for a volume pair."""
    output_path = ensure_output_dir(output_dir)
    return {
        "mip": save_mip_comparison(
            output_path / f"{name}_mip.png",
            reference=reference,
            candidate=candidate,
            name=name,
            clip_percentiles=clip_percentiles,
        ),
        "hist": save_histogram_comparison(
            output_path / f"{name}_hist.png",
            reference=reference,
            candidate=candidate,
            name=name,
        ),
    }


def ridge_mask(
    volume: ArrayLike,
    *,
    percentile: float = 99.0,
    positive_only: bool = True,
) -> np.ndarray:
    """Return a boolean sparse-ridge mask selected by value percentile.

    With ``positive_only=True``, zero and negative values are never selected,
    which keeps all-zero fault-likelihood volumes from becoming all-ridge masks.
    """
    return top_percentile_mask(
        np.asarray(volume),
        percentile,
        positive_only=positive_only,
    )


def save_ridge_overlay_slice(
    output_path: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    axis: str | int,
    index: int,
    percentile: float = 99.0,
    buffer_radius: float = 0.0,
    title: str | None = None,
) -> Path:
    """Save a static RGB ridge-overlap overlay for one 3D slice.

    Color policy is local to this helper: reference-only ridge samples are red,
    candidate-only samples are blue, exact overlap is white, buffered matches
    are cyan, and the background is black.
    """
    masks = _ridge_overlay_masks(
        reference,
        candidate,
        percentile=percentile,
        buffer_radius=buffer_radius,
    )
    axis_name, _ = _normalize_axis(axis)
    rgb = _ridge_overlay_rgb(
        slice_2d(masks["reference"], axis_name, index),
        slice_2d(masks["candidate"], axis_name, index),
        reference_buffer=slice_2d(masks["reference_buffer"], axis_name, index),
        candidate_buffer=slice_2d(masks["candidate_buffer"], axis_name, index),
        has_buffer=buffer_radius > 0.0,
    )
    return _save_ridge_overlay_rgb(
        output_path,
        rgb,
        title=title if title is not None else f"ridge overlay {axis_name}={index}",
    )


def save_buffered_ridge_overlay_slices(
    output_dir: str | Path,
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    name: str,
    slice_indices: dict[str, int] | None = None,
    percentile: float = 99.0,
    buffer_radius: float = 2.0,
) -> dict[str, Path]:
    """Save buffered ridge-overlap overlays for the center slice of each axis."""
    masks = _ridge_overlay_masks(
        reference,
        candidate,
        percentile=percentile,
        buffer_radius=buffer_radius,
    )
    indices = select_center_slices(masks["reference"].shape)
    if slice_indices is not None:
        for axis, index in slice_indices.items():
            axis_name, _ = _normalize_axis(axis)
            indices[axis_name] = index

    output_path = ensure_output_dir(output_dir)
    written: dict[str, Path] = {}
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        rgb = _ridge_overlay_rgb(
            slice_2d(masks["reference"], axis, index),
            slice_2d(masks["candidate"], axis, index),
            reference_buffer=slice_2d(masks["reference_buffer"], axis, index),
            candidate_buffer=slice_2d(masks["candidate_buffer"], axis, index),
            has_buffer=buffer_radius > 0.0,
        )
        panel_path = output_path / f"{name}_ridge_overlay_{axis}_{index}.png"
        written[axis] = _save_ridge_overlay_rgb(
            panel_path,
            rgb,
            title=f"{name} ridge overlay {axis}={index}",
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


def _validate_volume_pair(
    reference: ArrayLike, candidate: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    reference_values = np.asarray(reference, dtype=np.float32)
    candidate_values = np.asarray(candidate, dtype=np.float32)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference_values.ndim != 3:
        raise ValueError("reference and candidate must be 3D (n3, n2, n1) arrays")
    if any(size <= 0 for size in reference_values.shape):
        raise ValueError("reference and candidate dimensions must be positive")
    return reference_values, candidate_values


def _finite_values(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return finite.astype(np.float32, copy=False)


def _validate_value_range(value_range: tuple[float, float]) -> None:
    if len(value_range) != 2:
        raise ValueError("value_range must contain two values")
    low, high = value_range
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("value_range values must be finite")
    if high <= low:
        raise ValueError("value_range high value must be greater than low value")


def _ridge_overlay_masks(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    percentile: float,
    buffer_radius: float,
) -> dict[str, np.ndarray]:
    _validate_buffer_radius(buffer_radius)
    reference_values = np.asarray(reference)
    candidate_values = np.asarray(candidate)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference_values.ndim != 3:
        raise ValueError("reference and candidate must be 3D (n3, n2, n1) arrays")

    reference_mask = ridge_mask(reference_values, percentile=percentile)
    candidate_mask = ridge_mask(candidate_values, percentile=percentile)
    reference_buffer = _dilate_ridge_mask(reference_mask, buffer_radius)
    candidate_buffer = _dilate_ridge_mask(candidate_mask, buffer_radius)
    return {
        "reference": reference_mask,
        "candidate": candidate_mask,
        "reference_buffer": reference_buffer,
        "candidate_buffer": candidate_buffer,
    }


def _ridge_overlay_rgb(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    reference_buffer: np.ndarray,
    candidate_buffer: np.ndarray,
    has_buffer: bool,
) -> np.ndarray:
    reference_values = np.asarray(reference_mask, dtype=bool)
    candidate_values = np.asarray(candidate_mask, dtype=bool)
    reference_buffer_values = np.asarray(reference_buffer, dtype=bool)
    candidate_buffer_values = np.asarray(candidate_buffer, dtype=bool)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("reference and candidate masks must have the same shape")
    if reference_values.shape != reference_buffer_values.shape:
        raise ValueError("reference mask and buffer must have the same shape")
    if candidate_values.shape != candidate_buffer_values.shape:
        raise ValueError("candidate mask and buffer must have the same shape")
    if reference_values.ndim != 2:
        raise ValueError("ridge overlay masks must be 2D")

    exact_overlap = reference_values & candidate_values
    buffered_match = np.zeros(reference_values.shape, dtype=bool)
    if has_buffer:
        candidate_in_reference_buffer = candidate_values & reference_buffer_values
        reference_in_candidate_buffer = reference_values & candidate_buffer_values
        buffered_match = (candidate_in_reference_buffer | reference_in_candidate_buffer) & (
            ~exact_overlap
        )

    reference_only = reference_values & (~candidate_values) & (~buffered_match)
    candidate_only = candidate_values & (~reference_values) & (~buffered_match)

    rgb = np.zeros(reference_values.shape + (3,), dtype=np.float32)
    rgb[reference_only] = _RIDGE_REFERENCE_ONLY_RGB
    rgb[candidate_only] = _RIDGE_CANDIDATE_ONLY_RGB
    rgb[buffered_match] = _RIDGE_BUFFERED_MATCH_RGB
    rgb[exact_overlap] = _RIDGE_EXACT_OVERLAP_RGB
    return rgb


def _save_ridge_overlay_rgb(
    output_path: str | Path,
    rgb: np.ndarray,
    *,
    title: str | None,
) -> Path:
    output_file = Path(output_path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    try:
        if title is not None:
            ax.set_title(title)
        ax.imshow(rgb, origin="upper", aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)

    return output_file


def _validate_buffer_radius(buffer_radius: float) -> None:
    if not np.isfinite(buffer_radius):
        raise ValueError("buffer_radius must be finite")
    if buffer_radius < 0.0:
        raise ValueError("buffer_radius must be non-negative")
