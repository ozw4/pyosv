"""Optional static-visualization helpers."""

from __future__ import annotations

import os
import sys
from numbers import Integral, Real
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

_PUBLIC_FVT_COLOR = "#ff453a"
_BASELINE_FVT_COLOR = "#00c7ff"
_CANDIDATE_FVT_COLOR = "#ffd60a"
_STANDARD_FVT_LINEWIDTH = 0.8
_CANDIDATE_FVT_LINEWIDTH = 2.0
_CONTEXT_FVT_COLOR = "#ff9f0a"
_RIDGE_OVERLAP_COLOR = "#ffffff"
_OUTLIER_POINT_COLOR = "#ff2dff"
_NEAREST_PUBLIC_POINT_COLOR = "#30d158"

_PLANE_AXES = {
    "i3": (0, 1, 2),
    "i2": (1, 0, 2),
    "i1": (2, 0, 1),
}
_ARRAY_AXIS_NAMES = ("i3", "i2", "i1")


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


def save_outlier_orthogonal_amplitude_overlay(
    output_path: str | Path,
    *,
    amplitude: ArrayLike,
    public_fvt_mask: ArrayLike,
    baseline_fvt_mask: ArrayLike,
    candidate_fvt_mask: ArrayLike,
    representative_coordinate: tuple[int, int, int],
    nearest_public_coordinate: tuple[int, int, int],
    crop_global_start: tuple[int, int, int],
    amplitude_clip: float,
    window_radius: int,
    crop_index: int,
    component_id: int,
    distance_to_public: float,
) -> Path:
    """Save a 3-by-5 signed-amplitude review at an outlier's three planes.

    All coordinates are ordered ``(i3, i2, i1)``. Ridge inputs are precomputed
    boolean display masks; their metric-versus-display percentile convention and
    interior ROI are selected by the report orchestrator. The nearest-public
    marker is projected onto each representative-point plane; the fixed-axis
    offset is shown in the corresponding row label.
    """

    values, masks = _validate_amplitude_and_masks(
        amplitude,
        {
            "public": public_fvt_mask,
            "baseline": baseline_fvt_mask,
            "candidate": candidate_fvt_mask,
        },
    )
    representative = _validate_coordinate(
        representative_coordinate,
        values.shape,
        "representative_coordinate",
    )
    nearest_public = _validate_coordinate(
        nearest_public_coordinate,
        values.shape,
        "nearest_public_coordinate",
    )
    global_start = _validate_global_start(crop_global_start)
    clip = _validate_positive_finite_value(amplitude_clip, "amplitude_clip")
    radius = _validate_nonnegative_integer(window_radius, "window_radius")
    crop_number = _validate_positive_integer(crop_index, "crop_index")
    component_number = _validate_positive_integer(component_id, "component_id")
    distance = _validate_nonnegative_finite_value(distance_to_public, "distance_to_public")

    output_file = _prepare_output_file(output_path)
    plt = require_matplotlib()
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(18.0, 10.5),
        squeeze=False,
        constrained_layout=True,
    )
    columns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("seismic amplitude", ()),
        ("seismic + public FVT top 1% ridge", ("public",)),
        ("seismic + baseline FVT top 1% ridge", ("baseline",)),
        ("seismic + candidate FVT top 5% ridge (display)", ("candidate",)),
        ("seismic + combined ridges", ("public", "baseline", "candidate")),
    )
    colors = {
        "public": _PUBLIC_FVT_COLOR,
        "baseline": _BASELINE_FVT_COLOR,
        "candidate": _CANDIDATE_FVT_COLOR,
    }
    linewidths = {
        "public": _STANDARD_FVT_LINEWIDTH,
        "baseline": _STANDARD_FVT_LINEWIDTH,
        "candidate": _CANDIDATE_FVT_LINEWIDTH,
    }
    global_representative = _global_coordinate(representative, global_start)
    try:
        fig.suptitle(
            f"crop {crop_number:03d}, component {component_number:03d}; "
            f"outlier global={global_representative}, distance={distance:.6g} samples"
        )
        for row, axis_name in enumerate(("i3", "i2", "i1")):
            fixed_axis, row_axis, column_axis = _PLANE_AXES[axis_name]
            row_slice, column_slice = _plane_window_slices(
                values.shape,
                representative,
                fixed_axis=fixed_axis,
                radius=radius,
                include=nearest_public,
            )
            fixed_index = representative[fixed_axis]
            fixed_global = global_start[fixed_axis] + fixed_index
            nearest_offset = nearest_public[fixed_axis] - fixed_index
            for column, (panel_title, overlay_names) in enumerate(columns):
                ax = axes[row, column]
                _draw_amplitude_plane(
                    ax,
                    values,
                    axis_name=axis_name,
                    fixed_index=fixed_index,
                    row_slice=row_slice,
                    column_slice=column_slice,
                    global_start=global_start,
                    amplitude_clip=clip,
                )
                for overlay_name in overlay_names:
                    _draw_mask_plane(
                        ax,
                        masks[overlay_name],
                        axis_name=axis_name,
                        fixed_index=fixed_index,
                        row_slice=row_slice,
                        column_slice=column_slice,
                        global_start=global_start,
                        color=colors[overlay_name],
                        linewidth=linewidths[overlay_name],
                    )
                _draw_projected_point_markers(
                    ax,
                    representative=representative,
                    nearest_public=nearest_public,
                    row_axis=row_axis,
                    column_axis=column_axis,
                    global_start=global_start,
                    show_representative=True,
                    show_nearest=True,
                )
                ax.set_title(panel_title)
                if column == 0:
                    offset_text = (
                        "on plane"
                        if nearest_offset == 0
                        else f"nearest-public projection Δ{axis_name}={nearest_offset:+d}"
                    )
                    ax.set_ylabel(
                        f"global {axis_name}={fixed_global}\n{offset_text}\n"
                        f"global {_ARRAY_AXIS_NAMES[row_axis]}"
                    )
        _add_bottom_figure_legend(
            fig,
            _outlier_legend_handles(include_all_ridges=True),
            columns=5,
        )
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)
    return output_file


def save_outlier_adjacent_slice_overlay(
    output_path: str | Path,
    *,
    amplitude: ArrayLike,
    public_fvt_mask: ArrayLike,
    baseline_fvt_mask: ArrayLike,
    candidate_fvt_mask: ArrayLike,
    representative_coordinate: tuple[int, int, int],
    nearest_public_coordinate: tuple[int, int, int],
    crop_global_start: tuple[int, int, int],
    amplitude_clip: float,
    window_radius: int,
    adjacent_slice_radius: int,
    axis: str | int,
    crop_index: int,
    component_id: int,
) -> Path:
    """Save signed-amplitude combined overlays on neighboring slices of one axis."""

    values, masks = _validate_amplitude_and_masks(
        amplitude,
        {
            "public": public_fvt_mask,
            "baseline": baseline_fvt_mask,
            "candidate": candidate_fvt_mask,
        },
    )
    representative = _validate_coordinate(
        representative_coordinate,
        values.shape,
        "representative_coordinate",
    )
    nearest_public = _validate_coordinate(
        nearest_public_coordinate,
        values.shape,
        "nearest_public_coordinate",
    )
    global_start = _validate_global_start(crop_global_start)
    clip = _validate_positive_finite_value(amplitude_clip, "amplitude_clip")
    radius = _validate_nonnegative_integer(window_radius, "window_radius")
    adjacent_radius = _validate_nonnegative_integer(
        adjacent_slice_radius,
        "adjacent_slice_radius",
    )
    crop_number = _validate_positive_integer(crop_index, "crop_index")
    component_number = _validate_positive_integer(component_id, "component_id")
    axis_name, fixed_axis = _normalize_axis(axis)
    _, row_axis, column_axis = _PLANE_AXES[axis_name]

    center_index = representative[fixed_axis]
    first_index = max(0, center_index - adjacent_radius)
    stop_index = min(values.shape[fixed_axis], center_index + adjacent_radius + 1)
    fixed_indices = list(range(first_index, stop_index))
    row_slice, column_slice = _plane_window_slices(
        values.shape,
        representative,
        fixed_axis=fixed_axis,
        radius=radius,
        include=nearest_public,
    )

    output_file = _prepare_output_file(output_path)
    plt = require_matplotlib()
    column_count = min(7, len(fixed_indices))
    row_count = (len(fixed_indices) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.25 * column_count, 3.4 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "public": _PUBLIC_FVT_COLOR,
        "baseline": _BASELINE_FVT_COLOR,
        "candidate": _CANDIDATE_FVT_COLOR,
    }
    linewidths = {
        "public": _STANDARD_FVT_LINEWIDTH,
        "baseline": _STANDARD_FVT_LINEWIDTH,
        "candidate": _CANDIDATE_FVT_LINEWIDTH,
    }
    try:
        fig.suptitle(
            f"crop {crop_number:03d}, component {component_number:03d}; "
            f"adjacent {axis_name} continuity"
        )
        for panel_number, fixed_index in enumerate(fixed_indices):
            ax = axes.flat[panel_number]
            _draw_amplitude_plane(
                ax,
                values,
                axis_name=axis_name,
                fixed_index=fixed_index,
                row_slice=row_slice,
                column_slice=column_slice,
                global_start=global_start,
                amplitude_clip=clip,
            )
            for overlay_name in ("public", "baseline", "candidate"):
                _draw_mask_plane(
                    ax,
                    masks[overlay_name],
                    axis_name=axis_name,
                    fixed_index=fixed_index,
                    row_slice=row_slice,
                    column_slice=column_slice,
                    global_start=global_start,
                    color=colors[overlay_name],
                    linewidth=linewidths[overlay_name],
                )
            _draw_projected_point_markers(
                ax,
                representative=representative,
                nearest_public=nearest_public,
                row_axis=row_axis,
                column_axis=column_axis,
                global_start=global_start,
                show_representative=fixed_index == representative[fixed_axis],
                show_nearest=fixed_index == nearest_public[fixed_axis],
            )
            fixed_global = global_start[fixed_axis] + fixed_index
            suffix = " (representative)" if fixed_index == center_index else ""
            ax.set_title(f"global {axis_name}={fixed_global}{suffix}")
        for ax in axes.flat[len(fixed_indices) :]:
            ax.set_visible(False)
        _add_bottom_figure_legend(
            fig,
            _outlier_legend_handles(include_all_ridges=True),
            columns=5,
        )
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)
    return output_file


def save_context_orthogonal_amplitude_comparison(
    output_path: str | Path,
    *,
    amplitude: ArrayLike,
    base_candidate_fvt_mask: ArrayLike,
    context_candidate_fvt_mask: ArrayLike,
    representative_coordinate: tuple[int, int, int],
    nearest_public_coordinate: tuple[int, int, int],
    crop_global_start: tuple[int, int, int],
    amplitude_clip: float,
    window_radius: int,
    crop_index: int,
    component_id: int,
) -> Path:
    """Save a 3-by-4 same-global-ROI base/context amplitude comparison."""

    values, masks = _validate_amplitude_and_masks(
        amplitude,
        {
            "base": base_candidate_fvt_mask,
            "context": context_candidate_fvt_mask,
        },
    )
    representative = _validate_coordinate(
        representative_coordinate,
        values.shape,
        "representative_coordinate",
    )
    nearest_public = _validate_coordinate(
        nearest_public_coordinate,
        values.shape,
        "nearest_public_coordinate",
    )
    global_start = _validate_global_start(crop_global_start)
    clip = _validate_positive_finite_value(amplitude_clip, "amplitude_clip")
    radius = _validate_nonnegative_integer(window_radius, "window_radius")
    crop_number = _validate_positive_integer(crop_index, "crop_index")
    component_number = _validate_positive_integer(component_id, "component_id")

    base_only = masks["base"] & (~masks["context"])
    context_only = masks["context"] & (~masks["base"])
    overlap = masks["base"] & masks["context"]
    output_file = _prepare_output_file(output_path)
    plt = require_matplotlib()
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15.0, 10.5),
        squeeze=False,
        constrained_layout=True,
    )
    global_representative = _global_coordinate(representative, global_start)
    try:
        fig.suptitle(
            f"crop {crop_number:03d}, component {component_number:03d}; "
            f"base/context same-global ROI at {global_representative}"
        )
        for row, axis_name in enumerate(("i3", "i2", "i1")):
            fixed_axis, row_axis, column_axis = _PLANE_AXES[axis_name]
            row_slice, column_slice = _plane_window_slices(
                values.shape,
                representative,
                fixed_axis=fixed_axis,
                radius=radius,
                include=nearest_public,
            )
            fixed_index = representative[fixed_axis]
            panels: tuple[tuple[str, tuple[tuple[np.ndarray, str], ...]], ...] = (
                (
                    "base candidate FVT top 5% ridge (display)",
                    ((masks["base"], _BASELINE_FVT_COLOR),),
                ),
                (
                    "context candidate FVT top 5% ridge (display)",
                    ((masks["context"], _CONTEXT_FVT_COLOR),),
                ),
                (
                    "base/context top 5% combined",
                    (
                        (base_only, _BASELINE_FVT_COLOR),
                        (context_only, _CONTEXT_FVT_COLOR),
                        (overlap, _RIDGE_OVERLAP_COLOR),
                    ),
                ),
                (
                    "base-only / context-only top 5%",
                    (
                        (base_only, _BASELINE_FVT_COLOR),
                        (context_only, _CONTEXT_FVT_COLOR),
                    ),
                ),
            )
            for column, (panel_title, overlays) in enumerate(panels):
                ax = axes[row, column]
                _draw_amplitude_plane(
                    ax,
                    values,
                    axis_name=axis_name,
                    fixed_index=fixed_index,
                    row_slice=row_slice,
                    column_slice=column_slice,
                    global_start=global_start,
                    amplitude_clip=clip,
                )
                for mask, color in overlays:
                    _draw_mask_plane(
                        ax,
                        mask,
                        axis_name=axis_name,
                        fixed_index=fixed_index,
                        row_slice=row_slice,
                        column_slice=column_slice,
                        global_start=global_start,
                        color=color,
                        linewidth=_CANDIDATE_FVT_LINEWIDTH,
                    )
                _draw_projected_point_markers(
                    ax,
                    representative=representative,
                    nearest_public=nearest_public,
                    row_axis=row_axis,
                    column_axis=column_axis,
                    global_start=global_start,
                    show_representative=True,
                    show_nearest=True,
                )
                ax.set_title(panel_title)
                if column == 0:
                    fixed_global = global_start[fixed_axis] + fixed_index
                    ax.set_ylabel(
                        f"global {axis_name}={fixed_global}\nglobal {_ARRAY_AXIS_NAMES[row_axis]}"
                    )
        _add_bottom_figure_legend(fig, _context_legend_handles(), columns=5)
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)
    return output_file


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


def _validate_amplitude_and_masks(
    amplitude: ArrayLike,
    masks: dict[str, ArrayLike],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(amplitude, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("amplitude must be a 3D (n3, n2, n1) array")
    if any(size <= 0 for size in values.shape):
        raise ValueError("amplitude dimensions must be positive")

    validated: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        mask_values = np.asarray(mask)
        if mask_values.shape != values.shape:
            raise ValueError(f"{name}_fvt_mask must have the same shape as amplitude")
        if mask_values.ndim != 3:
            raise ValueError(f"{name}_fvt_mask must be a 3D (n3, n2, n1) array")
        validated[name] = mask_values.astype(bool, copy=False)
    return values, validated


def _validate_coordinate(
    coordinate: tuple[int, int, int],
    shape: tuple[int, int, int],
    name: str,
) -> tuple[int, int, int]:
    if len(coordinate) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    validated: list[int] = []
    for axis, (index, size) in enumerate(zip(coordinate, shape, strict=True)):
        if not isinstance(index, Integral) or isinstance(index, bool):
            raise TypeError(f"{name}[{axis}] must be an integer")
        integer = int(index)
        if integer < 0 or integer >= size:
            raise ValueError(f"{name}[{axis}] must be between 0 and {size - 1}")
        validated.append(integer)
    return tuple(validated)


def _validate_global_start(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError("crop_global_start must contain exactly three coordinates")
    starts: list[int] = []
    for axis, start in enumerate(value):
        if not isinstance(start, Integral) or isinstance(start, bool):
            raise TypeError(f"crop_global_start[{axis}] must be an integer")
        integer = int(start)
        if integer < 0:
            raise ValueError(f"crop_global_start[{axis}] must be non-negative")
        starts.append(integer)
    return tuple(starts)


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _validate_positive_finite_value(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _validate_nonnegative_finite_value(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _prepare_output_file(path: str | Path) -> Path:
    output_file = Path(path)
    if output_file.parent != Path(""):
        output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


def _plane_window_slices(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    *,
    fixed_axis: int,
    radius: int,
    include: tuple[int, int, int],
) -> tuple[slice, slice]:
    _, row_axis, column_axis = _PLANE_AXES[_ARRAY_AXIS_NAMES[fixed_axis]]

    def bounds(axis: int) -> slice:
        start = max(0, center[axis] - radius)
        stop = min(shape[axis], center[axis] + radius + 1)
        # The requested radius is the minimum window. Expand only when needed
        # so the projected nearest-public marker is always reviewable.
        start = min(start, include[axis])
        stop = max(stop, include[axis] + 1)
        return slice(start, stop)

    return bounds(row_axis), bounds(column_axis)


def _global_coordinate(
    local: tuple[int, int, int],
    start: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(index + offset for index, offset in zip(local, start, strict=True))


def _plane_extent(
    *,
    row_slice: slice,
    column_slice: slice,
    row_axis: int,
    column_axis: int,
    global_start: tuple[int, int, int],
) -> tuple[float, float, float, float]:
    left = global_start[column_axis] + int(column_slice.start) - 0.5
    right = global_start[column_axis] + int(column_slice.stop) - 0.5
    top = global_start[row_axis] + int(row_slice.start) - 0.5
    bottom = global_start[row_axis] + int(row_slice.stop) - 0.5
    return float(left), float(right), float(bottom), float(top)


def _draw_amplitude_plane(
    ax: Any,
    amplitude: np.ndarray,
    *,
    axis_name: str,
    fixed_index: int,
    row_slice: slice,
    column_slice: slice,
    global_start: tuple[int, int, int],
    amplitude_clip: float,
) -> None:
    _, row_axis, column_axis = _PLANE_AXES[axis_name]
    panel = slice_2d(amplitude, axis_name, fixed_index)[row_slice, column_slice]
    display = np.where(np.isfinite(panel), panel, np.float32(0.0))
    extent = _plane_extent(
        row_slice=row_slice,
        column_slice=column_slice,
        row_axis=row_axis,
        column_axis=column_axis,
        global_start=global_start,
    )
    ax.imshow(
        display,
        cmap="gray",
        vmin=-amplitude_clip,
        vmax=amplitude_clip,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )
    ax.set_xlabel(f"global {_ARRAY_AXIS_NAMES[column_axis]}")
    ax.set_ylabel(f"global {_ARRAY_AXIS_NAMES[row_axis]}")


def _draw_mask_plane(
    ax: Any,
    mask: np.ndarray,
    *,
    axis_name: str,
    fixed_index: int,
    row_slice: slice,
    column_slice: slice,
    global_start: tuple[int, int, int],
    color: str,
    linewidth: float = 0.8,
) -> None:
    line_width = _validate_positive_finite_value(linewidth, "linewidth")
    _, row_axis, column_axis = _PLANE_AXES[axis_name]
    panel = slice_2d(mask, axis_name, fixed_index)[row_slice, column_slice].astype(
        bool,
        copy=False,
    )
    if not np.any(panel):
        return

    x = np.arange(column_slice.start, column_slice.stop, dtype=np.float64)
    x += global_start[column_axis]
    y = np.arange(row_slice.start, row_slice.stop, dtype=np.float64)
    y += global_start[row_axis]
    if min(panel.shape) >= 2 and not np.all(panel):
        ax.contour(
            x,
            y,
            panel.astype(np.float32, copy=False),
            levels=(0.5,),
            colors=(color,),
            linewidths=line_width,
            alpha=0.95,
        )
        return

    # ``contour`` needs at least a 2-by-2 non-constant panel. A small hollow
    # square keeps radius-zero and crop-edge reviews valid without hiding xs.
    rows, columns = np.nonzero(panel)
    ax.scatter(
        x[columns],
        y[rows],
        s=12,
        marker="s",
        facecolors="none",
        edgecolors=color,
        linewidths=line_width,
        zorder=3,
    )


def _draw_projected_point_markers(
    ax: Any,
    *,
    representative: tuple[int, int, int],
    nearest_public: tuple[int, int, int],
    row_axis: int,
    column_axis: int,
    global_start: tuple[int, int, int],
    show_representative: bool,
    show_nearest: bool,
) -> None:
    if show_representative:
        ax.scatter(
            representative[column_axis] + global_start[column_axis],
            representative[row_axis] + global_start[row_axis],
            s=64,
            marker="*",
            c=_OUTLIER_POINT_COLOR,
            edgecolors="#111111",
            linewidths=0.6,
            zorder=5,
        )
    if show_nearest:
        ax.scatter(
            nearest_public[column_axis] + global_start[column_axis],
            nearest_public[row_axis] + global_start[row_axis],
            s=48,
            marker="x",
            c=_NEAREST_PUBLIC_POINT_COLOR,
            linewidths=1.2,
            zorder=5,
        )


def _add_bottom_figure_legend(fig: Any, handles: list[Any], *, columns: int) -> None:
    try:
        fig.legend(handles=handles, loc="outside lower center", ncol=columns)
    except ValueError:
        # Matplotlib before 3.7 does not support the ``outside`` location. Keep
        # the optional visualization dependency usable there as well.
        fig.legend(handles=handles, loc="lower center", ncol=columns)


def _outlier_legend_handles(*, include_all_ridges: bool) -> list[Any]:
    from matplotlib.lines import Line2D

    handles: list[Any] = []
    if include_all_ridges:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=_PUBLIC_FVT_COLOR,
                    lw=_STANDARD_FVT_LINEWIDTH,
                    label="public FVT top 1% ridge",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_BASELINE_FVT_COLOR,
                    lw=_STANDARD_FVT_LINEWIDTH,
                    label="baseline FVT top 1% ridge",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_CANDIDATE_FVT_COLOR,
                    lw=_CANDIDATE_FVT_LINEWIDTH,
                    label="candidate FVT top 5% ridge (display)",
                ),
            ]
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=_OUTLIER_POINT_COLOR,
                marker="*",
                markerfacecolor=_OUTLIER_POINT_COLOR,
                markeredgecolor="#111111",
                linestyle="none",
                markersize=9,
                label="representative outlier",
            ),
            Line2D(
                [0],
                [0],
                color=_NEAREST_PUBLIC_POINT_COLOR,
                marker="x",
                linestyle="none",
                markersize=7,
                label="nearest public FVT point",
            ),
        ]
    )
    return handles


def _context_legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=_BASELINE_FVT_COLOR,
            lw=_CANDIDATE_FVT_LINEWIDTH,
            label="base candidate FVT top 5% ridge (display)",
        ),
        Line2D(
            [0],
            [0],
            color=_CONTEXT_FVT_COLOR,
            lw=_CANDIDATE_FVT_LINEWIDTH,
            label="context candidate FVT top 5% ridge (display)",
        ),
        Line2D(
            [0],
            [0],
            color=_RIDGE_OVERLAP_COLOR,
            lw=_CANDIDATE_FVT_LINEWIDTH,
            label="exact top 5% mask overlap",
        ),
        *_outlier_legend_handles(include_all_ridges=False),
    ]


def _validate_buffer_radius(buffer_radius: float) -> None:
    if not np.isfinite(buffer_radius):
        raise ValueError("buffer_radius must be finite")
    if buffer_radius < 0.0:
        raise ValueError("buffer_radius must be non-negative")
